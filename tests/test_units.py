"""CPU checks of FLUX.2 [klein] and of everything both families share that need no weights (a tiny random trunk
stands in for FLUX.2; tests/test_units_flux3.py does the same for FLUX 3).

    python tests/test_units.py                         # everything
    FLUX2_ROOT=/path/to/black-forest-labs/flux2 python tests/test_units.py    # also compares with BFL's model.py

  layouts   schedule end points, realizable token counts, exact budgets, tilings, the training mixture
  trunk     this code == BFL's reference model == diffusers' Flux2Transformer2DModel (same random weights)
  model     dense layout == the released model; zero-init == mean pooling in, upsampling + skip out;
            padding, fused guidance, last-block shortcut, checkpointing and gradients
  convert   the dit/ export names and shapes of the full-size model, and the round trip
  train     how a step's examples are grouped into student passes, the EMA shards, checkpoints, the validation suite
  dmd       stage 2 on a tiny trunk: the LoRA taps, the discriminator's features, rollouts, both losses' gradients
"""
import importlib.util
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch
import torch.nn.functional as F

from budgetlatent.models import convert
from budgetlatent.models import added_modules as M
from budgetlatent.models.added_modules import hf_skip_gain
from budgetlatent.models.budgetpix import BPConfig, BudgetPixFlux2, count_parameters
from budgetlatent.models.flux2_trunk import Flux2Trunk, TrunkConfig, rope_tables
from budgetlatent.tokenization import layout as L

TINY = TrunkConfig(in_channels=16, context_in_dim=40, hidden_size=96, num_heads=3, depth=2, depth_single_blocks=3,
                   axes_dim=(8, 8, 8, 8), theta=2000, mlp_ratio=3.0)
OK = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAILED {name} {detail}")
    OK.append(name)
    print(f"  ok  {name} {detail}", flush=True)


def rand_pyr(b, gen):
    """Entropy-like maps with ties and flat regions."""
    pyr = {}
    for s in L.COARSE:
        g = L.GRID // s
        v = 9.0 * torch.rand(b, g, g, generator=gen)
        v = torch.where(torch.rand(b, g, g, generator=gen) < 0.25, torch.zeros_like(v), v)      # flat patches (ties)
        pyr[s] = v
    return pyr


# ---------------------------------------------------------------------------------------------------------
def test_layouts():
    print("[layouts]")
    ts = L.flux2_klein_timesteps(50, 1024)
    check("mu(1024, 50)", abs(L.compute_empirical_mu(1024, 50) - 1.701956) < 1e-6, f"{L.compute_empirical_mu(1024, 50):.6f}")
    check("schedule start", all(abs(a - b) < 1e-6 for a, b in zip(ts[:4], [0.0, 0.003707, 0.007540, 0.011504])), str([round(v, 6) for v in ts[:4]]))
    check("schedule end", all(abs(a - b) < 1e-6 for a, b in zip(ts[-4:], [0.740694, 0.813982, 0.899336, 1.0])), str([round(v, 6) for v in ts[-4:]]))
    check("schedule monotone", len(ts) == 51 and all(b > a for a, b in zip(ts, ts[1:])))
    table = {1.0: 1024, 0.9: 922, 0.8: 820, 0.7: 718, 0.6: 613, 0.5: 511, 0.4: 409, 0.3: 307, 0.2: 205, 0.1: 103,
             0.08: 82, 0.06: 61, 0.04: 40, 0.02: 19}
    check("token counts of the 14 budgets", all(L.snap_tokens(b) == n for b, n in table.items()))

    gen = torch.Generator().manual_seed(0)
    pyr = rand_pyr(6, gen)
    for b in list(table) + [0.015625, 0.33, 0.777, 0.999]:
        act = L.build_exact_budget_layout_s8(pyr, b)
        lay = L.pack(act, check=True)
        assert bool((lay.n == L.snap_tokens(b)).all()), (b, lay.n)
        assert not lay.padded
    check("exact budget layouts", True, "N = 16 + 3k for every image, tiling valid")
    for n in (16, 19, 1021, 1024):
        act = L.exact_layout(pyr, torch.full((6,), n))
        assert bool((L.count_tokens(act) == n).all())
    a16, a1024 = L.exact_layout(pyr, torch.full((6,), 16)), L.exact_layout(pyr, torch.full((6,), 1024))
    check("exact layout extremes", bool(a16[8].all()) and bool(a1024[1].all()))
    n_var = torch.tensor([16, 103, 511, 1024, 820, 40])
    check("per-image token counts", bool((L.count_tokens(L.exact_layout(pyr, n_var)) == n_var).all()))
    # degenerate maps: all equal (every rank is a tie) and all zero
    for flat in (0.0, 5.0):
        fp = {s: torch.full((2, L.GRID // s, L.GRID // s), flat) for s in L.COARSE}
        assert bool((L.count_tokens(L.build_exact_budget_layout_s8(fp, 0.5)) == 511).all())
    check("exact budget on flat maps", True)
    # higher budget -> the fine region only grows?  not guaranteed by construction; check the split counts instead
    lay = L.pack(L.threshold_layout(pyr, torch.tensor([-9.0, -1.0, 0.0, 1.0, 4.0, 20.0])), check=True)
    check("threshold layouts", int(lay.n[0]) == 1024 and int(lay.n[-1]) == 16 and bool((lay.n[:-1] >= lay.n[1:]).all()), str(lay.n.tolist()))
    check("padding mask", lay.padded and bool((lay.valid.sum(1) == lay.n).all()) and bool(((lay.scale > 0) == lay.valid).all()))
    cover = torch.zeros(6, L.GRID, L.GRID, dtype=torch.long)
    for s, (bi, ti, gy, gx) in lay.groups.items():
        o = torch.arange(s)
        cover[bi[:, None, None], (gy[:, None] + o)[:, :, None], (gx[:, None] + o)[:, None, :]] += 1
        assert bool((lay.scale[bi, ti] == s).all()) and bool((lay.gy[bi, ti] == gy).all())
    check("groups tile the grid", bool((cover == 1).all()))

    mix = L.LayoutMix()
    gen = torch.Generator().manual_seed(1)
    modes, toks, fields, slack = [], [], [], []
    for _ in range(40):
        pyr = rand_pyr(64, gen)
        plan = L.draw_plan(64, mix, gen)
        act, mode = L.training_layouts(pyr, mix, gen, plan)
        lay = L.pack(act, check=True)
        modes.append(mode)
        toks.append(lay.n)
        fields.append(plan.field)
        slack.append((plan.target - lay.n)[mode == 2])
        assert bool((lay.n[mode == 0] == 1024).all())
        assert bool(torch.isin(lay.n[mode == 1], torch.tensor([16, 64, 256])).all())
        assert bool(((lay.n - 16) % 3 == 0).all()) and bool((plan.field[mode < 2] == 0).all())
    modes, toks, fields, slack = torch.cat(modes), torch.cat(toks).float(), torch.cat(fields), torch.cat(slack).float()
    frac = [float((modes == k).float().mean()) for k in range(4)]
    check("training mixture shares", all(abs(f - p) < 0.03 for f, p in zip(frac, (0.10, 0.10, 0.55, 0.25))), str([round(f, 3) for f in frac]))
    check("budget mode stays within its token target", bool((slack >= 0).all()), f"target - tokens: median {float(slack.median()):.0f}, max {float(slack.max()):.0f}")
    fr = [float((fields[modes >= 2] == k).float().mean()) for k in range(4)]
    check("detail-map sources of the adaptive modes", all(abs(f - p) < 0.03 for f, p in zip(fr, (0.80, 0.067, 0.067, 0.066))),
          ", ".join(f"{n} {100 * f:.1f} %" for n, f in zip(L.FIELDS, fr)))
    print(f"      mean tokens per mode: " + ", ".join(f"{L.MODES[k]} {float(toks[modes == k].mean()):.0f}" for k in range(4))
          + f"; overall {float(toks.mean()):.0f}")
    # a plan drawn first says which images read their detail map; the others may hold anything
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    pyr = rand_pyr(64, torch.Generator().manual_seed(6))
    plan = L.draw_plan(64, mix, g1)
    junk = {s: torch.where(plan.needs_detail[:, None, None], v, torch.full_like(v, float("nan"))) for s, v in pyr.items()}
    a1, _ = L.training_layouts(pyr, mix, g1, plan)
    a2, _ = L.training_layouts(junk, mix, g2, L.draw_plan(64, mix, g2))
    check("layout plan", all(torch.equal(a1[s], a2[s]) for s in L.SCALES) and 0.3 < float(plan.needs_detail.float().mean()) < 0.9,
          f"{100 * float(plan.needs_detail.float().mean()):.0f} % of the images need their detail map")
    plan = L.draw_plan(256, mix, gen)
    aug = L.augmented_pyramid(rand_pyr(256, gen), plan, mix, gen)
    check("augmented detail maps", all(aug[s].shape == (256, L.GRID // s, L.GRID // s) and float(aug[s].min()) >= 0 and float(aug[s].max()) <= 9.0 for s in L.COARSE))
    t, first = L.sample_times(200000, gen)
    check("training times", abs(float(first.float().mean()) - 0.10) < 0.005 and bool((t[first] == 0).all()) and float(t.max()) < 1.0
          and float(t[~first].min()) >= 0.0, f"{100 * float(first.float().mean()):.1f} % at t = 0, median of the rest {float(t[~first].median()):.3f}")

    img = torch.rand(2, 3, 512, 512, generator=gen) * 2 - 1
    img[:, :, :256] = 0.3                                              # a flat half: zero entropy
    pe = L.pixel_entropy_pyramid_s8(img)
    check("pixel entropy", all(pe[s].shape == (2, 32 // s, 32 // s) for s in L.COARSE) and float(pe[2][:, :8].max()) < 1e-6
          and 8.0 < float(pe[8][:, 2:].min()) <= 9.0, f"flat {float(pe[2][:, :8].max()):.3f} bits, noise {float(pe[8][:, 2:].mean()):.2f} bits")
    ne = L.noise_latent_pyramid_s8(torch.randn(2, 32, 32, 128, generator=gen))
    check("noise pyramid", all(ne[s].shape == (2, 32 // s, 32 // s) for s in L.COARSE) and abs(float(ne[8].mean()) - 4.8) < 0.1, f"mean {float(ne[8].mean()):.2f} bits at s = 8")
    ov = L.overlay(img, L.build_exact_budget_layout_s8(pe, 0.3))
    check("overlay", ov.shape == (2, 512, 512, 3) and ov.dtype == torch.uint8)

    s = torch.tensor([1.0, 1.0 / 17.0, 0.0, 0.5], dtype=torch.float64)
    w = hf_skip_gain(s)
    grid = torch.linspace(0, 1, 170001, dtype=torch.float64)
    peak = hf_skip_gain(grid)
    check("wiener gain", abs(float(w[0]) - 1.0) < 1e-6 and abs(float(w[1])) < 1e-6 and abs(float(w[2]) + 1.0) < 1e-6
          and abs(float(peak.max()) - 2.125) < 1e-6 and abs(float(grid[peak.argmax()]) - 5.0 / 17.0) < 1e-4,
          f"w(1) = {float(w[0]):+.4f}, w(1/17) = {float(w[1]):+.1e}, w(0) = {float(w[2]):+.4f}; maximum {float(peak.max()):.4f} at sigma = {float(grid[peak.argmax()]):.4f} (5/17)")


# ---------------------------------------------------------------------------------------------------------
def tiny_inputs(b=2, g=32, gen=None, cfg=TINY):
    gen = gen or torch.Generator().manual_seed(0)
    z = torch.randn(b, g, g, cfg.in_channels, generator=gen)
    t = torch.rand(b, generator=gen) * 0.9 + 0.05
    ctx = torch.randn(b, 12, cfg.context_in_dim, generator=gen)
    return z, t, ctx


def randomize(model, gen, std=0.05):
    """Break the zero initialisations so every added module contributes."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if convert.is_adapter(n):
                p.add_(std * torch.randn(p.shape, generator=gen))
    return model


def test_trunk_references():
    print("[trunk vs references]")
    torch.manual_seed(0)
    trunk = Flux2Trunk(TINY).eval()
    z, t, ctx = tiny_inputs(g=8)
    with torch.no_grad():
        mine = trunk.forward_dense(z, t, ctx)

    root = os.environ.get("FLUX2_ROOT") or os.path.join(os.path.dirname(HERE), "third_party", "flux2")
    path = os.path.join(root, "src", "flux2", "model.py")
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location("bfl_flux2_model", path)
        bfl = importlib.util.module_from_spec(spec)
        sys.modules["bfl_flux2_model"] = bfl
        spec.loader.exec_module(bfl)
        params = bfl.Klein4BParams(in_channels=TINY.in_channels, context_in_dim=TINY.context_in_dim, hidden_size=TINY.hidden_size,
                                   num_heads=TINY.num_heads, depth=TINY.depth, depth_single_blocks=TINY.depth_single_blocks,
                                   axes_dim=list(TINY.axes_dim), theta=TINY.theta, mlp_ratio=TINY.mlp_ratio, use_guidance_embed=False)
        ref = bfl.Flux2(params).eval()
        missing, unexpected = ref.load_state_dict(trunk.state_dict(), strict=True)
        b, g = z.shape[0], z.shape[1]
        x_ids = torch.cartesian_prod(torch.arange(1), torch.arange(g), torch.arange(g), torch.arange(1))[None].expand(b, -1, -1)
        c_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(ctx.shape[1]))[None].expand(b, -1, -1)
        with torch.no_grad():
            raw = ref(x=z.flatten(1, 2), x_ids=x_ids, timesteps=1.0 - t, ctx=ctx, ctx_ids=c_ids, guidance=None)
        d = float((mine + raw.view_as(mine)).abs().max())
        check("== BFL reference model.py (strict state dict)", d < 2e-5, f"max |diff| {d:.1e} (output scale {float(mine.abs().max()):.2f})")
    else:
        print(f"  --  BFL reference not found at {path}; skipped")

    try:
        from diffusers import Flux2Transformer2DModel
    except Exception as e:                                             # pragma: no cover
        print(f"  --  diffusers not importable ({e}); skipped")
        return
    dm = Flux2Transformer2DModel(patch_size=1, in_channels=TINY.in_channels, num_layers=TINY.depth,
                                 num_single_layers=TINY.depth_single_blocks, attention_head_dim=TINY.hidden_size // TINY.num_heads,
                                 num_attention_heads=TINY.num_heads, joint_attention_dim=TINY.context_in_dim,
                                 timestep_guidance_channels=256, mlp_ratio=TINY.mlp_ratio, axes_dims_rope=TINY.axes_dim,
                                 rope_theta=TINY.theta).eval()
    sd = dm.state_dict()
    guidance = [k for k in sd if "guidance_embedder" in k]             # klein has no guidance embedding: silence it
    with torch.no_grad():
        for k in guidance:
            if k.endswith("linear_2.weight"):
                sd[k].zero_()
    conv = convert.diffusers_to_bfl({k: v for k, v in sd.items() if k not in guidance})
    trunk2 = Flux2Trunk(TINY).eval()
    trunk2.load_state_dict(conv, strict=True)
    b, g = z.shape[0], z.shape[1]
    img_ids = torch.cartesian_prod(torch.arange(1), torch.arange(g), torch.arange(g), torch.arange(1)).float()
    txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(ctx.shape[1])).float()
    with torch.no_grad():
        raw = dm(hidden_states=z.flatten(1, 2), encoder_hidden_states=ctx, timestep=1.0 - t, img_ids=img_ids, txt_ids=txt_ids,
                 guidance=torch.zeros(b), return_dict=False)[0]
        mine2 = trunk2.forward_dense(z, t, ctx)
    d = float((mine2 + raw.view_as(mine2)).abs().max())
    check("== diffusers Flux2Transformer2DModel (converted keys)", d < 2e-5, f"max |diff| {d:.1e} (output scale {float(mine2.abs().max()):.2f})")


# ---------------------------------------------------------------------------------------------------------
def test_model():
    print("[model]")
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    model = BudgetPixFlux2(TINY, BPConfig()).eval()
    z, t, ctx = tiny_inputs(b=3, gen=gen)
    with torch.no_grad():
        dense = model.forward_dense(z, t, ctx)
        d = float((model(z, t, ctx, L.pack(L.dense_layout(3))) - dense).abs().max())
    check("dense layout == released model", d == 0.0, f"max |diff| {d:.1e}")

    # zero init: mean pooling in, bilinear (= constant per token) + Wiener skip out
    with torch.no_grad():
        for s in L.COARSE:
            lay = L.pack(L.uniform_layout(3, s))
            got = model(z, t, ctx, lay)
            g = L.GRID // s
            cells = model.img_in(z.flatten(1, 2)).view(3, L.GRID, L.GRID, -1)
            tok = cells.view(3, g, s, g, s, -1).mean(dim=(2, 4)).flatten(1, 2)
            sigma = 1.0 - t
            vec = model.time_vec(sigma)
            pe = model.join_pe(model.text_pe(ctx.shape[1], z.device), model.layout_pe(lay))
            _, base = model.final_layer(model.run_blocks(tok, model.txt_in(ctx), vec, pe), vec)
            up = base.view(3, g, g, -1).repeat_interleave(s, 1).repeat_interleave(s, 2)
            zc = z.view(3, g, s, g, s, -1)
            detail = (zc - zc.mean(dim=(2, 4), keepdim=True)).view_as(z)
            want = -(up + hf_skip_gain(sigma)[:, None, None, None] * detail)
            d = float((got - want).abs().max())
            check(f"zero init at uniform s = {s}", d < 2e-5, f"max |diff| {d:.1e}")

    randomize(model, gen)
    mixer = model.mixers["2"]
    h = torch.randn(7, 2, 2, mixer.dw.weight.shape[0], generator=gen)
    ref = F.conv2d(F.pad(h.permute(0, 3, 1, 2), (0, 1, 0, 1)), mixer.dw.weight, mixer.dw.bias, groups=h.shape[-1]).permute(0, 2, 3, 1)
    d = float((mixer.depthwise(h) - ref).detach().abs().max())
    check("depthwise 2x2 written out == conv2d with SAME padding", d < 1e-5, f"max |diff| {d:.1e}")
    pyr = rand_pyr(3, gen)
    act = L.exact_layout(pyr, torch.tensor([1024, 409, 61]))
    lay = L.pack(act, check=True)
    with torch.no_grad():
        both = model(z, t, ctx, lay)
        d = 0.0
        for i in range(3):
            one = model(z[i:i + 1], t[i:i + 1], ctx[i:i + 1], L.pack(L.select(act, slice(i, i + 1))))
            d = max(d, float((both[i:i + 1] - one).abs().max()))
    check("padded batch == one image at a time", d < 2e-5, f"max |diff| {d:.1e}, tokens {lay.n.tolist()}")

    act = L.build_exact_budget_layout_s8(pyr, 0.3)
    lay = L.pack(act)
    with torch.no_grad():
        txt = model.project_text(ctx)
        fused = model.forward_cfg(z, t, txt, torch.flip(txt, dims=[0]), lay, 4.0)
        v_c = model(z, t, txt, lay, ctx_projected=True)
        v_u = model(z, t, torch.flip(txt, dims=[0]), lay, ctx_projected=True)
        d = float((fused - (v_u + 4.0 * (v_c - v_u))).abs().max())
        check("fused guidance == two passes", d < 1e-4, f"max |diff| {d:.1e}")
        vec = model.time_vec(1.0 - t)
        pe = model.join_pe(model.text_pe(ctx.shape[1], z.device), model.layout_pe(lay))
        tok = model.embed_tokens(z, lay)
        d = float((model.run_blocks(tok, txt, vec, pe, img_only=True) - model.run_blocks(tok, txt, vec, pe, img_only=False)).abs().max())
    check("last block without text queries", d < 1e-5, f"max |diff| {d:.1e}")
    with torch.no_grad():
        moved = float((model(z, t, ctx, lay) - model.forward_dense(z, t, ctx)).abs().mean())
    check("coarse layout differs from dense", moved > 1e-3, f"mean |diff| {moved:.3f}")

    # gradients: every added parameter is reached when all scales are present; checkpointing does not change them
    act = L.exact_layout(pyr, torch.tensor([205, 103, 40]))
    lay = L.pack(act)
    assert set(lay.groups) == {1, 2, 4, 8}
    model.train()
    grads = []
    for ck in (0, model.num_blocks, 2):                              # none, all, only the first two blocks
        model.checkpoint_blocks = ck
        model.zero_grad(set_to_none=True)
        (model(z, t, ctx, lay) - dense).pow(2).mean().backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    dead = [n for n, p in model.named_parameters() if n not in grads[0] or float(grads[0][n].abs().max()) == 0.0]
    check("gradients reach every parameter", dead == ["scale_emb"] or not dead,
          f"({len(grads[0])} tensors)" if not dead else f"no gradient: {dead}")
    d = max(float((grads[0][n] - g[n]).abs().max()) for g in grads[1:] for n in grads[0])
    check("gradient checkpointing (all blocks, some blocks)", d < 1e-6, f"max |grad diff| {d:.1e}")
    model.checkpoint_blocks = 0
    g0 = grads[0]["scale_emb"]
    check("scale_emb row 0 unused", float(g0[0].abs().max()) == 0.0 and float(g0[1:].abs().min(dim=1).values.max()) > 0)

    # the sampling loop (latent layouts, so no VAE is needed)
    from budgetlatent.inference.sampling import sample
    model.eval()
    noise = torch.randn(2, 32, 32, TINY.in_channels, generator=gen)
    with torch.no_grad():
        txt, null = model.project_text(ctx[:2]), model.project_text(torch.zeros_like(ctx[:2]))
        kw = dict(steps=6, cfg=4.0, relayout_every=2, layout_source="latent", noise=noise, amp=False, decode=False)
        r1, r2 = sample(model, None, txt, null, budget=0.3, **kw), sample(model, None, txt, null, budget=0.3, **kw)
        full = sample(model, None, txt, null, budget=1.0, **kw)
        z, ts = noise.clone(), L.flux2_klein_timesteps(6, 1024)
        for i in range(6):
            t_ = torch.full((2,), ts[i])
            v_c, v_u = model.forward_dense(z, t_, txt, ctx_projected=True), model.forward_dense(z, t_, null, ctx_projected=True)
            z = z + (ts[i + 1] - ts[i]) * (v_u + 4.0 * (v_c - v_u))
    check("sampler at a budget", torch.equal(r1["latent"], r2["latent"]) and bool(torch.isfinite(r1["latent"]).all())
          and bool((L.count_tokens(r1["layout"]) == 307).all()) and abs(r1["tokens"] - 307) < 1e-6, "307 tokens at every step, deterministic")
    d = float((full["latent"] - z).abs().max())
    check("sampler at 100 % == plain guided Euler", d < 1e-4 and full["layout"] is None, f"max |diff| {d:.1e}")

    # a layout prepared once (what the sampler keeps between two refreshes) gives the per-step path's numbers exactly
    z, t, ctx = tiny_inputs(b=3, gen=gen)
    with torch.no_grad():
        txt = model.project_text(ctx)
        worst, cases = 0.0, 0
        for act in (L.exact_layout(pyr, torch.tensor([1024, 409, 61])), L.build_exact_budget_layout_s8(pyr, 0.3),
                    L.exact_layout(pyr, torch.tensor([205, 103, 40])), L.uniform_layout(3, 8), L.dense_layout(3)):
            lay = L.pack(act)
            prep = model.prepare(lay, ctx.shape[1])
            want = model(z, t, ctx, lay)
            for _ in range(2):                                       # the second call reuses the layout-only embeddings
                worst = max(worst, float((model(z, t, ctx, prep) - want).abs().max()))
            want = model.forward_cfg(z, t, txt, torch.flip(txt, dims=[0]), lay, 4.0)
            got = model.forward_cfg(z, t, txt, torch.flip(txt, dims=[0]), model.prepare(lay, txt.shape[1], repeat=2), 4.0)
            worst = max(worst, float((got - want).abs().max()))
            cases += 1
        try:
            model(z[:2], t[:2], ctx[:2], model.prepare(lay, ctx.shape[1]))
            mismatch = False
        except ValueError:
            mismatch = True
    check("prepared layout == per-step path", worst == 0.0 and mismatch,
          f"{cases} layouts (padded, exact, mixed, coarsest, dense), plain and guided: max |diff| {worst:.1e}")
    lay = L.pack(L.exact_layout(pyr, torch.tensor([1024, 409, 61])))
    exact = L.pack(L.build_exact_budget_layout_s8(pyr, 0.3))
    check("pack knows whether a layout is padded", lay.known_padded is True and exact.known_padded is False
          and lay.repeat(2).padded and not exact.repeat(2).padded and bool((lay.valid.sum(1) == lay.n).all()))

    class PixelVAE:                                                  # a fixed map from cells to 16 x 16 px, for previews
        def __init__(self, c):
            self.w = 0.3 * torch.randn(c, 3 * 16 * 16, generator=gen)

        def decode(self, zz, bf16=False):
            b, g = zz.shape[0], zz.shape[1]
            x = (zz.float() @ self.w).view(b, g, g, 3, 16, 16).permute(0, 3, 1, 4, 2, 5).reshape(b, 3, 16 * g, 16 * g)
            return torch.tanh(x)

    vae = PixelVAE(TINY.in_channels)
    noise = torch.randn(2, 32, 32, TINY.in_channels, generator=gen)
    with torch.no_grad():
        txt, null = model.project_text(ctx[:2]), model.project_text(torch.zeros_like(ctx[:2]))
        same = True
        for budget, source in ((0.3, "pixel"), (0.05, "pixel"), (0.6, "latent")):
            kw = dict(steps=7, cfg=4.0, relayout_every=2, layout_source=source, noise=noise, amp=False, decode=True)
            old = sample(model, vae, txt, null, budget=budget, reuse_layout=False, **kw)
            new = sample(model, vae, txt, null, budget=budget, **kw)
            same = same and torch.equal(old["latent"], new["latent"]) and torch.equal(old["image"], new["image"]) \
                and all(torch.equal(old["layout"][k], new["layout"][k]) for k in old["layout"])
    check("sampler with prepared layouts == per-step sampler", same, "pixel previews at 30 % and 5 %, latent at 60 %")


# ---------------------------------------------------------------------------------------------------------
DIT_SHAPES = {                                                         # representative rows of the dit/ export
    "dit/img_in/kernel": (128, 3072), "dit/txt_in/kernel": (7680, 3072),
    "dit/time_in/in_layer/kernel": (256, 3072), "dit/time_in/out_layer/kernel": (3072, 3072),
    "dit/double_stream_modulation_img/lin/kernel": (3072, 18432), "dit/single_stream_modulation/lin/kernel": (3072, 9216),
    "dit/double_blocks_4/img_attn/qkv/kernel": (3072, 9216), "dit/double_blocks_0/img_attn/norm/query_norm/scale": (128,),
    "dit/double_blocks_0/txt_attn/norm/key_norm/scale": (128,), "dit/double_blocks_2/txt_attn/proj/kernel": (3072, 3072),
    "dit/double_blocks_0/img_mlp/layers_0/kernel": (3072, 18432), "dit/double_blocks_0/txt_mlp/layers_2/kernel": (9216, 3072),
    "dit/single_blocks_19/linear1/kernel": (3072, 27648), "dit/single_blocks_0/linear2/kernel": (12288, 3072),
    "dit/single_blocks_7/norm/query_norm/scale": (128,), "dit/final_layer/adaLN_modulation/layers_1/kernel": (3072, 6144),
    "dit/final_layer/linear/kernel": (3072, 128), "dit/scale_emb": (4, 3072), "dit/mixers_2/pos": (1, 1, 2, 2, 3072),
    "dit/mixers_4/down/kernel": (3072, 256), "dit/mixers_4/down/bias": (256,), "dit/mixers_8/dw/kernel": (2, 2, 1, 256),
    "dit/mixers_8/dw/bias": (256,), "dit/mixers_2/up/kernel": (1024, 3072), "dit/mixers_2/up/bias": (3072,),
    "dit/ipe_2d/mlp_0/kernel": (64, 256), "dit/ipe_2d/mlp_0/bias": (256,), "dit/ipe_2d/mlp_1/kernel": (256, 3072),
    "dit/ipe_2d/mlp_1/bias": (3072,), "dit/refiners_8/embed/kernel": (128, 128), "dit/refiners_8/embed/bias": (128,),
    "dit/refiners_4/slots": (2, 2, 128), "dit/refiners_2/blocks_1/ada/kernel": (3072, 768), "dit/refiners_2/blocks_1/ada/bias": (768,),
    "dit/refiners_2/blocks_0/n1/scale": (128,), "dit/refiners_2/blocks_0/n2/scale": (128,), "dit/refiners_2/blocks_0/qkv/kernel": (128, 384),
    "dit/refiners_2/blocks_0/qkv/bias": (384,), "dit/refiners_2/blocks_0/qn/scale": (32,), "dit/refiners_2/blocks_0/kn/scale": (32,),
    "dit/refiners_2/blocks_0/proj/kernel": (128, 128), "dit/refiners_2/blocks_0/proj/bias": (128,), "dit/refiners_2/blocks_0/w12/kernel": (128, 704),
    "dit/refiners_2/blocks_0/w3/kernel": (352, 128), "dit/refiners_4/norm/scale": (128,), "dit/refiners_4/out/kernel": (128, 128),
    "dit/refiners_4/out/bias": (128,),
}


def dit_shape(name, shape):
    if name.endswith(".weight"):
        return (shape[2], shape[3], shape[1], shape[0]) if len(shape) == 4 else tuple(reversed(shape))
    return tuple(shape)


def test_convert():
    print("[convert]")
    with torch.device("meta"):
        full = BudgetPixFlux2(TrunkConfig(), BPConfig())
    trunk, added = count_parameters(full)
    check("trunk parameters", trunk == 3_875_544_576, f"{trunk:,}")
    check("added parameters", added == 28_136_192, f"{added:,} (+{100 * added / trunk:.3f} %)")
    per = {k: sum(p.numel() for n, p in full.named_parameters() if n.startswith(k)) for k in ("scale_emb", "mixers.2.", "ipe_2d.", "refiners.2.", "double_blocks.0.", "single_blocks.0.")}
    check("per-module counts", per == {"scale_emb": 12288, "mixers.2.": 3_949_056, "ipe_2d.": 806_144, "refiners.2.": 5_156_864,
                                        "double_blocks.0.": 245_367_296, "single_blocks.0.": 122_683_648}, str(per))
    shapes = {convert.torch_key_to_dit(n): dit_shape(n, tuple(p.shape)) for n, p in full.state_dict().items()}
    bad = {k: (shapes.get(k), v) for k, v in DIT_SHAPES.items() if shapes.get(k) != v}
    check("dit/ export names and shapes", not bad, f"{len(shapes)} keys" if not bad else str(bad))
    check("key map is a bijection", all(convert.dit_key_to_torch(convert.torch_key_to_dit(n)) == n for n in full.state_dict()))

    torch.manual_seed(0)
    tiny = BudgetPixFlux2(TINY, BPConfig())
    randomize(tiny, torch.Generator().manual_seed(0))
    sd = tiny.state_dict()
    back = convert.to_bfl(convert.torch_to_dit(sd))
    check("torch -> dit/ -> torch round trip", set(back) == set(sd) and all(torch.equal(back[k], sd[k]) for k in sd))
    j = convert.torch_to_dit(sd)
    w = sd["mixers.2.dw.weight"]
    check("depthwise kernel layout", torch.equal(j["dit/mixers_2/dw/kernel"][1, 0, 0, 5], w[5, 0, 1, 0]) and j["dit/mixers_2/dw/kernel"].shape == (2, 2, 1, w.shape[0]))


def test_train_helpers():
    print("[train helpers]")
    from budgetlatent.training import trainlib as T
    gen = torch.Generator().manual_seed(3)
    n_tok = torch.cat([torch.full((5,), 1024), torch.full((4,), 16), torch.randint(16, 1025, (23,), generator=gen)]).tolist()
    plan = T.plan_micro_batches(n_tok, 512, cap_tokens=2 * 1536, cap_batch=8)
    flat = sorted(i for g in plan for i in g)
    sizes = [len(g) * (512 + max(n_tok[i] for i in g)) for g in plan]
    check("student passes cover every example once", flat == list(range(len(n_tok))))
    check("student passes respect the caps", all(len(g) <= 8 for g in plan) and all(sz <= 2 * 1536 or len(g) == 1 for g, sz in zip(plan, sizes))
          and sizes == sorted(sizes, reverse=True), f"{len(plan)} passes for {len(n_tok)} examples, sizes {min(sizes)}..{max(sizes)} tokens")
    pad = sum(len(g) * max(n_tok[i] for i in g) - sum(n_tok[i] for i in g) for g in plan) / sum(n_tok)
    naive = T.plan_micro_batches(n_tok, 512, 0, 0, bucket=False, fixed=2)
    pad0 = sum(len(g) * max(n_tok[i] for i in g) - sum(n_tok[i] for i in g) for g in naive) / sum(n_tok)
    check("grouping by length cuts the padding", pad < 0.5 * pad0, f"{100 * pad:.1f} % of the image tokens, against {100 * pad0:.1f} % for pairs in arrival order")
    check("fixed passes", T.plan_micro_batches(n_tok, 512, 0, 0, bucket=False, fixed=2)[:2] == [[0, 1], [2, 3]])
    torch.manual_seed(0)
    model = BudgetPixFlux2(TINY, BPConfig())
    named = list(model.named_parameters())
    for world in (1, 4, 8):
        own = T.split_owners(named, world)
        load = [sum(p.numel() for n, p in named if own[n] == r) for r in range(world)]
        assert set(own) == {n for n, _ in named} and max(load) - min(load) <= max(p.numel() for _, p in named)
    check("EMA split across ranks", True, f"8 ranks: {min(load):,} .. {max(load):,} parameters each")
    check("learning-rate warm-up", abs(T.lr_scale(0, 500) - 1e-4) < 1e-12 and T.lr_scale(500, 500) == 1.0 and 0.49 < T.lr_scale(250, 500) < 0.51)

    # EMA shards: two decays, written by 3 ranks, read back by 1 and by 2; the checkpoint directory; the export
    cpu = torch.device("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        ranks = [T.EmaShards(named, r, 3, (0.9999, 0.999), warmup=False) for r in range(3)]
        randomize(model, gen)
        for r, sh in enumerate(ranks):
            sh.update(0)
            sh.save(tmp, r)
        one = T.EmaShards(named, 0, 1, (0.9999, 0.999), warmup=False)
        one.load(tmp)
        two = [T.EmaShards(named, r, 2, (0.9999, 0.999), warmup=False) for r in range(2)]
        for sh in two:
            sh.load(tmp)
        by_name = {n: v for sh in ranks for n, v in zip(sh.names, sh.avg[1])}
        ok = all(torch.equal(v, by_name[n]) for n, v in zip(one.names, one.avg[1])) \
            and all(torch.equal(v, by_name[n]) for sh in two for n, v in zip(sh.names, sh.avg[1])) \
            and sorted(n for sh in ranks for n in sh.names) == sorted(n for n, _ in named)
        check("EMA shards survive a change of the number of ranks", ok, f"{len(named)} tensors, 2 decays, 3 -> 1 and 3 -> 2 ranks")

        rt = T.Runtime(1, 0, 0, cpu, False)
        ck = T.Checkpoint(rt, tmp)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model(*tiny_inputs(), L.pack(L.uniform_layout(2, 2))).pow(2).mean().backward()
        opt.step()
        ema = T.EmaShards(named, 0, 1, (0.9999, 0.999), warmup=True)
        ema.update(3)
        ck.save(model, opt, ema, {"step": 7, "guard": {"ema": 1.5, "accepted": 6, "skipped": 1}}, extra={"critic.pt": {"x": torch.ones(3)}})
        torch.manual_seed(1)
        other = BudgetPixFlux2(TINY, BPConfig())
        opt2 = torch.optim.AdamW(other.parameters(), lr=1e-3)
        ema2 = T.EmaShards(list(other.named_parameters()), 0, 1, (0.9999, 0.999), warmup=True)
        meta, files = ck.load(other, opt2, ema2, extra=("critic.pt",))
        same = all(torch.equal(a_, b_) for a_, b_ in zip(model.parameters(), other.parameters())) \
            and all(torch.equal(a_, b_) for k in range(2) for a_, b_ in zip(ema.avg[k], ema2.avg[k])) \
            and opt2.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]
        check("checkpoint round trip", ck.exists() and same and meta["step"] == 7 and meta["guard"]["accepted"] == 6 and bool((files["critic.pt"]["x"] == 1).all())
              and not os.path.exists(ck.tmp))
        path = os.path.join(tmp, "ema-s7.safetensors")
        ck.export_ema(model, path, 1, metadata={"step": 7, "chat_template": "no_think"})
        from budgetlatent.models.weights import checkpoint_family, checkpoint_template, read_state
        sd, meta = read_state(path)
        d = max(float((sd[n].float() - v.float()).abs().max()) for n, v in zip(ema.names, ema.avg[1]))
        check("EMA export", set(sd) == {n for n, _ in named} and d < 0.05 and checkpoint_template(path) == "no_think" and checkpoint_template(None) == "bfl_nothink"
              and checkpoint_family(path) == "flux2",
              f"bf16 file of EMA 1, max |diff| to the fp32 shards {d:.1e}; the template travels with the weights")

        # the validation suite on the tiny model (stand-ins for the VAE and the text encoder)
        from budgetlatent.inference import validate as V
        model.eval()
        perf = V.render(model, StubVAE(), StubText(TINY.context_in_dim), V.PROMPTS[:3], os.path.join(tmp, "val"), budgets=(1.0, 0.3, 0.02),
                        indices=[0, 2], steps=2, batch=2, thumb=32, amp=False)
        V.write_index(os.path.join(tmp, "val"), V.PROMPTS[:3], (1.0, 0.3, 0.02), "unit test")
        from PIL import Image
        strip = Image.open(os.path.join(tmp, "val", "002.jpg"))
        check("validation suite", len(V.PROMPTS) == 32 and len(V.BUDGETS) == 10 and strip.size == (3 * 32, 2 * 32) and not os.path.exists(os.path.join(tmp, "val", "001.jpg"))
              and [perf[k]["tokens"] for k in ("1", "0.3", "0.02")] == [1024, 307, 19] and perf["0.3"]["images"] == 2 and abs(perf["0.3"]["mean_tokens"] - 307) < 1e-6,
              "32 prompts x 10 budgets; strips written for this rank's prompts only")


class StubVAE:
    """(B, g, g, C) -> (B, 3, 16 g, 16 g) in [-1, 1]: enough for the layout and overlay code paths."""

    def decode(self, z, bf16=False):
        return torch.tanh(F.interpolate(z[..., :3].float().permute(0, 3, 1, 2), scale_factor=16, mode="nearest"))


class StubText:
    def __init__(self, dim, length=12):
        self.dim, self.length = dim, length

    def __call__(self, prompts):
        return torch.stack([torch.randn(self.length, self.dim, generator=torch.Generator().manual_seed(len(p))) for p in prompts])

    def encode(self, prompts):                                       # encoders.text.Qwen3Text.encode: (states, no windows)
        return self(prompts), None


# ---------------------------------------------------------------------------------------------------------
def test_dmd():
    print("[stage 2]")
    from budgetlatent.training import dmd as D
    gen = torch.Generator().manual_seed(21)
    torch.manual_seed(0)
    teacher = Flux2Trunk(TINY).eval().requires_grad_(False)
    student = BudgetPixFlux2(TINY, BPConfig())
    student.load_state_dict(teacher.state_dict(), strict=False)
    cfg = D.DMDConfig(cls_channels=32, lora_rank=4, lora_alpha=4.0)
    lora, head = D.FakeScoreLoRA.for_trunk(teacher, cfg.lora_rank, cfg.lora_alpha), D.DMD2DiscriminatorHead.for_trunk(teacher, cfg)
    z, t, ctx = tiny_inputs(b=3, gen=gen)
    with torch.no_grad():
        plain = teacher.forward_dense(z, t, ctx)
        v0, aux = teacher.forward_dense(z, t, ctx, lora=lora, return_aux=True)
    check("LoRA at B = 0: the fake denoiser is the real one", torch.equal(plain, v0))
    check("discriminator features", aux["mid_spatial"].shape == (3, 32, 32, 96) == aux["final_spatial"].shape and aux["vec"].shape == (3, 96)
          and abs(float(aux["mid_spatial"].mean(-1).abs().max())) < 1e-4 and abs(float(aux["mid_spatial"].var(-1).mean()) - 1) < 0.05,
          "mid (layer-normed) and final spatial maps, time embedding")
    with torch.no_grad():
        for b_ in lora.b.values():
            b_.add_(0.05 * torch.randn(b_.shape, generator=gen))
        moved = float((teacher.forward_dense(z, t, ctx, lora=lora) - plain).abs().mean())
        d = float((teacher.forward_dense(z, t, ctx, lora=lora) - teacher.forward_dense(z, t, ctx, lora=lora, return_aux=True)[0]).abs().max())
    check("LoRA changes the denoiser; asking for the features does not", moved > 1e-3 and d == 0.0, f"mean |change| {moved:.3f}")

    # the LoRA's gradient through the frozen teacher, with and without checkpointing the blocks
    grads = []
    for ck in (0, teacher.num_blocks):
        teacher.checkpoint_blocks = ck
        lora.zero_grad(set_to_none=True)
        teacher.forward_dense(z, t, ctx, lora=lora).pow(2).mean().backward()
        grads.append([p.grad.clone() for p in lora.parameters()])
    d = max(float((a_ - b_).abs().max()) for a_, b_ in zip(*grads))
    live = sum(float(g_.abs().max()) > 0 for g_ in grads[0])
    check("LoRA gradients, checkpointed teacher == plain", d < 1e-6 and live == len(grads[0]) and all(p.grad is None for p in teacher.parameters()),
          f"max |grad diff| {d:.1e}; all {live} LoRA tensors reached, none of the teacher's")
    lora.zero_grad(set_to_none=True)

    vae, mix = StubVAE(), L.LayoutMix()
    for use_student in (False, True):
        zc = D.rollout(student, teacher, vae, ctx, use_student, gen, steps=3, amp=False)
        assert zc.shape == z.shape and bool(torch.isfinite(zc).all())
    z_t, tt, v_real, x0_real, active, mode = D.synthesize(student, teacher, vae, ctx, True, mix, cfg, gen, amp=False)
    lay = L.pack(active, check=True)
    check("simulated training state", z_t.shape == z.shape == v_real.shape == x0_real.shape and tt.shape == (3,) and mode.shape == (3,)
          and not z_t.requires_grad, f"3-step rollouts (teacher and student), tokens {lay.n.tolist()}")

    pooled = D.pooled_text(ctx, torch.tensor([[5] * 8 + [0] * 4] * 3), pad_id=0)
    check("pooled text", pooled.shape == (3, TINY.context_in_dim) and float((pooled - ctx[:, :8].mean(1)).abs().max()) < 1e-5)
    w = D.coarse_weights(lay.cell_scale, 0.5)
    check("coarse-cell loss weights", float((w.mean(dim=(1, 2)) - 1).abs().max()) < 1e-5 and bool(((w > 1) == ((lay.cell_scale > 1) & (w != 1))).all()))

    # the student's pass: gradients reach the student only, through all three terms
    student.train()
    randomize(student, gen)
    D.set_trainable(False, lora, head)
    act = L.exact_layout(rand_pyr(3, gen), torch.tensor([205, 103, 40]))
    lay = L.pack(act)
    w = D.coarse_weights(lay.cell_scale, 0.5)
    parts = {}
    for name, eff in (("distill", 0.0), ("all", 1.0)):
        student.zero_grad(set_to_none=True)
        terms, x0_fake = D.generator_loss(student(z_t, tt, ctx, lay), z_t, tt, v_real, w, ctx, pooled, teacher, lora, head, cfg, eff, gen, amp=False)
        per = terms["distill"] + (eff * (cfg.dmd_weight * terms["dmd"] + cfg.gan_weight * terms["gan"]) if eff else 0.0)
        per.sum().backward()
        parts[name] = (terms, torch.cat([p.grad.flatten() for p in student.parameters() if p.grad is not None]))
    terms = parts["all"][0]
    check("student loss terms", set(parts["distill"][0]) == {"distill"} and set(terms) == {"distill", "dmd", "gan"} and all(v.shape == (3,) and bool(torch.isfinite(v).all()) for v in terms.values())
          and not x0_fake.requires_grad and all(p.grad is None for m in (teacher, lora, head) for p in m.parameters())
          and float((parts["all"][1] - parts["distill"][1]).abs().max()) > 0,
          ", ".join(f"{k} {float(v.mean()):.2e}" for k, v in terms.items()) + "; the DMD and GAN terms add gradient, the critics and the teacher get none")
    # the GAN term alone must reach the student through the teacher's features
    student.zero_grad(set_to_none=True)
    terms, _ = D.generator_loss(student(z_t, tt, ctx, lay), z_t, tt, v_real, w, ctx, pooled, teacher, lora, head, cfg, 1.0, gen, amp=False)
    terms["gan"].sum().backward()
    g_gan = max(float(p.grad.abs().max()) for p in student.parameters() if p.grad is not None)
    check("generator GAN term reaches the student", g_gan > 0, f"max |grad| {g_gan:.1e}")

    # the critics' pass: gradients reach the LoRA (score term) and the head (GAN term), nothing else
    D.set_trainable(True, lora, head)
    student.zero_grad(set_to_none=True)
    ws = D.score_weights(tt)
    terms = D.critic_loss(x0_fake, x0_real, tt, ws, ctx, pooled, teacher, lora, head, cfg, gen, amp=False)
    (terms["score"] + cfg.d_gan_loss_weight * terms["gan"]).sum().backward()
    g_lora = all(p.grad is not None and float(p.grad.abs().max()) > 0 for p in lora.parameters())
    g_head = all(p.grad is not None for p in head.parameters()) and max(float(p.grad.abs().max()) for p in head.parameters()) > 0
    check("critic loss terms", g_lora and g_head and all(v.shape == (3,) and bool(torch.isfinite(v).all()) for v in terms.values())
          and abs(float(ws.mean()) - 1) < 1e-5 and all(p.grad is None for p in list(teacher.parameters()) + list(student.parameters())),
          f"score {float(terms['score'].mean()):.4f}, gan {float(terms['gan'].mean()):.4f} (2 ln 2 = 1.386 for an untrained head: {abs(float(terms['gan'].mean()) - 1.386) < 0.5})")
    # with gan_feature_lora_grad = False the discriminator term must leave the LoRA alone
    lora.zero_grad(set_to_none=True)
    D.critic_loss(x0_fake, x0_real, tt, ws, ctx, pooled, teacher, lora, head, cfg, gen, amp=False)["gan"].sum().backward()
    check("discriminator features carry no gradient to the LoRA", all(p.grad is None for p in lora.parameters()))

    # a few real updates of both players on the tiny models: finite, and the critics learn their own objective
    D.set_trainable(True, lora, head)
    opt_c = torch.optim.AdamW(list(lora.parameters()) + list(head.parameters()), lr=5e-4, betas=cfg.critic_betas)
    first = last = None
    for it in range(20):
        opt_c.zero_grad(set_to_none=True)
        terms = D.critic_loss(x0_fake, x0_real, tt, ws, ctx, pooled, teacher, lora, head, cfg, torch.Generator().manual_seed(100), amp=False)
        loss = (terms["score"] + cfg.d_gan_loss_weight * terms["gan"]).mean()
        loss.backward()
        opt_c.step()
        first, last = (float(loss) if first is None else first), float(loss)
    check("critics train", last < first and math.isfinite(last), f"critic objective {first:.4f} -> {last:.4f} in 20 updates on one batch")
    teacher.checkpoint_blocks = 0


def test_compiled_layouts():
    """fast.compile_layouts: the compiled layout builder makes the same masks (needs a working torch.compile; on a GPU
    if there is one).  Not part of the default run: python tests/test_units.py compile"""
    print("[compiled layout builder]")
    from budgetlatent.inference import fast
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator().manual_seed(3)
    fast.compile_layouts()
    try:
        bad = 0
        for b, budget in ((1, 0.5), (1, 0.1), (8, 0.3), (8, 0.9), (8, 0.02), (2, 0.7)):
            pyr = {s: v.to(dev) for s, v in rand_pyr(b, gen).items()}
            n = torch.full((b,), L.snap_tokens(budget), dtype=torch.long, device=dev)
            want, got = L._exact_layout(pyr, n), L.EXACT_LAYOUT_FN(pyr, n)
            bad += sum(not torch.equal(want[s], got[s]) for s in want)
            bad += int((L.count_tokens(got) != n).sum())
    finally:
        L.EXACT_LAYOUT_FN = None
    check("compiled layout builder == eager", bad == 0, f"6 batches on {dev.type}, GRID {L.GRID}")


def test_vae_chunks():
    """The VAE decodes one 1024 x 1024 image per pass: the same images as four per pass, in fp32 and with the bf16
    decoder copy; and the bf16 copy against fp32 (needs a GPU and the released VAE).  python tests/test_units.py vae"""
    print("[VAE, one image per pass]")
    from budgetlatent.encoders.vae import Flux2VAE
    from budgetlatent.models.weights import klein_dir
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    vae = Flux2VAE(klein_dir(), dev)
    gen = torch.Generator().manual_seed(5)
    z = torch.randn(4, 64, 64, 128, generator=gen).to(dev)           # 4 latents of 1024 x 1024 images
    check("one image per pass at 1024 by default", vae._chunk(1024) == 1 and vae._chunk(512) == 4, "")
    vae.bf16_preview()
    for name, bf16 in (("fp32", False), ("bf16 copy", "weights")):
        vae.chunk = 4
        four = vae.decode(z, bf16=bf16)
        vae.chunk = None
        one = vae.decode(z, bf16=bf16)
        d, m = float((four - one).abs().max()), float((four - one).abs().mean())
        # (TF32 convolutions: cuDNN picks its algorithm per batch size, a few 1e-3 apart; 1 / 255 = 7.8e-3 in [-1, 1])
        check(f"{name} decode: one image per pass == four per pass", d < (2e-2 if bf16 is False else 6e-2) and m < 1e-3,
              f"max |diff| {d:.1e}, mean {m:.1e}")
    d = float((vae.decode(z, bf16="weights") - vae.decode(z)).abs().mean())
    check("bf16 copy close to the fp32 decoder", d < 2e-2, f"mean |diff| {d:.1e} (images in [-1, 1])")


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    which = sys.argv[1:] or ["layouts", "trunk", "model", "convert", "train", "dmd"]
    if "vae" in which:
        test_vae_chunks()
    if "compile" in which:
        test_compiled_layouts()
    if "layouts" in which:
        test_layouts()
    if "trunk" in which:
        test_trunk_references()
    if "model" in which:
        test_model()
    if "convert" in which:
        test_convert()
    if "train" in which:
        test_train_helpers()
    if "dmd" in which:
        test_dmd()
    print(f"\n{len(OK)} checks passed")
