"""CPU checks of FLUX 3 Image Klein that need no weights (a tiny random FLUX 3 trunk stands in for the release).

    python tests/test_units_flux3.py                     # everything that is importable
    PYTHONPATH=<diffusers >= 0.42 environment> python tests/test_units_flux3.py   # + diffusers' Flux3Transformer2DModel
    FLUX3_ROOT=/path/to/flux-3-image-klein python tests/test_units_flux3.py      # BFL's reference (default third_party/)

  trunk     this code == BFL's reference dit.py (strict state dict, its packed per-sample attention, one text length
            per sample) == diffusers' Flux3Transformer2DModel (converted keys); a padded batch with per-row text
            windows == each row alone, with the boolean mask and with the unmasked per-group kernel (VARLEN)
  model     BudgetPixFlux3: dense layout == the released model; zero init == mean pooling in, upsampling + skip out;
            padding + text windows; fused guidance with texts of different windows; last-block shortcut; gradients
            reach every parameter, checkpointing does not change them; the sampler on FLUX 3's schedule
  sizes     the full-size model's parameter counts, the release's names, the text helpers, the schedule
"""
import importlib.util
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch

from budgetlatent.models import convert
from budgetlatent.models import flux2_trunk as T
from budgetlatent.models.added_modules import hf_skip_gain
from budgetlatent.models.budgetpix import BPConfig, BudgetPixFlux3, count_parameters
from budgetlatent.models.flux3_trunk import Flux3Config, Flux3Trunk, KeyRuns
from budgetlatent.tokenization import layout as L
from test_units import OK, check, rand_pyr, randomize

TINY3 = Flux3Config(in_channels=16, cond_in_channels=16, context_in_dim=40, hidden_size=96, num_heads=3, depth=2,
                    depth_single_blocks=3, axes_dim=(8, 8, 8, 8), theta=10000, mlp_ratio=3.0, image_rope_scale=0.5)


def perturb_norms(module, gen, std=0.2):
    """The RMS-norm gains start at 1: move them, so that a swapped query / key gain would show."""
    with torch.no_grad():
        for n, p in module.named_parameters():
            if n.endswith(".scale"):
                p.add_(std * torch.randn(p.shape, generator=gen))
    return module


def inputs3(b=2, g=8, lens=(7, 12), gen=None, cfg=TINY3):
    gen = gen or torch.Generator().manual_seed(0)
    z = torch.randn(b, g, g, cfg.in_channels, generator=gen)
    t = torch.rand(b, generator=gen) * 0.9 + 0.05
    ctx = torch.randn(b, max(lens), cfg.context_in_dim, generator=gen)
    return z, t, ctx, torch.tensor(lens[:b])


def bfl_dit():
    root = os.environ.get("FLUX3_ROOT")
    cands = [root] if root else [os.path.join(os.path.dirname(HERE), "third_party", "flux-3-image-klein")]
    for r in cands:
        path = os.path.join(r, "src", "flux3", "dit.py")
        if r and os.path.exists(path):
            spec = importlib.util.spec_from_file_location("bfl_flux3_dit", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["bfl_flux3_dit"] = mod
            spec.loader.exec_module(mod)
            return mod, path
    return None, cands


# ---------------------------------------------------------------------------------------------------------
def test_trunk():
    print("[FLUX 3 trunk vs references]")
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    cfg = Flux3Config(**{**TINY3.to_dict(), "image_cond": True})          # strict state dicts: the reference has the branch
    trunk = perturb_norms(Flux3Trunk(cfg).eval(), gen)
    z, t, ctx, lens = inputs3(gen=gen)
    b, g = z.shape[0], z.shape[1]
    with torch.no_grad():
        mine = trunk.forward_dense(z, t, ctx, txt_len=lens)
        alone = torch.cat([trunk.forward_dense(z[i:i + 1], t[i:i + 1], ctx[i:i + 1, :int(lens[i])]) for i in range(b)])
        T.VARLEN_ATTENTION = True
        try:
            mine_vl = trunk.forward_dense(z, t, ctx, txt_len=lens)
        finally:
            T.VARLEN_ATTENTION = False
    d = float((mine - alone).abs().max())
    check("text windows: padded batch (mask) == each row alone", d < 1e-5, f"max |diff| {d:.1e}, windows {lens.tolist()}")
    d = float((mine_vl - alone).abs().max())
    check("text windows: padded batch (unmasked per-group kernel) == each row alone", d < 1e-5, f"max |diff| {d:.1e}")

    dit, where = bfl_dit()
    if dit is not None:
        for k, v in dict(IN_CHANNELS={"image": cfg.in_channels, "image_cond": cfg.cond_in_channels, "text": cfg.context_in_dim},
                         HIDDEN=cfg.hidden_size, HEADS=cfg.num_heads, DEPTH=cfg.depth, DEPTH_SINGLE=cfg.depth_single_blocks,
                         AXES_DIM=tuple(cfg.axes_dim), MLP_RATIO=cfg.mlp_ratio, THETA=cfg.theta).items():
            setattr(dit, k, v)
        ref = dit.DiT(flow_map=False).eval()
        ref.load_state_dict(trunk.state_dict(), strict=True)
        n = g * g
        rows, cols = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
        img_ids = torch.stack([torch.zeros_like(rows), rows, cols, torch.zeros_like(rows)], -1).reshape(n, 4).float()
        img_ids[:, 1:3] *= cfg.image_rope_scale
        txt = [ctx[i, :int(lens[i])] for i in range(b)]
        txt_ids = [torch.stack([torch.zeros(len(x)), torch.zeros(len(x)), torch.zeros(len(x)), torch.arange(len(x)).float()], -1) for x in txt]
        offs = [0] + list(torch.cumsum(lens, 0).tolist())
        packed = dit.Packed(
            data={"image": z.flatten(1, 2).reshape(b * n, -1), "image_cond": torch.zeros(0, cfg.cond_in_channels),
                  "text": torch.cat(txt)},
            ids={"image": img_ids.repeat(b, 1), "image_cond": torch.zeros(0, 4), "text": torch.cat(txt_ids)},
            timesteps={"image": (1.0 - t).repeat_interleave(n), "image_cond": torch.zeros(0), "text": torch.zeros(int(lens.sum()))},
            mode_indices={"image": [(i * n, (i + 1) * n) for i in range(b)], "image_cond": [(0, 0)] * b,
                          "text": [(offs[i], offs[i + 1]) for i in range(b)]})
        with torch.no_grad():
            raw = ref(packed)
        d = float((mine + raw.view_as(mine)).abs().max())
        check("== BFL reference dit.py (strict state dict, per-sample text lengths)", d < 2e-5,
              f"max |diff| {d:.1e} (output scale {float(mine.abs().max()):.2f})")
    else:
        print(f"  --  BFL reference dit.py not found ({where}); skipped")

    try:
        from diffusers import Flux3Transformer2DModel
    except Exception as e:                                           # noqa: BLE001
        print(f"  --  diffusers' Flux3Transformer2DModel not importable ({type(e).__name__}); skipped (needs diffusers >= 0.42)")
        return
    torch.manual_seed(1)
    dm = Flux3Transformer2DModel(in_channels=cfg.in_channels, cond_in_channels=cfg.cond_in_channels, num_layers=cfg.depth,
                                 num_single_layers=cfg.depth_single_blocks, attention_head_dim=cfg.hidden_size // cfg.num_heads,
                                 num_attention_heads=cfg.num_heads, joint_attention_dim=cfg.context_in_dim, timestep_channels=256,
                                 mlp_ratio=cfg.mlp_ratio, axes_dims_rope=tuple(cfg.axes_dim), rope_theta=cfg.theta).eval()
    perturb_norms(dm, gen)
    with torch.no_grad():                                            # the final modulation's two halves must differ to test the swap
        dm.norm_out.linear.weight.add_(0.05 * torch.randn(dm.norm_out.linear.weight.shape, generator=gen))
    conv = convert.flux3_diffusers_to_bfl(dm.state_dict())
    trunk2 = Flux3Trunk(cfg).eval()
    trunk2.load_state_dict(conv, strict=True)
    d = 0.0
    with torch.no_grad():
        for i in range(b):
            lt = int(lens[i])
            img_ids = torch.cartesian_prod(torch.arange(1), torch.arange(g), torch.arange(g), torch.arange(1)).float()
            img_ids[:, 1:3] *= cfg.image_rope_scale
            txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(lt)).float()
            raw = dm(hidden_states=z[i:i + 1].flatten(1, 2), encoder_hidden_states=ctx[i:i + 1, :lt], timestep=1.0 - t[i:i + 1],
                     img_ids=img_ids, txt_ids=txt_ids, return_dict=False)[0]
            mine2 = trunk2.forward_dense(z[i:i + 1], t[i:i + 1], ctx[i:i + 1, :lt])
            d = max(d, float((mine2 + raw.view_as(mine2)).abs().max()))
    check("== diffusers Flux3Transformer2DModel (converted keys)", d < 2e-5, f"max |diff| {d:.1e}")


# ---------------------------------------------------------------------------------------------------------
def test_model():
    print("[BudgetPixFlux3]")
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    model = perturb_norms(BudgetPixFlux3(TINY3, BPConfig()).eval(), gen)
    check("family", model.family == "flux3" and not any(convert.is_flux3_cond(n) for n, _ in model.named_parameters()),
          "text-to-image only: no reference-image branch")
    z, t, ctx, lens = inputs3(b=3, g=32, lens=(7, 12, 9), gen=gen)
    with torch.no_grad():
        dense = model.forward_dense(z, t, ctx, txt_len=lens)
        d = float((model(z, t, ctx, L.pack(L.dense_layout(3)), txt_len=lens) - dense).abs().max())
    check("dense layout == released model", d == 0.0, f"max |diff| {d:.1e}")

    with torch.no_grad():                                            # zero init: mean pooling in, bilinear + Wiener skip out
        for s in L.COARSE:
            lay = L.pack(L.uniform_layout(3, s))
            got = model(z, t, ctx, lay, txt_len=lens)
            g = L.GRID // s
            cells = model.embed_cells(z.flatten(1, 2)).view(3, L.GRID, L.GRID, -1)
            tok = cells.view(3, g, s, g, s, -1).mean(dim=(2, 4)).flatten(1, 2)
            sigma = 1.0 - t
            vec = model.time_vec(sigma)
            img = model.run_trunk(tok, model.project_text(ctx), vec, model.layout_pe(lay), model.text_pe(ctx.shape[1], z.device), txt_len=lens)
            _, base = model.final(img, vec)
            up = base.view(3, g, g, -1).repeat_interleave(s, 1).repeat_interleave(s, 2)
            zc = z.view(3, g, s, g, s, -1)
            detail = (zc - zc.mean(dim=(2, 4), keepdim=True)).view_as(z)
            want = -(up + hf_skip_gain(sigma)[:, None, None, None] * detail)
            d = float((got - want).abs().max())
            check(f"zero init at uniform s = {s}", d < 2e-5, f"max |diff| {d:.1e}")

    # positions: FLUX 3 scales the image ids by 0.5, and the box variance by 0.25
    cy, cx = torch.tensor([[3.5]], dtype=torch.float64), torch.tensor([[1.5]], dtype=torch.float64)
    a = model.image_pe(cy, cx, torch.tensor([[1.25]], dtype=torch.float64))
    pos = torch.tensor([[[0.0, 1.75, 0.75, 0.0]]], dtype=torch.float64)
    b_ = T.rope_tables(pos, TINY3.axes_dim, TINY3.theta, torch.tensor([[[0.0, 0.3125, 0.3125, 0.0]]], dtype=torch.float64))
    check("image rope scale 0.5 (positions and box variance)", all(torch.equal(x, y) for x, y in zip(a, b_)))

    randomize(model, gen)
    pyr = rand_pyr(3, gen)
    act = L.exact_layout(pyr, torch.tensor([1024, 409, 61]))
    lay = L.pack(act, check=True)
    for varlen in (False, True):
        T.VARLEN_ATTENTION = varlen
        try:
            with torch.no_grad():
                both = model(z, t, ctx, lay, txt_len=lens)
                d = 0.0
                for i in range(3):
                    one = model(z[i:i + 1], t[i:i + 1], ctx[i:i + 1, :int(lens[i])], L.pack(L.select(act, slice(i, i + 1))))
                    d = max(d, float((both[i:i + 1] - one).abs().max()))
        finally:
            T.VARLEN_ATTENTION = False
        check(f"padded batch + text windows == one image at a time ({'per-group kernel' if varlen else 'mask'})", d < 2e-5,
              f"max |diff| {d:.1e}, tokens {lay.n.tolist()}, windows {lens.tolist()}")

    act = L.build_exact_budget_layout_s8(pyr, 0.3)
    lay = L.pack(act)
    with torch.no_grad():
        txt = model.project_text(ctx)
        unc = model.project_text(torch.randn(3, 5, TINY3.context_in_dim, generator=gen))       # an 80-token-like shorter window
        fused = model.forward_cfg(z, t, txt, unc, lay, 3.5, lens, None)
        v_c = model(z, t, txt, lay, ctx_projected=True, txt_len=lens)
        v_u = model(z, t, unc, lay, ctx_projected=True)
        d = float((fused - (v_u + 3.5 * (v_c - v_u))).abs().max())
        check("fused guidance (texts of different windows) == two passes", d < 1e-4, f"max |diff| {d:.1e}")
        vec = model.time_vec(1.0 - t)
        pe_img, pe_txt = model.layout_pe(lay), model.text_pe(ctx.shape[1], z.device)
        tok = model.embed_tokens(z, lay)
        a1 = model.run_trunk(tok, txt, vec, pe_img, pe_txt, None, lens, img_only=True)
        a2 = model.run_trunk(tok, txt, vec, pe_img, pe_txt, None, lens, img_only=False)
        d = float((a1 - a2).abs().max())
    check("last block without text queries", d < 1e-5, f"max |diff| {d:.1e}")

    act = L.exact_layout(pyr, torch.tensor([205, 103, 40]))
    lay = L.pack(act)
    assert set(lay.groups) == {1, 2, 4, 8}
    model.train()
    grads = []
    for ck in (0, model.num_blocks, 3):
        model.checkpoint_blocks = ck
        model.zero_grad(set_to_none=True)
        (model(z, t, ctx, lay, txt_len=lens) - dense).pow(2).mean().backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
    dead = [n for n, p in model.named_parameters() if n not in grads[0] or float(grads[0][n].abs().max()) == 0.0]
    check("gradients reach every parameter", dead == ["scale_emb"] or not dead,
          f"({len(grads[0])} tensors)" if not dead else f"no gradient: {dead}")
    d = max(float((grads[0][n] - g_[n]).abs().max()) for g_ in grads[1:] for n in grads[0])
    check("gradient checkpointing (all blocks, some blocks)", d < 1e-6, f"max |grad diff| {d:.1e}")
    model.checkpoint_blocks = 0

    from budgetlatent.inference.sampling import sample
    model.eval()
    noise = torch.randn(2, L.GRID, L.GRID, TINY3.in_channels, generator=gen)
    with torch.no_grad():
        txt, null = model.project_text(ctx[:2]), model.project_text(torch.zeros(2, 5, TINY3.context_in_dim))
        kw = dict(steps=6, relayout_every=2, layout_source="latent", noise=noise, amp=False, decode=False, len_cond=lens[:2])
        r1 = sample(model, None, txt, null, budget=0.3, **kw)
        full = sample(model, None, txt, null, budget=1.0, **kw)
        zz, ts = noise.clone(), L.schedule(6, "flux3")
        for i in range(6):
            t_ = torch.full((2,), ts[i])
            v_c = model.forward_dense(zz, t_, txt, ctx_projected=True, txt_len=lens[:2])
            v_u = model.forward_dense(zz, t_, null, ctx_projected=True)
            zz = zz + (ts[i + 1] - ts[i]) * (v_u + 3.5 * (v_c - v_u))
    check("sampler at a budget", bool(torch.isfinite(r1["latent"]).all()) and bool((L.count_tokens(r1["layout"]) == 307).all()),
          "307 tokens at every step")
    d = float((full["latent"] - zz).abs().max())
    check("sampler at 100 % == plain guided Euler (FLUX 3: shift 6.93, guidance 3.5)", d < 1e-4 and full["layout"] is None, f"max |diff| {d:.1e}")


# ---------------------------------------------------------------------------------------------------------
def test_sizes():
    print("[sizes, names, text, schedule]")
    with torch.device("meta"):
        full = BudgetPixFlux3(Flux3Config(), BPConfig())
        with_cond = Flux3Trunk(Flux3Config(image_cond=True))
    trunk, added = count_parameters(full)
    n_cond = sum(p.numel() for p in with_cond.parameters())
    check("parameters", trunk == 4_868_023_808 and added == 28_136_192 and n_cond == 5_538_458_368,
          f"trunk {trunk:,} (text-to-image; with the reference branch {n_cond:,} = the release's file), added {added:,} "
          f"(+{100 * added / trunk:.3f} %)")
    names = set(with_cond.state_dict())
    check("BFL names", "content_mode_blocks.image.4.q_proj.weight" in names and "single_blocks.27.norm.key_norm.scale" in names
          and "final_layer.image.adaLN_modulation.1.weight" in names and len(names) == 357, f"{len(names)} tensors (the release has 357)")
    check("blocks in forward order", full.num_blocks == 2 * 5 + 28)

    from budgetlatent.encoders.text import F3_PAD_MULTIPLE, window, wrap_flux3
    check("text windows", [window(n) for n in (1, 8, 80, 81, 160, 161)] == [80, 80, 80, 160, 160, 240] and window(9000) == 8192
          and int(window(torch.tensor([8, 200]), 160).max()) == 160 and F3_PAD_MULTIPLE == 80)
    check("chat template", wrap_flux3("a fox") == "<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n")

    ts = L.schedule(50, "flux3")
    sig = torch.linspace(1.0, 0.0, 51, dtype=torch.float64)
    want = 6.93 * sig / (1.0 + 5.93 * sig)                            # BFL's flux3/sampling.py formula, the release's alpha
    d = max(abs((1.0 - a) - float(b)) for a, b in zip(ts, want))
    check("schedule (constant shift 6.93)", d < 1e-12 and ts[0] == 0.0 and ts[-1] == 1.0,
          f"sigma {1 - ts[1]:.6f}, {1 - ts[2]:.6f} ... {1 - ts[-2]:.6f}; resolution independent")
    gen = torch.Generator().manual_seed(0)
    t, first = L.sample_times(40000, gen, shift=L.time_shift("flux3"))
    u = 1.0 - t[~first]
    med = float(u.median())
    check("training times through the FLUX 3 shift", abs(float(first.float().mean()) - 0.10) < 0.01
          and abs(med - 6.93 * 0.5 / (1 + 5.93 * 0.5)) < 0.01, f"median sigma {med:.3f} (shifted 0.5: {6.93 * 0.5 / (1 + 5.93 * 0.5):.3f})")

    from budgetlatent.training import trainlib as Tl
    tl = [80, 160, 80, 240, 80, 80]
    plan = Tl.plan_micro_batches([1024, 1024, 100, 50, 50, 16], tl, cap_tokens=2 * (160 + 1024), cap_batch=8)
    flat = sorted(i for gr in plan for i in gr)
    sizes = [len(gr) * (max(tl[i] for i in gr) + max([1024, 1024, 100, 50, 50, 16][i] for i in gr)) for gr in plan]
    check("student passes with per-example text windows", flat == list(range(6)) and all(sz <= 2 * (160 + 1024) or len(gr) == 1 for gr, sz in zip(plan, sizes)),
          f"{plan}")
    k = KeyRuns.build([(0, [5, 3], 5), (5, [4, 2], 4)], 2)
    check("key runs", k.runs == (((0, 9),), ((0, 3), (5, 2))) and k.mask(9, "cpu")[1, 0, 0].tolist() == [1, 1, 1, 0, 0, 1, 1, 0, 0])


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    which = sys.argv[1:] or ["trunk", "model", "sizes"]
    if "trunk" in which:
        test_trunk()
    if "model" in which:
        test_model()
    if "sizes" in which:
        test_sizes()
    print(f"\n{len(OK)} checks passed")
