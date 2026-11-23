"""Reference images (editing, multi-reference generation): models/refs.py through both trunks, the budget models and
the sampler, on tiny random trunks (no GPU, no weights).

    python tests/test_units_refs.py
    FLUX3_ROOT=/path/to/flux-3-image-klein python tests/test_units_refs.py     # + BFL's dit.py with an image_cond pack

  flux2  the trunk with two references (different sizes) == diffusers' Flux2Transformer2DModel on [latents, references]
         with Flux2KleinPipeline's ids (10 + 10 i, h, w, 0), read on the latents; the budget model's dense layout == the
         trunk; a layout prepared once == the per-step layout (forward and forward_cfg); padded rows are refused
  flux3  the trunk with two references == BFL's dit.py (image_cond mode, per-sample text lengths), Klein Base and the
         flow map; the boolean-mask and the unmasked per-group kernels agree; the reference blocks cached once per
         sample == run at every pass; the budget model's dense layout == the trunk; a padded batch == each row alone
  both   the sampler runs with references at 100 % and at a budget (FLUX 3 Base with and without its cache, the flow
         map, FLUX.2 Base with guidance); references change the output; without them nothing changes
  files  prepare_images (the scripts' --ref): the first reference must be the grid's size, the others are fitted under
         1 MP and cropped to 16 px, at most 10; encode_images gives the VAE's latents in order, with each family's ids
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import torch

from budgetlatent.inference import sampling as S
from budgetlatent.models import convert
from budgetlatent.models import flux2_trunk as T
from budgetlatent.models.budgetpix import BudgetPixFlux2, BudgetPixFlux3
from budgetlatent.models.flux2_trunk import Flux2Trunk
from budgetlatent.models.flux3_trunk import Flux3Config, Flux3Trunk
from budgetlatent.models.refs import MAX_REFERENCES, References, encode_images, prepare_images
from budgetlatent.tokenization import layout as L
from test_units import OK, TINY, check, rand_pyr, randomize
from test_units_flux3 import TINY3, bfl_dit, inputs3, perturb_norms

SIZES = ((6, 8), (4, 4))                # latent cells of the two references


def ref_latents(c, gen):
    return [torch.randn(h, w, c, generator=gen) for h, w in SIZES]


def ids_flux2(sizes):
    return torch.cat([torch.cartesian_prod(torch.tensor([10 + 10 * i]), torch.arange(h), torch.arange(w), torch.arange(1))
                      for i, (h, w) in enumerate(sizes)]).float()


# ---------------------------------------------------------------------------------------------------------
def test_flux2():
    print("[FLUX.2 with references]")
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    b, g = 2, 8
    z = torch.randn(b, g, g, TINY.in_channels, generator=gen)
    t = torch.rand(b, generator=gen) * 0.9 + 0.05
    ctx = torch.randn(b, 12, TINY.context_in_dim, generator=gen)
    lat = ref_latents(TINY.in_channels, gen)
    refs = References.from_latents(lat, "flux2")
    check("References: tokens and ids", refs.length == sum(h * w for h, w in SIZES)
          and torch.equal(refs.positions.float(), ids_flux2(SIZES)), f"{refs.length} tokens")
    try:
        from diffusers import Flux2Transformer2DModel
    except Exception as e:                                           # pragma: no cover
        print(f"  --  diffusers not importable ({e}); skipped")
        Flux2Transformer2DModel = None
    if Flux2Transformer2DModel is not None:
        dm = Flux2Transformer2DModel(patch_size=1, in_channels=TINY.in_channels, num_layers=TINY.depth,
                                     num_single_layers=TINY.depth_single_blocks, attention_head_dim=TINY.hidden_size // TINY.num_heads,
                                     num_attention_heads=TINY.num_heads, joint_attention_dim=TINY.context_in_dim,
                                     timestep_guidance_channels=256, mlp_ratio=TINY.mlp_ratio, axes_dims_rope=TINY.axes_dim,
                                     rope_theta=TINY.theta).eval()
        sd = dm.state_dict()
        guidance = [k for k in sd if "guidance_embedder" in k]
        with torch.no_grad():
            for k in guidance:
                if k.endswith("linear_2.weight"):
                    sd[k].zero_()
        trunk = Flux2Trunk(TINY).eval()
        trunk.load_state_dict(convert.diffusers_to_bfl({k: v for k, v in sd.items() if k not in guidance}), strict=True)
        img_ids = torch.cartesian_prod(torch.arange(1), torch.arange(g), torch.arange(g), torch.arange(1)).float()
        txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(ctx.shape[1])).float()
        hs = torch.cat((z.flatten(1, 2), refs.tokens[None].expand(b, -1, -1)), dim=1)
        with torch.no_grad():
            raw = dm(hidden_states=hs, encoder_hidden_states=ctx, timestep=1.0 - t, img_ids=torch.cat((img_ids, ids_flux2(SIZES))),
                     txt_ids=txt_ids, guidance=torch.zeros(b), return_dict=False)[0][:, :g * g]
            mine = trunk.forward_dense(z, t, ctx, refs=refs)
            plain = trunk.forward_dense(z, t, ctx)
        d = float((mine + raw.view_as(mine)).abs().max())
        check("trunk + 2 references == diffusers Flux2Transformer2DModel on [latents, references] (Flux2KleinPipeline ids)",
              d < 2e-5, f"max |diff| {d:.1e} (output scale {float(mine.abs().max()):.2f})")
        check("the references change the prediction", float((mine - plain).abs().max()) > 1e-3)

    gl = L.GRID
    model = randomize(BudgetPixFlux2(TINY).eval(), gen)
    z = torch.randn(b, gl, gl, TINY.in_channels, generator=gen)
    with torch.no_grad():
        dense = model.forward_dense(z, t, ctx, refs=refs)
        d = float((model(z, t, ctx, L.pack(L.dense_layout(b)), refs=refs) - dense).abs().max())
        check("budget model, dense layout + references == the trunk + references", d < 1e-5, f"max |diff| {d:.1e}")
        for name, act in (("uniform 4 x 4", L.uniform_layout(b, 4)),
                          ("exact 30 %", L.build_exact_budget_layout_s8(rand_pyr(b, gen), 0.3))):
            lay = L.pack(act)
            check(f"{name}: unpadded", not lay.padded)
            per = model(z, t, ctx, lay, refs=refs)
            prep = model(z, t, ctx, model.prepare(lay, ctx.shape[1]), refs=refs)
            d = float((per - prep).abs().max())
            check(f"{name}: layout prepared once == per-step layout, with references", d < 1e-5, f"max |diff| {d:.1e}")
            txt = model.project_text(ctx)
            c1 = model.forward_cfg(z, t, txt, torch.flip(txt, dims=[0]), lay, 4.0, refs=refs)
            c2 = model.forward_cfg(z, t, txt, torch.flip(txt, dims=[0]), model.prepare(lay, txt.shape[1], repeat=2), 4.0, refs=refs)
            d = float((c1 - c2).abs().max())
            check(f"{name}: forward_cfg prepared == per-step, with references", d < 1e-5, f"max |diff| {d:.1e}")
        pyr = rand_pyr(b, gen)
        padded = L.pack(L.exact_layout(pyr, torch.tensor([205, 103])))
        try:
            model(z, t, ctx, padded, refs=refs)
            raised = False
        except ValueError:
            raised = True
        check("padded rows (different token counts) with references are refused", raised)


# ---------------------------------------------------------------------------------------------------------
def bfl_pack(dit, cfg, z, t, ctx, lens, refs):
    b, g = z.shape[0], z.shape[1]
    n = g * g
    rows, cols = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    img_ids = torch.stack([torch.zeros_like(rows), rows, cols, torch.zeros_like(rows)], -1).reshape(n, 4).float()
    img_ids[:, 1:3] *= cfg.image_rope_scale
    rid = []
    for i, (h, w) in enumerate(SIZES):                               # BFL pipeline.encode_references: scaled ids, l = -(1 + i)
        r, c = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        ids = torch.stack([torch.zeros_like(r), r, c, torch.zeros_like(r)], -1).reshape(h * w, 4).float()
        ids[:, 1:3] *= cfg.image_rope_scale
        ids[:, 3] = -(1 + i)
        rid.append(ids)
    rid = torch.cat(rid)
    nr = rid.shape[0]
    txt = [ctx[i, :int(lens[i])] for i in range(b)]
    txt_ids = [torch.stack([torch.zeros(len(x)), torch.zeros(len(x)), torch.zeros(len(x)), torch.arange(len(x)).float()], -1) for x in txt]
    offs = [0] + list(torch.cumsum(lens, 0).tolist())
    return dit.Packed(
        data={"image": z.flatten(1, 2).reshape(b * n, -1), "image_cond": refs.tokens.repeat(b, 1), "text": torch.cat(txt)},
        ids={"image": img_ids.repeat(b, 1), "image_cond": rid.repeat(b, 1), "text": torch.cat(txt_ids)},
        timesteps={"image": (1.0 - t).repeat_interleave(n), "image_cond": torch.zeros(b * nr), "text": torch.zeros(int(lens.sum()))},
        mode_indices={"image": [(i * n, (i + 1) * n) for i in range(b)], "image_cond": [(i * nr, (i + 1) * nr) for i in range(b)],
                      "text": [(offs[i], offs[i + 1]) for i in range(b)]})


def test_flux3():
    print("[FLUX 3 with references]")
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    cfg = Flux3Config(**{**TINY3.to_dict(), "image_cond": True})
    cfgf = Flux3Config(**{**cfg.to_dict(), "flow_map": True})
    z, t, ctx, lens = inputs3(gen=gen)
    b = z.shape[0]
    refs = References.from_latents(ref_latents(cfg.cond_in_channels, gen), "flux3")
    t_ref = torch.clamp(t + 0.1, max=1.0)
    trunk = perturb_norms(Flux3Trunk(cfg).eval(), gen)
    trunkf = perturb_norms(Flux3Trunk(cfgf).eval(), gen)
    with torch.no_grad():
        mine = trunk.forward_dense(z, t, ctx, txt_len=lens, refs=refs)
        minef = trunkf.forward_dense(z, t, ctx, txt_len=lens, t_ref=t_ref, refs=refs)
        T.VARLEN_ATTENTION = True
        try:
            mine_vl = trunk.forward_dense(z, t, ctx, txt_len=lens, refs=refs)
        finally:
            T.VARLEN_ATTENTION = False
        cache = {}
        c1 = trunk.forward_dense(z, t, ctx, txt_len=lens, refs=refs, txt_cache=cache)
        c2 = trunk.forward_dense(z, t * 0.5, ctx, txt_len=lens, refs=refs, txt_cache=cache)
        n2 = trunk.forward_dense(z, t * 0.5, ctx, txt_len=lens, refs=refs)
        plain = trunk.forward_dense(z, t, ctx, txt_len=lens)
    d = float((mine_vl - mine).abs().max())
    check("references: unmasked per-group kernel == boolean mask (per-sample text lengths)", d < 1e-5, f"max |diff| {d:.1e}")
    check("reference blocks cached once per sample == run at every pass", "ref_mode_blocks" in cache
          and float((c1 - mine).abs().max()) < 1e-6 and float((c2 - n2).abs().max()) < 1e-6)
    check("the references change the prediction", float((mine - plain).abs().max()) > 1e-3)

    dit, where = bfl_dit()
    if dit is not None:
        for k, v in dict(IN_CHANNELS={"image": cfg.in_channels, "image_cond": cfg.cond_in_channels, "text": cfg.context_in_dim},
                         HIDDEN=cfg.hidden_size, HEADS=cfg.num_heads, DEPTH=cfg.depth, DEPTH_SINGLE=cfg.depth_single_blocks,
                         AXES_DIM=tuple(cfg.axes_dim), MLP_RATIO=cfg.mlp_ratio, THETA=cfg.theta).items():
            setattr(dit, k, v)
        for name, tr, flow, out in (("Klein Base", trunk, False, mine), ("the flow map", trunkf, True, minef)):
            ref = dit.DiT(flow_map=flow).eval()
            ref.load_state_dict(tr.state_dict(), strict=True)
            with torch.no_grad():
                raw = ref(bfl_pack(dit, cfg, z, t, ctx, lens, refs), **({"timesteps_ref": 1.0 - t_ref} if flow else {}))
            d = float((out + raw.view_as(out)).abs().max())
            check(f"{name} + 2 references == BFL reference dit.py (image_cond pack, per-sample text lengths)", d < 2e-5,
                  f"max |diff| {d:.1e} (output scale {float(out.abs().max()):.2f})")
    else:
        print(f"  --  BFL reference dit.py not found ({where}); skipped")

    try:
        Flux3Trunk(TINY3).forward_dense(z, t, ctx, txt_len=lens, refs=refs)
        raised = False
    except ValueError:
        raised = True
    check("a trunk without the image_cond branch refuses references", raised)

    model = randomize(BudgetPixFlux3(cfg).eval(), gen)
    g = L.GRID
    z = torch.randn(b, g, g, cfg.in_channels, generator=gen)
    with torch.no_grad():
        dense = model.forward_dense(z, t, ctx, txt_len=lens, refs=refs)
        d = float((model(z, t, ctx, L.pack(L.dense_layout(b)), txt_len=lens, refs=refs) - dense).abs().max())
        check("budget model, dense layout + references == the trunk + references", d < 1e-5, f"max |diff| {d:.1e}")
        act = L.exact_layout(rand_pyr(b, gen), torch.tensor([205, 103]))
        lay = L.pack(act)
        both = model(z, t, ctx, lay, txt_len=lens, refs=refs)
        d = max(float((both[i:i + 1] - model(z[i:i + 1], t[i:i + 1], ctx[i:i + 1, :int(lens[i])],
                                              L.pack(L.select(act, slice(i, i + 1))), refs=refs)).abs().max()) for i in range(b))
        check("padded batch (205 and 103 tokens) + references == each row alone", d < 1e-5, f"max |diff| {d:.1e}")


# ---------------------------------------------------------------------------------------------------------
def test_sampler():
    print("[the sampler with references]")
    gen = torch.Generator().manual_seed(1)
    g = L.GRID
    cfg = Flux3Config(**{**TINY3.to_dict(), "image_cond": True})
    lat = ref_latents(cfg.cond_in_channels, gen)
    m3 = randomize(BudgetPixFlux3(cfg).eval(), gen)
    m3.family = "flux3"
    m3f = randomize(BudgetPixFlux3(Flux3Config(**{**cfg.to_dict(), "flow_map": True})).eval(), gen)
    m3f.family = "flux3_distilled"
    m2 = randomize(BudgetPixFlux2(TINY).eval(), gen)
    m2.family = "flux2"
    ctx3 = torch.randn(1, 9, cfg.context_in_dim, generator=gen)
    ctx2 = torch.randn(1, 12, TINY.context_in_dim, generator=gen)
    kw = dict(steps=4, layout_source="latent", decode=False, amp=False, seed=3)
    with torch.no_grad():
        for name, m, fam, ctx, guided in (("FLUX 3 Base", m3, "flux3", ctx3, True), ("FLUX 3 flow map", m3f, "flux3", ctx3, False),
                                          ("FLUX.2 Base", m2, "flux2", ctx2, True)):
            refs = References.from_latents(lat, fam)
            txt = m.project_text(ctx)
            unc = m.project_text(torch.zeros_like(ctx)) if guided else None
            cfg_ = 3.5 if guided else 1.0
            for budget in (1.0, 0.25):
                out = S.sample(m, None, txt, unc, budget=budget, cfg=cfg_, refs=refs, **kw)
                base = S.sample(m, None, txt, unc, budget=budget, cfg=cfg_, **kw)
                ok = out["latent"].shape == (1, g, g, m.in_channels) and bool(torch.isfinite(out["latent"]).all())
                check(f"{name}, {budget:.0%}: samples with 2 references (and they change the result)",
                      ok and float((out["latent"] - base["latent"]).abs().max()) > 1e-4)
            if name == "FLUX 3 Base":
                S.CACHE_TEXT = False
                try:
                    nocache = S.sample(m, None, txt, unc, budget=0.25, cfg=cfg_, refs=refs, **kw)
                finally:
                    S.CACHE_TEXT = True
                d = float((nocache["latent"] - out["latent"]).abs().max())
                check("FLUX 3 Base: text and reference blocks cached per sample == run at every step", d < 1e-5, f"max |diff| {d:.1e}")


# ---------------------------------------------------------------------------------------------------------
def test_files():
    print("[reference image files: prepare_images, encode_images]")
    from PIL import Image
    size = 16 * L.GRID
    first = Image.new("RGB", (size, size), (200, 10, 10))
    def refused(images):
        try:
            prepare_images(images, size)
            return False
        except ValueError:
            return True
    check("a first reference of another size is refused (it sets the output size)",
          refused([Image.new("RGB", (size, size + 16))]) and refused([Image.new("RGB", (size // 2, size // 2))]))
    check(f"more than {MAX_REFERENCES} references are refused", refused([first] * (MAX_REFERENCES + 1)))
    big, odd = Image.new("RGB", (1500, 1000), (0, 90, 0)), Image.new("L", (333, 250), 128)
    out = prepare_images([first, big, odd], size)
    sizes = [im.size for im in out]
    ok = (sizes[0] == (size, size) and all(im.mode == "RGB" for im in out)
          and all(w % 16 == 0 and h % 16 == 0 for w, h in sizes[1:]) and sizes[1][0] * sizes[1][1] <= 1024 * 1024
          and sizes[1] == (1248, 832) and sizes[2] == (320, 240))
    check("first kept as is, others fitted under 1 MP (sides floored) and centre-cropped to 16 px, RGB", ok, f"{sizes}")

    class FakeVAE:                                                   # encode: (B, 3, H, W) -> (B, H / 16, W / 16, 4)
        mean = torch.zeros(1)
        def encode(self, x):
            return torch.nn.functional.avg_pool2d(x, 16).repeat(1, 2, 1, 1)[:, :4].permute(0, 2, 3, 1)
    for fam, axis in (("flux2", 0), ("flux3_distilled", 3)):
        refs = encode_images(out, FakeVAE(), fam)
        cells = [(h // 16, w // 16) for w, h in sizes]
        ok = refs.sizes == tuple(cells) and refs.length == sum(h * w for h, w in cells) and refs.family == fam[:5]
        first_val = float(refs.tokens[0, 0])                         # red 200 -> 200 / 127.5 - 1
        lab = sorted({float(v) for v in refs.positions[:, axis]})
        want = [10.0, 20.0, 30.0] if fam == "flux2" else [-3.0, -2.0, -1.0]
        check(f"encode_images ({fam}): sizes, token count, values in order, reference ids {want}",
              ok and abs(first_val - (200 / 127.5 - 1)) < 1e-6 and lab == want, f"{refs.sizes}, ids {lab}")


if __name__ == "__main__":
    test_flux2()
    test_flux3()
    test_sampler()
    test_files()
    print(f"\nall {len(OK)} reference-image checks passed")
