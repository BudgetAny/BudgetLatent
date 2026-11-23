"""CPU checks of FLUX 3 Image Klein, BFL's step-distilled release ("flux3_distilled", a flow map), that need no weights
(a tiny random FLUX 3 trunk with the flow map's ref_time_in stands in for it).

    python tests/test_units_flux3_distilled.py
    FLUX3_ROOT=/path/to/flux-3-image-klein python tests/test_units_flux3_distilled.py     # + BFL's DiT(flow_map=True)
    PYTHONPATH=<diffusers >= 0.42 environment> python tests/test_units_flux3_distilled.py  # + diffusers' flow-map transformer

  family     the family table: the release's repository, FLUX 3's trunk, classes, chat template and text encoder, the
             sampler defaults (16 steps, no guidance, the layout refreshes), a snapshot of the other FLUX 3 release
             refused either way, the BudgetLatent repositories of all four families, the parameter counts
  trunk      t_ref: ref_time_in is built for the flow map only; the flow map refuses a pass without t_ref, the base
             model one with it; t_ref changes the output; the flow map == the base trunk whose time embedding of every
             mode (the image at t, the text at 0) has ref_time_in(1 - t_ref) added -- BFL's DiT(flow_map=True) -- dense
             and at a budget (BudgetPixFlux3), with per-row text windows; == BFL's dit.py and diffusers' transformer
             when they are importable
  convert    diffusers' target_timestep_embedder.linear_{1,2} -> BFL's ref_time_in.{in,out}_layer
  detection  family_from_keys / checkpoint_family: ref_time_in -> flux3_distilled, a file without it never; the metadata
             "flux3_distilled", and the trainer's "flux3" next to ref_time_in; load_model gives the model its family and
             refuses Klein Base's trunk for Klein and Klein's for Klein Base, with the --family to use
  schedule   schedule(16, "flux3_distilled") = 1 - BFL's timesteps(16, 6.93), the base model's formula at 16 steps
  sampler    no empty prompt encoded, one row per image, t_ref = the next time of the schedule at every step, a cfg
             other than 1 refused, the layout from the noise at step 0 and from decoded previews (the family's size)
             at the family's steps; the defaults equal the explicit settings; 100 % == plain unguided flow-map Euler;
             fast.warmup and the validation suite take the same path
  scripts    sample.py, gallery.py, bench.py and fetch_weights.py --help offer the family
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch
import torch.nn as nn

from budgetlatent.inference import sampling as S
from budgetlatent.models import convert
from budgetlatent.models import families as F
from budgetlatent.models.budgetpix import BPConfig, BudgetPixFlux2, BudgetPixFlux3, count_parameters
from budgetlatent.models.flux2_trunk import timestep_embedding
from budgetlatent.models.flux3_trunk import Flux3Config, Flux3Trunk
from budgetlatent.tokenization import layout as L
from test_units import OK, TINY, check, rand_pyr, randomize
from test_units_flux2_distilled import PixelVAE, SpyText
from test_units_flux3 import TINY3, bfl_dit, inputs3, perturb_norms

DISTILLED = "flux3_distilled"
TINY3D = Flux3Config(**{**TINY3.to_dict(), "flow_map": True})
REF_KEYS = {"ref_time_in.in_layer.weight", "ref_time_in.out_layer.weight"}
# the sampler defaults of the family: steps, guidance, relayout_every, relayout_until, preview_px
DEFAULTS = (16, 1.0, 8, 8, 256)


def raises(fn, exc=ValueError):
    """The message of the exception fn() raises ("" if it does not)."""
    try:
        fn()
    except exc as e:
        return str(e) or type(e).__name__
    return ""


def tiny_dit(cfg, flow_map):
    """config.json's "dit" section of a release, at the tiny sizes."""
    d = {"hidden": cfg.hidden_size, "heads": cfg.num_heads, "depth": cfg.depth, "depth_single": cfg.depth_single_blocks,
         "mlp_ratio": cfg.mlp_ratio, "axes_dim": list(cfg.axes_dim), "theta": cfg.theta,
         "in_channels": {"image": cfg.in_channels, "image_cond": cfg.cond_in_channels, "text": cfg.context_in_dim},
         "out_modes": ["image"], "vec_in_dim": None, "image_rope_scale": cfg.image_rope_scale}
    if flow_map:
        d["flow_map"] = True
    return d


def tiny_snapshot(root, flow_map, seed=0):
    """A tiny stand-in for a release snapshot: config.json and BFL's single file model.safetensors (the reference
    branch included, as released).  Returns its state dict."""
    from safetensors.torch import save_file
    os.makedirs(root, exist_ok=True)
    cfg = Flux3Config(**{**TINY3.to_dict(), "image_cond": True, "flow_map": flow_map})
    torch.manual_seed(seed)
    sd = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in Flux3Trunk(cfg).state_dict().items()}
    save_file(sd, os.path.join(root, "model.safetensors"))
    json.dump({"format_version": 1, "dit": tiny_dit(TINY3, flow_map),
               "sampler": {"steps": 16 if flow_map else 50, "guidance": None if flow_map else 3.5, "alpha": 6.93}},
              open(os.path.join(root, "config.json"), "w"))
    return sd


# ---------------------------------------------------------------------------------------------------------
def test_family():
    print("[family]")
    fam, f3, f2, f2d = (F.family_of(n) for n in (DISTILLED, "flux3", "flux2", "flux2_distilled"))
    check("table", fam.name == DISTILLED and fam.label == "FLUX 3 Image Klein" and fam.repo == "black-forest-labs/flux-3-image-klein"
          and fam.arch == "flux3" and fam.trunk_cls is Flux3Trunk and fam.model_cls is BudgetPixFlux3
          and fam.probe == "emb_in.image.weight" and fam.template == "flux3" and fam.trunk_files == f3.trunk_files and fam.flow_map,
          f"{fam.label} ({fam.repo}): FLUX 3's trunk, classes, chat template and files, a flow map")
    check("sampler defaults", (fam.steps, fam.guidance, fam.relayout_every, fam.relayout_until, fam.preview_px) == DEFAULTS
          and not fam.guided, f"{fam.steps} steps, guidance {fam.guidance} (no unconditional pass), layouts every "
          f"{fam.relayout_every} steps up to step {fam.relayout_until}, from a {fam.preview_px} px preview")
    check("the other families unchanged",
          (f2.steps, f2.guidance, f2.relayout_every, f2.relayout_until, f2.preview_px, f2.arch, f2.flow_map) == (50, 4.0, 4, 20, 512, "flux2", False)
          and (f2d.steps, f2d.guidance, f2d.relayout_every, f2d.relayout_until, f2d.preview_px, f2d.arch, f2d.flow_map) == (4, 1.0, 1, 1, 256, "flux2", False)
          and (f3.steps, f3.guidance, f3.relayout_every, f3.relayout_until, f3.preview_px, f3.arch, f3.flow_map) == (50, 3.5, 4, 20, 512, "flux3", False)
          and f3.repo == "black-forest-labs/flux-3-image-klein-base" and f3.label == "FLUX 3 Image Klein Base")
    check("BudgetLatent repositories", F.BUDGETLATENT_REPOS == {
        "flux2": "BudgetAny/BudgetLatent-FLUX.2-klein-base-4B", "flux2_distilled": "BudgetAny/BudgetLatent-FLUX.2-klein-4B",
        "flux3": "BudgetAny/BudgetLatent-FLUX-3-Image-Klein-Base", "flux3_distilled": "BudgetAny/BudgetLatent-FLUX-3-Image-Klein"}
          and set(F.BUDGETLATENT_REPOS) == set(F.FAMILIES), "BFL's naming: Image-Klein-Base the base model, Image-Klein the distilled one")
    check("trunk config: a flow map for this family only", fam.config().flow_map and not f3.config().flow_map
          and fam.config() == Flux3Config(flow_map=True) and f3.config() == Flux3Config()
          and fam.config(None, {"hidden_size": 96, "num_heads": 3, "axes_dim": (8, 8, 8, 8)}).flow_map)

    with tempfile.TemporaryDirectory() as tmp:
        dist, base = os.path.join(tmp, "klein"), os.path.join(tmp, "klein_base")
        tiny_snapshot(dist, True)
        tiny_snapshot(base, False)
        c_dist, c_base = fam.config(dist), f3.config(base)
        check("the release's config.json read", c_dist.flow_map and not c_base.flow_map and c_dist.hidden_size == 96
              and Flux3Config.from_snapshot(dist).flow_map and c_dist == Flux3Config(**{**c_base.to_dict(), "flow_map": True}))
        m1, m2 = raises(lambda: f3.config(dist), SystemExit), raises(lambda: fam.config(base), SystemExit)
        check("the other release's snapshot refused, with the family to use", "--family flux3_distilled" in m1
              and "--family flux3" in m2 and "--family flux3_distilled" not in m2, f"{m1[-70:]!r}; {m2[-70:]!r}")
        diffusers_cfg = {"in_channels": 16, "joint_attention_dim": 40, "num_attention_heads": 3, "attention_head_dim": 32,
                         "num_layers": 2, "num_single_layers": 3, "axes_dims_rope": [8, 8, 8, 8], "rope_theta": 10000,
                         "mlp_ratio": 3.0, "flow_map": True}
        check("diffusers' transformer/config.json with flow_map", Flux3Config.from_dict(diffusers_cfg).flow_map
              and not Flux3Config.from_dict({**diffusers_cfg, "flow_map": False}).flow_map)

        from budgetlatent.models.weights import expected_counts, trunk_file
        counts = expected_counts(DISTILLED, dist)
        with torch.device("meta"):
            want = count_parameters(BudgetPixFlux3(c_dist, BPConfig()))
        n_ref = sum(p.numel() for p in Flux3Trunk(TINY3D).ref_time_in.parameters())
        check("parameter counts (released header vs built model)", counts == want and n_ref == 256 * 96 + 96 * 96
              and counts[0] == expected_counts("flux3", base)[0] + n_ref, f"{counts}: Klein Base's trunk + ref_time_in")
        check("released trunk file", os.path.basename(trunk_file(dist, DISTILLED)) == "model.safetensors")

    import budgetlatent.encoders.text as TX
    built = []
    orig = TX.Qwen3VLText
    TX.Qwen3VLText = lambda *a, **k: built.append(("Qwen3VLText", a[3] if len(a) > 3 else k.get("template"))) or "vl"
    try:
        enc = TX.text_encoder(DISTILLED, "snapshot")
    finally:
        TX.Qwen3VLText = orig
    from budgetlatent.models.weights import checkpoint_template
    check("text encoder: FLUX 3's Qwen3-VL-4B", enc == "vl" and built == [("Qwen3VLText", "flux3")] and DISTILLED in TX.FLUX3_FAMILIES
          and checkpoint_template(None, family=DISTILLED) == "flux3", f"{built}")
    check("unknown family refused", bool(raises(lambda: F.family_of("flux3-distilled"))))


# ---------------------------------------------------------------------------------------------------------
class PlusRef(nn.Module):
    """A time embedder that adds a fixed (B, D) vector to every embedding it makes: the base trunk with this as its
    time_in is BFL's flow map with that vector as ref_time_in(1 - t_ref) -- added to every mode's time vector."""

    def __init__(self, inner, add):
        super().__init__()
        self.inner, self.in_layer, self.add = inner, inner.in_layer, add

    def forward(self, x):
        return self.inner(x) + self.add


def without_ref(sd):
    return {k: v for k, v in sd.items() if k not in REF_KEYS}


def test_trunk():
    print("[the flow map's t_ref]")
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    trunk = perturb_norms(Flux3Trunk(TINY3D).eval(), gen)
    base = Flux3Trunk(TINY3).eval()
    check("ref_time_in for the flow map only", set(trunk.state_dict()) - set(base.state_dict()) == REF_KEYS
          and set(base.state_dict()) <= set(trunk.state_dict()) and not hasattr(base, "ref_time_in"))
    base.load_state_dict(without_ref(trunk.state_dict()), strict=True)
    z, t, ctx, lens = inputs3(gen=gen)
    t_ref = torch.clamp(t + 0.2, max=1.0)
    with torch.no_grad():
        m1 = raises(lambda: trunk.forward_dense(z, t, ctx, txt_len=lens))
        m2 = raises(lambda: base.forward_dense(z, t, ctx, txt_len=lens, t_ref=t_ref))
        check("a flow map without t_ref, and the base model with one, refused", "required by, and only by" in m1 and "required by" in m2)
        mine = trunk.forward_dense(z, t, ctx, txt_len=lens, t_ref=t_ref)
        other = trunk.forward_dense(z, t, ctx, txt_len=lens, t_ref=torch.ones_like(t))
        d = float((mine - other).abs().max())
        check("t_ref changes the output", d > 1e-3, f"max |diff| {d:.2e} between t_ref and 1")
        vref = trunk.ref_time_in(timestep_embedding(1.0 - t_ref.float()))
        base.time_in = PlusRef(base.time_in, vref)
        want = base.forward_dense(z, t, ctx, txt_len=lens)
        d = float((mine - want).abs().max())
        check("== the base trunk with ref_time_in(1 - t_ref) added to every mode's time vector (text at 0 too)", d < 1e-6,
              f"max |diff| {d:.1e} (output scale {float(mine.abs().max()):.2f})")
        base.time_in = base.time_in.inner
        only_img = PlusRef(base.time_in, vref)
        calls = []
        base.time_in = only_img
        orig_tv = base.time_vec

        def first_only(sigma, embedder=None):                       # the image's time vector only: not BFL's flow map
            calls.append(1)
            return orig_tv(sigma, embedder) if len(calls) % 2 == 1 else orig_tv(sigma, only_img.inner)
        base.time_vec = first_only
        partial = base.forward_dense(z, t, ctx, txt_len=lens)
        del base.time_vec
        base.time_in = base.time_in.inner
        d_text = float((mine - partial).abs().max())
        check("... and the text's: without it the output differs", d_text > 1e-4, f"max |diff| {d_text:.2e}")
        alone = torch.cat([trunk.forward_dense(z[i:i + 1], t[i:i + 1], ctx[i:i + 1, :int(lens[i])], t_ref=t_ref[i:i + 1])
                           for i in range(z.shape[0])])
        d = float((mine - alone).abs().max())
        check("padded batch with text windows and per-row t_ref == each row alone", d < 1e-5, f"max |diff| {d:.1e}")

    print("[BudgetPixFlux3 as a flow map]")
    model = randomize(perturb_norms(BudgetPixFlux3(TINY3D, BPConfig()).eval(), gen), gen)
    base_m = BudgetPixFlux3(TINY3, BPConfig()).eval()
    base_m.load_state_dict(without_ref(model.state_dict()), strict=True)
    z, t, ctx, lens = inputs3(b=3, g=L.GRID, lens=(7, 12, 9), gen=gen)
    t_ref = torch.clamp(t + 0.15, max=1.0)
    lay = L.pack(L.build_exact_budget_layout_s8(rand_pyr(3, gen), 0.3))
    with torch.no_grad():
        check("model.flow_map", model.flow_map and not base_m.flow_map and not BudgetPixFlux2(TINY, BPConfig()).flow_map)
        dense = model.forward_dense(z, t, ctx, txt_len=lens, t_ref=t_ref)
        d = float((model(z, t, ctx, L.pack(L.dense_layout(3)), txt_len=lens, t_ref=t_ref) - dense).abs().max())
        check("dense layout == the released flow map", d == 0.0, f"max |diff| {d:.1e}")
        got = model(z, t, ctx, lay, txt_len=lens, t_ref=t_ref)
        vref = model.ref_time_in(timestep_embedding(1.0 - t_ref.float()))
        base_m.time_in = PlusRef(base_m.time_in, vref)
        want = base_m(z, t, ctx, lay, txt_len=lens)
        base_m.time_in = base_m.time_in.inner
        d = float((got - want).abs().max())
        check("at a budget (30 %, padded text windows) == the base model with ref_time_in added to every mode", d < 1e-6,
              f"max |diff| {d:.1e}")
        f2 = BudgetPixFlux2(TINY, BPConfig()).eval()
        z2 = torch.randn(1, L.GRID, L.GRID, TINY.in_channels, generator=gen)
        msgs = [raises(lambda: model(z, t, ctx, lay, txt_len=lens)),
                raises(lambda: base_m(z, t, ctx, lay, txt_len=lens, t_ref=t_ref)),
                raises(lambda: f2(z2, t[:1], torch.randn(1, 12, TINY.context_in_dim), None, t_ref=t_ref[:1])),
                raises(lambda: model.forward_cfg(z, t, model.project_text(ctx), model.project_text(ctx[:, :5]), lay, 3.5, lens, None))]
        check("refused: the flow map without t_ref, t_ref to Klein Base or FLUX.2, guidance on the flow map", all(msgs),
              "; ".join(m[:48] for m in msgs))

    dit, where = bfl_dit()
    if dit is not None:
        cfg = Flux3Config(**{**TINY3D.to_dict(), "image_cond": True})
        trunk = perturb_norms(Flux3Trunk(cfg).eval(), gen)
        z, t, ctx, lens = inputs3(gen=gen)
        t_ref = torch.clamp(t + 0.2, max=1.0)
        b, g = z.shape[0], z.shape[1]
        for k, v in dict(IN_CHANNELS={"image": cfg.in_channels, "image_cond": cfg.cond_in_channels, "text": cfg.context_in_dim},
                         HIDDEN=cfg.hidden_size, HEADS=cfg.num_heads, DEPTH=cfg.depth, DEPTH_SINGLE=cfg.depth_single_blocks,
                         AXES_DIM=tuple(cfg.axes_dim), MLP_RATIO=cfg.mlp_ratio, THETA=cfg.theta).items():
            setattr(dit, k, v)
        ref = dit.DiT(flow_map=True).eval()
        ref.load_state_dict(trunk.state_dict(), strict=True)
        n = g * g
        rows, cols = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
        img_ids = torch.stack([torch.zeros_like(rows), rows, cols, torch.zeros_like(rows)], -1).reshape(n, 4).float()
        img_ids[:, 1:3] *= cfg.image_rope_scale
        txt = [ctx[i, :int(lens[i])] for i in range(b)]
        txt_ids = [torch.stack([torch.zeros(len(x)), torch.zeros(len(x)), torch.zeros(len(x)), torch.arange(len(x)).float()], -1) for x in txt]
        offs = [0] + list(torch.cumsum(lens, 0).tolist())
        packed = dit.Packed(
            data={"image": z.flatten(1, 2).reshape(b * n, -1), "image_cond": torch.zeros(0, cfg.cond_in_channels), "text": torch.cat(txt)},
            ids={"image": img_ids.repeat(b, 1), "image_cond": torch.zeros(0, 4), "text": torch.cat(txt_ids)},
            timesteps={"image": (1.0 - t).repeat_interleave(n), "image_cond": torch.zeros(0), "text": torch.zeros(int(lens.sum()))},
            mode_indices={"image": [(i * n, (i + 1) * n) for i in range(b)], "image_cond": [(0, 0)] * b,
                          "text": [(offs[i], offs[i + 1]) for i in range(b)]})
        with torch.no_grad():
            raw = ref(packed, timesteps_ref=1.0 - t_ref)
            mine = trunk.forward_dense(z, t, ctx, txt_len=lens, t_ref=t_ref)
        d = float((mine + raw.view_as(mine)).abs().max())
        check("== BFL reference dit.py DiT(flow_map=True) (strict state dict, per-sample t_ref and text lengths)", d < 2e-5,
              f"max |diff| {d:.1e} (output scale {float(mine.abs().max()):.2f})")
    else:
        print(f"  --  BFL reference dit.py not found ({where}); skipped")

    try:
        from diffusers import Flux3Transformer2DModel
    except Exception as e:                                           # noqa: BLE001
        print(f"  --  diffusers' Flux3Transformer2DModel not importable ({type(e).__name__}); skipped (needs diffusers >= 0.42)")
        return
    cfg = Flux3Config(**{**TINY3D.to_dict(), "image_cond": True})
    torch.manual_seed(1)
    dm = Flux3Transformer2DModel(in_channels=cfg.in_channels, cond_in_channels=cfg.cond_in_channels, num_layers=cfg.depth,
                                 num_single_layers=cfg.depth_single_blocks, attention_head_dim=cfg.hidden_size // cfg.num_heads,
                                 num_attention_heads=cfg.num_heads, joint_attention_dim=cfg.context_in_dim, timestep_channels=256,
                                 mlp_ratio=cfg.mlp_ratio, axes_dims_rope=tuple(cfg.axes_dim), rope_theta=cfg.theta, flow_map=True).eval()
    perturb_norms(dm, gen)
    with torch.no_grad():
        dm.norm_out.linear.weight.add_(0.05 * torch.randn(dm.norm_out.linear.weight.shape, generator=gen))
    sd = dm.state_dict()
    conv = convert.flux3_diffusers_to_bfl(sd)
    trunk2 = Flux3Trunk(cfg).eval()
    trunk2.load_state_dict(conv, strict=True)
    z, t, ctx, lens = inputs3(gen=gen)
    t_ref = torch.clamp(t + 0.2, max=1.0)
    d = 0.0
    with torch.no_grad():
        for i in range(z.shape[0]):
            lt, g = int(lens[i]), z.shape[1]
            img_ids = torch.cartesian_prod(torch.arange(1), torch.arange(g), torch.arange(g), torch.arange(1)).float()
            img_ids[:, 1:3] *= cfg.image_rope_scale
            txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(lt)).float()
            raw = dm(hidden_states=z[i:i + 1].flatten(1, 2), encoder_hidden_states=ctx[i:i + 1, :lt], timestep=1.0 - t[i:i + 1],
                     img_ids=img_ids, txt_ids=txt_ids, target_timestep=1.0 - t_ref[i:i + 1], return_dict=False)[0]
            mine2 = trunk2.forward_dense(z[i:i + 1], t[i:i + 1], ctx[i:i + 1, :lt], t_ref=t_ref[i:i + 1])
            d = max(d, float((mine2 + raw.view_as(mine2)).abs().max()))
    check("== diffusers Flux3Transformer2DModel(flow_map=True) (converted keys, target_timestep)", d < 2e-5
          and F.family_from_keys(sd) == DISTILLED, f"max |diff| {d:.1e}")


# ---------------------------------------------------------------------------------------------------------
def to_diffusers(sd):
    """BFL's names -> Flux3Transformer2DModel's (the inverse of convert.flux3_diffusers_to_bfl), written out here."""
    inner = {v: k for k, v in convert._F3_INNER.items()}
    top = {"txt_in.weight": "context_embedder.weight", "time_in.in_layer.weight": "timestep_embedder.linear_1.weight",
           "time_in.out_layer.weight": "timestep_embedder.linear_2.weight", "final_layer.image.linear.weight": "proj_out.weight",
           "ref_time_in.in_layer.weight": "target_timestep_embedder.linear_1.weight",
           "ref_time_in.out_layer.weight": "target_timestep_embedder.linear_2.weight"}
    out = {}
    for k, v in sd.items():
        p = k.split(".")
        if k in top:
            out[top[k]] = v
        elif k == "final_layer.image.adaLN_modulation.1.weight":
            shift, scale = v.chunk(2, dim=0)
            out["norm_out.linear.weight"] = torch.cat([scale, shift], dim=0)
        elif p[0] == "emb_in":
            out[f"x_embedders.{p[1]}.weight"] = v
        elif p[0].endswith("_stream_modulations"):
            out[f"{p[0]}.{p[1]}.linear.weight"] = v
        elif p[0] == "content_mode_blocks":
            out[f"mode_blocks.{p[1]}.{p[2]}.{inner['.'.join(p[3:])]}"] = v
        elif p[0] == "txt_mode_blocks":
            out[f"txt_mode_blocks.{p[1]}.{inner['.'.join(p[2:])]}"] = v
        elif p[0] == "single_blocks":
            out[f"single_transformer_blocks.{p[1]}.{inner['.'.join(p[2:])]}"] = v
        else:
            raise KeyError(k)
    return out


def test_convert():
    print("[diffusers names]")
    torch.manual_seed(0)
    sd = Flux3Trunk(Flux3Config(**{**TINY3D.to_dict(), "image_cond": True})).state_dict()
    dif = to_diffusers(sd)
    back = convert.to_bfl(dif)
    check("target_timestep_embedder.linear_{1,2} -> ref_time_in.{in,out}_layer", convert.detect(dif) == "flux3_diffusers"
          and set(back) == set(sd) and all(torch.equal(back[k], sd[k]) for k in sd)
          and convert.flux3_key("target_timestep_embedder.linear_1.weight") == "ref_time_in.in_layer.weight"
          and convert.flux3_key("target_timestep_embedder.linear_2.weight") == "ref_time_in.out_layer.weight",
          f"{len(sd)} tensors, round trip exact")
    base = convert.to_bfl(to_diffusers(Flux3Trunk(Flux3Config(**{**TINY3.to_dict(), "image_cond": True})).state_dict()))
    check("Klein Base's names unchanged", not (set(base) & REF_KEYS) and F.family_from_keys(base) == "flux3"
          and F.family_from_keys(dif) == DISTILLED and F.family_from_keys(sd) == DISTILLED)


# ---------------------------------------------------------------------------------------------------------
def save(path, sd, meta=None):
    from safetensors.torch import save_file
    save_file({k: v.detach().to(torch.bfloat16).contiguous() for k, v in sd.items()}, path, metadata=meta)


def test_detection():
    print("[detection]")
    from budgetlatent.models.weights import checkpoint_family, checkpoint_template, load_model
    meta = lambda fam: {"family": fam, "chat_template": "flux3", "format": "budgetpix_flux3_s8", "resolution": str(L.GRID * L.CELL_PX)}
    with tempfile.TemporaryDirectory() as tmp:
        dist, base = os.path.join(tmp, "klein"), os.path.join(tmp, "klein_base")
        released = tiny_snapshot(dist, True, seed=1)
        tiny_snapshot(base, False, seed=2)
        torch.manual_seed(0)
        ref = randomize(BudgetPixFlux3(TINY3D, BPConfig()), torch.Generator().manual_seed(4))
        sd = ref.state_dict()
        added = {k: v for k, v in sd.items() if convert.is_adapter(k)}
        base_sd = without_ref(sd)
        paths = {n: os.path.join(tmp, f"{n}.safetensors") for n in
                 ("dist", "trainer", "nometa", "base", "base_nometa", "added", "added_nometa")}
        save(paths["dist"], sd, meta(DISTILLED))
        save(paths["trainer"], sd, meta("flux3"))                     # what the trainer of the distilled run records
        save(paths["nometa"], sd)
        save(paths["base"], base_sd, meta("flux3"))
        save(paths["base_nometa"], base_sd)
        save(paths["added"], added, meta(DISTILLED))
        save(paths["added_nometa"], added)
        fams = {n: checkpoint_family(p) for n, p in paths.items()}
        check("checkpoint_family", fams == {"dist": DISTILLED, "trainer": DISTILLED, "nometa": DISTILLED, "base": "flux3",
                                            "base_nometa": "flux3", "added": DISTILLED, "added_nometa": "flux2"}
              and checkpoint_family(os.path.join(dist, "model.safetensors")) == DISTILLED
              and checkpoint_family(os.path.join(base, "model.safetensors")) == "flux3",
              "ref_time_in or the record says flux3_distilled; Klein Base's files (no ref_time_in) stay flux3")
        check("chat template", checkpoint_template(paths["dist"], family=DISTILLED) == "flux3")

        model = load_model(paths["dist"], weights=dist, family=checkpoint_family(paths["dist"]))
        d = max(float((model.state_dict()[k].float() - v.to(torch.bfloat16).float()).abs().max()) for k, v in sd.items())
        check("load_model gives the model its family", model.family == DISTILLED and type(model) is BudgetPixFlux3 and model.flow_map
              and S.sampler_defaults(model) == DEFAULTS[:4] and S.preview_side(model) == DEFAULTS[4] and d == 0.0,
              f"model.family {model.family!r}; the sampler's defaults {S.sampler_defaults(model)}, preview {S.preview_side(model)} px")
        again = load_model(paths["trainer"], weights=dist, family=checkpoint_family(paths["trainer"]))
        check("a file the trainer recorded as flux3 loads as flux3_distilled", again.family == DISTILLED
              and torch.equal(again.ref_time_in.in_layer.weight, model.ref_time_in.in_layer.weight))
        rel = load_model(None, weights=dist, family=DISTILLED)
        top = load_model(paths["added"], weights=dist, family=DISTILLED)
        check("the released flow map (added modules at zero, or from a file of added modules only)",
              torch.equal(rel.ref_time_in.out_layer.weight, released["ref_time_in.out_layer.weight"].float())
              and torch.equal(top.ref_time_in.out_layer.weight, released["ref_time_in.out_layer.weight"].float())
              and float(rel.scale_emb.detach().abs().max()) == 0.0 and torch.equal(top.scale_emb, ref.scale_emb.detach().to(torch.bfloat16).float()))
        msgs = [raises(lambda: load_model(paths["dist"], weights=base, family="flux3"), SystemExit),
                raises(lambda: load_model(paths["base"], weights=dist, family=DISTILLED), SystemExit),
                raises(lambda: load_model(paths["dist"], weights=dist, family="flux3"), SystemExit),
                raises(lambda: load_model(None, weights=base, family=DISTILLED), SystemExit)]
        check("the wrong trunk refused, with the family to use", "--family flux3_distilled" in msgs[0]
              and "--family flux3" in msgs[1] and "--family flux3_distilled" in msgs[2] and "--family flux3" in msgs[3],
              " | ".join(m[-60:] for m in msgs))
        b3 = load_model(paths["base"], weights=base, family="flux3")
        check("Klein Base itself unchanged", b3.family == "flux3" and not b3.flow_map and not hasattr(b3, "ref_time_in"))


# ---------------------------------------------------------------------------------------------------------
def test_schedule():
    print("[schedule]")
    ts = L.schedule(16, DISTILLED)
    sig = torch.linspace(1.0, 0.0, 17, dtype=torch.float64)
    want = 6.93 * sig / (1.0 + 5.93 * sig)                            # BFL's flux3/sampling.py timesteps(16, alpha), config.json's alpha
    d = max(abs((1.0 - a) - float(b)) for a, b in zip(ts, want))
    check("schedule(16, flux3_distilled) == 1 - BFL timesteps(16, 6.93)", d < 1e-12 and ts == L.schedule(16, "flux3")
          and L.time_shift(DISTILLED) == L.time_shift("flux3") == 6.93 and ts[0] == 0.0 and ts[-1] == 1.0,
          f"t = {', '.join(f'{v:.4f}' for v in ts[:4])} ... {ts[-2]:.4f}, 1")


# ---------------------------------------------------------------------------------------------------------
class SpyText3(SpyText):
    """FLUX 3's encoder returns a text window per prompt."""

    def encode(self, prompts):
        states, _ = super().encode(prompts)
        return states, torch.tensor([7 + 5 * (i % 2) for i in range(len(prompts))])


def spy(model):
    """Record every trunk pass (batch, text rows, layout kind, t, t_ref) and every layout build; refuse forward_cfg."""
    log = {"forward": [], "builds": 0, "build_steps": []}
    fwd = model.forward

    def forward(z, t, ctx, layout=None, **kw):
        tr = kw.get("t_ref")
        log["forward"].append((z.shape[0], ctx.shape[0], type(layout).__name__, tuple(t.tolist()),
                               None if tr is None else tuple(tr.tolist())))
        return fwd(z, t, ctx, layout, **kw)

    def forward_cfg(*a, **kw):
        raise AssertionError("forward_cfg called on the flow map")
    model.forward, model.forward_cfg = forward, forward_cfg
    build = S.build_exact_budget_layout_s8

    def counted(pyr, budget):
        log["builds"] += 1
        log["build_steps"].append(len(log["forward"]))
        return build(pyr, budget)
    S.build_exact_budget_layout_s8 = counted
    return log, lambda: setattr(S, "build_exact_budget_layout_s8", build)


def test_sampler():
    print("[sampler: the flow map, no guidance]")
    fam = F.family_of(DISTILLED)
    gen = torch.Generator().manual_seed(3)
    torch.manual_seed(0)
    model = randomize(BudgetPixFlux3(TINY3D, BPConfig()).eval(), gen)
    model.family = DISTILLED                                         # (weights.load_model sets it)
    text = SpyText3(TINY3D.context_in_dim)
    prompts = ["a red fox", "a corgi on the moon"]
    cond, uncond, lens = S.encode_prompts(text, model, prompts, amp=False)
    check("no empty prompt encoded", uncond is None and lens["len_uncond"] is None and lens["len_cond"].tolist() == [7, 12]
          and text.seen == prompts and cond.shape[0] == 2, f"the text encoder saw {text.seen}")

    vae = PixelVAE(TINY3D.in_channels, gen)
    noise = torch.randn(2, L.GRID, L.GRID, TINY3D.in_channels, generator=gen)
    ts = L.schedule(fam.steps, DISTILLED)
    refresh = [i for i in range(fam.steps) if i % fam.relayout_every == 0 and (fam.relayout_until is None or i <= fam.relayout_until)]
    side = 16 * L.GRID
    factor = max(1, side // fam.preview_px)
    log, restore = spy(model)
    try:
        with torch.no_grad():
            res = S.sample(model, vae, cond, uncond, budget=0.3, noise=noise, amp=False, **lens)
            fwd = list(log["forward"])
            check(f"defaults: {fam.steps} steps, layouts at steps {refresh}, previews at 1/{factor} of {side} px",
                  len(fwd) == fam.steps and log["build_steps"] == refresh and vae.calls == len(refresh) - 1 + 1
                  and vae.downscales[:-1] == [factor] * (len(refresh) - 1) and vae.downscales[-1] == 1
                  and bool((L.count_tokens(res["layout"]) == L.snap_tokens(0.3)).all()) and abs(res["tokens"] - L.snap_tokens(0.3)) < 1e-6,
                  f"{len(fwd)} trunk passes, layouts before steps {log['build_steps']} (step 0 from the noise), "
                  f"{vae.calls - 1} previews, {L.snap_tokens(0.3)} tokens")
            f32 = lambda v: (float(torch.tensor(v, dtype=torch.float32)),) * 2                # the sampler's (B,) float32 times
            check("one row per image, t_ref = the next time of the schedule", {f[:3] for f in fwd} == {(2, 2, "Layout")}
                  and all(f[3] == f32(ts[i]) and f[4] == f32(ts[i + 1]) for i, f in enumerate(fwd)) and fwd[-1][4] == (1.0, 1.0),
                  f"step 0: t {fwd[0][3][0]:.4f} -> t_ref {fwd[0][4][0]:.4f}; last: t_ref 1 (clean)")
            explicit = S.sample(model, vae, cond, uncond, budget=0.3, noise=noise, amp=False, steps=fam.steps, cfg=1.0,
                                relayout_every=fam.relayout_every, relayout_until=fam.relayout_until, preview_px=fam.preview_px, **lens)
            check("defaults == the explicit settings", torch.equal(res["latent"], explicit["latent"]) and torch.equal(res["image"], explicit["image"])
                  and all(torch.equal(res["layout"][s], explicit["layout"][s]) for s in res["layout"]))
            m = raises(lambda: S.sample(model, vae, cond, model.project_text(torch.zeros(2, 5, TINY3D.context_in_dim)),
                                        budget=0.3, noise=noise, amp=False, cfg=3.5, **lens))
            check("a cfg other than 1 refused", "flow map" in m, m[:70])
            with_null = S.sample(model, vae, cond, model.project_text(torch.zeros(2, 5, TINY3D.context_in_dim)), budget=0.3,
                                 noise=noise, amp=False, **lens)
            check("an unconditional text is ignored (cfg 1.0)", torch.equal(res["latent"], with_null["latent"]))
            log["build_steps"].clear()
            log["forward"].clear()
            every = S.sample(model, vae, cond, uncond, budget=0.3, noise=noise, amp=False, relayout_every=4, relayout_until=None,
                             preview_px=side, **lens)
            check("the training validation's schedule on request (steps 0, 4, 8, 12, full previews)", log["build_steps"] == [0, 4, 8, 12]
                  and vae.downscales[-4:-1] == [1, 1, 1] and torch.isfinite(every["latent"]).all().item())

            log["forward"].clear()
            full = S.sample(model, None, cond, uncond, budget=1.0, noise=noise, amp=False, decode=False, **lens)
            z = noise.clone()
            for i in range(fam.steps):
                z = z + (ts[i + 1] - ts[i]) * model.forward_dense(z, torch.full((2,), ts[i]), cond, ctx_projected=True,
                                                                  txt_len=lens["len_cond"], t_ref=torch.full((2,), ts[i + 1]))
            d = float((full["latent"] - z).abs().max())
            check("100 % == plain unguided flow-map Euler (t_ref the next time)", d < 1e-5 and full["layout"] is None
                  and [f[:2] for f in log["forward"]] == [(2, 2)] * fam.steps, f"max |diff| {d:.1e}")

            from budgetlatent.inference.fast import warmup
            log["forward"].clear()
            log["build_steps"].clear()
            warmup(model, vae, cond, uncond, [0.5, 0.1], noise=noise, amp=False, **lens)
            n = fam.relayout_every + 2
            per = [i for i in range(n) if i % fam.relayout_every == 0 and (fam.relayout_until is None or i <= fam.relayout_until)]
            check("fast.warmup: the family's cadence, one row, t_ref", len(log["forward"]) == 2 * n
                  and log["build_steps"] == per + [n + i for i in per] and {f[:2] for f in log["forward"]} == {(2, 2)}
                  and all(f[4] is not None for f in log["forward"]), f"{n} steps (relayout_every {fam.relayout_every} + 2) per budget")

            from budgetlatent.inference import validate as V
            text.seen.clear()
            with tempfile.TemporaryDirectory() as tmp:
                perf = V.render(model, vae, text, V.PROMPTS[:3], tmp, budgets=(1.0, 0.1), batch=2, thumb=32, amp=False)
                strips = sorted(os.listdir(tmp))
            check("validation suite: the family's sampler, no empty prompt", "" not in text.seen and len(text.seen) == 3
                  and strips == ["000.jpg", "001.jpg", "002.jpg"] and perf["0.1"]["tokens"] == L.snap_tokens(0.1),
                  f"{len(text.seen)} prompts encoded, strips {strips}")
    finally:
        restore()


def test_scripts():
    print("[scripts]")
    for rel in ("scripts/inference/sample.py", "scripts/inference/gallery.py", "scripts/inference/bench.py", "scripts/fetch_weights.py"):
        out = subprocess.run([sys.executable, os.path.join(ROOT, rel), "--help"], capture_output=True, text=True)
        check(f"{rel} --family flux3_distilled", out.returncode == 0 and DISTILLED in out.stdout)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import fetch_weights
    check("fetch_weights: Klein's files as Klein Base's, gated", fetch_weights.PATTERNS[DISTILLED] == fetch_weights.PATTERNS["flux3"]
          and DISTILLED in fetch_weights.GATED and not any(p.startswith("variants") for p in fetch_weights.PATTERNS[DISTILLED]))


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    which = sys.argv[1:] or ["family", "trunk", "convert", "detection", "schedule", "sampler", "scripts"]
    for name in which:
        globals()[f"test_{name}"]()
    print(f"\n{len(OK)} checks passed")
