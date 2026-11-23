"""Key maps between the checkpoint layouts of the trunks and the BudgetPix modules.

  bfl        the names used in this code: BFL's reference names for the trunk (`double_blocks.0.img_attn.qkv.weight`)
             plus `scale_emb`, `mixers.{2,4,8}.*`, `ipe_2d.*`, `refiners.{8,4,2}.*`
  diffusers  `Flux2Transformer2DModel` (trunk only): split q / k / v, and the final modulation stored (scale, shift)
  dit        the dit/ export, `dit/<path>/{kernel,bias,scale}`: Linear kernels (in, out), conv kernels HWIO

FLUX 3 (flux3_trunk.py): `bfl` is the release's single file `model.safetensors` (BFL's dit.py names:
`content_mode_blocks.image.0.q_proj.weight`, ...), `diffusers` is `Flux3Transformer2DModel` (flux3_diffusers_to_bfl).
FLUX 3 Image Klein (the step-distilled flow map) adds one time embedder: BFL's `ref_time_in.{in,out}_layer`, diffusers'
`target_timestep_embedder.linear_{1,2}`.  The dit/ export is FLUX.2's only.
"""
import re

import torch

ADAPTER_PREFIXES = ("scale_emb", "mixers.", "ipe_2d.", "refiners.")
_INDEXED = ("double_blocks", "single_blocks", "mixers", "refiners", "blocks", "mlp")
_DIT_INDEXED = re.compile(rf"^({'|'.join(_INDEXED)})_(\d+)$")
_DIT_LAYER = re.compile(r"^layers_(\d+)$")


def is_adapter(name: str) -> bool:
    return name.startswith(ADAPTER_PREFIXES)


def detect(keys) -> str:
    keys = set(keys)
    if any(k.startswith("dit/") for k in keys):
        return "dit"
    if "x_embedder.weight" in keys:
        return "diffusers"
    if "x_embedders.image.weight" in keys:
        return "flux3_diffusers"
    if "img_in.weight" in keys or "emb_in.image.weight" in keys:
        return "bfl"
    if any(is_adapter(k) for k in keys) and all(is_adapter(k) for k in keys):
        return "bfl"                                                 # added modules only
    raise ValueError("unknown checkpoint layout (neither BFL, diffusers nor the dit/ export)")


# ---------------------------------------------------------------------------------------------------------
# diffusers -> bfl (trunk)
# ---------------------------------------------------------------------------------------------------------
def diffusers_to_bfl(sd: dict) -> dict:
    out = {}

    def take(src, dst):
        out[dst] = sd[src]

    take("x_embedder.weight", "img_in.weight")
    take("context_embedder.weight", "txt_in.weight")
    t = "time_guidance_embed" if "time_guidance_embed.timestep_embedder.linear_1.weight" in sd else "time_embed"
    take(f"{t}.timestep_embedder.linear_1.weight", "time_in.in_layer.weight")
    take(f"{t}.timestep_embedder.linear_2.weight", "time_in.out_layer.weight")
    for name in ("double_stream_modulation_img", "double_stream_modulation_txt", "single_stream_modulation"):
        take(f"{name}.linear.weight", f"{name}.lin.weight")
    n_double = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("transformer_blocks."))
    n_single = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("single_transformer_blocks."))
    for i in range(n_double):
        s, d = f"transformer_blocks.{i}", f"double_blocks.{i}"
        out[f"{d}.img_attn.qkv.weight"] = torch.cat([sd[f"{s}.attn.to_{c}.weight"] for c in "qkv"], dim=0)
        out[f"{d}.txt_attn.qkv.weight"] = torch.cat([sd[f"{s}.attn.add_{c}_proj.weight"] for c in "qkv"], dim=0)
        take(f"{s}.attn.norm_q.weight", f"{d}.img_attn.norm.query_norm.scale")
        take(f"{s}.attn.norm_k.weight", f"{d}.img_attn.norm.key_norm.scale")
        take(f"{s}.attn.norm_added_q.weight", f"{d}.txt_attn.norm.query_norm.scale")
        take(f"{s}.attn.norm_added_k.weight", f"{d}.txt_attn.norm.key_norm.scale")
        take(f"{s}.attn.to_out.0.weight", f"{d}.img_attn.proj.weight")
        take(f"{s}.attn.to_add_out.weight", f"{d}.txt_attn.proj.weight")
        take(f"{s}.ff.linear_in.weight", f"{d}.img_mlp.0.weight")
        take(f"{s}.ff.linear_out.weight", f"{d}.img_mlp.2.weight")
        take(f"{s}.ff_context.linear_in.weight", f"{d}.txt_mlp.0.weight")
        take(f"{s}.ff_context.linear_out.weight", f"{d}.txt_mlp.2.weight")
    for i in range(n_single):
        s, d = f"single_transformer_blocks.{i}", f"single_blocks.{i}"
        take(f"{s}.attn.to_qkv_mlp_proj.weight", f"{d}.linear1.weight")
        take(f"{s}.attn.to_out.weight", f"{d}.linear2.weight")
        take(f"{s}.attn.norm_q.weight", f"{d}.norm.query_norm.scale")
        take(f"{s}.attn.norm_k.weight", f"{d}.norm.key_norm.scale")
    scale, shift = sd["norm_out.linear.weight"].chunk(2, dim=0)          # diffusers: (scale, shift) -> (shift, scale)
    out["final_layer.adaLN_modulation.1.weight"] = torch.cat([shift, scale], dim=0)
    take("proj_out.weight", "final_layer.linear.weight")
    return out


# ---------------------------------------------------------------------------------------------------------
# the dit/ export <-> bfl
# ---------------------------------------------------------------------------------------------------------
def dit_key_to_torch(key: str) -> str:
    parts = key.split("/")
    if parts[0] != "dit":
        raise ValueError(f"expected a dit/ key, got {key}")
    parts = parts[1:]
    out = []
    for p in parts[:-1]:
        m = _DIT_INDEXED.match(p)
        if m:
            out += [m.group(1), m.group(2)]
            continue
        m = _DIT_LAYER.match(p)
        out.append(m.group(1) if m else p)
    out.append({"kernel": "weight"}.get(parts[-1], parts[-1]))
    return ".".join(out)


def torch_key_to_dit(key: str) -> str:
    parts = key.split(".")
    out = []
    for p in parts[:-1]:
        if p.isdigit():
            if out and out[-1] in _INDEXED:
                out[-1] = f"{out[-1]}_{p}"
            else:
                out.append(f"layers_{p}")
        else:
            out.append(p)
    out.append({"weight": "kernel"}.get(parts[-1], parts[-1]))
    return "dit/" + "/".join(out)


def dit_to_torch(flat: dict) -> dict:
    """The dit/ export -> a state dict for BudgetPixFlux2 (Linear kernels transposed, conv kernels HWIO -> OIHW)."""
    out = {}
    for k, v in flat.items():
        name = dit_key_to_torch(k)
        if name.endswith(".weight"):
            v = v.permute(3, 2, 0, 1) if v.ndim == 4 else v.t()
        out[name] = v.contiguous()
    return out


def torch_to_dit(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k.endswith(".weight"):
            v = v.permute(2, 3, 1, 0) if v.ndim == 4 else v.t()
        out[torch_key_to_dit(k)] = v.contiguous()
    return out


def to_bfl(sd: dict) -> dict:
    """Any of the layouts -> the names of this code."""
    kind = detect(sd.keys())
    if kind == "dit":
        return dit_to_torch(sd)
    if kind == "diffusers":
        return diffusers_to_bfl(sd)
    if kind == "flux3_diffusers":
        return flux3_diffusers_to_bfl(sd)
    return sd


# ---------------------------------------------------------------------------------------------------------
# FLUX 3: diffusers -> bfl (trunk)
# ---------------------------------------------------------------------------------------------------------
_F3_TOP = [
    (re.compile(r"^x_embedders\.(\w+)\.weight$"), r"emb_in.\1.weight"),
    (re.compile(r"^context_embedder\.weight$"), "txt_in.weight"),
    (re.compile(r"^timestep_embedder\.linear_1\.weight$"), "time_in.in_layer.weight"),
    (re.compile(r"^timestep_embedder\.linear_2\.weight$"), "time_in.out_layer.weight"),
    (re.compile(r"^target_timestep_embedder\.linear_1\.weight$"), "ref_time_in.in_layer.weight"),      # the flow map's
    (re.compile(r"^target_timestep_embedder\.linear_2\.weight$"), "ref_time_in.out_layer.weight"),
    (re.compile(r"^(early|single)_stream_modulations\.(\w+)\.linear\.weight$"), r"\1_stream_modulations.\2.lin.weight"),
    (re.compile(r"^proj_out\.weight$"), "final_layer.image.linear.weight"),
]
_F3_BLOCK = re.compile(r"^(mode_blocks\.\w+|txt_mode_blocks|single_transformer_blocks)\.(\d+)\.(.+)$")
_F3_INNER = {"attn.to_q.weight": "q_proj.weight", "attn.to_k.weight": "k_proj.weight", "attn.to_v.weight": "v_proj.weight",
             "attn.to_out.0.weight": "attn_out.weight", "attn.norm_q.weight": "norm.query_norm.scale",
             "attn.norm_k.weight": "norm.key_norm.scale", "ff.linear_in.weight": "mlp_in.weight",
             "ff.linear_out.weight": "mlp_out.weight"}


def flux3_key(k: str) -> str:
    """One Flux3Transformer2DModel name -> BFL's (norm_out is handled by flux3_diffusers_to_bfl)."""
    for pat, rep in _F3_TOP:
        if pat.match(k):
            return pat.sub(rep, k)
    m = _F3_BLOCK.match(k)
    if m and m.group(3) in _F3_INNER:
        group = m.group(1)
        group = "content_" + group if group.startswith("mode_blocks.") else ("single_blocks" if group.startswith("single") else group)
        return f"{group}.{m.group(2)}.{_F3_INNER[m.group(3)]}"
    raise KeyError(f"unknown Flux3Transformer2DModel key {k}")


def flux3_diffusers_to_bfl(sd: dict) -> dict:
    """A Flux3Transformer2DModel state dict (Klein Base, or the flow map Klein with its target_timestep_embedder) ->
    BFL's names."""
    out = {flux3_key(k): v for k, v in sd.items() if k != "norm_out.linear.weight"}
    scale, shift = sd["norm_out.linear.weight"].chunk(2, dim=0)          # diffusers: (scale, shift) -> BFL (shift, scale)
    out["final_layer.image.adaLN_modulation.1.weight"] = torch.cat([shift, scale], dim=0)
    return out


FLUX3_COND_PREFIXES = ("emb_in.image_cond.", "content_mode_blocks.image_cond.", "early_stream_modulations.image_cond.",
                       "single_stream_modulations.image_cond.")


def is_flux3_cond(name: str) -> bool:
    """The reference-image (editing) branch of FLUX 3, which text-to-image BudgetPix does not build."""
    return name.startswith(FLUX3_COND_PREFIXES)
