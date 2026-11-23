"""The FLUX 3 Image Klein DiT trunk (klein: 5 per-mode blocks per token mode + 28 single blocks, 3072 wide).

Written to the module names of BFL's reference implementation (github.com/black-forest-labs/flux-3-image-klein,
src/flux3/dit.py), so the released single file `model.safetensors` loads with strict key matching (convert.py maps
diffusers' Flux3Transformer2DModel names).  The sizes come from the release's config.json (Flux3Config.from_snapshot);
nothing below is specific to klein.

What FLUX 3 computes, and how it differs from FLUX.2 (flux2_trunk.py):

  * three token modes -- the noisy image, optional reference images ("image_cond", editing) and the text -- each
    with its own input projection and its own `depth` blocks that attend *within the mode only* (no image-text
    attention in the first blocks), then `depth_single` blocks shared by all modes over the joint sequence;
  * one block type everywhere: attention and SwiGLU MLP in parallel from the same modulated input, one gate,
    separate q / k / v projections; QK RMS norm and the rotary products in fp32 with one cast back;
  * modulation per token and per mode: the image tokens are modulated by the flow time, the text (and references)
    by time 0, with separate heads for the per-mode and the single blocks (no shared double/single heads);
  * image positions (0, h, w, 0) scaled by image_rope_scale (0.5), text (0, 0, 0, l); rope theta 10000;
  * the text window is the prompt's own length rounded up to 80 tokens, and the transformer attends to the padding
    inside that window -- so the window is part of the input (encoders/text.py).

What is different here is what BudgetPix needs around it (the same list as flux2_trunk.py): rotary tables passed in per
batch row (fractional centres, box damping), key masks / variable lengths (padded image tokens, and per-row text
windows), per-block gradient checkpointing, the final layer's modulated hidden state, and the last single block
querying the image tokens only.  Only the text-to-image path is built by default (image_cond=False): the reference
branch is 670 M parameters that text to image never runs; with image_cond=True it is built and loaded, and run_trunk
takes reference images (editing, multi-reference; models/refs.py) through it, dense, at time 0.

Time: the trunk takes the flow time sigma in [0, 1] (1 = noise) and returns the raw velocity eps - x, as FLUX.2.

FLUX 3 Image Klein, BFL's step-distilled release (config.json "flow_map": true, Flux3Config.flow_map), is a flow map:
the same trunk plus one more time embedder, ref_time_in (diffusers: target_timestep_embedder), for the time each
sampling step jumps to.  Its embedding is added to the time vector of every mode -- the image tokens', and the text's
(and references') at time 0 -- so it reaches the per-mode blocks, the single blocks and the final layer (BFL's
DiT(flow_map=True)); the output is then the average velocity from t to that time.  forward_dense / run_trunk take it
as t_ref; the base model has no ref_time_in and refuses a t_ref, the flow map requires one.
Token order in the single blocks: [image, text], with references [image, image_cond, text] (BFL's order; diffusers
concatenates [text, image] -- attention is order-free, only the summation order of the kernel differs).
"""
import json
import os
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import flux2_trunk as T          # the shared switches (SDPA_BACKEND, VARLEN_ATTENTION) and helpers
from .flux2_trunk import _eager, apply_rope, rope_tables, timestep_embedding

FAMILY = "flux3"

# The official bf16 paths put the time input on the bf16 grid before the sinusoid: diffusers 0.42.1 casts the timestep
# to the model dtype and multiplies by 1000 in it, BFL's sampler keeps the timesteps in a bf16 tensor (1000 t in bf16).
# This code keeps the time in fp32 (training and sampling alike) -- the exact schedule; TIME_BF16 = True reproduces the
# official rounding (tests/reference_flux3.py measures what it changes).  Off by default.
TIME_BF16 = False


@dataclass
class Flux3Config:                       # defaults: FLUX 3 Image Klein (config.json "dit" of the release)
    in_channels: int = 128               # image tokens: packed VAE latents (32 x 2 x 2)
    cond_in_channels: int = 128          # reference-image tokens (editing)
    context_in_dim: int = 20480          # 8 stacked Qwen3-VL layers x 2560
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5                       # per-mode blocks, for each mode
    depth_single_blocks: int = 28
    axes_dim: tuple = (32, 32, 32, 32)   # rotary dims of the (t, h, w, l) position axes
    theta: int = 10000
    mlp_ratio: float = 3.0
    image_rope_scale: float = 0.5        # image (h, w) position ids are multiplied by this
    image_cond: bool = False             # build the reference-image branch (editing: run_trunk's refs)
    flow_map: bool = False               # FLUX 3 Image Klein (step-distilled): ref_time_in embeds the time each step
                                         # jumps to (config.json "dit" "flow_map"; BFL's DiT(flow_map=True))

    @classmethod
    def from_dict(cls, d: dict, **over):
        """The release's config.json ("dit" section, BFL) or transformer/config.json (diffusers)."""
        d = dict(d.get("dit", d))
        if "hidden" in d:                                            # BFL
            ch = d.get("in_channels", {})
            kw = dict(in_channels=ch.get("image", 128), cond_in_channels=ch.get("image_cond", 128),
                      context_in_dim=ch.get("text", 20480), hidden_size=d["hidden"], num_heads=d["heads"],
                      depth=d["depth"], depth_single_blocks=d["depth_single"], axes_dim=tuple(d["axes_dim"]),
                      theta=d["theta"], mlp_ratio=d["mlp_ratio"], image_rope_scale=d.get("image_rope_scale", 0.5),
                      flow_map=bool(d.get("flow_map", False)))
        else:                                                        # diffusers
            kw = dict(in_channels=d["in_channels"], cond_in_channels=d.get("cond_in_channels", 128),
                      context_in_dim=d["joint_attention_dim"], hidden_size=d["num_attention_heads"] * d["attention_head_dim"],
                      num_heads=d["num_attention_heads"], depth=d["num_layers"], depth_single_blocks=d["num_single_layers"],
                      axes_dim=tuple(d["axes_dims_rope"]), theta=d["rope_theta"], mlp_ratio=d["mlp_ratio"],
                      flow_map=bool(d.get("flow_map", False)))
        kw.update(over)
        return cls(**kw)

    @classmethod
    def from_snapshot(cls, root, **over):
        d = snapshot_config(root)
        return cls(**over) if d is None else cls.from_dict(d, **over)

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


def snapshot_config(root):
    """The trunk config a snapshot ships: BFL's config.json (with its "dit" section), else diffusers'
    transformer/config.json; None if it has neither."""
    for rel in ("config.json", os.path.join("transformer", "config.json")):
        p = os.path.join(root, rel)
        if os.path.exists(p):
            d = json.load(open(p))
            if rel == "config.json" and "dit" not in d:
                continue
            return d
    return None


def modes(cfg):
    """The content modes the trunk builds."""
    return ("image", "image_cond") if cfg.image_cond else ("image",)


# ---------------------------------------------------------------------------------------------------------
# attention over variable-length rows
# ---------------------------------------------------------------------------------------------------------
class KeyRuns:
    """Which keys of each batch row are real, as (start, length) runs; e.g. ((0, n_b), (N, w_b)) for the
    [image, text] sequence of the single blocks.  Host integers (built once per forward), so the attention can
    run one unmasked call per group of rows with the same runs (VARLEN_ATTENTION) or build a boolean mask."""
    __slots__ = ("runs",)

    def __init__(self, runs):
        self.runs = tuple(tuple((int(a), int(n)) for a, n in r if int(n) > 0) for r in runs)

    @classmethod
    def build(cls, segments, batch):
        """segments: the parts of the sequence in order, (offset, lengths per row or None = all real, size)."""
        rows = [[] for _ in range(batch)]
        for off, lengths, size in segments:
            for r in range(batch):
                n = size if lengths is None else lengths[r]
                if rows[r] and rows[r][-1][0] + rows[r][-1][1] == off:          # adjacent runs -> one
                    rows[r][-1] = (rows[r][-1][0], rows[r][-1][1] + n)
                else:
                    rows[r].append((off, n))
        return cls(rows)

    def full(self, length):
        return all(r == ((0, length),) for r in self.runs)

    def mask(self, length, device):
        """(B, 1, 1, length) bool, True = attend."""
        m = torch.zeros(len(self.runs), length, dtype=torch.bool)
        for i, r in enumerate(self.runs):
            for a, n in r:
                m[i, a:a + n] = True
        return m.to(device)[:, None, None, :]


@torch.compiler.disable
def runs_attention(q, k, v, keys: KeyRuns):
    """Unpadded attention: row b attends over its real keys only; the queries are the first Lq positions (Lq < Lk
    when the last block queries the image tokens only).  Rows with the same runs share one call; a prefix needs
    no gather.  Query rows that are not real come out 0.  Always eager (a graph break under torch.compile)."""
    b, h, lq, d = q.shape
    out, i = [], 0
    while i < b:
        j, runs = i + 1, keys.runs[i]
        while j < b and keys.runs[j] == runs:
            j += 1
        if len(runs) == 1 and runs[0][0] == 0:
            n = runs[0][1]
            nq = min(n, lq)
            o = T._sdpa(q[i:j, :, :nq], k[i:j, :, :n], v[i:j, :, :n])
            out.append(F.pad(o, (0, 0, 0, lq - nq)) if nq < lq else o)
        else:
            kpos = [p for a, n in runs for p in range(a, a + n)]
            kidx = torch.tensor(kpos, device=q.device)
            qidx = torch.tensor([p for p in kpos if p < lq], device=q.device)
            o = T._sdpa(q[i:j].index_select(2, qidx), k[i:j].index_select(2, kidx), v[i:j].index_select(2, kidx))
            full = o.new_zeros(j - i, h, lq, d)
            out.append(full.index_copy(2, qidx, o))
        i = j
    return out[0] if len(out) == 1 else torch.cat(out)


def attend(q, k, v, keys=None):
    """q (B, H, Lq, D), k / v (B, H, Lk, D), already normed and rotated; keys None (all real), a bool mask
    (B, 1, 1, Lk) or KeyRuns.  Returns (B, Lq, H * D)."""
    if isinstance(keys, KeyRuns):
        out = runs_attention(q, k, v, keys)
    elif keys is None:
        out = T._sdpa(q, k, v)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=keys)
    return out.transpose(1, 2).flatten(2)


# ---------------------------------------------------------------------------------------------------------
# modules (BFL's names)
# ---------------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)


def rmsnorm_rope(x, scale, pe, eps=1e-6):
    """RMS norm and rotation in fp32, one cast back to x's dtype (BFL's rmsnorm_rope)."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * scale.float()
        return apply_rope(xf, *pe).to(x.dtype)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.out_layer(F.silu(self.in_layer(x)))


class ModeModulation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, vec):
        """vec (B, D) or (B, L, D) -> (shift, scale, gate), each (B, 1 or L, D)."""
        out = self.lin(F.silu(vec))
        return tuple((out if out.dim() == 3 else out[:, None]).chunk(3, dim=-1))


class Block(nn.Module):
    """FLUX 3's one block type: parallel attention + SwiGLU MLP from the same modulated input, one gate."""

    def __init__(self, hidden, heads, mlp_ratio):
        super().__init__()
        self.heads = heads
        mlp_hidden = int(hidden * mlp_ratio)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_in = nn.Linear(hidden, mlp_hidden * 2, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, hidden, bias=False)
        self.norm = QKNorm(hidden // heads)
        # fast.fuse_projections: the six projections as two matmuls.  w_in stacks the k, v, q and mlp_in weights
        # (rows), w_out puts attn_out and mlp_out side by side (columns); the six Linear weights become views of them,
        # so the parameters and the state dict are unchanged.  None: the six separate matmuls (as trained).
        self.w_in = None
        self.w_out = None

    def forward(self, x, mod, pe, keys=None, n_query=0):
        """x (B, L, D); mod (shift, scale, gate), each (B, 1 or L, D); pe (cos, sin) (B or 1, 1, L, Dh / 2);
        keys see `attend`; n_query > 0: only the first n_query tokens are queried and returned."""
        shift, scale, gate = mod
        b, length, _ = x.shape
        h = (1.0 + scale) * F.layer_norm(x, x.shape[-1:], eps=1e-6) + shift
        split = lambda t: t.view(b, t.shape[1], self.heads, -1).transpose(1, 2)
        if self.w_in is not None:
            return self._forward_fused(x, h, gate, pe, keys, n_query, split)
        k = rmsnorm_rope(split(self.k_proj(h)), self.norm.key_norm.scale, pe)
        v = split(self.v_proj(h))
        hq, pe_q = h, pe
        if n_query:
            hq, x = h[:, :n_query], x[:, :n_query]
            pe_q = (pe[0][:, :, :n_query], pe[1][:, :, :n_query])
            gate = gate[:, :n_query] if gate.shape[1] > 1 else gate
        q = rmsnorm_rope(split(self.q_proj(hq)), self.norm.query_norm.scale, pe_q)
        attn = attend(q, k, v.to(q.dtype), keys)
        gate_in, up = self.mlp_in(hq).chunk(2, dim=-1)
        return x + gate * (self.attn_out(attn) + self.mlp_out(F.silu(gate_in) * up))

    def _forward_fused(self, x, h, gate, pe, keys, n_query, split):
        """forward with the projections fused (fast.fuse_projections): one matmul for k, v, q and the MLP input, one
        for attn_out + mlp_out over [attention, MLP activation].  The same function of the weights; the last bits
        differ (summation order, and the two output projections summed in fp32 before rounding)."""
        d = self.w_out.shape[0]
        if n_query:                                   # keys and values over every token, queries and MLP on the first n_query
            kv = F.linear(h, self.w_in[:2 * d])
            hq, x = h[:, :n_query], x[:, :n_query]
            pe_q = (pe[0][:, :, :n_query], pe[1][:, :, :n_query])
            gate = gate[:, :n_query] if gate.shape[1] > 1 else gate
            qm = F.linear(hq, self.w_in[2 * d:])
        else:
            y = F.linear(h, self.w_in)
            kv, qm, pe_q = y[..., :2 * d], y[..., 2 * d:], pe
        k = rmsnorm_rope(split(kv[..., :d]), self.norm.key_norm.scale, pe)
        v = split(kv[..., d:])
        q = rmsnorm_rope(split(qm[..., :d]), self.norm.query_norm.scale, pe_q)
        attn = attend(q, k, v.to(q.dtype), keys)
        gate_in, up = qm[..., d:].chunk(2, dim=-1)
        return x + gate * F.linear(torch.cat((attn, F.silu(gate_in) * up), dim=-1), self.w_out)


class LastLayer(nn.Module):
    def __init__(self, hidden, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False))

    def forward(self, x, vec):
        """-> (the modulated hidden state (B, N, D), the prediction (B, N, C))."""
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        if shift.dim() == 2:
            shift, scale = shift[:, None], scale[:, None]
        hidden = (1.0 + scale) * F.layer_norm(x, x.shape[-1:], eps=1e-6) + shift
        return hidden, self.linear(hidden)


# ---------------------------------------------------------------------------------------------------------
# the trunk
# ---------------------------------------------------------------------------------------------------------
class Flux3Trunk(nn.Module):
    family = FAMILY
    # the text-mode blocks see the text alone at a constant time (0) unless a flow map adds t_ref: their output is the
    # same at every step, so `sample` can pass a txt_cache and they run once per sample (run_trunk)
    caches_text = True

    def __init__(self, cfg: Flux3Config = Flux3Config()):
        super().__init__()
        if cfg.hidden_size % cfg.num_heads or sum(cfg.axes_dim) != cfg.hidden_size // cfg.num_heads:
            raise ValueError(f"bad trunk geometry: {cfg}")
        self.cfg = cfg
        self.in_channels, self.hidden_size, self.num_heads = cfg.in_channels, cfg.hidden_size, cfg.num_heads
        dim, ms = cfg.hidden_size, modes(cfg)
        chans = {"image": cfg.in_channels, "image_cond": cfg.cond_in_channels}
        self.emb_in = nn.ModuleDict({m: nn.Linear(chans[m], dim, bias=False) for m in ms})
        self.txt_in = nn.Linear(cfg.context_in_dim, dim, bias=False)
        self.time_in = MLPEmbedder(256, dim)
        if cfg.flow_map:
            self.ref_time_in = MLPEmbedder(256, dim)
        self.early_stream_modulations = nn.ModuleDict({m: ModeModulation(dim) for m in (*ms, "txt")})
        self.single_stream_modulations = nn.ModuleDict({m: ModeModulation(dim) for m in (*ms, "txt")})
        block = lambda: Block(dim, cfg.num_heads, cfg.mlp_ratio)
        self.content_mode_blocks = nn.ModuleDict({m: nn.ModuleList([block() for _ in range(cfg.depth)]) for m in ms})
        self.txt_mode_blocks = nn.ModuleList([block() for _ in range(cfg.depth)])
        self.single_blocks = nn.ModuleList([block() for _ in range(cfg.depth_single_blocks)])
        self.final_layer = nn.ModuleDict({"image": LastLayer(dim, cfg.in_channels)})
        self.checkpoint_blocks = 0           # how many blocks (in forward order: depth image + text pairs, single) are checkpointed
        self._pe_cache = {}

    @property
    def num_blocks(self):
        """Blocks in forward order: the per-mode image and text blocks (one pair per depth), then the single blocks."""
        return 2 * self.cfg.depth + len(self.single_blocks)

    # ---- the interface BudgetPix uses (budgetpix.py; flux2_trunk.Flux2Trunk has the same) ---------------------
    def embed_cells(self, x):
        """(B, L, C) latent cells -> (B, L, D): FLUX 3's own image projection."""
        return self.emb_in["image"](x)

    def time_vec(self, sigma, embedder=None):
        """The time embedding of sigma (B,) by time_in, or by `embedder` (a flow map's ref_time_in, see ref_vec)."""
        embedder = embedder or self.time_in
        if TIME_BF16:
            emb = timestep_embedding((sigma.to(torch.bfloat16) * 1000).float(), time_factor=1.0)
        else:
            emb = timestep_embedding(sigma)
        if not torch.is_autocast_enabled():
            emb = emb.to(embedder.in_layer.weight.dtype)
        return embedder(emb)

    def ref_vec(self, t_ref):
        """Flow map: the embedding of the time each sample jumps to (t_ref (B,), 0 = noise, 1 = clean; the trunk's own
        sigma_ref = 1 - t_ref), which is added to the time vector of every mode.  None for the base model, which takes
        no t_ref; either mismatch raises, as BFL's DiT.forward does."""
        if (t_ref is not None) != self.cfg.flow_map:
            raise ValueError("t_ref is required by, and only by, a flow-map (step-distilled) FLUX 3 trunk")
        return None if t_ref is None else self.time_vec(1.0 - t_ref.float(), self.ref_time_in)

    def project_text(self, ctx):
        """(B, Lt, 20480) Qwen3-VL states -> (B, Lt, hidden); depends on the caption only."""
        return self.txt_in(ctx)

    def text_pe(self, length, device):
        """Text ids (0, 0, 0, l): (1, length, Dh / 2) each."""
        key = ("txt", length, str(device))
        if key not in self._pe_cache:
            pos = torch.zeros(1, length, 4, dtype=torch.float64, device=device)
            pos[0, :, 3] = torch.arange(length, dtype=torch.float64, device=device)
            self._pe_cache[key] = rope_tables(pos, self.cfg.axes_dim, self.cfg.theta)
        return self._pe_cache[key]

    def image_pe(self, cy, cx, var=None):
        """Rotary tables of image tokens centred at (cy, cx) cells (B, N) float64; var (B, N) the box variance in
        cells^2 (None: no damping).  FLUX 3 scales the image ids by image_rope_scale, so a variance by its square."""
        r = float(self.cfg.image_rope_scale)
        pos = torch.zeros(*cy.shape, 4, dtype=torch.float64, device=cy.device)
        pos[..., 1], pos[..., 2] = r * cy, r * cx
        v = None
        if var is not None:
            v = torch.zeros_like(pos)
            v[..., 1] = v[..., 2] = (r * r) * var
        return rope_tables(pos, self.cfg.axes_dim, self.cfg.theta, v)

    def grid_pe(self, gh, gw, device):
        key = ("img", gh, gw, str(device))
        if key not in self._pe_cache:
            cy = torch.arange(gh, dtype=torch.float64, device=device)[:, None].expand(gh, gw).reshape(1, -1)
            cx = torch.arange(gw, dtype=torch.float64, device=device)[None, :].expand(gh, gw).reshape(1, -1)
            self._pe_cache[key] = self.image_pe(cy, cx)
        return self._pe_cache[key]

    def final(self, img, vec):
        return self.final_layer["image"](img, vec)

    # ---- the blocks ----------------------------------------------------------------------------------------
    def run_trunk(self, img, txt, vec, pe_img, pe_txt, img_valid=None, txt_len=None, img_only=True, vref=None,
                  txt_cache=None, refs=None):
        """img (B, N, D) embedded image tokens, txt (B, Lt, D) projected text, vec (B, D) time embedding of the
        image tokens, pe_img / pe_txt rotary tables (B or 1, N or Lt, Dh / 2); img_valid (B, N) bool for padded
        image tokens (real ones first), txt_len (B,) the text window of each row (None: all Lt).  Returns the image
        tokens after the last single block, (B, N, D).  vref (B, D): a flow map's ref_vec, already added to `vec`;
        added here to the text's time vector too.  txt_cache: a dict that lives as long as one sample (the same text
        at every call): the text-mode blocks' output is kept in it and reused -- the same tensor they would compute
        again, since they see the text alone at time 0 (not with vref, which changes every step).

        refs (models/refs.References): reference images (editing), BFL's "image_cond" mode -- its own projection,
        per-mode blocks and modulations at time 0 (plus vref), then [image, image_cond, text] in the single blocks;
        never padded, and like the text their per-mode blocks run once per sample through txt_cache (base model)."""
        b, n, lt = img.shape[0], img.shape[1], txt.shape[1]
        nr = 0 if refs is None else refs.length
        if refs is not None and "image_cond" not in self.emb_in:
            raise ValueError("reference images need the image_cond branch: build the trunk with image_cond=True")
        vec0 = self.time_vec(torch.zeros(b, device=img.device))       # text (and references) sit at time 0
        if vref is not None:
            vec0 = vec0 + vref
        e_img, e_txt = self.early_stream_modulations["image"](vec), self.early_stream_modulations["txt"](vec0)
        s_img, s_txt = self.single_stream_modulations["image"](vec), self.single_stream_modulations["txt"](vec0)
        n_img = None if img_valid is None else [int(v) for v in img_valid.sum(dim=1).tolist()]
        w_txt = None if txt_len is None else [int(v) for v in torch.as_tensor(txt_len).tolist()]
        if n_img is not None and all(v == n for v in n_img):
            n_img = None
        if w_txt is not None and all(v == lt for v in w_txt):
            w_txt = None
        keys_txt = None if w_txt is None else KeyRuns.build([(0, w_txt, lt)], b)
        keys_img = None if n_img is None else KeyRuns.build([(0, n_img, n)], b)
        segs = [(0, n_img, n)] + ([(n, None, nr)] if nr else []) + [(n + nr, w_txt, lt)]
        keys_joint = None if n_img is None and w_txt is None else KeyRuns.build(segs, b)
        if not T.VARLEN_ATTENTION:                                   # boolean masks, built once (compiled blocks never see the runs)
            keys_txt, keys_img, keys_joint = (None if k is None else k.mask(size, img.device)
                                              for k, size in ((keys_txt, lt), (keys_img, n), (keys_joint, n + nr + lt)))
        add_head = lambda pe: (pe[0][:, None], pe[1][:, None])
        pi, pt = add_head(pe_img), add_head(pe_txt)
        if nr:
            pr = add_head(refs.pe(self, img.device))
            pj = tuple(torch.cat((a.expand(b, -1, -1, -1), r.expand(b, -1, -1, -1), c.expand(b, -1, -1, -1)), dim=2)
                       for a, r, c in zip(pi, pr, pt))
        else:
            pj = tuple(torch.cat((a.expand(b, -1, -1, -1), c.expand(b, -1, -1, -1)), dim=2) for a, c in zip(pi, pt))
        n_ckpt = self.checkpoint_blocks if torch.is_grad_enabled() else 0

        def run(i, block, *args, eager=False):
            if i < n_ckpt:
                return checkpoint(_eager(block), *args, use_reentrant=False)
            return _eager(block)(*args) if eager else block(*args)

        cache = txt_cache if txt_cache is not None and vref is None and not n_ckpt else None
        cached = None if cache is None else cache.get("txt_mode_blocks")
        if cached is not None and cached.shape != txt.shape:
            raise ValueError(f"txt_cache holds text of shape {tuple(cached.shape)}, this call has {tuple(txt.shape)}: "
                             f"one cache per sample")
        ref = None
        if nr:
            ref = None if cache is None else cache.get("ref_mode_blocks")
            if ref is not None and ref.shape[:2] != (b, nr):
                raise ValueError("txt_cache holds other references: one cache per sample")
            run_ref = ref is None
            if run_ref:
                proj = self.emb_in["image_cond"]
                ref = proj(refs.on(img.device, proj.weight.dtype)).expand(b, -1, -1).to(img.dtype)
                e_ref = self.early_stream_modulations["image_cond"](vec0)
            s_ref = self.single_stream_modulations["image_cond"](vec0)
        for d in range(self.cfg.depth):
            img = run(2 * d, self.content_mode_blocks["image"][d], img, e_img, pi, keys_img)
            if cached is None:
                txt = run(2 * d + 1, self.txt_mode_blocks[d], txt, e_txt, pt, keys_txt)
            if nr and run_ref:
                ref = _eager(self.content_mode_blocks["image_cond"][d])(ref, e_ref, pr, None)
        if cached is not None:
            txt = cached
        elif cache is not None:
            cache["txt_mode_blocks"] = txt
        if nr and run_ref and cache is not None:
            cache["ref_mode_blocks"] = ref
        x = torch.cat((img, txt), dim=1) if not nr else torch.cat((img, ref, txt), dim=1)
        if nr:
            mod = tuple(torch.cat((a.expand(b, n, -1), r.expand(b, nr, -1), c.expand(b, lt, -1)), dim=1)
                        for a, r, c in zip(s_img, s_ref, s_txt))
        else:
            mod = tuple(torch.cat((a.expand(b, n, -1), c.expand(b, lt, -1)), dim=1) for a, c in zip(s_img, s_txt))
        first, last = 2 * self.cfg.depth, len(self.single_blocks) - 1
        for i, block in enumerate(self.single_blocks):
            # the last block queries the image tokens only; it runs its plain forward even when the blocks are compiled
            # (inductor, torch 2.9: slicing by the symbolic query count after the graph break of the per-group attention
            # fails to codegen -- "expected [s..] to have been codegen-ed")
            nq = n if (img_only and i == last) else 0
            x = run(first + i, block, x, mod, pj, keys_joint, nq, eager=bool(nq))
        return x[:, :n]

    # ---- the released model -------------------------------------------------------------------------------
    def forward_dense(self, z, t, ctx, ctx_projected=False, txt_len=None, t_ref=None, txt_cache=None, refs=None):
        """The released model on the dense grid.  z (B, gh, gw, C) noisy latent, t (B,) in [0, 1] with 0 = noise and
        1 = clean, ctx (B, Lt, context_in_dim) (or projected), txt_len (B,) text windows, t_ref (B,) the time a flow
        map jumps to (None for the base model).  Returns dz/dt = x - eps, (B, gh, gw, C) fp32 -- for a flow map the
        average over [t, t_ref]: z(t_ref) = z + (t_ref - t) v."""
        b, gh, gw, _ = z.shape
        sigma = 1.0 - t.float()
        vref = self.ref_vec(t_ref)
        vec = self.time_vec(sigma) if vref is None else self.time_vec(sigma) + vref
        txt = ctx if ctx_projected else self.txt_in(ctx)
        img = self.run_trunk(self.embed_cells(z.flatten(1, 2)), txt, vec, self.grid_pe(gh, gw, z.device),
                             self.text_pe(txt.shape[1], z.device), txt_len=txt_len, vref=vref, txt_cache=txt_cache,
                             **({} if refs is None else {"refs": refs}))
        _, raw = self.final(img, vec)
        return -raw.float().view(b, gh, gw, -1)

    def forward(self, z, t, ctx, layout=None, ctx_projected=False, txt_len=None, t_ref=None, refs=None):
        if layout is not None:
            raise ValueError("the plain trunk only runs the dense grid; layouts need BudgetPixFlux3")
        return self.forward_dense(z, t, ctx, ctx_projected, txt_len, t_ref=t_ref, refs=refs)


def released_param_count(cfg: Flux3Config) -> int:
    """Parameters of the trunk this config builds (meta tensors: no memory)."""
    with torch.device("meta"):
        return sum(p.numel() for p in Flux3Trunk(cfg).parameters())
