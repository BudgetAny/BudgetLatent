"""The FLUX.2 [klein] MMDiT trunk (5 double-stream + 20 single-stream blocks at 4B).

Written to the module names of BFL's reference implementation (github.com/black-forest-labs/flux2,
src/flux2/model.py), so the released single-file checkpoint `flux-2-klein-base-4b.safetensors` loads with strict
key matching.  It computes what the reference computes; what is different is what BudgetPix needs around it:

  * rotary positions are passed in as (cos, sin) tables per batch row, so tokens can sit at fractional centres
    and coarse tokens can damp their high frequencies (budgetpix.py builds the tables);
  * an optional key mask, for batches whose images have different token counts;
  * gradient checkpointing per block (all blocks, or only the first `checkpoint_blocks`);
  * the final layer also returns the modulated hidden state of every token (the refiners condition on it);
  * the last single-stream block skips the text queries, whose outputs nothing reads;
  * stage 2 only (training/dmd.py): optional low-rank residuals around the blocks, and the features its discriminator reads.

Time: the trunk takes FLUX.2's own sigma in [0, 1] (1 = noise) and returns FLUX.2's raw prediction eps - x.
Precision: call it under torch.autocast(bfloat16) for bf16 compute (weights may be stored bf16 or fp32); layer
norms, RMS norms, the time sinusoid and the rotary products are fp32 either way.  Without autocast and with fp32
weights everything is fp32.
"""
import functools
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:                      # torch < 2.3: no kernel selection
    SDPBackend = sdpa_kernel = None


def _eager(block):
    """The block's plain (uncompiled) forward.  A gradient-checkpointed block runs eagerly: torch.compile'd blocks
    (fast.compile_blocks) inside non-reentrant checkpointing can recompute a different graph in the backward than the
    forward ran (CheckpointError: recomputed tensors' metadata differ), so only the blocks that keep their
    activations are compiled -- which also loses nothing: their recompute is the cheap part."""
    return functools.partial(type(block).forward, block)


@dataclass
class TrunkConfig:                       # FLUX.2 [klein] 4B (Klein4BParams of the reference)
    in_channels: int = 128
    context_in_dim: int = 7680
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 20
    axes_dim: tuple = (32, 32, 32, 32)   # rotary dims of the (t, h, w, l) position axes
    theta: int = 2000
    mlp_ratio: float = 3.0


# Which scaled-dot-product kernel the unmasked attention uses: None = PyTorch's choice (FlashAttention),
# "cudnn" = cuDNN's fused kernel (faster on Hopper; see inference/fast.py).  Masked attention always uses PyTorch's
# choice.  The FLUX 3 trunk (flux3_trunk.py) reads these switches and the helpers below from this module.
SDPA_BACKEND = None

# Padded batches (the student's passes): False = one masked SDPA call per layer -- a mask rules FlashAttention out,
# so PyTorch runs its memory-efficient kernel; True = one unmasked call per run of equally long images, on the
# valid prefix [text, image tokens] of each (FlashAttention), padded query rows set to 0.  The valid tokens get the
# same attention either way; nothing reads the padded rows.  run_blocks turns the mask (valid tokens first, padding
# at the end, as layout.pack makes it) into host lengths once per forward (one device sync).  Off by default;
# scripts/train/train.py --varlen_attention.
VARLEN_ATTENTION = False

# The same fp32 rotation as one complex product (3 kernels instead of ~9 in eager mode); torch.compile fuses the
# real form anyway, so compiled code keeps it.  Off by default.
ROPE_COMPLEX = False


class KeyLengths:
    """A padded batch's key mask as host integers: the valid (text + image) length of every batch row.  Not a
    tuple, so that torch.compile passes it through to varlen_attention without guarding on the lengths."""
    __slots__ = ("lengths",)

    def __init__(self, lengths):
        self.lengths = tuple(int(n) for n in lengths)


def timestep_embedding(t, dim=256, max_period=10000, time_factor=1000.0):
    """Sinusoidal embedding of sigma (fp32): [cos, sin] halves."""
    t = time_factor * t.float()
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def rope_frequencies(axis_dim, theta, device=None):
    """omega_j = theta^(-2j / axis_dim), j = 0 .. axis_dim / 2 - 1 (float64)."""
    scale = torch.arange(0, axis_dim, 2, dtype=torch.float64, device=device) / axis_dim
    return 1.0 / (theta ** scale)


def rope_tables(pos, axes_dim, theta, var=None):
    """Rotary (cos, sin) tables.  pos (..., A) positions on the A axes, var (..., A) optional box variance of
    the token along each axis: both cos and sin of plane j are scaled by exp(-omega_j^2 var / 2), the expected
    rotation over a uniform box (var = 0 leaves the plain rotation).  Returns float32 (..., sum(axes_dim) / 2)."""
    cos, sin = [], []
    pos = pos.double()
    for a, d in enumerate(axes_dim):
        omega = rope_frequencies(d, theta, pos.device)
        ang = pos[..., a, None] * omega
        c, s = torch.cos(ang), torch.sin(ang)
        if var is not None:
            damp = torch.exp(-0.5 * omega ** 2 * var[..., a, None].double())
            c, s = c * damp, s * damp
        cos.append(c)
        sin.append(s)
    return torch.cat(cos, dim=-1).float(), torch.cat(sin, dim=-1).float()


def apply_rope(x, cos, sin):
    """Rotate adjacent pairs (2j, 2j + 1) of x (B, H, L, D) by the tables (B or 1, 1, L, D / 2), in fp32."""
    if ROPE_COMPLEX and not torch.compiler.is_compiling():
        xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)).contiguous())
        return torch.view_as_real(xc * torch.complex(cos, sin)).flatten(-2).to(x.dtype)
    x1, x2 = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((cos * x1 - sin * x2, sin * x1 + cos * x2), dim=-1).flatten(-2).to(x.dtype)


def _sdpa(q, k, v):
    if SDPA_BACKEND == "cudnn":
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v)
    return F.scaled_dot_product_attention(q, k, v)


@torch.compiler.disable
def varlen_attention(q, k, v, keys):
    """Unpadded attention of a padded batch (keys: KeyLengths): row b attends over its first keys.lengths[b] keys;
    its queries are the last Lq of those (Lq = Lk when every token queries, Lk - text length in the last
    single-stream block).  Rows with equal lengths share one call.  Query rows beyond a row's own length come out
    0.  Always eager: under torch.compile it is a graph break, so the lengths never become guards."""
    lengths = keys.lengths
    b, _, lq, _ = q.shape
    off = k.shape[2] - lq
    outs, i = [], 0
    while i < b:
        j, n = i + 1, lengths[i]
        while j < b and lengths[j] == n:
            j += 1
        o = _sdpa(q[i:j, :, :n - off], k[i:j, :, :n], v[i:j, :, :n])
        outs.append(F.pad(o, (0, 0, 0, lq - (n - off))) if n - off < lq else o)
        i = j
    return outs[0] if len(outs) == 1 else torch.cat(outs)


def attention(q, k, v, pe_q, pe_k, mask=None):
    """q (B, H, Lq, D), k / v (B, H, Lk, D); mask (B, 1, 1, Lk) bool, True = attend, or KeyLengths (see
    VARLEN_ATTENTION).  Returns (B, Lq, H * D)."""
    q = apply_rope(q, *pe_q)
    k = apply_rope(k, *pe_k)
    if isinstance(mask, KeyLengths):
        out = varlen_attention(q, k, v, mask)
    elif mask is None and SDPA_BACKEND == "cudnn":
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2).flatten(2)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.scale.float()).to(x.dtype)


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k):
        return self.query_norm(q), self.key_norm(k)


class SelfAttention(nn.Module):
    """Holds the projections of one stream; the joint attention itself runs in the block."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm = QKNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim, bias=False)

    def qkv_heads(self, x):
        b, l, _ = x.shape
        q, k, v = self.qkv(x).view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k)
        return q, k, v


class SiLUActivation(nn.Module):
    """SwiGLU without its own projection: SiLU(first half) * second half."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class Modulation(nn.Module):
    def __init__(self, dim, double):
        super().__init__()
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=False)

    def forward(self, vec):
        out = self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (out[:3], out[3:]) if self.multiplier == 6 else out      # (shift, scale, gate) triples


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.out_layer(self.silu(self.in_layer(x)))


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(hidden_size, num_heads)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden * 2, bias=False), SiLUActivation(),
                                     nn.Linear(mlp_hidden, hidden_size, bias=False))
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(hidden_size, num_heads)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden * 2, bias=False), SiLUActivation(),
                                     nn.Linear(mlp_hidden, hidden_size, bias=False))

    def forward(self, img, txt, mod_img, mod_txt, pe, mask=None):
        (i_sh1, i_sc1, i_g1), (i_sh2, i_sc2, i_g2) = mod_img
        (t_sh1, t_sc1, t_g1), (t_sh2, t_sc2, t_g2) = mod_txt
        img_q, img_k, img_v = self.img_attn.qkv_heads((1 + i_sc1) * self.img_norm1(img) + i_sh1)
        txt_q, txt_k, txt_v = self.txt_attn.qkv_heads((1 + t_sc1) * self.txt_norm1(txt) + t_sh1)
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        attn = attention(q, k, v, pe, pe, mask)
        lt = txt.shape[1]
        img = img + i_g1 * self.img_attn.proj(attn[:, lt:])
        img = img + i_g2 * self.img_mlp((1 + i_sc2) * self.img_norm2(img) + i_sh2)
        txt = txt + t_g1 * self.txt_attn.proj(attn[:, :lt])
        txt = txt + t_g2 * self.txt_mlp((1 + t_sc2) * self.txt_norm2(txt) + t_sh2)
        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio):
        super().__init__()
        self.hidden_size, self.num_heads = hidden_size, num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim * 2, bias=False)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=False)
        self.norm = QKNorm(hidden_size // num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation()

    def forward(self, x, mod, pe, mask=None, skip=0):
        """skip > 0: only the tokens from `skip` on are queried and returned (the keys are still all tokens)."""
        shift, scale, gate = mod
        b, l, _ = x.shape
        qkv, mlp = torch.split(self.linear1((1 + scale) * self.pre_norm(x) + shift),
                               [3 * self.hidden_size, 2 * self.mlp_hidden_dim], dim=-1)
        q, k, v = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k)
        pe_q = pe
        if skip:
            q, mlp, x = q[:, :, skip:], mlp[:, skip:], x[:, skip:]
            pe_q = (pe[0][:, :, skip:], pe[1][:, :, skip:])
        attn = attention(q, k, v, pe_q, pe, mask)
        return x + gate * self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=2))


class LastLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        hidden = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        return hidden, self.linear(hidden)


class Flux2Trunk(nn.Module):
    def __init__(self, cfg: TrunkConfig = TrunkConfig()):
        super().__init__()
        if cfg.hidden_size % cfg.num_heads or sum(cfg.axes_dim) != cfg.hidden_size // cfg.num_heads:
            raise ValueError(f"bad trunk geometry: {cfg}")
        self.cfg = cfg
        self.in_channels, self.hidden_size, self.num_heads = cfg.in_channels, cfg.hidden_size, cfg.num_heads
        self.img_in = nn.Linear(cfg.in_channels, cfg.hidden_size, bias=False)
        self.time_in = MLPEmbedder(256, cfg.hidden_size)
        self.txt_in = nn.Linear(cfg.context_in_dim, cfg.hidden_size, bias=False)
        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth)])
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth_single_blocks)])
        self.double_stream_modulation_img = Modulation(cfg.hidden_size, double=True)
        self.double_stream_modulation_txt = Modulation(cfg.hidden_size, double=True)
        self.single_stream_modulation = Modulation(cfg.hidden_size, double=False)
        self.final_layer = LastLayer(cfg.hidden_size, cfg.in_channels)
        self.checkpoint_blocks = 0           # how many blocks (in forward order) are gradient-checkpointed
        self._pe_cache = {}

    @property
    def num_blocks(self):
        return len(self.double_blocks) + len(self.single_blocks)

    # ---- positions ---------------------------------------------------------------------------------------
    def text_pe(self, length, device):
        """Rotary tables of the text tokens, ids (0, 0, 0, l): (1, length, D / 2) each."""
        key = ("txt", length, str(device))
        if key not in self._pe_cache:
            pos = torch.zeros(1, length, 4, dtype=torch.float64, device=device)
            pos[0, :, 3] = torch.arange(length, dtype=torch.float64, device=device)
            self._pe_cache[key] = rope_tables(pos, self.cfg.axes_dim, self.cfg.theta)
        return self._pe_cache[key]

    def grid_pe(self, gh, gw, device):
        """Rotary tables of the dense grid in row-major order, ids (0, h, w, 0): (1, gh * gw, D / 2) each."""
        key = ("img", gh, gw, str(device))
        if key not in self._pe_cache:
            pos = torch.zeros(1, gh, gw, 4, dtype=torch.float64, device=device)
            pos[0, :, :, 1] = torch.arange(gh, dtype=torch.float64, device=device)[:, None]
            pos[0, :, :, 2] = torch.arange(gw, dtype=torch.float64, device=device)[None, :]
            self._pe_cache[key] = rope_tables(pos.flatten(1, 2), self.cfg.axes_dim, self.cfg.theta)
        return self._pe_cache[key]

    @staticmethod
    def join_pe(pe_txt, pe_img):
        """[txt, img] tables with a head axis: (B or 1, 1, Lt + N, D / 2) each."""
        b = max(pe_txt[0].shape[0], pe_img[0].shape[0])
        return tuple(torch.cat((t.expand(b, -1, -1), i.expand(b, -1, -1)), dim=1)[:, None] for t, i in zip(pe_txt, pe_img))

    # ---- the interface BudgetPix uses (budgetpix.py; flux3_trunk.Flux3Trunk has the same) ---------------------
    family = "flux2"

    def embed_cells(self, x):
        """(B, L, C) latent cells -> (B, L, D): FLUX.2's own img_in."""
        return self.img_in(x)

    def image_pe(self, cy, cx, var=None):
        """Rotary tables of image tokens centred at (cy, cx) cells (B, N) float64, ids (0, cy, cx, 0); var (B, N) the
        box variance in cells^2 that damps the spatial planes (None: plain rotation)."""
        pos = torch.zeros(*cy.shape, 4, dtype=torch.float64, device=cy.device)
        pos[..., 1], pos[..., 2] = cy, cx
        v = None
        if var is not None:
            v = torch.zeros_like(pos)
            v[..., 1] = v[..., 2] = var
        return rope_tables(pos, self.cfg.axes_dim, self.cfg.theta, v)

    def run_trunk(self, img, txt, vec, pe_img, pe_txt, img_valid=None, txt_len=None, img_only=True, refs=None):
        """The blocks on embedded image tokens and projected text (see run_blocks).  FLUX.2's text window is fixed
        (512 tokens, all attended), so txt_len must be None or the full length.  refs (models/refs.References):
        reference images, appended to the image stream (with_refs); the image tokens alone are returned."""
        if txt_len is not None and bool((torch.as_tensor(txt_len) != txt.shape[1]).any()):
            raise ValueError("FLUX.2 attends to its whole 512-token text window; per-row text lengths are FLUX 3's")
        if refs is None:
            return self.run_blocks(img, txt, vec, self.join_pe(pe_txt, pe_img), img_valid, img_only=img_only)
        if img_valid is not None:
            raise ValueError("reference images need unpadded image tokens (every row the same token count)")
        n = img.shape[1]
        img, pe_img = self.with_refs(img, pe_img, refs)
        out = self.run_blocks(img, txt, vec, self.join_pe(pe_txt, pe_img), None, img_only=img_only)
        return out[:, :n]

    def with_refs(self, img, pe_img, refs):
        """[image tokens, reference tokens] and their rotary tables: the references through img_in, as the released
        pipeline (Flux2KleinPipeline) concatenates them after the latents, ids (10 + 10 i, h, w, 0)."""
        b = img.shape[0]
        emb = self.img_in(refs.on(img.device, self.img_in.weight.dtype))
        cat = lambda a, c: torch.cat((a.expand(b, -1, -1), c.expand(b, -1, -1)), dim=1)
        pr = refs.pe(self, img.device)
        return cat(img, emb.to(img.dtype)), tuple(cat(a, c) for a, c in zip(pe_img, pr))

    def final(self, img, vec):
        return self.final_layer(img, vec)

    # ---- the pieces of a forward -------------------------------------------------------------------------
    def time_vec(self, sigma):
        emb = timestep_embedding(sigma)
        if not torch.is_autocast_enabled():                          # under autocast the Linear casts it itself
            emb = emb.to(self.time_in.in_layer.weight.dtype)
        return self.time_in(emb)

    def project_text(self, ctx):
        """(B, Lt, 7680) text-encoder states -> (B, Lt, hidden).  Depends on the caption only: a sampler calls it
        once and passes the result with ctx_projected=True."""
        return self.txt_in(ctx)

    def run_blocks(self, img, txt, vec, pe, mask=None, img_only=True, lora=None, aux=None):
        """img (B, N, D) embedded image tokens, txt (B, Lt, D) projected text, pe joint [txt, img] tables,
        mask (B, N) bool for padded image tokens (None = no padding).  Returns the image tokens.

        Stage 2: `lora` (dmd.FakeScoreLoRA) adds lora.tap(key, block input, modulation) to the image output of every
        double-stream block and to the output of every single-stream block; `aux`, a dict, receives the image
        tokens after the middle single-stream block ("mid")."""
        mod_img = self.double_stream_modulation_img(vec)
        mod_txt = self.double_stream_modulation_txt(vec)
        mod_single = self.single_stream_modulation(vec)
        lt = txt.shape[1]
        if mask is not None and VARLEN_ATTENTION:
            mask = KeyLengths(n + lt for n in mask.sum(dim=1).tolist())
        elif mask is not None:
            mask = torch.cat((mask.new_ones(mask.shape[0], lt), mask), dim=1)[:, None, None, :]
        n_ckpt = self.checkpoint_blocks if torch.is_grad_enabled() else 0
        for i, block in enumerate(self.double_blocks):
            delta = None if lora is None else lora.tap(f"double_{i}", img, mod_img[0])
            if i < n_ckpt:
                img, txt = checkpoint(_eager(block), img, txt, mod_img, mod_txt, pe, mask, use_reentrant=False)
            else:
                img, txt = block(img, txt, mod_img, mod_txt, pe, mask)
            if delta is not None:
                img = img + delta
        x = torch.cat((txt, img), dim=1)
        first, last, mid = len(self.double_blocks), len(self.single_blocks) - 1, len(self.single_blocks) // 2 - 1
        for i, block in enumerate(self.single_blocks):
            skip = lt if (img_only and i == last) else 0
            delta = None if lora is None else lora.tap(f"single_{i}", x[:, skip:], mod_single)
            if first + i < n_ckpt:
                x = checkpoint(_eager(block), x, mod_single, pe, mask, skip, use_reentrant=False)
            else:
                x = block(x, mod_single, pe, mask, skip)
            if delta is not None:
                x = x + delta
            if aux is not None and i == mid:
                aux["mid"] = x[:, lt:]
        return x if img_only else x[:, lt:]

    # ---- the released model -------------------------------------------------------------------------------
    def forward_tokens(self, x, sigma, ctx, pe, mask=None, ctx_projected=False, lora=None, aux=None):
        """Raw FLUX.2 on a token sequence: x (B, N, 128) -> (hidden (B, N, D), prediction (B, N, 128))."""
        vec = self.time_vec(sigma)
        txt = ctx if ctx_projected else self.txt_in(ctx)
        img = self.run_blocks(self.img_in(x), txt, vec, pe, mask, lora=lora, aux=aux)
        hidden, out = self.final_layer(img, vec)
        if lora is not None:
            out = out + lora.delta("final", hidden)
        if aux is not None:
            aux["vec"] = vec
        return hidden, out

    def forward_dense(self, z, t, ctx, ctx_projected=False, lora=None, return_aux=False, txt_len=None, refs=None):
        """The released model on the dense grid.  z (B, gh, gw, 128) noisy latent, t (B,) in [0, 1] with 0 = noise
        and 1 = clean, ctx (B, Lt, 7680).  Returns the velocity dz/dt = x - eps, (B, gh, gw, 128) fp32.

        Stage 2: with `lora` it is the fake-score denoiser; return_aux=True also returns the discriminator's
        features, fp32: mid_spatial (B, gh, gw, D), the layer-normed image tokens after the middle single-stream
        block; final_spatial (B, gh, gw, D), the modulated final hidden states; vec (B, D), the time embedding."""
        b, gh, gw, _ = z.shape
        if refs is not None:                                         # reference images: the released pipeline's way
            if lora is not None or return_aux:
                raise ValueError("reference images: plain sampling only (no lora, no aux)")
            sigma = 1.0 - t.float()
            vec = self.time_vec(sigma)
            txt = ctx if ctx_projected else self.txt_in(ctx)
            n = gh * gw
            img = self.run_trunk(self.img_in(z.flatten(1, 2)), txt, vec, self.grid_pe(gh, gw, z.device),
                                 self.text_pe(txt.shape[1], z.device), refs=refs)
            _, raw = self.final_layer(img[:, :n], vec)
            return -raw.float().view(b, gh, gw, -1)
        pe = self.join_pe(self.text_pe(ctx.shape[1], z.device), self.grid_pe(gh, gw, z.device))
        aux = {} if return_aux else None
        hidden, raw = self.forward_tokens(z.flatten(1, 2), 1.0 - t.float(), ctx, pe, ctx_projected=ctx_projected, lora=lora, aux=aux)
        v = -raw.float().view(b, gh, gw, -1)
        if not return_aux:
            return v
        mid = aux.pop("mid").float()
        return v, {"mid_spatial": F.layer_norm(mid, mid.shape[-1:], eps=1e-6).view(b, gh, gw, -1),
                   "final_spatial": hidden.float().view(b, gh, gw, -1), "vec": aux["vec"].float()}

    def forward(self, z, t, ctx, layout=None, ctx_projected=False, txt_len=None, refs=None):
        if layout is not None:
            raise ValueError("the plain trunk only runs the dense grid; layouts need BudgetPixFlux2")
        return self.forward_dense(z, t, ctx, ctx_projected, refs=refs)
