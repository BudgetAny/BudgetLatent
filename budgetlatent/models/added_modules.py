"""The BudgetPix s8 modules added to the FLUX.2 [klein] trunk (28,136,192 parameters at 4B).

Every module only touches coarse tokens (s in 2, 4, 8) and its last projection is zero-initialised, so

  * at the dense layout nothing here runs and the model is the released FLUX.2;
  * at step 0 a coarse token is the mean of the cell embeddings it covers on the way in, and its prediction
    upsampled bilinearly (plus the analytic high-frequency skip) on the way out.

The arithmetic and the initialisations are those of the reference classes of the method.  Parameter names follow the dit/
export of the reference implementation, with torch conventions for Linear / Conv weights; convert.py maps between
the two.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.scale.float()).to(x.dtype)


class CascadeMixer2D(nn.Module):
    """One stage of the token encoder: a 2 x 2 group of embeddings -> one embedding (mean + zero-init residual).

    mixers[2] merges cells into 2x tokens, mixers[4] merges those into 4x tokens, mixers[8] into 8x tokens."""

    def __init__(self, dim, bottleneck=256):
        super().__init__()
        self.pos = nn.Parameter(torch.empty(1, 1, 2, 2, dim))
        self.down = nn.Linear(dim, bottleneck)
        self.dw = nn.Conv2d(bottleneck, bottleneck, kernel_size=2, groups=bottleneck)
        self.up = nn.Linear(4 * bottleneck, dim)
        nn.init.normal_(self.pos, std=0.02)
        for layer in (self.down, self.dw):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.up.weight)                               # the stage starts as the exact 2 x 2 mean
        nn.init.zeros_(self.up.bias)

    def depthwise(self, h):
        """The depthwise 2x2 convolution with "SAME" padding on a (M, 2, 2, C) map: out[y, x] = sum over (dy, dx) of
        w[dy, dx] * h[y + dy, x + dx], zero beyond the bottom row and the right column.  Written out, because the
        map is always 2x2 and M changes with every layout (a cuDNN convolution would be re-planned every time)."""
        w = self.dw.weight[:, 0].to(h.dtype)                         # (C, 2, 2)
        h00, h01, h10, h11 = h[:, 0, 0], h[:, 0, 1], h[:, 1, 0], h[:, 1, 1]
        top = torch.stack((h00 * w[:, 0, 0] + h01 * w[:, 0, 1] + h10 * w[:, 1, 0] + h11 * w[:, 1, 1],
                           h01 * w[:, 0, 0] + h11 * w[:, 1, 0]), dim=1)
        bottom = torch.stack((h10 * w[:, 0, 0] + h11 * w[:, 0, 1], h11 * w[:, 0, 0]), dim=1)
        return torch.stack((top, bottom), dim=1) + self.dw.bias.to(h.dtype)

    def forward(self, sub):                                          # (M, 2, 2, dim) -> (M, dim)
        mean = sub.float().mean(dim=(1, 2)).to(sub.dtype)
        h = F.gelu(self.down(sub + self.pos.view(1, 2, 2, -1).to(sub.dtype)))
        h = F.gelu(self.depthwise(h))
        return mean + self.up(h.flatten(1)).to(mean.dtype)


class MipNeRFIPE2DV3(nn.Module):
    """Integrated positional encoding of a token's box (position and extent only; zero-init output).

    The box of a token of scale s is a uniform distribution over its footprint; each sinusoid is replaced by its
    expectation over that box (the Mip-NeRF damping), with one copy damped by the full box variance and one by
    the excess over a single cell, which vanishes for s = 1."""

    def __init__(self, dim, grid=32, octaves=8, hidden=256):
        super().__init__()
        self.grid, self.octaves = grid, octaves
        self.mlp = nn.ModuleList([nn.Linear(8 * octaves, hidden), nn.Linear(hidden, dim)])
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

    def features(self, cy, cx, s):
        """cy, cx (M,) token centres in cell units, s the token scale -> (M, 64) fp32.  Octave by octave:
        [d_full sin, d_full cos, d_exc sin, d_exc cos], each for (y, x) -- the order of the reference listing."""
        delta = 2.0 * math.pi / self.grid
        coords = torch.stack((cy, cx), dim=-1).double()                                  # (M, 2): y, x
        mu = 2.0 * math.pi * ((coords + 0.5) / self.grid - 0.5)
        omega = (2.0 ** torch.arange(self.octaves, dtype=torch.float64, device=coords.device))[:, None]   # (8, 1)
        ang = mu[:, None, :] * omega                                                     # (M, 8, 2)
        d_full = torch.exp(-0.5 * omega ** 2 * (s * delta) ** 2 / 12.0)
        d_exc = torch.expm1(-0.5 * omega ** 2 * (s * s - 1.0) * delta ** 2 / 12.0)
        sin, cos = torch.sin(ang), torch.cos(ang)
        return torch.cat((d_full * sin, d_full * cos, d_exc * sin, d_exc * cos), dim=-1).flatten(1).float()

    def forward(self, cy, cx, s):
        return self.mlp[1](F.silu(self.mlp[0](self.features(cy, cx, s))))


def subgrid_pos_emb_2d(gh, gw, width, device=None):
    """Fixed sinusoidal embedding of the cells of a (gh, gw) sub-grid: (gh * gw, width) = [e_h, e_w], fp32."""
    quarter = width // 4
    freqs = math.pi * torch.exp(-math.log(10000.0) * torch.arange(quarter, dtype=torch.float32, device=device) / quarter)

    def axis(n):
        coords = (torch.arange(n, dtype=torch.float32, device=device) + 0.5) / n * 2.0 - 1.0
        ang = coords[:, None] * freqs
        return torch.cat((torch.sin(ang), torch.cos(ang)), dim=-1)                       # (n, width / 2)

    e_h, e_w = axis(gh), axis(gw)
    return torch.cat((e_h[:, None, :].expand(gh, gw, -1), e_w[None, :, :].expand(gh, gw, -1)), dim=-1).reshape(gh * gw, width)


class RefineBlock2D(nn.Module):
    """A small transformer block over the sub-cells of one token, modulated by that token's hidden state."""

    def __init__(self, width, cond_dim, heads=4, mlp_hidden=352):
        super().__init__()
        self.heads = heads
        self.ada = nn.Linear(cond_dim, 6 * width)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        self.n1, self.n2 = RMSNorm(width), RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.qn, self.kn = RMSNorm(width // heads), RMSNorm(width // heads)
        self.proj = nn.Linear(width, width)
        self.w12 = nn.Linear(width, 2 * mlp_hidden, bias=False)
        self.w3 = nn.Linear(mlp_hidden, width, bias=False)

    def forward(self, x, cond):                                      # (M, P, width), (M, cond_dim)
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(F.silu(cond))[:, None, :].chunk(6, dim=-1)
        m, p, w = x.shape
        q, k, v = self.qkv(self.n1(x) * (1 + sc1) + sh1).view(m, p, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(self.qn(q), self.kn(k), v)
        x = x + g1 * self.proj(a.transpose(1, 2).reshape(m, p, w))
        gate, val = self.w12(self.n2(x) * (1 + sc2) + sh2).chunk(2, dim=-1)
        return x + g2 * self.w3(F.silu(gate) * val)


class CascadeRefineStage2D(nn.Module):
    """One stage of the latent decoder: a (g, g) patch of one token -> (2g, 2g), bilinear upsample + zero-init
    residual.  refiners[8] opens an 8x token into 2 x 2, refiners[4] takes 4x cells to 2x, refiners[2] to 1x.
    Runs in fp32 (it is tiny: width 128)."""

    def __init__(self, channels, cond_dim, width=128, depth=2, heads=4, mlp_hidden=352):
        super().__init__()
        self.width = width
        self.embed = nn.Linear(channels, width)
        self.slots = nn.Parameter(torch.empty(2, 2, width))
        nn.init.normal_(self.slots, std=0.02)
        self.blocks = nn.ModuleList([RefineBlock2D(width, cond_dim, heads, mlp_hidden) for _ in range(depth)])
        self.norm = RMSNorm(width)
        self.out = nn.Linear(width, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, lat, feat, cond):
        """lat (M, g, g, C) fp32, feat (M, g, g, width) of the parent stage or None, cond (M, cond_dim) fp32."""
        m, g, _, c = lat.shape
        n = 4 * g * g
        up = F.interpolate(lat.permute(0, 3, 1, 2), scale_factor=2, mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        x = self.embed(up.reshape(m, n, c)) + self.slots.repeat(g, g, 1).reshape(1, n, self.width)
        if feat is not None:
            x = x + feat.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2).reshape(m, n, self.width)
        if n > 4:
            x = x + subgrid_pos_emb_2d(2 * g, 2 * g, self.width, lat.device)
        for blk in self.blocks:
            x = blk(x, cond)
        return up + self.out(self.norm(x)).reshape(m, 2 * g, 2 * g, c), x.reshape(m, 2 * g, 2 * g, self.width)


def hf_skip_gain(sigma, rho=0.25):
    """Linear-MMSE gain for the raw velocity's detail inside a coarse token.

    With z = (1 - sigma) x + sigma eps, unit-variance noise detail and clean detail of std rho, the best linear
    estimate of the detail of FLUX.2's raw target eps - x from the detail of z is w * detail(z) with

        w(sigma) = (sigma - rho^2 (1 - sigma)) / (sigma^2 + rho^2 (1 - sigma)^2).

    w(1) = +1 (pure noise: the detail is the noise), w = 0 at sigma = rho^2 / (1 + rho^2) (1 / 17 for rho = 0.25),
    w(0) = -1 (clean: the detail is -x's), with the maximum (1 + rho^2) / (2 rho) = 2.125 at sigma = 5 / 17."""
    r2 = rho * rho
    return (sigma - r2 * (1.0 - sigma)) / torch.clamp(sigma * sigma + r2 * (1.0 - sigma) ** 2, min=1e-6)
