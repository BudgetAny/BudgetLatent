"""Token layouts for BudgetPix on FLUX.2 [klein] (s8: 16 / 32 / 64 / 128 px tokens at 512 x 512).

The latent grid is 32 x 32 cells (one cell = one FLUX.2 token = 16 x 16 px).  A layout is a quadtree over it: a
token of scale s in {1, 2, 4, 8} covers s x s cells.  Starting from the 4 x 4 grid of s = 8 tokens, every split
replaces one token by four, so a layout always has N = 16 + 3k tokens (16 .. 1024).

Other resolutions: the environment variable BP_GRID, read once when this module is imported, sets the cells per
side -- 32 (512 x 512, the reference resolution) by default, 64 for 1024 x 1024 (8 x 8 = 64 .. 4096 tokens, N = 64 + 3k).
Every number above scales with it: GRID, N_DENSE = GRID^2, N_MIN = (GRID / 8)^2 and the schedule's shift mu(N_DENSE).

  flux2_klein_timesteps          FLUX.2's sampler time grid (t = 0 noise -> t = 1 clean)
  schedule, time_shift           the sampler time grid of each family (FLUX.2 [klein], base and distilled: BFL's
                                 get_schedule(steps, tokens); FLUX 3: a constant shift, 6.93)
  sample_times                   training times: t = 0 for a share, else a uniform draw through the schedule's shift
  pixel_entropy_pyramid_s8       detail score from decoded pixels (512-bin luma entropy)
  noise_latent_pyramid_s8        detail proxy from the latent itself (used at t = 0, before anything is decoded)
  threshold_layout               leaf masks from the thresholds shifted by an offset (N varies per image)
  calibrate_offset               the offset whose quadtree stays within a token target (24-step bisection)
  build_exact_budget_layout_s8   leaf masks with exactly N tokens for every image of the batch (sampling)
  draw_plan, training_layouts    the 4-mode "v3" training mixture (the plan says which images need a detail map)
  pack                           leaf masks -> the token index tensors the model reads

These follow the reference functions of the method.  Differences in form only: random numbers come from a generator that
is passed in (reproducible per rank and step), and the mixture is drawn in two parts -- first the plan, which
needs no detail map, then the layouts -- so the trainer decodes only the images whose layout will read them.
"""
import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

GRID =int(os.environ.get("BP_GRID", "32"))   # latent cells per side: 32 at 512 x 512 (default), 64 at 1024 x 1024
if GRID < 8 or GRID % 8:
    raise ValueError(f"BP_GRID={GRID}: the grid must be a positive multiple of 8 (the 8 x 8-cell tokens tile it)")
CELL_PX = 16                            # pixels per cell
SCALES = (1, 2, 4, 8)                   # token side in cells, fine -> coarse
COARSE = (2, 4, 8)
THRESHOLDS = {8: 2.0, 4: 4.0, 2: 6.0}   # bits: a token of scale s splits when its entropy >= THRESHOLDS[s] + offset
N_DENSE, N_MIN = GRID * GRID, (GRID // 8) ** 2
MAX_SPLITS = (N_DENSE - N_MIN) // 3     # 336 at GRID 32
ENTROPY_BINS = 512
MAX_BITS = math.log2(ENTROPY_BINS)      # 9.0


# ---------------------------------------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------------------------------------
def compute_empirical_mu(image_seq_len: int = 1024, num_steps: int = 50) -> float:
    """FLUX.2 [klein] empirical time-shift exponent mu (1.701956 at 1024 tokens, 50 steps)."""
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def flux2_klein_timesteps(num_steps: int = 50, image_seq_len: int = 1024) -> list:
    """num_steps + 1 increasing times t in [0, 1] (0 = noise, 1 = clean); the model's own sigma is 1 - t."""
    e_mu = math.exp(compute_empirical_mu(image_seq_len, num_steps))
    ts = [0.0]
    for i in range(1, num_steps):
        sigma_raw = 1.0 - i / float(num_steps)
        sigma_shifted = e_mu / (e_mu + (1.0 / sigma_raw - 1.0))
        ts.append(1.0 - sigma_shifted)
    ts.append(1.0)
    return ts


def shift_sigma(sigma_raw, mu):
    """The same shift on a tensor of raw sigmas in (0, 1]."""
    e_mu = math.exp(mu)
    return e_mu / (e_mu + (1.0 / sigma_raw - 1.0))


# FLUX 3 Image Klein Base: the release's config.json sampler "alpha" and diffusers 0.42.1's scheduler shift.  (BFL's
# GitHub code samples the base model with alpha 16.0 and guidance 4.0 instead -- Pipeline.generate's default and
# MODELS["base"] -- while config.json and diffusers say 6.93 and 3.5; this code follows the release's config.json.
# configs/flux3_*.yaml model.time_shift overrides it, for the training times and the sampler alike.)
FLUX3_SHIFT = 6.93


FLUX3_SHIFT_FAMILIES = ("flux3", "flux3_distilled")  # FLUX 3 Image Klein Base and Klein (config.json "sampler" alpha 6.93)


def time_shift(family: str = "flux2", num_steps: int = 50, image_seq_len: int = None) -> float:
    """The factor a of the schedule's shift sigma -> a sigma / (1 + (a - 1) sigma): FLUX.2 [klein] exp(mu(tokens,
    steps)) (resolution dependent), FLUX 3 (Image Klein Base, and the step-distilled Klein at its 16 steps) a constant
    6.93 at every resolution."""
    if family in FLUX3_SHIFT_FAMILIES:
        return FLUX3_SHIFT
    return math.exp(compute_empirical_mu(image_seq_len or N_DENSE, num_steps))


FLUX2_SCHEDULE = ("flux2", "flux2_distilled")       # the families sampled on flux2_klein_timesteps(steps, N_DENSE)


def schedule(num_steps: int = 50, family: str = "flux2", shift: float = None) -> list:
    """num_steps + 1 increasing times t in [0, 1] (0 = noise, 1 = clean): the raw sigmas 1, 1 - 1/n, ..., 1/n shifted,
    then 0.  FLUX.2 [klein] (4B Base, and the distilled 4B at its 4 steps) is flux2_klein_timesteps, i.e. 1 - BFL's
    get_schedule(num_steps, N_DENSE) (at 1024 x 1024: get_schedule(4, 4096) = t 0, 0.0326, 0.0919, 0.2328, 1); FLUX 3 is
    diffusers' FlowMatchEulerDiscreteScheduler(shift=6.93) on linspace(1, 1/n, n), the same formula as BFL's
    timesteps(steps, alpha) (whose default alpha is 16, see FLUX3_SHIFT) -- Image Klein Base at 50 steps, the
    step-distilled Klein (a flow map) at 16."""
    if family in FLUX2_SCHEDULE and shift is None:
        return flux2_klein_timesteps(num_steps, N_DENSE)
    a = float(shift if shift is not None else time_shift(family, num_steps))
    ts = [0.0]
    for i in range(1, num_steps):
        s = 1.0 - i / float(num_steps)
        ts.append(1.0 - a * s / (1.0 + (a - 1.0) * s))
    ts.append(1.0)
    return ts


ASYNC_COPY = False      # scripts/train/train.py --fast_prep: what is drawn on the host reaches the GPU without a synchronisation


def to_device(x, device):
    """x.to(device); with ASYNC_COPY a host tensor goes to a GPU through pinned memory without waiting for the GPU
    (the same values: only the copy is asynchronous)."""
    if ASYNC_COPY and x.device.type == "cpu" and torch.device(device).type == "cuda":
        return x.pin_memory().to(device, non_blocking=True)
    return x.to(device)


def _rand(shape, gen, device):
    return to_device(torch.rand(shape, generator=gen), device)


def _randn(shape, gen, device):
    return to_device(torch.randn(shape, generator=gen), device)


def sample_times(n: int, gen=None, device="cpu", first_p: float = 0.10, num_steps: int = 50, uniform_frac: float = 0.0,
                 shift: float = None):
    """Training times (sample_training_times): t = 0 exactly with probability first_p, otherwise a
    uniform draw pushed through the shift of the num_steps-step schedule (shift: the factor a = exp(mu); default
    FLUX.2's mu(N_DENSE), FLUX 3 passes time_shift("flux3")).  uniform_frac > 0 (stage 2) replaces that share of the
    other examples by t ~ U[0, 1).  Returns (t (n,), first (n,) bool)."""
    mu = compute_empirical_mu(N_DENSE, num_steps) if shift is None else math.log(shift)
    first = _rand((n,), gen, device) < first_p
    u = _rand((n,), gen, device) * (1.0 - 1e-4)
    t = torch.where(first, torch.zeros_like(u), 1.0 - shift_sigma(1.0 - u, mu))
    if uniform_frac > 0:
        t_uniform = _rand((n,), gen, device) * (1.0 - 1e-4)
        t = torch.where(~first & (_rand((n,), gen, device) < uniform_frac), t_uniform, t)
    return t, first


# ---------------------------------------------------------------------------------------------------------
# detail pyramids: {s: (B, 32 // s, 32 // s)} in bits, for s in (2, 4, 8)
# ---------------------------------------------------------------------------------------------------------
def pixel_entropy_pyramid_s8(rgb_m1_p1: torch.Tensor) -> dict:
    """512-bin luma Shannon entropy (bits) of every 32 / 64 / 128 px block of (B, 3, 512, 512) RGB in [-1, 1]."""
    x = torch.nan_to_num(rgb_m1_p1.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    img255 = torch.floor((x.clamp(-1.0, 1.0) + 1.0) * 127.5)
    gray = 0.2989 * img255[:, 0] + 0.5870 * img255[:, 1] + 0.1140 * img255[:, 2]
    bsz, h, w = gray.shape
    gh, gw = h // CELL_PX, w // CELL_PX
    cells = gray.view(bsz, gh, CELL_PX, gw, CELL_PX).permute(0, 1, 3, 2, 4).reshape(bsz * gh * gw, CELL_PX * CELL_PX)
    bins = (cells.clamp(0.0, 255.0) * (ENTROPY_BINS / 256.0)).long().clamp(0, ENTROPY_BINS - 1)
    hist = torch.zeros(bsz * gh * gw, ENTROPY_BINS, device=gray.device, dtype=torch.float32)
    hist.scatter_add_(1, bins, torch.ones_like(bins, dtype=torch.float32))
    hist = hist.view(bsz, gh, gw, ENTROPY_BINS)
    pyr = {}
    for s in COARSE:
        gh, gw = hist.shape[1] // 2, hist.shape[2] // 2
        hist = hist.view(bsz, gh, 2, gw, 2, ENTROPY_BINS).sum(dim=(2, 4))
        p = hist / float((CELL_PX * s) * (CELL_PX * s))
        pyr[s] = -(p * torch.log2(p + 1e-10)).sum(dim=-1)
    return pyr


def noise_latent_pyramid_s8(z_nhwc: torch.Tensor) -> dict:
    """Step-0 proxy on (B, 32, 32, 128) latents: spatial std of the 2x2-unpacked latent, mapped to bits."""
    bsz, g = z_nhwc.shape[0], z_nhwc.shape[1]
    c = z_nhwc.shape[-1] // 4
    z = z_nhwc.float().view(bsz, g, g, c, 2, 2).permute(0, 1, 4, 2, 5, 3).reshape(bsz, 2 * g, 2 * g, c)
    pyr = {}
    for s in COARSE:
        gs = g // s
        b = 2 * g // gs
        blk = z.view(bsz, gs, b, gs, b, c).permute(0, 1, 3, 2, 4, 5)
        sp_std = blk.std(dim=(3, 4), correction=0).mean(dim=-1)
        norm_std = sp_std / (sp_std.mean(dim=(1, 2), keepdim=True) + 1e-5)
        pyr[s] = (4.8 + 2.2 * torch.log2(norm_std.clamp(min=1e-3))).clamp(0.0, 9.0)
    return pyr


# ---------------------------------------------------------------------------------------------------------
# leaf masks: {s: (B, 32 // s, 32 // s) bool}, True where a token of scale s sits
# ---------------------------------------------------------------------------------------------------------
def snap_tokens(budget_ratio: float) -> int:
    """The realizable token count N = N_MIN + 3k closest to budget_ratio * N_DENSE."""
    raw = max(N_MIN, min(int(round(budget_ratio * N_DENSE)), N_DENSE))
    return N_MIN + 3 * max(0, min(int(round((raw - N_MIN) / 3.0)), MAX_SPLITS))


def _up2(mask):
    return mask.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)


def uniform_layout(bsz: int, s: int, device="cpu") -> dict:
    """Every token at scale s (s = 1 is the dense layout)."""
    return {k: torch.full((bsz, GRID // k, GRID // k), k == s, dtype=torch.bool, device=device) for k in SCALES}


def dense_layout(bsz: int, device="cpu") -> dict:
    return uniform_layout(bsz, 1, device)


def threshold_layout(pyr: dict, offset) -> dict:
    """Top-down quadtree: an existing token of scale s splits when pyr[s] >= THRESHOLDS[s] + offset."""
    bsz, device = pyr[8].shape[0], pyr[8].device
    offset = torch.as_tensor(offset, dtype=torch.float32, device=device).expand(bsz)
    alive = torch.ones_like(pyr[8], dtype=torch.bool)
    active = {}
    for s in (8, 4, 2):
        sp = alive & (pyr[s] >= THRESHOLDS[s] + offset[:, None, None])
        active[s] = alive & ~sp
        alive = _up2(sp)
    active[1] = alive
    return active


def calibrate_offset(pyr: dict, n_target) -> torch.Tensor:
    """Per image, the threshold offset (24 bisection steps on [-16, 16] bits) at which the quadtree has as many
    tokens as it can without exceeding n_target[b]."""
    bsz, device = pyr[8].shape[0], pyr[8].device
    n_target = torch.as_tensor(n_target, dtype=torch.long, device=device).expand(bsz)
    lo = torch.full((bsz,), -16.0, device=device)
    hi = torch.full((bsz,), 16.0, device=device)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        over = count_tokens(threshold_layout(pyr, mid)) > n_target
        lo = torch.where(over, mid, lo)
        hi = torch.where(over, hi, mid)
    return hi


# exact_layout's tensor work without its argument check; fast.compile_layouts() puts a compiled copy here (same
# comparisons, counts and sorts, so the same masks).  None: the plain function.
EXACT_LAYOUT_FN = None


def exact_layout(pyr: dict, n_exact: torch.Tensor) -> dict:
    """Leaf masks with exactly n_exact[b] tokens (each of the form 16 + 3k).

    A bisection finds the largest threshold offset whose quadtree does not exceed the count; the splits still
    missing are then handed out top-down by entropy rank, keeping every level able to absorb what is left."""
    bsz, device = pyr[8].shape[0], pyr[8].device
    n_exact = torch.as_tensor(n_exact, dtype=torch.long, device=device).expand(bsz)
    if bool(((n_exact - N_MIN) % 3 != 0).any()) or int(n_exact.min()) < N_MIN or int(n_exact.max()) > N_DENSE:
        raise ValueError("token counts must be of the form 16 + 3k, between 16 and 1024")
    return (EXACT_LAYOUT_FN or _exact_layout)(pyr, n_exact)


def _exact_layout(pyr: dict, n_exact: torch.Tensor) -> dict:
    bsz, device = pyr[8].shape[0], pyr[8].device
    hi = calibrate_offset(pyr, n_exact)

    k_rem = (n_exact - N_MIN) // 3
    alive = torch.ones_like(pyr[8], dtype=torch.bool)
    active = {}
    for s in (8, 4, 2):
        sp_b = alive & (pyr[s] >= THRESHOLDS[s] + hi[:, None, None])
        k_init = sp_b.sum(dim=(1, 2))
        n_alive = alive.sum(dim=(1, 2))
        max_mult = (s * s - 1) // 3  # splits one token of this scale can still yield: 21 / 5 / 1
        k_lo = (k_rem + max_mult - 1) // max_mult
        k_s = torch.minimum(torch.maximum(k_init, k_lo), torch.minimum(n_alive, k_rem))
        flat = alive.reshape(bsz, -1)
        masked = torch.where(flat, pyr[s].reshape(bsz, -1).float(), torch.tensor(-1e30, device=device))
        order = torch.argsort(-masked, dim=1, stable=True)
        rank = torch.argsort(order, dim=1)
        sp_s = ((rank < k_s[:, None]) & flat).reshape(alive.shape)
        active[s] = alive & ~sp_s
        alive = _up2(sp_s)
        k_rem = k_rem - k_s
    active[1] = alive
    return active


def build_exact_budget_layout_s8(pyr: dict, budget_ratio: float) -> dict:
    """Exact, zero-padding leaf masks for one budget: every image gets N = snap_tokens(budget_ratio) tokens."""
    bsz, device = pyr[8].shape[0], pyr[8].device
    n_exact = snap_tokens(budget_ratio)
    if n_exact == N_DENSE:
        return uniform_layout(bsz, 1, device)
    if n_exact == N_MIN:
        return uniform_layout(bsz, 8, device)
    # snap_tokens only returns valid counts: straight to the tensor work (the check would wait for the GPU)
    return (EXACT_LAYOUT_FN or _exact_layout)(pyr, torch.full((bsz,), n_exact, dtype=torch.long, device=device))


def count_tokens(active: dict) -> torch.Tensor:
    return sum(active[s].flatten(1).sum(1) for s in SCALES)


# ---------------------------------------------------------------------------------------------------------
# training mixture (4 modes) and its layout augmentation: sample_training_layouts_v3_s8
# ---------------------------------------------------------------------------------------------------------
@dataclass
class LayoutMix:
    anchor_p: float = 0.10              # dense s = 1 layout
    uniform_coarse_p: float = 0.10      # every token at one scale s ~ U{2, 4, 8}
    budget_p: float = 0.55              # thresholds calibrated to a token target ~ LogUniform(budget_range) x 1024
    budget_range: tuple = (1.0 / 64.0, 1.0)
    offset_range: tuple = (-1.0, 4.0)   # the remaining share: thresholds shifted by delta ~ U(offset_range) bits
    random_placement_p: float = 0.20    # of the budget + offset examples: detail map replaced by a synthetic field
    smooth_scramble_frac: float = 0.67  # of those: half of this share blobs, half spotlight / ramp; the rest i.i.d.
    pyramid_noise_std: float = 0.6      # otherwise: per-cell Gaussian noise on the detail map (bits)
    level_jitter_std: float = 0.75      # ... and one Gaussian shift per image and pyramid level (bits)

    @classmethod
    def from_config(cls, d):
        d = dict(d or {})
        kw = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        for k in ("budget_range", "offset_range"):
            if k in kw:
                kw[k] = tuple(float(v) for v in kw[k])
        return cls(**kw)


MODES = ("anchor", "uniform", "budget", "offset")
FIELDS = ("image", "blob", "geometric", "iid")      # what the two adaptive modes read as their detail map


@dataclass
class LayoutPlan:
    """What each image of a batch will be laid out with, drawn before any detail map exists -- so the caller only
    has to compute (decode) detail maps for the images that will read them (`needs_detail`)."""
    mode: torch.Tensor          # (B,) index into MODES
    field: torch.Tensor         # (B,) index into FIELDS (always 0 for the anchor and uniform modes)
    target: torch.Tensor        # (B,) token target of the budget mode, 16 .. 1024
    delta: torch.Tensor         # (B,) threshold offset of the offset mode
    s_uniform: torch.Tensor     # (B,) scale of the uniform mode

    @property
    def needs_detail(self):
        return (self.mode >= 2) & (self.field == 0)


def draw_plan(bsz: int, mix: LayoutMix = LayoutMix(), gen=None, device="cpu") -> LayoutPlan:
    u = _rand((bsz,), gen, device)
    e1 = mix.anchor_p + mix.uniform_coarse_p
    mode = (u >= mix.anchor_p).long() + (u >= e1).long() + (u >= e1 + mix.budget_p).long()
    scramble = (mode >= 2) & (_rand((bsz,), gen, device) < mix.random_placement_p)
    su, half = _rand((bsz,), gen, device), 0.5 * mix.smooth_scramble_frac
    field = torch.where(scramble, 1 + (su >= half).long() + (su >= 2.0 * half).long(), torch.zeros_like(mode))
    lo, hi = math.log(mix.budget_range[0]), math.log(mix.budget_range[1])
    target = torch.round(torch.exp(lo + (hi - lo) * _rand((bsz,), gen, device)) * N_DENSE).long().clamp(N_MIN, N_DENSE)
    delta = mix.offset_range[0] + (mix.offset_range[1] - mix.offset_range[0]) * _rand((bsz,), gen, device)
    s_uniform = to_device(torch.tensor(COARSE), device)[to_device(torch.randint(0, len(COARSE), (bsz,), generator=gen), device)]
    return LayoutPlan(mode, field, target, delta, s_uniform)


def augmented_pyramid(pyr: dict, plan: LayoutPlan, mix: LayoutMix = LayoutMix(), gen=None) -> dict:
    """The detail maps the adaptive modes threshold, in bits on [0, 9]: the image's own with one Gaussian shift per
    level and Gaussian noise per cell, or (plan.field) a synthetic one that ignores the image --

      blob        a 4 x 4 uniform map upsampled bilinearly to the level's grid
      geometric   a Gaussian spotlight, or with probability 0.4 a linear ramp; inverted with probability 0.5
      iid         uniform noise per cell

    The blob and geometric fields are one function per image, sampled at every level's resolution, with a little
    of that level's i.i.d. map mixed in (0.10 and 0.08)."""
    bsz, device = pyr[8].shape[0], pyr[8].device
    r = lambda *shape: _rand(shape, gen, device)
    blob4 = MAX_BITS * r(bsz, 1, 4, 4)
    cy, cx = -0.8 + 1.6 * r(bsz, 1, 1), -0.8 + 1.6 * r(bsz, 1, 1)
    sigma2 = (0.25 + 0.70 * r(bsz, 1, 1)) ** 2
    ang = 2.0 * math.pi * r(bsz, 1, 1)
    use_ramp, invert = r(bsz, 1, 1) < 0.4, r(bsz, 1, 1) < 0.5
    field = plan.field[:, None, None]
    aug = {}
    for s in COARSE:
        p = pyr[s].float()
        g = p.shape[1]
        iid = MAX_BITS * r(bsz, g, g)
        blob = F.interpolate(blob4, size=(g, g), mode="bilinear", align_corners=False)[:, 0] + 0.10 * (iid - 0.5 * MAX_BITS)
        ax = torch.linspace(-1.0, 1.0, g, device=device)
        ys, xs = ax[None, :, None], ax[None, None, :]
        spot = torch.exp(-0.5 * ((ys - cy) ** 2 + (xs - cx) ** 2) / sigma2)
        ramp = (0.5 * (torch.cos(ang) * ys + torch.sin(ang) * xs + 1.0)).clamp(0.0, 1.0)
        f01 = torch.where(use_ramp, ramp, spot)
        geom = MAX_BITS * torch.where(invert, 1.0 - f01, f01) + 0.08 * (iid - 0.5 * MAX_BITS)
        own = p + mix.level_jitter_std * _randn((bsz, 1, 1), gen, device) + mix.pyramid_noise_std * _randn((bsz, g, g), gen, device)
        aug[s] = torch.where(field == 0, own, torch.where(field == 1, blob, torch.where(field == 2, geom, iid))).clamp(0.0, MAX_BITS)
    return aug


def training_layouts(pyr: dict, mix: LayoutMix = LayoutMix(), gen=None, plan: LayoutPlan = None):
    """Per-image layouts of the training mixture.  pyr: the images' detail maps (only read where
    plan.needs_detail).  Returns (leaf masks, mode index per image (see MODES)).

    The budget mode thresholds the augmented map at the offset that brings the token count as close to its target
    as a single offset can from below; the count is not forced to the target (sampling does that, exact_layout)."""
    bsz, device = pyr[8].shape[0], pyr[8].device
    plan = plan or draw_plan(bsz, mix, gen, device)
    aug = augmented_pyramid(pyr, plan, mix, gen)
    adaptive = threshold_layout(aug, torch.where(plan.mode == 2, calibrate_offset(aug, plan.target), plan.delta))
    m = plan.mode[:, None, None]
    active = {}
    for s in SCALES:
        fixed = torch.where(m == 0, torch.full_like(adaptive[s], s == 1), (plan.s_uniform == s)[:, None, None].expand_as(adaptive[s]))
        active[s] = torch.where(m <= 1, fixed, adaptive[s])
    return active, plan.mode


# ---------------------------------------------------------------------------------------------------------
# packing: what the model reads
# ---------------------------------------------------------------------------------------------------------
@dataclass
class Layout:
    """A batch of layouts as token index tensors.  Tokens are ordered by scale (1, 2, 4, 8), row-major inside a
    scale; images with fewer tokens than the longest are padded at the end (`valid` False).

    scale (B, N)   token side in cells, 0 where padded        gy, gx (B, N)   top-left cell
    n (B,)         tokens per image                             cell_scale (B, G, G)  scale of the token on each cell
    groups {s: (b, t, gy, gx)}  flat indices of every token of scale s in the batch (b = image, t = token slot)
    """
    scale: torch.Tensor
    gy: torch.Tensor
    gx: torch.Tensor
    valid: torch.Tensor
    n: torch.Tensor
    cell_scale: torch.Tensor
    groups: dict
    known_padded: object = None          # pack sets it from counts it already has on the host: `padded` needs no wait

    @property
    def batch(self):
        return self.scale.shape[0]

    @property
    def length(self):
        return self.scale.shape[1]

    @property
    def padded(self):
        if self.known_padded is not None:
            return self.known_padded
        return not bool(self.valid.all())

    @property
    def is_dense(self):
        return set(self.groups) == {1}

    def centers(self):
        """Token centres in cell units (what the positions use): g + (s - 1) / 2."""
        half = 0.5 * (self.scale.clamp(min=1) - 1).double()
        return self.gy.double() + half, self.gx.double() + half

    def box_var(self):
        """Variance of a uniform box of side s cells, minus that of one cell: (s^2 - 1) / 12 (0 for s = 1)."""
        s = self.scale.clamp(min=1).double()
        return (s * s - 1.0) / 12.0

    def repeat(self, k: int):
        """The same layouts k times along the batch (for a fused [cond; uncond] forward)."""
        if k == 1:
            return self
        b = self.batch
        groups = {s: (torch.cat([g[0] + i * b for i in range(k)]), g[1].repeat(k), g[2].repeat(k), g[3].repeat(k))
                  for s, g in self.groups.items()}
        return Layout(self.scale.repeat(k, 1), self.gy.repeat(k, 1), self.gx.repeat(k, 1), self.valid.repeat(k, 1),
                      self.n.repeat(k), self.cell_scale.repeat(k, 1, 1), groups, self.known_padded)

    def to(self, device):
        mv = lambda v: v.to(device)
        return Layout(mv(self.scale), mv(self.gy), mv(self.gx), mv(self.valid), mv(self.n), mv(self.cell_scale),
                      {s: tuple(mv(v) for v in g) for s, g in self.groups.items()}, self.known_padded)


def pack(active: dict, check: bool = False) -> Layout:
    bsz, device = active[1].shape[0], active[1].device
    grid = active[1].shape[1]
    counts = {s: active[s].flatten(1).sum(1) for s in SCALES}
    n = sum(counts.values())
    host = torch.stack([counts[s] for s in SCALES] + [n]).tolist()   # one wait for the GPU, not five
    length = max(host[-1])
    scale = torch.zeros(bsz, length, dtype=torch.long, device=device)
    gy, gx = torch.zeros_like(scale), torch.zeros_like(scale)
    cell = torch.zeros(bsz, grid, grid, dtype=torch.long, device=device)
    cover = torch.zeros_like(cell)
    offset = torch.zeros(bsz, dtype=torch.long, device=device)
    groups = {}
    for k, s in enumerate(SCALES):
        m = active[s]
        if sum(host[k]):
            b, i, j = m.nonzero(as_tuple=True)                      # sorted by image, then row-major
            starts = torch.cumsum(counts[s], 0) - counts[s]
            t = offset[b] + torch.arange(len(b), device=device) - starts[b]
            scale[b, t], gy[b, t], gx[b, t] = s, i * s, j * s
            groups[s] = (b, t, i * s, j * s)
            up = m.repeat_interleave(s, dim=1).repeat_interleave(s, dim=2).long()
            cell, cover = cell + s * up, cover + up
        offset = offset + counts[s]
    if check and not bool((cover == 1).all()):
        raise ValueError("leaf masks do not tile the grid")
    valid = torch.arange(length, device=device)[None, :] < n[:, None]
    return Layout(scale, gy, gx, valid, n, cell, groups, min(host[-1]) != length)


def select(active: dict, index) -> dict:
    """Leaf masks of a subset of the batch."""
    return {s: m[index] for s, m in active.items()}


# ---------------------------------------------------------------------------------------------------------
# overlay (the gallery legend: green 1x / 16 px, amber 2x / 32 px, blue 4x / 64 px, purple 8x / 128 px)
# ---------------------------------------------------------------------------------------------------------
OVERLAY_RGB = {1: (46, 160, 67), 2: (255, 179, 0), 4: (33, 150, 243), 8: (156, 39, 176)}


def overlay(images_m1_p1: torch.Tensor, active: dict, alpha: float = 0.38) -> torch.Tensor:
    """(B, 3, H, W) images in [-1, 1] + leaf masks -> uint8 (B, H, W, 3) with the token grid drawn on top."""
    img = ((images_m1_p1.float().clamp(-1, 1) + 1) * 127.5).permute(0, 2, 3, 1)
    bsz, h, w, _ = img.shape
    px = h // active[1].shape[1]
    tint = torch.zeros_like(img)
    edge = torch.zeros(bsz, h, w, dtype=torch.bool, device=img.device)
    for s in SCALES:
        m = active[s].repeat_interleave(s * px, dim=1).repeat_interleave(s * px, dim=2)
        tint = tint + m[..., None] * torch.tensor(OVERLAY_RGB[s], dtype=torch.float32, device=img.device)
        yy = torch.arange(h, device=img.device) % (s * px)
        xx = torch.arange(w, device=img.device) % (s * px)
        border = (yy[:, None] == 0) | (xx[None, :] == 0) | (yy[:, None] == s * px - 1) | (xx[None, :] == s * px - 1)
        edge = edge | (m & border[None])
    out = (1 - alpha) * img + alpha * tint
    out = torch.where(edge[..., None], 0.35 * out + 0.65 * tint, out)
    return out.round().clamp(0, 255).to(torch.uint8)
