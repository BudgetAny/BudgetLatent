"""The sampler of BudgetPix at a token budget, for every family (scripts/inference/sample.py is its command line).

    cond, uncond, lens = encode_prompts(text, model, prompts)           # uncond None for a family without guidance
    out = sample(model, vae, cond, uncond, budget=0.5, **lens)          # latent, layout, tokens, image

Steps, guidance and layout refreshes default to the model's family (families.Family, `sampler_defaults`):

  FLUX.2 [klein] 4B Base   50 Euler steps, the empirical shift mu(tokens), guidance 4.0, layouts at steps 0, 4, ..., 20
  FLUX.2 [klein] 4B        4 Euler steps on BFL's get_schedule(4, tokens), no guidance, layouts at steps 0 and 1 (256 px preview)
  FLUX 3 Image Klein Base  50 Euler steps, the constant shift 6.93, guidance 3.5, layouts at steps 0, 4, ..., 20
  FLUX 3 Image Klein       16 flow-map steps on the same schedule, no guidance, layouts at steps 0, 8 (256 px preview)

With guidance there is one fused [cond; uncond] pass per step (FLUX 3: the empty prompt's 80-token window is padded
to the prompt's and masked); without it (FLUX.2 [klein] 4B, FLUX 3 Image Klein, or cfg=1.0) one row per image goes
through the trunk and no unconditional text is encoded.  FLUX 3 Image Klein is a flow map (BFL's flux3/sampling.py
denoise_flow_map): each step also passes the time it jumps to, t_ref = the next time of the schedule (1, clean, after
the last), and moves z by (t_next - t) times the average velocity the model returns; it has no unconditional branch,
so a cfg other than 1.0 is refused.  The layouts read the clean estimate z + (1 - t) v, as for the other families.  With a budget below 1 the first layout comes from the noise itself (no dense
warm-up step) and is then re-derived from the entropy of the decoded running estimate of the clean image at steps
4, 8, ..., 20 (RELAYOUT_UNTIL; the layout of step 20 is kept to the end; FLUX.2 [klein] 4B: at step 1 only, from a 256 px
preview; FLUX 3 Image Klein: once, at step 8, from a 256 px preview).
Refreshing every 4 steps to the end (relayout_until=None, 12 decodes instead of 5) changed the 50-step images by less
than 0.1 dB of PSNR against the dense image of the same seed, for 6 - 16 % more time per image
(scripts/inference/bench.py --schedules).  Every image of a batch gets exactly the same number of tokens,
N = N_MIN + 3k (64 + 3k at 1024 x 1024), so there is no padding.

FLUX.2 (base and distilled): between two layout refreshes the steps share one model.prepare(layout): rotary tables, index tensors and the
layout-only embeddings are made once per layout instead of once per step, and no step waits for the GPU (the
images are those of the per-step path, reuse_layout=False).  FLUX 3 runs the per-step path.  `timer` (PhaseTimer)
splits a run's time into the layout preview, the layout itself, the model and the update -- it synchronises at
every phase, so it is for profiling only.
"""
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

from ..tokenization.layout import GRID, N_DENSE, build_exact_budget_layout_s8, noise_latent_pyramid_s8, pack, \
    pixel_entropy_pyramid_s8, schedule, snap_tokens

RELAYOUT_EVERY, RELAYOUT_UNTIL = 4, 20                       # layout refreshes at steps 0, 4, ..., 20 (the 50-step families)
FAMILY = "family"                                            # a sampler argument left to the model's family


def sampler_defaults(model, steps=None, cfg=None, relayout_every=None, relayout_until=FAMILY):
    """(steps, cfg, relayout_every, relayout_until) with None (relayout_until: FAMILY) replaced by the defaults of the
    model's family (families.Family; model.family, set by weights.load_model)."""
    from ..models.families import family_of
    fam = family_of(getattr(model, "family", "flux2"))
    return (fam.steps if steps is None else int(steps), fam.guidance if cfg is None else cfg,
            fam.relayout_every if relayout_every is None else int(relayout_every),
            fam.relayout_until if isinstance(relayout_until, str) and relayout_until == FAMILY else relayout_until)


def preview_side(model, preview_px=None):
    """The side (px) of the decoded preview layouts are computed from: preview_px, or the family's (512 for the 50-step
    models, 256 for the step-distilled ones; 1024 = the full image)."""
    from ..models.families import family_of
    return int(preview_px) if preview_px else family_of(getattr(model, "family", "flux2")).preview_px


def decode_preview(vae, z0_hat, side_px, bf16=False):
    """The running estimate decoded for a layout refresh, at full size (16 * grid px): decoded directly when side_px is
    the full size, else decoded from the latent average-pooled to side_px and upsampled (bilinear) back to full size,
    so the pixel-entropy pyramid is computed as usual."""
    full = 16 * z0_hat.shape[1]
    factor = max(1, full // int(side_px))               # a preview at least as large as the image: the full decode
    if int(side_px) < full and (full % int(side_px) or factor & (factor - 1)):
        raise ValueError(f"preview_px must divide the image side {full} by a power of 2 ({full}, {full // 2}, {full // 4}, "
                         f"{full // 8}, ...), not {side_px}")
    if factor == 1:
        return vae.decode(z0_hat, bf16=bf16)
    small = vae.decode(z0_hat, bf16=bf16, downscale=factor)
    return torch.nn.functional.interpolate(small, size=(full, full), mode="bilinear", align_corners=False)


def initial_noise(batch, grid=None, channels=128, seed=42, device="cpu", offset=0):
    """Image k of a run always gets the noise of seed + offset + k, whatever the batching.  grid: layout.GRID."""
    grid, out = grid or GRID, []
    for k in range(batch):
        g = torch.Generator(device="cpu").manual_seed(int(seed) + offset + k)
        out.append(torch.randn((grid, grid, channels), generator=g, dtype=torch.float32))
    return torch.stack(out).to(device)


class PhaseTimer:
    """Wall time per phase of `sample` (seconds, summed over calls), with a CUDA synchronisation at every phase
    boundary: phases are measured one after the other, so the total is a little above an untimed run's."""

    def __init__(self, cuda=True):
        self.cuda, self.seconds, self.calls = cuda and torch.cuda.is_available(), defaultdict(float), defaultdict(int)

    @contextmanager
    def __call__(self, name):
        if self.cuda:
            torch.cuda.synchronize()
        tic = time.perf_counter()
        yield
        if self.cuda:
            torch.cuda.synchronize()
        self.seconds[name] += time.perf_counter() - tic
        self.calls[name] += 1


@contextmanager
def _untimed(name):
    yield


# The matmul library of each sampling shape, by timing (fast.warmup(..., blas="timed") fills it; empty: PyTorch's
# default).  Key: blas_key(...).  cuBLAS's heuristics pick a much slower kernel for some row counts (FLUX 3 Image Klein,
# one row of 409 image + 80 text tokens at 10 %: about 2.5x slower 3072 x 3072 projections), cuBLASLt for others.
BLAS_CHOICE = {}
# FLUX 3 Image Klein Base: run the text-mode blocks once per sample (exact: their input does not change between steps)
CACHE_TEXT = True


def blas_key(model, n_tokens, rows, text_len):
    """The shape a sample's matmuls run at: family, image tokens per row, rows through the trunk, text window."""
    return getattr(model, "family", "flux2"), int(n_tokens), int(rows), int(text_len)


def _guided(model, txt_uncond, cfg):
    """Whether `sample` runs the unconditional row too (as it decides it)."""
    return txt_uncond is not None and sampler_defaults(model, cfg=cfg)[1] != 1.0


def _text_len(model, txt_cond, txt_uncond, cfg):
    return max(txt_cond.shape[1], txt_uncond.shape[1]) if _guided(model, txt_uncond, cfg) else txt_cond.shape[1]


@contextmanager
def blas_library(name):
    """Run the block with torch.backends.cuda.preferred_blas_library(name) ("cublas" / "cublaslt"; None: unchanged)."""
    if name is None or not torch.cuda.is_available():
        yield
        return
    previous = torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library(name)
    try:
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(previous)


@torch.inference_mode()
def sample(model, vae, txt_cond, txt_uncond=None, *, budget=1.0, steps=None, cfg=None, relayout_every=None,
           relayout_until=FAMILY, layout_source="pixel", first_layout="noise", dense_steps=0, noise=None, seed=42,
           amp=True, decode=True, len_cond=None, len_uncond=None, shift=None, final_decode="fp32", reuse_layout=True,
           timer=None, memory=None, preview_px=None, refs=None):
    """txt_cond / txt_uncond: (B, Lt, D) projected text (model.project_text); len_cond / len_uncond (B,) their text
    windows (FLUX 3; None = the whole tensor; encode_prompts returns them).  steps / cfg None: the family's (FLUX.2
    [klein] 4B Base 50 / 4.0, FLUX.2 [klein] 4B 4 / 1.0, FLUX 3 Image Klein Base 50 / 3.5, FLUX 3 Image Klein 16 / 1.0);
    cfg 1.0 (or no txt_uncond) samples one row per image, without an unconditional pass (FLUX 3 Image Klein, a flow
    map: always, and with t_ref); shift: the schedule's shift factor (None: the family's).

    relayout_every / relayout_until: the layout is rebuilt at steps 0, every, 2 every, ... up to `until` (None: to
    the end; the default, FAMILY, is the family's: 4 / 20, FLUX.2 [klein] 4B 1 / 1, FLUX 3 Image Klein 8 / 8).  layout_source: "pixel" (entropy of the decoded running estimate; the bf16
    decoder copy of vae.bf16_preview if it was made, else the fp32 decoder under bf16 autocast) or "latent" (no
    decode: the spatial spread of the running estimate itself).  preview_px: the side of that decoded preview (None:
    the family's, 512, FLUX.2 [klein] 4B 256, FLUX 3 Image Klein 256: the latent pooled before the decode, then upsampled, see decode_preview).  dense_steps: steps run on the dense grid first (0
    with the noise layout).  reuse_layout=False runs the per-step path on FLUX.2 too (the layout's tables rebuilt at
    every step), for comparisons.  final_decode: "fp32" (the VAE as released) or "bf16" (the bf16 decoder copy of
    vae.bf16_preview, about a third of the fp32 decoder's activation memory; the images differ from fp32 by the
    bf16 rounding of the decoder).  memory: a dict that receives, on a GPU, torch's allocator peak (bytes) of the
    denoising loop ("loop") and of the final decode alone ("decode"); the caller resets the peak before the call.

    refs (models/refs.References): reference images for editing / multi-reference generation, the same for every
    row; they stay dense (the budget covers the generated image), so they add their tokens to every pass.

    Returns a dict: latent (B, g, g, 128); image (B, 3, 16 g, 16 g) in [-1, 1] if decode; layout, the final leaf
    masks (None when dense); tokens, the mean number of image tokens per step."""
    family = getattr(model, "family", "flux2")
    flow_map = bool(getattr(getattr(model, "cfg", None), "flow_map", False))     # FLUX 3 Image Klein: passes take t_ref
    if flow_map and cfg not in (None, 1.0):
        raise ValueError(f"FLUX 3 Image Klein is a flow map and samples without classifier-free guidance; cfg = {cfg}")
    steps, cfg, relayout_every, relayout_until = sampler_defaults(model, steps, cfg, relayout_every, relayout_until)
    preview_px = preview_side(model, preview_px)
    b, device = txt_cond.shape[0], txt_cond.device
    amp = amp and device.type == "cuda"
    tm = timer or _untimed
    z = initial_noise(b, channels=model.in_channels, seed=seed, device=device) if noise is None else noise.to(device, torch.float32)
    ts = schedule(steps, family, shift)
    n_tokens = snap_tokens(budget)
    adaptive, guided = n_tokens < N_DENSE, txt_uncond is not None and cfg != 1.0
    prepared = reuse_layout and getattr(model, "supports_prepare", False)
    text_len = max(txt_cond.shape[1], txt_uncond.shape[1]) if guided else txt_cond.shape[1]
    active, layout, z0_hat, used = None, None, z, 0
    blas = BLAS_CHOICE.get(blas_key(model, n_tokens, b * (2 if guided else 1), text_len)) if device.type == "cuda" else None
    # FLUX 3 Image Klein Base: the text-mode blocks run once per sample (their output does not change between steps)
    cache = {"txt_cache": {}} if CACHE_TEXT and getattr(model, "caches_text", False) and not flow_map else {}
    if refs is not None:
        cache["refs"] = refs
        blas = None                     # (the timed table holds the text-to-image shapes)
    with blas_library(blas):
        for i in range(steps):
            t_c, t_n = ts[i], ts[i + 1]
            if adaptive and i >= dense_steps and (i - dense_steps) % relayout_every == 0 \
                    and (active is None or relayout_until is None or i <= relayout_until):
                with tm("preview"):
                    if i == 0 and first_layout == "noise":
                        pyr = noise_latent_pyramid_s8(z)
                    elif layout_source == "pixel":
                        pyr = pixel_entropy_pyramid_s8(decode_preview(vae, z0_hat, preview_px, bf16="weights" if amp else False))
                    else:
                        pyr = noise_latent_pyramid_s8(z0_hat)
                with tm("layout"):
                    active = build_exact_budget_layout_s8(pyr, budget)
                    layout = pack(active)
                    if prepared:
                        layout = model.prepare(layout, text_len, repeat=2 if guided else 1)
            with tm("model"):
                t = torch.full((b,), t_c, device=device, dtype=torch.float32)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    if guided:
                        v = model.forward_cfg(z, t, txt_cond, txt_uncond, layout, cfg, len_cond, len_uncond, **cache)
                    elif flow_map:
                        v = model(z, t, txt_cond, layout, ctx_projected=True, txt_len=len_cond,
                                  t_ref=torch.full((b,), t_n, device=device, dtype=torch.float32),
                                  **({} if refs is None else {"refs": refs}))
                    else:
                        v = model(z, t, txt_cond, layout, ctx_projected=True, txt_len=len_cond, **cache)
            with tm("update"):
                used += N_DENSE if layout is None else n_tokens
                z0_hat = z + (1.0 - t_c) * v
                z = z + (t_n - t_c) * v
    out = {"latent": z, "layout": active, "tokens": used / steps}
    track = memory is not None and device.type == "cuda"
    if track:
        memory["loop"] = torch.cuda.max_memory_allocated(device)
    if decode:
        if final_decode not in ("fp32", "bf16"):
            raise ValueError(f"final_decode must be 'fp32' or 'bf16', not {final_decode!r}")
        if track:
            torch.cuda.reset_peak_memory_stats(device)
        with tm("decode"):
            out["image"] = vae.decode(z, bf16="weights" if final_decode == "bf16" and amp else False).clamp(-1.0, 1.0)
        if track:
            memory["decode"] = torch.cuda.max_memory_allocated(device)
    return out


TEXT_TOKENS_HELP = ("FLUX.2: how many of the 512 text tokens the model sees: 512 (as trained, the default), 'auto' (the "
                    "longest prompt's own tokens, empty prompt included, rounded up to 16) or a number (never fewer "
                    "than that): faster at small budgets, the images change (the padding is part of the trained input)")


def text_window(text, prompts, uncond, tokens):
    """FLUX.2's text tokens to keep (None: all 512, as trained) for `tokens` = 512 / None, 'auto' or a number.  Qwen3
    is causal, so the first L states of the padded encoding are those of any longer one: keeping L drops padding
    tokens only, never a prompt token (a number below the longest prompt keeps the prompt)."""
    from ..encoders.text import MAX_LENGTH
    if tokens is None or str(tokens) == str(MAX_LENGTH):
        return None
    need = int(text.tokenize(list(prompts))[1].sum(1).max())
    if uncond:
        need = max(need, int(text.tokenize([""])[1].sum()))
    n = -(-need // 16) * 16 if str(tokens) == "auto" else max(int(tokens), need)
    return min(n, MAX_LENGTH)


@torch.inference_mode()
def encode_prompts(text, model, prompts, amp=True, uncond=None, text_tokens=None):
    """Prompts -> (projected cond text, projected uncond text, {"len_cond": .., "len_uncond": ..}) for `sample`:
    sample(model, vae, cond, uncond, **lens, ...).  The lengths are None for FLUX.2 (its fixed 512-token window).
    uncond: whether to encode the empty prompt too (None: if the model's family samples with guidance; FLUX.2 [klein]
    4B does not, and its uncond and len_uncond are None -- pass uncond=True to sample it with a cfg of your own).
    text_tokens (FLUX.2): keep that many of the 512 text tokens (`text_window`; None / 512: all, as trained)."""
    from ..models.families import family_of
    device = next(model.parameters()).device
    amp = amp and device.type == "cuda"
    fam = family_of(getattr(model, "family", "flux2"))
    if uncond is None:
        uncond = fam.guided
    project = lambda s: model.project_text(s.to(device) if amp else s.to(device).float())
    c, lc = text.encode(list(prompts))
    u, lu = text.encode([""] * len(prompts)) if uncond else (None, None)
    if text_tokens is not None and str(text_tokens) != "512":
        if fam.arch != "flux2":
            raise ValueError(f"text_tokens is FLUX.2's (its padded 512-token window); {fam.label} uses each prompt's own window")
        keep = text_window(text, prompts, uncond, text_tokens)
        if keep is not None:
            c, u = c[:, :keep], None if u is None else u[:, :keep]
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        cond, null = project(c), None if u is None else project(u)
    lens = {"len_cond": None if lc is None else lc.cpu(), "len_uncond": None if lu is None else lu.cpu()}
    return cond, null, lens


def to_uint8(images):
    """(B, 3, H, W) in [-1, 1] -> uint8 (B, H, W, 3) on the CPU."""
    return ((images.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu()
