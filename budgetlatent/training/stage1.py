"""Stage 1's preparation of a step (scripts/train/train.py): teacher targets and token layouts for this rank's examples.

    prepare = make_prepare(loader, text, teacher, vae, gen, mix, dcfg, empty_ids, empty_mask, n_rank, prep_batch, 50, rt, clock)
    z_t, t, ctx, v, active, mode, lens = prepare()

One example: a clean latent is noised to a time t, the frozen dense teacher predicts the velocity, the image the
teacher implies is decoded where the layout mode reads it, and its entropy decides a token layout (one of four
modes).  Both families predict the flow velocity eps - x, so the target is the dense teacher's velocity for either;
FLUX 3 carries a text window per example (lens), FLUX.2's 512-token window is fixed (lens None).
"""
import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from ..tokenization import layout
from ..tokenization.layout import COARSE, GRID, draw_plan, noise_latent_pyramid_s8, pixel_entropy_pyramid_s8, sample_times, \
    training_layouts
from .data import to_unit


def text_mask_4d(mask):
    """(B, L) padding mask -> the (B, 1, L, L) causal + padding mask Qwen3's sdpa path builds from it.  Passed in
    directly, transformers skips its own construction and the host check in it (padding_mask.all())."""
    length = mask.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool, device=mask.device).tril_()
    return causal[None, None] & mask.bool()[:, None, None, :]


def make_prepare(loader, text, teacher, vae, gen, mix, dcfg, empty_ids, empty_mask, n_rank, prep_batch, num_steps, rt, clock,
                 data_dir="", fast=False, decode_bf16=None, teacher_context=nullcontext, shift=None, parked=()):
    """prepare(): the teacher targets and layouts for this rank's examples of one step -> (z_t, t, ctx, v, leaf
    masks, mode, text windows (B,) on the host or None).  empty_ids / empty_mask: the empty prompt on the host.
    shift: the training times' schedule shift (layout.sample_times; None = FLUX.2's).  parked: trainlib.Parked frozen
    models (teacher, text encoder), brought to the GPU for the preparation and parked again after it.

    fast (train.py --fast_prep; GPU only): the same numbers without a single host synchronisation in the hot path,
    so the host queues the whole preparation while the GPU still works and the GPU never waits for it --
      * the host's random numbers and the token ids reach the GPU through pinned memory, non-blocking (the same
        draws in the same order: the random stream is unchanged);
      * the layout plan is drawn before the teacher (it consumes no random number the teacher would), so which
        images need a detail map reaches the host while the GPU runs the teacher, and the decode takes those rows
        by index (no boolean indexing, which waits for the GPU);
      * Qwen3 gets its causal + padding mask ready-made (text_mask_4d) instead of building it with a host check.
    decode_bf16: the VAE preview's precision (vae.decode's bf16; default: bf16 autocast on a GPU).
    """
    device, amp = rt.device, rt.cuda
    dec_bf16 = amp if decode_bf16 is None else decode_bf16
    fast = bool(fast) and rt.cuda
    layout.ASYNC_COPY = fast                                         # sample_times, draw_plan, training_layouts
    h2d = (lambda x: x.pin_memory().to(device, non_blocking=True)) if fast else (lambda x: x.to(device))

    def encode_batch():
        """One loader batch -> (clean latents, text states, text windows or None) on the GPU."""
        batch = next(loader)
        clock.lap("data")
        if "latent" in batch:
            z_clean = batch["latent"].to(device, non_blocking=True).float()
        else:
            z_clean = vae.encode(to_unit(batch["pixel_values"], device))
        if z_clean.shape[1] != GRID:
            raise SystemExit(f"{data_dir}: {z_clean.shape[1]} x {z_clean.shape[2]} latents, the layout grid is {GRID} "
                             f"(the data and BP_GRID / the config's resolution disagree)")
        b = z_clean.shape[0]
        clock.lap("encode")
        drop = torch.rand(b, generator=gen) < dcfg["class_dropout_prob"]              # on the host, as the token ids
        ids = torch.where(drop[:, None], empty_ids.expand(b, -1), batch["input_ids"])
        mask = torch.where(drop[:, None], empty_mask.expand(b, -1), batch["attention_mask"])
        lens = text.lengths(mask)                                    # FLUX 3: each caption's window (host); FLUX.2: None
        ids, mask = text.trim(ids, mask)                             # FLUX 3: the batch's longest window
        ids, mask = h2d(ids), h2d(mask)
        with rt.autocast():
            ctx = text.encode_ids(ids, text_mask_4d(mask) if fast else mask)
        clock.lap("text")
        return z_clean, ctx, lens

    def teacher_batch(z_clean, ctx, lens):
        """Noise one batch to its training times and run the dense teacher -> the state for layout_batch."""
        b = z_clean.shape[0]
        t, first = sample_times(b, gen, device, float(dcfg["rollout_first_p"]), num_steps, shift=shift)
        eps = h2d(torch.randn(z_clean.shape, generator=gen))
        tb = t[:, None, None, None]
        z_t = tb * z_clean + (1.0 - tb) * eps
        plan = need_host = need_ready = None
        if fast:
            plan = draw_plan(b, mix, gen, device)
            need_host = torch.empty(b, dtype=torch.bool, pin_memory=True)
            need_host.copy_(plan.needs_detail & ~first, non_blocking=True)
            need_ready = torch.cuda.Event()
            need_ready.record()
        with rt.autocast(), teacher_context():
            v = teacher.forward_dense(z_t, t, ctx, txt_len=lens)
        clock.lap("teacher")
        return z_t, t, first, tb, ctx, v, lens, plan, need_host, need_ready

    def layout_batch(z_t, t, first, tb, ctx, v, lens, plan, need_host, need_ready):
        """The detail maps (decoding the teacher's clean estimate where the layout reads it) and the layouts."""
        b = z_t.shape[0]
        # the layout reads the image the teacher implies -- only where the mode uses it; at t = 0 the noise itself
        pyr = noise_latent_pyramid_s8(z_t)
        if fast:
            need_ready.synchronize()                                 # done long ago: it was queued before the teacher
            rows = need_host.nonzero().flatten()
            if len(rows):
                idx = h2d(rows)
                x0 = z_t.index_select(0, idx) + (1.0 - tb.index_select(0, idx)) * v.index_select(0, idx)
                pix = pixel_entropy_pyramid_s8(vae.decode(x0, bf16=dec_bf16))
                for s in COARSE:
                    pyr[s].index_copy_(0, idx, pix[s])
        else:
            plan = draw_plan(b, mix, gen, device)
            need = plan.needs_detail & ~first
            if bool(need.any()):
                pix = pixel_entropy_pyramid_s8(vae.decode((z_t + (1.0 - tb) * v)[need], bf16=dec_bf16))
                for s in COARSE:
                    pyr[s][need] = pix[s]
        active, mode = training_layouts(pyr, mix, gen, plan)
        clock.lap("layout")
        return z_t, t, ctx, v, active, mode, lens

    @torch.no_grad()
    def prepare():
        n_batches = math.ceil(n_rank / prep_batch)
        if parked:
            # one large consumer of memory on the GPU at a time: every caption with the text encoder, then every teacher
            # pass with the teacher, then the decodes and layouts with neither (the order of the random draws differs
            # from the single pass below; the distributions do not)
            text_parked, teacher_parked = parked[1], parked[0]
            text_parked.load()
            encoded = [encode_batch() for _ in range(n_batches)]
            text_parked.park()
            teacher_parked.load()
            states = []
            for k in range(n_batches):
                states.append(teacher_batch(*encoded[k]))
                encoded[k] = None
            teacher_parked.park()
            parts = [layout_batch(*st) for st in states]
            del states
        else:
            parts = [layout_batch(*teacher_batch(*encode_batch())) for _ in range(n_batches)]
        cat = lambda k: torch.cat([p[k] for p in parts])[:n_rank]
        active = {s: torch.cat([p[4][s] for p in parts])[:n_rank] for s in parts[0][4]}
        width = max(p[2].shape[1] for p in parts)                    # FLUX 3: the windows differ between the batches
        ctx = torch.cat([F.pad(p[2], (0, 0, 0, width - p[2].shape[1])) for p in parts])[:n_rank]
        lens = None if parts[0][6] is None else cat(6)
        return cat(0), cat(1), ctx, cat(3), active, cat(5), lens

    return prepare
