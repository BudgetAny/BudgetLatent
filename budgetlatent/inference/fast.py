"""Speed settings for sampling (FLUX.2 [klein] 4B Base and 4B, FLUX 3 Image Klein Base and Klein).  The weights never change; the token layouts
change only with `preview`.

    from budgetlatent.inference.fast import optimize_for_inference, warmup
    optimize_for_inference(model, vae)                       # after loading, before sampling
    warmup(model, vae, cond, uncond, budgets)                 # before anything is timed

  * TF32 for the fp32 matmuls and convolutions that remain (refiners, VAE);
  * the VAE with cuDNN autotuning, channels-last unless channels_last=False (vae.Flux2VAE.optimize): its
    convolutions are what the layout previews cost, 5 decodes per image at the default relayout schedule.  --fast
    uses NCHW: with the compiled bf16 preview decoder it was a little faster than channels-last on an L40S and
    needed 1.3 instead of 1.5 GiB per 1024 x 1024 preview (the group norms need no layout copies);
  * torch.compile of the trunk's blocks (both families' block types).  dynamic=None (PyTorch's default): the
    first two distinct token counts compile, the second time with a symbolic sequence length that later budgets
    reuse; dynamic=False: one graph per token count, compiled the first time that count runs;
  * compile_layout: the layout builder's tensor work (24 bisection steps of the threshold quadtree and the top-down
    split ranking, ~700 small kernels per refresh) compiled into a few kernels -- same comparisons, counts and
    sorts, so the same layouts;
  * preview="bf16" / "bf16_compiled": the layout previews run an all-bf16 copy of the VAE decoder (compiled, for
    the second), instead of the fp32 decoder under bf16 autocast.  The final images still come from the fp32
    decoder; a layout can differ where two cell entropies were within bf16 rounding of each other;
  * fuse=True (`fuse_projections`, FLUX 3 only): each block's k, v, q and MLP-input projections as one matmul over the
    shared modulated input, and its attention and MLP output projections as one over [attention, MLP activation] --
    two matmuls instead of six; the same function of the same parameters, the last bits differ (summation order),
    which over FLUX 3 Image Klein Base's 50 guided steps grows into small differences in fine detail, as large as
    those between cuBLAS and cuBLASLt (median PSNR 32-35 dB against the six matmuls at 50 / 25 / 10 %, same content).
    FLUX.2's single blocks are fused as released and its double blocks cannot be (the MLP reads the attention's output);
  * warmup(..., blas="timed") (`tune_blas`): each budget's matmul library -- cuBLAS or cuBLASLt -- by timing, kept in
    sampling.BLAS_CHOICE for that shape (budget, rows, text window).  cuBLAS's heuristics pick a slow kernel for some
    row counts: FLUX 3 Image Klein at 10 % (409 image + 80 text tokens, one row) ran its 3072 x 3072 projections about
    2.5x slower than cuBLASLt, which made 10 % slower than 12.5 %; elsewhere cuBLAS is as fast or faster, and stays;
  * the text-mode blocks of FLUX 3 Image Klein Base run once per sample (sampling.CACHE_TEXT; always on, exact): they
    see the text alone at time 0, so their output is the same at every step (not for the flow map: t_ref changes the
    text's time vector);
  * optionally cuDNN's attention kernel instead of FlashAttention (attention="cudnn"); scripts/inference/bench.py
    --ablation measures all of this, --variants the last three against --fast without them;
  * optionally (deterministic=True, --deterministic) kernel choices that do not depend on timing
    (`deterministic_kernels`) and static graphs (dynamic=False unless given), so that compiled sampling of the 50-step
    models (FLUX.2 [klein] 4B Base, FLUX 3 Image Klein Base: two rows per image, guidance) gives the same bits in every
    process -- with a warm, a shared or a fresh inductor cache.  The 4-step FLUX.2 [klein] 4B (one row per image) is
    not fully covered: at 100 % and 10 % about a third of fresh processes gave a second, slightly different result
    (the 16-step FLUX 3 Image Klein, also one row per image, was not measured).
    It costs time: batch 1, --fast,
    one RTX PRO 6000 Blackwell, FLUX.2 +1.2 % at 100 % and +6.8 % at 10 %, FLUX 3 +1.6 % and +6.7 %
    (scripts/inference/bench.py --deterministic), and a compilation for every new token count, text length and
    batch size (seconds each once the inductor cache holds the kernels).

Without it (the default), PyTorch picks some kernels by timing the candidates the first time they run: the block
sizes of inductor's Triton pointwise and reduction kernels, kept for the life of the inductor cache
(TORCHINDUCTOR_CACHE_DIR), and cuDNN's convolution algorithms (the VAE), timed again in every process.  Results are
the same within a process; across processes only as far as these picks agree -- with a fresh inductor cache they
can differ.  Candidates that time within noise of each other win in different processes, and a reduction with
another block size sums in another order (torch 2.9.1: the layer norms over the 3072-wide hidden state at the full
grid, R0_BLOCK 2048 or 4096, from identical kernel sources): the last bits of the first velocity change, then,
through the layout thresholds, the layouts and the images.  `deterministic_kernels` turns the timing off:
  * torch._inductor.config.triton.autotune_pointwise = False: one heuristic config per pointwise and reduction kernel;
  * torch._inductor.config.triton.coalesce_tiling_analysis = False: without it, the wide reductions above still get a
    second, persistent-block candidate and are still timed (torch.use_deterministic_algorithms does not change that
    either, in torch 2.9.1);
  * torch._inductor.config.shape_padding = False: with static graphs, inductor would time each matmul whose row
    count is not a multiple of 8 padded and unpadded (it kept every one unpadded here) -- now never padded;
  * torch.backends.cudnn.benchmark = False: cuDNN's heuristic convolution algorithm.
With these, the symbolic-shape graph that dynamic=None compiles at the second token count still gave different
bits in some fresh processes (FLUX.2 at 10 % after 100 %: 5 of 19 processes, at least 3 different results, from
identical generated code and no timed kernel; the image stream of the first double block, stable within a
process); static graphs gave the same bits in every process (the 50-step models), hence dynamic=False with
deterministic=True.
Reproducible means: same GPU model, same torch / Triton / cuDNN versions, same settings and inputs.  The inductor
settings are process-wide and apply to graphs compiled after the call (compilation happens at the first call).

warmup runs every budget once, briefly, so that compilation, cuDNN's autotuning (if on) and the allocator's first growth
happen before any timing starts (a first call can take minutes; its time says nothing about the budget).  It samples
with the rows the timed runs will use: a family without guidance (FLUX.2 [klein] 4B, FLUX 3 Image Klein; uncond None) compiles the
single-row batch only.

`compile_blocks` alone is what scripts/train/train.py uses for the frozen teacher, whose input shape never changes
(and, with a symbolic shape, for the student: --compile_student).
"""
import types

import torch

from ..models import flux2_trunk as trunk          # SDPA_BACKEND, shared by both trunks
from ..models.flux2_trunk import DoubleStreamBlock, SingleStreamBlock
from ..models.flux3_trunk import Block as Flux3Block
from ..tokenization import layout as L

BLOCK_TYPES = (DoubleStreamBlock, SingleStreamBlock, Flux3Block)

_COMPILED = {}
PREVIEWS = ("autocast", "bf16", "bf16_compiled")


def compile_blocks(model, dynamic=None, mode=None):
    """Bind compiled forwards to the transformer blocks of `model` -- and only of `model`: other instances of the
    same classes (a student next to its teacher) keep the plain Python forward."""
    import torch._dynamo as dynamo
    # one entry per block instance and shape (static graphs: per token count, text length and batch size -- past
    # the limit dynamo would run the blocks eagerly)
    static = dynamic is False
    dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, 4096 if static else 512)
    if hasattr(dynamo.config, "accumulated_cache_size_limit"):
        dynamo.config.accumulated_cache_size_limit = max(dynamo.config.accumulated_cache_size_limit, 16384 if static else 4096)
    for mod in model.modules():
        if isinstance(mod, BLOCK_TYPES):
            key = (type(mod), dynamic, mode)
            if key not in _COMPILED:
                _COMPILED[key] = torch.compile(type(mod).forward, dynamic=dynamic, mode=mode)
            mod.forward = types.MethodType(_COMPILED[key], mod)
    return model


def compile_layouts():
    """The layout builder's tensor work, compiled (one graph per batch size; made once per process)."""
    if "layouts" not in _COMPILED:
        _COMPILED["layouts"] = torch.compile(L._exact_layout, dynamic=False)
    L.EXACT_LAYOUT_FN = _COMPILED["layouts"]


def deterministic_kernels():
    """Kernel choices by heuristics instead of by timing (see the module docstring): inductor's Triton configs for the
    graphs compiled from now on in this process, and cuDNN's convolution algorithms."""
    import torch._inductor.config as inductor_config
    inductor_config.triton.autotune_pointwise = False
    inductor_config.triton.coalesce_tiling_analysis = False
    inductor_config.shape_padding = False
    torch.backends.cudnn.benchmark = False


def fuse_projections(model, enable=True):
    """FLUX 3 blocks (both FLUX 3 families): the six projections of each block as two matmuls (Flux3Block.w_in /
    w_out; the Linear weights become views of the fused tensors, so the parameters and the state dict are unchanged).
    enable=False goes back to the six matmuls (the fused tensors stay; the views are the weights).  FLUX.2's single
    blocks are fused as released, and its double blocks cannot be (the MLP reads the attention's output).  Returns the
    number of blocks fused."""
    n = 0
    for mod in model.modules():
        if not isinstance(mod, Flux3Block):
            continue
        if enable and mod.w_in is None:
            d = mod.q_proj.weight.shape[0]
            if not hasattr(mod, "_w_in"):
                with torch.no_grad():
                    w_in = torch.cat([mod.k_proj.weight, mod.v_proj.weight, mod.q_proj.weight, mod.mlp_in.weight])
                    w_out = torch.cat([mod.attn_out.weight, mod.mlp_out.weight], dim=1)
                parts = ((mod.k_proj, w_in[:d]), (mod.v_proj, w_in[d:2 * d]), (mod.q_proj, w_in[2 * d:3 * d]),
                         (mod.mlp_in, w_in[3 * d:]), (mod.attn_out, w_out[:, :d]), (mod.mlp_out, w_out[:, d:]))
                for lin, view in parts:                              # (the separate weights are freed)
                    lin.weight = torch.nn.Parameter(view, requires_grad=lin.weight.requires_grad)
                mod._w_in, mod._w_out = w_in, w_out
            mod.w_in, mod.w_out = mod._w_in, mod._w_out
            n += 1
        elif not enable:
            mod.w_in = mod.w_out = None
    return n


def tune_blas(model, vae, cond, uncond, budget, steps, rounds=2, margin=0.02, say=None, **sample_kw):
    """Time this sampling shape (budget, rows, text window) with cuBLAS and with cuBLASLt, `rounds` times each after
    one untimed run, and keep the faster in sampling.BLAS_CHOICE -- cuBLASLt only if it is at least `margin` faster
    (else PyTorch's default stays).  Each run is `steps` steps without the final decode."""
    import time

    from . import sampling as S
    key = S.blas_key(model, S.snap_tokens(budget), cond.shape[0] * (2 if S._guided(model, uncond, sample_kw.get("cfg")) else 1),
                     S._text_len(model, cond, uncond, sample_kw.get("cfg")))
    times = {"cublas": [], "cublaslt": []}
    for r in range(rounds + 1):
        for lib in times:
            S.BLAS_CHOICE[key] = lib
            torch.cuda.synchronize()
            tic = time.perf_counter()
            S.sample(model, vae, cond, uncond, budget=budget, **dict(sample_kw, steps=steps, decode=False))
            torch.cuda.synchronize()
            if r:
                times[lib].append(time.perf_counter() - tic)
    best = "cublaslt" if min(times["cublaslt"]) < (1 - margin) * min(times["cublas"]) else "cublas"
    S.BLAS_CHOICE[key] = best
    if say:
        say(f"[blas] budget {budget:.1%}: cuBLAS {min(times['cublas']):.3f} s, cuBLASLt {min(times['cublaslt']):.3f} s"
            f" ({steps} steps) -> {best}")
    return best


def optimize_for_inference(model, vae=None, compile=True, attention=None, compile_vae=False, dynamic=None,
                           compile_layout=False, preview="autocast", channels_last=True, deterministic=False, fuse=False):
    """The settings of the module docstring.  deterministic=True: `deterministic_kernels` and, unless `dynamic` is
    given, static graphs (the same bits in every process for the 50-step models, a few percent slower); else inductor
    and cuDNN time their candidates.  fuse: `fuse_projections` (FLUX 3)."""
    if preview not in PREVIEWS:
        raise ValueError(f"preview must be one of {PREVIEWS}, not {preview!r}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    trunk.SDPA_BACKEND = attention
    if deterministic and dynamic is None:
        dynamic = False
    if fuse:
        fuse_projections(model)
    if compile:
        compile_blocks(model, dynamic=dynamic)
    if compile_layout:
        compile_layouts()
    if vae is not None:
        vae.optimize(compile_decoder=compile_vae, channels_last=channels_last)
        if preview != "autocast":
            vae.bf16_preview(compile=preview == "bf16_compiled")
    if deterministic:                                                # (after vae.optimize, which turns cuDNN autotuning on)
        deterministic_kernels()
    return model


BLAS_MODES = ("timed", "cublas", "cublaslt")


def add_arguments(ap, skip=()):
    """The fast path's parts as command-line switches, for sample.py / gallery.py (and bench.py): each defaults to what
    --fast (and --deterministic) choose, and can be turned on or off alone.  Read back with `settings(a)` (bench.py:
    `overrides(a)`).  skip: switches the script already has (e.g. ("attention",))."""
    g = ap.add_argument_group("fast path, part by part (default: as --fast sets them; budgetlatent/inference/fast.py)")
    g.add_argument("--fuse", dest="fuse", action="store_true", default=None,
                   help="FLUX 3: each block's six projections as two matmuls (default: on with --fast)")
    g.add_argument("--no_fuse", dest="fuse", action="store_false", help="FLUX 3: the six projections as released")
    g.add_argument("--blas", default=None, choices=BLAS_MODES,
                   help="matmul library: 'timed' = per budget, the faster of cuBLAS and cuBLASLt at the warm-up "
                        "(default with --fast, not with --deterministic), or one for everything (default: cublas)")
    g.add_argument("--text_cache", dest="text_cache", action="store_true", default=None,
                   help="FLUX 3 Image Klein Base: the text-mode blocks once per image (default: on; exact)")
    g.add_argument("--no_text_cache", dest="text_cache", action="store_false",
                   help="FLUX 3 Image Klein Base: the text-mode blocks at every step, as released")
    g.add_argument("--preview_decoder", default=None, choices=PREVIEWS,
                   help="decoder of the layout previews: a compiled bf16 copy (default with --fast), a bf16 copy, or "
                        "the fp32 decoder under autocast")
    g.add_argument("--compile_layout", dest="compile_layout", action="store_true", default=None,
                   help="compile the layout builder (default: on with --fast)")
    g.add_argument("--no_compile_layout", dest="compile_layout", action="store_false")
    if "attention" not in skip:
        g.add_argument("--attention", default=None, choices=("flash", "cudnn"),
                       help="attention kernel: FlashAttention (default) or cuDNN's")
    return g


def overrides(a):
    """The switches of `add_arguments` given on the command line, as keys of bench.py's settings dicts."""
    out = {}
    if a.fuse is not None:
        out["fuse"] = a.fuse
    if a.blas:
        out["blas"] = a.blas
    if a.text_cache is not None:
        out["cache_text"] = a.text_cache
    if a.preview_decoder:
        out["preview"] = a.preview_decoder
    if a.compile_layout is not None:
        out["compile_layout"] = a.compile_layout
    return out


def settings(a):
    """The resolved fast-path settings of `add_arguments`' switches (with a.fast, a.compile, a.deterministic and
    a.final_decode): a dict for `apply_settings`."""
    fast, det = bool(getattr(a, "fast", False)), bool(getattr(a, "deterministic", False))
    blas = a.blas or ("timed" if fast and not det else "cublas")
    if det and blas == "timed":
        raise SystemExit("--blas timed picks the library by timing; with --deterministic pick one (cublas or cublaslt)")
    final_bf16 = getattr(a, "final_decode", None) == "bf16"
    return dict(compile=fast or bool(getattr(a, "compile", False)),
                compile_layout=fast if a.compile_layout is None else a.compile_layout,
                preview=a.preview_decoder or ("bf16_compiled" if fast else ("bf16" if final_bf16 else "autocast")),
                channels_last=not fast, deterministic=det,
                fuse=fast if a.fuse is None else a.fuse,
                attention="cudnn" if a.attention == "cudnn" else None,
                text_cache=True if a.text_cache is None else a.text_cache, blas=blas)


def apply_settings(model, vae, s):
    """optimize_for_inference with the settings of `settings`, the text-mode cache switch and a fixed matmul library;
    returns the warm-up's blas argument ("timed" or None)."""
    from . import sampling
    optimize_for_inference(model, vae, compile=s["compile"], attention=s["attention"], compile_layout=s["compile_layout"],
                           preview=s["preview"], channels_last=s["channels_last"], deterministic=s["deterministic"],
                           fuse=s["fuse"])
    sampling.CACHE_TEXT = s["text_cache"]
    if s["blas"] != "timed" and torch.cuda.is_available():
        torch.backends.cuda.preferred_blas_library(s["blas"])
    return "timed" if s["blas"] == "timed" else None


def warmup(model, vae, cond, uncond, budgets, relayout_every=None, say=None, blas=None, **sample_kw):
    # (relayout_every + 2 steps: one refresh after the noise layout, whatever relayout_until is, unless it is 0)
    """Run each budget for relayout_every + 2 steps (a noise layout, a forward, a previewed refresh, another
    forward, the final decode), with the batch size and text that the timed runs will use.  relayout_every None:
    the family's (4; FLUX.2 [klein] 4B: 1, so 3 steps).  blas="timed": then pick the matmul library of each budget by
    timing (`tune_blas`; on a GPU, not with deterministic kernels)."""
    import time

    from .sampling import sample, sampler_defaults
    relayout_every = sampler_defaults(model, relayout_every=relayout_every)[2]
    for b in budgets:
        tic = time.time()
        sample(model, vae, cond, uncond, budget=b, relayout_every=relayout_every,
               **dict(sample_kw, steps=relayout_every + 2, decode=True))
        if cond.is_cuda:
            torch.cuda.synchronize()
        if say:
            say(f"[warmup] budget {b:.0%}: {time.time() - tic:.1f} s")
        if blas == "timed" and cond.is_cuda:
            tune_blas(model, vae, cond, uncond, b, relayout_every + 2, say=say, relayout_every=relayout_every, **sample_kw)


def describe(compile=True, attention=None, compile_vae=False, dynamic=None, compile_layout=False, preview="autocast",
             channels_last=True, deterministic=False, fuse=False, blas=None):
    parts = ["bf16 weights and compute", "cuDNN attention" if attention == "cudnn" else "FlashAttention"]
    if compile:
        parts.append("torch.compile (blocks" + (", static shapes" if dynamic is False else "") +
                     (", VAE decoder" if compile_vae else "") + (", layout builder" if compile_layout else "") + ")")
    elif compile_layout:
        parts.append("torch.compile (layout builder)")
    parts.append({"autocast": "previews: fp32 VAE under bf16 autocast", "bf16": "previews: bf16 VAE copy",
                  "bf16_compiled": "previews: compiled bf16 VAE copy"}[preview])
    parts.append("channels-last VAE" if channels_last else "NCHW VAE")
    if fuse:
        parts.append("fused FLUX 3 projections")
    if blas == "timed" and not deterministic:
        parts.append("matmul library by timing")
    elif blas == "cublaslt":
        parts.append("cuBLASLt matmuls")
    parts.append("kernels by heuristics (deterministic)" if deterministic else "kernels by timing")
    return ", ".join(parts)
