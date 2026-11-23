"""Latency, token count and memory per budget and where the time goes.

    python scripts/inference/bench.py --ckpt <ckpt> --out bench_flux2 [--compile] [--fast]    # the table
    BP_GRID=64 python scripts/inference/bench.py --ckpt BudgetAny/BudgetLatent-FLUX.2-klein-4B --fast --out bench_flux2_distilled
    python scripts/inference/bench.py --ckpt <FLUX 3 ckpt> --fast --out bench_flux3              # the family comes from the checkpoint
    BP_GRID=64 python scripts/inference/bench.py --ckpt BudgetAny/BudgetLatent-FLUX-3-Image-Klein --fast --out bench_flux3_distilled
    python scripts/inference/bench.py --ablation --out bench_flux2_ablation        # what each speed setting buys, at 3 budgets
    python scripts/inference/bench.py --profile --budgets 1.0 0.5 0.1              # seconds per phase of the sampler
    python scripts/inference/bench.py --compare --batch 1 8                        # the previous sampler settings against --fast
    python scripts/inference/bench.py --memory --batch 1 8                         # where the memory goes, per budget

bf16 weights, the step-0 noise layout and the family's sampler: FLUX.2 [klein] 4B Base, 50 Euler steps with fused
guidance 4.0 (two rows through the trunk) and a pixel-entropy relayout at steps 4, 8, ..., 20 (5 VAE decodes of the
running estimate; "previous": every 4 steps to step 48, 12 decodes); FLUX.2 [klein] 4B (step-distilled), 4 Euler
steps without guidance (one row through the trunk) and one relayout, at step 1 (1 VAE decode of a 256 px preview); FLUX 3 Image Klein
Base, as FLUX.2 [klein] 4B Base with guidance 3.5; FLUX 3 Image Klein (step-distilled, a flow map), 16 steps without
guidance (one row through the trunk) and one relayout, at step 8 (1 VAE decode of a 256 px preview).  The text is
projected once (FLUX.2: padded to 512 tokens as in training; FLUX 3: the prompt's own window).  Every budget is warmed up first (fast.warmup: compilation, cuDNN
autotuning if on, the allocator's first growth), then run once more untimed; the time is the median of --repeats
runs of the denoising loop, layouts included, without the text encoder and the final decode.  Memory is torch's
allocator peak; "activations" is that peak minus what is resident before the loop (the weights).

Settings ("previous" = what galleries and validation ran before; "--fast" = all of fast.py except cuDNN attention):
  previous   eager blocks, channels-last VAE decoding 4 images per pass, previews by the fp32 VAE under bf16
             autocast, layout tables per step, a layout refresh every 4 steps to the end
  --fast     compiled blocks, compiled layout builder, compiled bf16 preview decoder, NCHW VAE decoding one
             1024 x 1024 image per pass, layouts prepared once per refresh (FLUX.2; FLUX 3 runs per step), the family's
             refreshes, fused FLUX 3 projections, each budget's matmul library by timing at the warm-up, FLUX 3 Image
             Klein Base's text-mode blocks once per sample
Inductor and cuDNN pick kernels by timing their candidates; --deterministic picks them by heuristics instead, for every
setting of the process, and compiles static graphs (fast.deterministic_kernels: the same bits across processes and
fresh inductor caches for the 50-step models).

--compare runs "previous", "--fast with the previous refresh schedule" and "--fast" on the same GPU, budget by budget, and checks
the images: the latent of every setting against "previous" (and "previous" against itself, run to run), whether the
final layouts are identical, and the PSNR of the decoded images.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))    # the repository root

import torch

from budgetlatent.inference.sampling import TEXT_TOKENS_HELP
from budgetlatent.models.families import FAMILIES

ALL_BUDGETS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
PREVIOUS = dict(compile=False, dynamic=None, compile_layout=False, preview="autocast", channels_last=True,
                reuse_layout=False, attention=None, chunk=4, relayout_until=None, final_decode="fp32", fuse=False, blas=None,
                cache_text=False)
FAST = dict(PREVIOUS, compile=True, compile_layout=True, preview="bf16_compiled", reuse_layout=True, channels_last=False,
            chunk=None, relayout_until="family", final_decode="bf16",   # the family's (20; FLUX.2 [klein] 4B: 1; FLUX 3 Image Klein: 8)
            fuse=True, blas="timed", cache_text=True)
# --variants: --fast as released without each implementation change since, then with each alone, then with all (the
# same images up to rounding; the text-mode cache is exact), alternated budget by budget.  (name, setting, applies to)
RELEASED = dict(FAST, fuse=False, blas=None, cache_text=False)
VARIANTS = [("fast as released", RELEASED, lambda fam: True),
            ("+ matmul library by timing", dict(RELEASED, blas="timed"), lambda fam: True),
            ("+ fused projections", dict(RELEASED, fuse=True), lambda fam: fam.arch == "flux3"),
            ("+ text-mode blocks once", dict(RELEASED, cache_text=True), lambda fam: fam.arch == "flux3" and not fam.flow_map),
            ("all (= --fast)", FAST, lambda fam: fam.arch == "flux3")]


class Bench:
    def __init__(self, a):
        from budgetlatent.encoders.text import text_encoder
        from budgetlatent.encoders.vae import Flux2VAE
        from budgetlatent.inference import fast, sampling
        from budgetlatent.inference.sampling import PhaseTimer, encode_prompts, initial_noise, sample
        from budgetlatent.models import flux2_trunk
        from budgetlatent.models.families import family_of
        from budgetlatent.models.weights import checkpoint_family, checkpoint_template, klein_dir, load_model, resolve_checkpoint
        from budgetlatent.tokenization import layout as L
        self.a, self.fast, self.L, self.trunk, self.sample, self.PhaseTimer = a, fast, L, flux2_trunk, sample, PhaseTimer
        self.sampling = sampling
        self.dev = torch.device("cuda")
        ckpt = resolve_checkpoint(a.ckpt)
        self.family = a.family or checkpoint_family(ckpt)
        self.fam = fam = family_of(self.family)
        if fam.flow_map and a.cfg not in (None, 1.0):
            raise SystemExit(f"--cfg {a.cfg}: {fam.label} is a flow map and samples without guidance (cfg 1.0)")
        a.steps, a.cfg = a.steps or fam.steps, fam.guidance if a.cfg is None else a.cfg
        a.relayout_every = a.relayout_every or fam.relayout_every
        root = klein_dir(a.weights or fam.repo)
        text = text_encoder(self.family, root, self.dev, template=checkpoint_template(ckpt, family=self.family))
        self.model = load_model(ckpt, root, self.dev, dtype=torch.bfloat16, family=self.family).eval().requires_grad_(False)
        prompts = [a.prompt] if a.prompt else [
            "a red fox standing in a snowy forest at golden hour, detailed fur",
            "a corgi wearing a spacesuit on the moon, the earth in the black sky",
            "a bowl of ramen with a soft-boiled egg, steam rising, top-down food photography",
            "a futuristic city skyline at night with neon signs reflected in wet streets",
            "a close-up of a mechanical watch movement, gears and jewels, studio lighting",
            "a street market in Marrakech with colorful spices and woven baskets",
            "a stained glass window depicting a peacock, vivid colours, light shining through",
            "a cross-section of a geode with purple amethyst crystals, studio photo"]
        self.text_states = {}
        for bs in sorted(set(a.batch)):
            ps = [prompts[k % len(prompts)] for k in range(bs)]
            self.text_states[bs] = encode_prompts(text, self.model, ps, uncond=a.cfg != 1.0, text_tokens=a.text_tokens)
        torch.cuda.synchronize()
        with_text = torch.cuda.memory_allocated()
        del text
        import gc
        gc.collect()                                                 # (the encoder's modules hold reference cycles)
        torch.cuda.empty_cache()
        self.text_mib = (with_text - torch.cuda.memory_allocated()) / 2 ** 20      # the text encoder's weights
        self.vae = Flux2VAE(root, self.dev)
        self.noise = {bs: initial_noise(bs, channels=self.model.in_channels, seed=42, device=self.dev) for bs in set(a.batch)}
        self.previews, self.current = {}, None

    # ---- settings ---------------------------------------------------------------------------------------------
    def apply(self, s):
        self.fast.fuse_projections(self.model, enable=s.get("fuse", False))
        self.sampling.BLAS_CHOICE.clear()                            # (warm fills it again when the setting times it)
        lib = s.get("blas")                                          # a fixed library: cublas (default) or cublaslt
        torch.backends.cuda.preferred_blas_library(lib if lib in ("cublas", "cublaslt") else "cublas")
        self.sampling.CACHE_TEXT = s.get("cache_text", True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if self.a.deterministic:                                     # (process-wide: inductor's part acts on what compiles next)
            self.fast.deterministic_kernels()
        else:
            torch.backends.cudnn.benchmark = True
        self.trunk.SDPA_BACKEND = s["attention"]
        for mod in self.model.modules():
            if isinstance(mod, self.fast.BLOCK_TYPES):
                mod.__dict__.pop("forward", None)                     # back to the plain forward
        if s["compile"]:
            self.fast.compile_blocks(self.model, dynamic=self.dynamic(s))
        self.L.EXACT_LAYOUT_FN = None
        if s["compile_layout"]:
            self.fast.compile_layouts()
        fmt = torch.channels_last if s["channels_last"] else torch.contiguous_format
        self.vae.vae.to(memory_format=fmt)
        self.vae.memory_format = fmt
        self.vae.chunk = s["chunk"]
        self.vae.preview = None
        if s["preview"] != "autocast":                               # (a bf16 final decode uses the same copy)
            key = (s["channels_last"], s["preview"])
            if key not in self.previews:
                self.vae.bf16_preview(compile=s["preview"] == "bf16_compiled")
                self.previews[key] = self.vae.preview
            self.vae.preview = self.previews[key]
        self.current = s

    def dynamic(self, s):
        """--deterministic compiles static graphs unless the setting names its own (as fast.optimize_for_inference)."""
        return False if self.a.deterministic and s["dynamic"] is None else s["dynamic"]

    def kw(self, **extra):
        a, s = self.a, self.current
        until = s["relayout_until"] if a.relayout_until == "setting" else a.relayout_until
        return dict(dict(steps=a.steps, cfg=a.cfg, relayout_every=a.relayout_every, relayout_until=until,
                         layout_source=a.layout_source, preview_px=a.preview_px, reuse_layout=s["reuse_layout"],
                         final_decode=s["final_decode"]), **extra)

    def warm(self, budgets, bs):
        cond, uncond, lens = self.text_states[bs]
        blas = None if self.a.deterministic or self.current.get("blas") != "timed" else "timed"
        self.fast.warmup(self.model, self.vae, cond, uncond, budgets, say=lambda m: print(m, flush=True), blas=blas, **lens,
                         **{k: v for k, v in self.kw(noise=self.noise[bs]).items() if k not in ("steps",)})

    def run(self, budget, bs, repeats=None, keep=False, untimed=True, **extra):
        """Median seconds per image over `repeats` timed runs (after one untimed run, unless untimed=False)."""
        cond, uncond, lens = self.text_states[bs]
        kw = self.kw(noise=self.noise[bs], decode=False, **lens, **extra)
        times, outs = [], []
        for r in range(0 if untimed else 1, (repeats or self.a.repeats) + 1):
            torch.cuda.synchronize()
            resident = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            tic = time.perf_counter()
            res = self.sample(self.model, self.vae, cond, uncond, budget=budget, **kw)
            torch.cuda.synchronize()
            if r:
                times.append((time.perf_counter() - tic) / bs)
                if keep:
                    outs.append(res)
        peak = torch.cuda.max_memory_allocated()
        n = self.L.snap_tokens(budget)
        row = {"tokens": n, "joint": cond.shape[1] + n, "mean_tokens": res["tokens"], "seconds": statistics.median(times),
               "all_seconds": times, "activation_mib": (peak - resident) / 2 ** 20, "total_mib": peak / 2 ** 20}
        return (row, outs) if keep else row

    def profile(self, budget, bs):
        cond, uncond, lens = self.text_states[bs]
        timer = self.PhaseTimer()
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        tic = time.perf_counter()
        self.sample(self.model, self.vae, cond, uncond, budget=budget, timer=timer, **lens, **self.kw(noise=self.noise[bs], decode=True))
        total = time.perf_counter() - tic
        peak = torch.cuda.max_memory_allocated()
        return {"total": total / bs, **{k: v / bs for k, v in timer.seconds.items()}, "calls": dict(timer.calls),
                "activation_mib": (peak - resident) / 2 ** 20, "total_mib": peak / 2 ** 20}

    def decode(self, latent, bf16=False):
        return self.vae.decode(latent, bf16="weights" if bf16 else False).clamp(-1, 1)

    def memory(self, budget, bs):
        """One warmed run with the final decode: MiB resident before it (the weights), and the activations (peak
        minus resident) of the denoising loop and of the final decode, each measured on its own."""
        cond, uncond, lens = self.text_states[bs]
        mem = {}
        torch.cuda.synchronize()
        resident = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        res = self.sample(self.model, self.vae, cond, uncond, budget=budget, memory=mem, **lens, **self.kw(noise=self.noise[bs], decode=True))
        torch.cuda.synchronize()
        return {"resident_mib": resident / 2 ** 20, "loop_mib": (mem["loop"] - resident) / 2 ** 20,
                "decode_mib": (mem["decode"] - resident) / 2 ** 20, "total_mib": max(mem.values()) / 2 ** 20}, res


def same_layout(x, y):
    """Per image: are the two runs' final leaf masks identical (True for two dense runs)."""
    if x is None or y is None:
        return [x is None and y is None]
    b = next(iter(x.values())).shape[0]
    return [all(bool((x[s][i] == y[s][i]).all()) for s in x) for i in range(b)]


def psnr(x, y):
    mse = float(((x - y) / 2).pow(2).mean())                          # images in [-1, 1] -> [0, 1]
    return math.inf if mse == 0 else 10 * math.log10(1.0 / mse)


SCHEDULES = [("every 4 steps to the end (the previous default): 12 previews", dict(relayout_until=None)),
             ("every 4 steps up to step 28: 7 previews", dict(relayout_until=28)),
             ("every 8 steps: 6 previews", dict(relayout_every=8, relayout_until=None)),
             ("every 4 steps up to step 20 (the default): 5 previews", dict(relayout_until=20)),
             ("every 8 steps up to step 32: 4 previews", dict(relayout_every=8, relayout_until=32)),
             ("every 4 steps up to step 12: 3 previews", dict(relayout_until=12)),
             ("every 6 steps up to step 18: 3 previews", dict(relayout_every=6, relayout_until=18)),
             ("steps 0, 5 and 10: 2 previews", dict(relayout_every=5, relayout_until=10)),
             ("the noise layout of step 0 only: no preview", dict(relayout_until=0))]


def schedule_sheet(path, title, named_rows, tile=192):
    """named_rows: [(label, images uint8 (B, H, W, 3), overlays uint8 (B, H, W, 3))] -> one sheet: per row label,
    the images, and under them the same images with their token layout."""
    from PIL import Image, ImageDraw
    label_w, head = 430, 56
    b = named_rows[0][1].shape[0]
    sheet = Image.new("RGB", (label_w + b * tile, head + len(named_rows) * 2 * tile), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((12, 12), title, fill=(20, 20, 20))
    for r, (label, imgs, overs) in enumerate(named_rows):
        y = head + r * 2 * tile
        for j in range(b):
            for k, src in enumerate((imgs, overs)):
                sheet.paste(Image.fromarray(src[j].numpy()).resize((tile, tile), Image.LANCZOS), (label_w + j * tile, y + k * tile))
        for li, line in enumerate(label.split("\n")):
            d.text((12, y + 12 + 16 * li), line, fill=(30, 30, 30))
        d.text((12, y + tile + 12), "token layout", fill=(120, 120, 120))
    sheet.save(path, quality=90)


def table(rows, budgets, title):
    base = rows[f"{budgets[0]:g}"]["seconds"]
    lines = [title, "", "| budget | image tokens | joint sequence | seconds / image | speed-up | peak activations | peak total |",
             "|---|---|---|---|---|---|---|"]
    for b in budgets:
        r = rows[f"{b:g}"]
        r["speedup"] = base / r["seconds"]
        lines.append(f"| {b:.0%} | {r['tokens']} | {r['joint']} | {r['seconds']:.3f} | {r['speedup']:.2f}x | "
                     f"{r['activation_mib']:.0f} MiB | {r['total_mib'] / 1024:.2f} GiB |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="a local .safetensors or a Hugging Face repo id (see sample.py)")
    ap.add_argument("--family", default=None, choices=sorted(FAMILIES), help="default: the checkpoint's (else flux2)")
    ap.add_argument("--weights", default=None, help="snapshot of the released base model (default: the HF cache)")
    ap.add_argument("--out", default="bench")
    ap.add_argument("--budgets", type=float, nargs="+", default=None)
    ap.add_argument("--steps", type=int, default=None, help="default: the family's (50; FLUX.2 [klein] 4B: 4; FLUX 3 Image Klein: 16)")
    ap.add_argument("--cfg", type=float, default=None,
                    help="default: the family's (FLUX.2 [klein] 4B Base 4.0, FLUX.2 [klein] 4B 1.0 = none, FLUX 3 Image Klein Base 3.5, "
                         "FLUX 3 Image Klein 1.0 = none, the only value it takes)")
    ap.add_argument("--relayout_every", type=int, default=None, help="default: the family's (4; FLUX.2 [klein] 4B: 1; FLUX 3 Image Klein: 8)")
    ap.add_argument("--relayout_until", type=lambda v: v if v in ("setting", "none") else int(v), default="setting",
                    help="default: each setting's own (previous: none, --fast: the family's, 20, FLUX.2 [klein] 4B 1, FLUX 3 Image Klein 8); "
                         "'none': every relayout_every steps to the end")
    ap.add_argument("--layout_source", default="pixel", choices=["pixel", "latent"])
    ap.add_argument("--preview_px", type=int, default=None, help="side (px) of the decoded preview the refreshed layouts are computed from (default: the family's, 512 for the 50-step models, 256 for FLUX.2 [klein] 4B and FLUX 3 Image Klein; 1024 = the full image)")
    ap.add_argument("--compile", action="store_true", help="the table with compiled blocks (fast.py)")
    ap.add_argument("--fast", action="store_true", help="the table with every setting of FAST (implies --compile)")
    ap.add_argument("--attention", default=None, choices=["cudnn"], help="cuDNN's attention kernel instead of FlashAttention")
    ap.add_argument("--deterministic", action="store_true",
                    help="kernels picked by heuristics, not by timing (fast.deterministic_kernels), and static graphs: the "
                         "same bits in every process for the 50-step models, a few percent slower")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--variants", action="store_true",
                    help="--fast as released against the matmul library by timing, fused FLUX 3 projections and both: "
                         "alternated budget by budget, with the image check")
    ap.add_argument("--schedules", action="store_true", help="how often the layout is refreshed: time and image change")
    ap.add_argument("--memory", action="store_true", help="weights, loop activations and final-decode activations per budget")
    ap.add_argument("--schedule_budgets", type=float, nargs="+", default=[0.5, 0.3, 0.1])
    ap.add_argument("--schedule_batch", type=int, default=None, help="default: the largest of --batch")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch", type=int, nargs="+", default=[1], help="batch sizes (--compare, --profile: each of them)")
    ap.add_argument("--compare_budgets_large", type=float, nargs="+", default=None,
                    help="--compare at batch sizes above 1: these budgets (default: the same as at batch 1)")
    ap.add_argument("--prompt", default=None, help="one prompt for every image (default: 8 different prompts)")
    ap.add_argument("--text_tokens", default="512", help=TEXT_TOKENS_HELP)
    from budgetlatent.inference import fast as F
    F.add_arguments(ap, skip=("attention",))
    a = ap.parse_args()
    if a.deterministic and a.blas == "timed":
        ap.error("--blas timed picks the library by timing; with --deterministic pick one (cublas or cublaslt)")
    if a.relayout_until == "none":
        a.relayout_until = None
    os.makedirs(a.out, exist_ok=True)
    B = Bench(a)
    meta = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "settings": vars(a)}
    kept = next(iter(B.text_states.values()))[0].shape[1]
    window = ("text in its own window" if B.fam.arch != "flux2" else "text padded to 512 tokens" if kept == 512 else
              f"text cut to {kept} of its 512 tokens (--text_tokens {a.text_tokens})")
    head = f"{B.family}, {meta['gpu']}, bf16, {a.steps} steps, guidance {a.cfg}" + (" (none: one row)" if a.cfg == 1.0 else "") + f", {window}"
    report, results = [], {}

    def save():                                                      # after every section: a crash keeps what was measured
        json.dump({"results": results, **meta}, open(os.path.join(a.out, "bench.json"), "w"), indent=1,
                  default=lambda o: str(o))
        open(os.path.join(a.out, "bench.md"), "w").write("\n".join(report) + "\n")

    def say(m):
        print(m, flush=True)
        report.append(m)
        save()

    def section(name, fn, *args):
        try:
            fn(*args)
        except Exception as e:                                       # noqa: BLE001 -- the next section still runs
            import traceback
            traceback.print_exc()
            say(f"### {name}: FAILED ({type(e).__name__}: {str(e)[:300]})\n")
        torch.cuda.empty_cache()

    def compare(bs, budgets):
        reps = a.repeats if bs == 1 else max(1, a.repeats - 1)
        settings = [("previous", PREVIOUS), ("fast, every 4 steps to the end", dict(FAST, relayout_until=None)), ("fast", FAST)]
        if bs > 1:                                                   # (batch 1 shows the schedule's share too)
            settings = [settings[0], settings[2]]
        runs = {}
        for name, st in settings:
            B.apply(st)
            B.warm(budgets, bs)
            runs[name] = {}
            for b in budgets:
                row, outs = B.run(b, bs, repeats=reps, keep=True)
                runs[name][f"{b:g}"] = (row, outs[0], outs[-1])
                print(f"[compare] batch {bs}, {name}, {b:.0%}: {row['seconds']:.3f} s/image", flush=True)
            results.setdefault(f"compare/b{bs}", {})[name] = {k: v[0] for k, v in runs[name].items()}
            say(table({k: v[0] for k, v in runs[name].items()}, budgets, f"### {name}, batch {bs}; {head}"))
        # which change moves the images: each pair differs in one group of settings
        pairs = [("previous", "previous", "run to run"),
                 ("fast, every 4 steps to the end", "previous", "the speed settings of --fast"),
                 ("fast", "fast, every 4 steps to the end", "refreshes up to step 20 only"),
                 ("fast", "previous", "all of --fast")]
        lines = [f"### same images? batch {bs} (the latent after {a.steps} steps, the final layout, the decoded image)", "",
                 "| budget | change | max abs latent diff | identical final layouts | PSNR of the images |", "|---|---|---|---|---|"]
        check = {}
        for b in budgets:
            k = f"{b:g}"
            decoded = {}
            for x, y, what in pairs:
                if x not in runs or y not in runs:
                    continue
                rx = runs[x][k][2] if x == y else runs[x][k][1]      # run to run: last against first
                ry = runs[y][k][1]
                for nm, r in ((f"{x}/{x == y}", rx), (f"{y}/False", ry)):
                    if nm not in decoded:
                        decoded[nm] = B.decode(r["latent"])
                d = float((rx["latent"] - ry["latent"]).abs().max())
                same = same_layout(ry["layout"], rx["layout"])
                p = psnr(decoded[f"{x}/{x == y}"], decoded[f"{y}/False"])
                lines.append(f"| {b:.0%} | {what} | {d:.2e} | {sum(same)}/{len(same)} | {'identical' if math.isinf(p) else f'{p:.1f} dB'} |")
                check.setdefault(k, {})[what] = {"max_abs": d, "same_layouts": sum(same), "images": len(same),
                                                 "psnr": None if math.isinf(p) else p}
        results.setdefault(f"compare/b{bs}", {})["check"] = check
        say("\n".join(lines) + "\n")

    def variants(bs, budgets):
        """VARIANTS on one GPU in one process: every variant warmed up (and its matmul libraries timed), then, budget by
        budget, the variants in turn, --repeats rounds, so a slower node or a busy neighbour hits them alike.  The
        images of every variant against the first (the latent, the final layout, the decoded image)."""
        rounds = a.repeats
        named = [(n, st) for n, st, applies in VARIANTS if applies(B.fam)]      # (FLUX.2: only the matmul library)
        choices = {}
        for name, st in named:                                       # compile every variant; keep each one's library choices
            B.apply(st)
            B.warm(budgets, bs)
            choices[name] = dict(B.sampling.BLAS_CHOICE)
        secs = {n: {} for n, _ in named}
        first = {}
        for b in budgets:
            k = f"{b:g}"
            for r in range(rounds + 1):
                for name, st in named:
                    B.fast.fuse_projections(B.model, enable=st.get("fuse", False))
                    B.sampling.CACHE_TEXT = st.get("cache_text", True)
                    B.sampling.BLAS_CHOICE.clear()
                    B.sampling.BLAS_CHOICE.update(choices[name])
                    B.current = st
                    row, outs = B.run(b, bs, repeats=1, keep=True, untimed=False)
                    if r == 0:                                       # (the first round is not counted)
                        first.setdefault(k, {})[name] = (row, outs[0])
                    else:
                        secs[name].setdefault(k, []).append(row["seconds"])
            line = ", ".join(f"{n} {statistics.median(secs[n][k]):.3f}" for n, _ in named)
            print(f"[variants] batch {bs}, {b:.1%}: {line}", flush=True)
            results.setdefault(f"variants/b{bs}", {})[k] = {n: {"seconds": statistics.median(secs[n][k]), "all": secs[n][k]}
                                                             for n, _ in named}
            save()
        base = named[0][0]
        lines = [f"### implementation variants, batch {bs}: seconds per image (median of {rounds} rounds, variants alternated "
                 f"within each budget); speed-up = the time of '{base}' / the variant's, at the same budget", "",
                 "| budget | " + " | ".join(n for n, _ in named) + " |", "|---|" + "---|" * len(named)]
        for b in budgets:
            k = f"{b:g}"
            t0 = statistics.median(secs[base][k])
            cells = [f"{statistics.median(secs[n][k]):.3f} ({t0 / statistics.median(secs[n][k]):.2f}x)" for n, _ in named]
            lines.append(f"| {b:.1%} | " + " | ".join(cells) + " |")
        lines += ["", f"speed-up vs 100 % of each variant at the lowest budget ({budgets[-1]:.1%}): " + ", ".join(
            f"{n} {statistics.median(secs[n][f'{budgets[0]:g}']) / statistics.median(secs[n][f'{budgets[-1]:g}']):.2f}x"
            for n, _ in named), ""]
        lines += [f"### same images? batch {bs}: each variant against '{base}'", "",
                  "| budget | variant | max abs latent diff | identical final layouts | PSNR of the images |", "|---|---|---|---|---|"]
        check = {}
        for b in budgets:
            k = f"{b:g}"
            ref = first[k][base][1]
            ref_img = B.decode(ref["latent"])
            for name, _ in named[1:]:
                o = first[k][name][1]
                d = float((o["latent"] - ref["latent"]).abs().max())
                same = same_layout(ref["layout"], o["layout"])
                p = psnr(B.decode(o["latent"]), ref_img)
                lines.append(f"| {b:.1%} | {name} | {d:.2e} | {sum(same)}/{len(same)} | {'identical' if math.isinf(p) else f'{p:.1f} dB'} |")
                check.setdefault(k, {})[name] = {"max_abs": d, "same_layouts": sum(same), "images": len(same),
                                                 "psnr": None if math.isinf(p) else p}
        results.setdefault(f"variants/b{bs}", {})["check"] = check
        results.setdefault(f"variants/b{bs}", {})["blas"] = {n: {str(key): v for key, v in c.items()} for n, c in choices.items()}
        say("\n".join(lines) + "\n")

    def schedules():
        from budgetlatent.inference.sampling import to_uint8
        from budgetlatent.tokenization.layout import dense_layout, overlay
        bs = a.schedule_batch or max(a.batch)
        if bs not in B.text_states:
            raise ValueError(f"--schedule_batch {bs} is not one of --batch {a.batch}")
        B.apply(FAST)
        B.warm([1.0] + a.schedule_budgets, bs)
        dense_res = B.run(1.0, bs, repeats=1, keep=True, untimed=False)[1][0]
        dense_img = B.decode(dense_res["latent"])
        dense_row = ("100 % (dense), for reference", to_uint8(dense_img), overlay(dense_img, dense_layout(bs, B.dev)).cpu())
        lines = [f"### how often the layout is refreshed: batch {bs}, --fast settings (seconds per image; PSNR = mean over the "
                 f"{bs} images; 'vs every 4' = against the schedule used so far, 'vs 100 %' = against the dense image of "
                 f"the same seed)", "",
                 "| budget | schedule | seconds / image | PSNR vs every 4 | PSNR vs 100 % | identical layouts vs every 4 | "
                 "peak activations | peak total |",
                 "|---|---|---|---|---|---|---|---|"]
        out = {}
        for b in a.schedule_budgets:
            k, ref_img, ref_layout, sheet_rows = f"{b:g}", None, None, [dense_row]
            for name, extra in SCHEDULES:
                row, outs = B.run(b, bs, repeats=1, keep=True, untimed=False, **extra)
                img = B.decode(outs[0]["latent"])
                if ref_img is None:
                    ref_img, ref_layout = img, outs[0]["layout"]
                p_ref = statistics.mean(psnr(img[i], ref_img[i]) if not torch.equal(img[i], ref_img[i]) else 99.0 for i in range(bs))
                p_dense = statistics.mean(psnr(img[i], dense_img[i]) for i in range(bs))
                same = same_layout(ref_layout, outs[0]["layout"])
                lines.append(f"| {b:.0%} | {name} | {row['seconds']:.3f} | {p_ref:.1f} dB | {p_dense:.1f} dB | {sum(same)}/{len(same)} | "
                             f"{row['activation_mib']:.0f} MiB | {row['total_mib'] / 1024:.2f} GiB |")
                out.setdefault(k, {})[name] = {"seconds": row["seconds"], "psnr_vs_every4": p_ref, "psnr_vs_dense": p_dense,
                                               "same_layouts": sum(same), "activation_mib": row["activation_mib"],
                                               "total_mib": row["total_mib"]}
                sheet_rows.append((f"{b:.0%}: {name}\n{row['seconds']:.2f} s / image\nPSNR {p_ref:.1f} dB vs every 4, "
                                   f"{p_dense:.1f} dB vs 100 %", to_uint8(img), overlay(img, outs[0]["layout"]).cpu()))
                print(f"[schedules] {b:.0%}, {name}: {row['seconds']:.3f} s/image, PSNR {p_ref:.1f} / {p_dense:.1f} dB", flush=True)
                results["schedules"] = out
                save()
            schedule_sheet(os.path.join(a.out, f"schedules_b{round(100 * b):03d}.jpg"),
                           f"Layout refresh schedules at {b:.0%} ({B.L.snap_tokens(b)} tokens), {meta['gpu']}", sheet_rows)
            del sheet_rows
        say("\n".join(lines) + "\n")

    def profile():
        budgets = a.budgets or [1.0, 0.5, 0.1]
        for name, st in (("previous", PREVIOUS), ("fast", FAST)):
            B.apply(st)
            for bs in a.batch:
                B.warm(budgets, bs)
                rows = {f"{b:g}": B.profile(b, bs) for b in budgets}
                rows = {k: B.profile(float(k), bs) for k in rows}    # the second pass is the one reported
                results[f"profile/{name}/b{bs}"] = rows
                phases = ["preview", "layout", "model", "update", "decode"]
                lines = [f"### where the time goes: {name}, batch {bs} (seconds per image, every phase synchronised; "
                         f"decode = the final image; memory: the whole run, final decode included)", "",
                         "| budget | " + " | ".join(phases) + " | total | peak activations | peak total |",
                         "|---|" + "---|" * (len(phases) + 3)]
                for k, r in rows.items():
                    lines.append(f"| {float(k):.0%} | " + " | ".join(f"{r.get(p, 0.0):.3f}" for p in phases) +
                                 f" | {r['total']:.3f} | {r['activation_mib']:.0f} MiB | {r['total_mib'] / 1024:.2f} GiB |")
                say("\n".join(lines) + "\n")

    def ablation():
        budgets = a.budgets or [1.0, 0.5, 0.1]
        old = dict(FAST, channels_last=True, chunk=4, relayout_until=None)     # the pieces before NCHW, chunks, schedule
        stages = [("previous: eager blocks, channels-last VAE, autocast previews, layout tables per step", PREVIOUS, {}),
                  ("+ compiled blocks", dict(PREVIOUS, compile=True), {}),
                  ("+ compiled blocks, static shapes", dict(PREVIOUS, compile=True, dynamic=False), {}),
                  ("+ compiled blocks, layouts prepared once per refresh", dict(PREVIOUS, compile=True, reuse_layout=True), {}),
                  ("  + compiled layout builder", dict(old, preview="autocast"), {}),
                  ("    + NCHW VAE", dict(old, preview="autocast", channels_last=False), {}),
                  ("    + bf16 preview decoder", dict(old, preview="bf16"), {}),
                  ("    + compiled bf16 preview decoder", old, {}),
                  ("    + compiled bf16 preview decoder, NCHW, one image per decode", dict(old, channels_last=False, chunk=None), {}),
                  ("      + refreshes up to step 20 (= --fast)", FAST, {}),
                  ("--fast + cuDNN attention", dict(FAST, attention="cudnn"), {}),
                  ("--fast, layout from pixels at steps 0, 5, 10 only (changes the images)", FAST,
                   dict(relayout_every=5, relayout_until=10)),
                  ("--fast, layout from the latent up to step 20 (changes the images)", FAST, dict(layout_source="latent")),
                  ("--fast, layout from the latent at steps 0, 5, 10 (changes the images)", FAST,
                   dict(layout_source="latent", relayout_every=5, relayout_until=10))]
        bs = a.batch[0]
        rows = {}
        for name, st, extra in stages:
            try:
                B.apply(st)
                bb = [b for b in budgets if not extra or b < 1.0]
                B.warm(bb, bs)
                rows[name] = {f"{b:g}": B.run(b, bs, repeats=2, **extra) for b in bb}
            except Exception as e:                                   # noqa: BLE001 -- a setting that does not run here is a result too
                rows[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"[bench] {name}: {rows[name]}", flush=True)
            results["ablation"] = rows
            save()
        b0, b1 = f"{budgets[0]:g}", f"{budgets[-1]:g}"
        fast_dense = rows.get("      + refreshes up to step 20 (= --fast)", {}).get(b0, {}).get("seconds")
        for what, key, fmt in (("seconds per image, median of 2 after warm-up", "seconds", lambda v: f"{v:.3f}"),
                               ("peak activations: allocator peak minus what was resident before the run", "activation_mib",
                                lambda v: f"{v:.0f} MiB"),
                               ("peak total GPU memory (weights included)", "total_mib", lambda v: f"{v / 1024:.2f} GiB")):
            speed = key == "seconds"
            lines = [f"### what each setting buys, batch {bs}: {what}", "",
                     "| setting | " + " | ".join(f"{b:.0%}" for b in budgets) +
                     (f" | {budgets[0]:.0%} / {budgets[-1]:.0%} |" if speed else " |"),
                     "|---|" + "---|" * (len(budgets) + (1 if speed else 0))]
            for name, r in rows.items():
                if "error" in r:
                    lines.append(f"| {name} | " + " | ".join("failed" for _ in budgets) + (f" | {r['error'][:80]} |" if speed else " |"))
                    continue
                cells = [fmt(r[f"{b:g}"][key]) if f"{b:g}" in r else "-" for b in budgets]
                if speed:
                    first = r[b0]["seconds"] if b0 in r else fast_dense   # schedule rows: against --fast's dense time
                    ratio = f"{first / r[b1]['seconds']:.2f}x" if first and b1 in r else "-"
                    lines.append(f"| {name} | " + " | ".join(cells) + f" | {ratio} |")
                else:
                    lines.append(f"| {name} | " + " | ".join(cells) + " |")
            say("\n".join(lines) + "\n")

    def memory():
        budgets = a.budgets or [1.0, 0.5, 0.1]
        fp32 = dict(FAST, final_decode="fp32")
        settings = [("previous", PREVIOUS), ("fast with the fp32 final decode", fp32), ("fast (bf16 final decode)", FAST)]
        say(f"### where the memory goes ({meta['gpu']}): the text encoder's weights are {B.text_mib / 1024:.2f} GiB; "
            f"they are freed (moved to the CPU in galleries) once the prompts are encoded, and are not counted below.\n")
        out = {}
        for bs in a.batch:
            lines = [f"### memory at batch {bs}: resident = weights (transformer + VAE); loop = activations of the {a.steps} steps, "
                     f"layout previews included; decode = activations of the final decode; peak = resident + the larger of the two", "",
                     "| setting | budget | resident | loop | final decode | peak total | bf16 decode vs fp32 |", "|---|---|---|---|---|---|---|"]
            fp32_images = {}
            for name, st in settings:
                B.apply(st)
                B.warm(budgets, bs)
                for b in budgets:
                    row, res = B.memory(b, bs)
                    k = f"{b:g}"
                    note = ""
                    if st is fp32:                           # (kept off the GPU: they would count as resident)
                        fp32_images[k] = res["latent"].cpu(), res["image"].cpu()
                    elif st is FAST and k in fp32_images:
                        lat, img32 = fp32_images[k]
                        img16 = B.decode(lat.to(B.dev), bf16=True).cpu()
                        p = psnr(img16, img32)
                        row["psnr_bf16_vs_fp32"] = p
                        note = f"{p:.1f} dB (same latent)"
                    del res
                    out.setdefault(f"b{bs}", {}).setdefault(name, {})[k] = row
                    print(f"[memory] batch {bs}, {name}, {b:.0%}: {row}", flush=True)
                    lines.append(f"| {name} | {b:.0%} | {row['resident_mib'] / 1024:.2f} GiB | {row['loop_mib'] / 1024:.2f} GiB | "
                                 f"{row['decode_mib'] / 1024:.2f} GiB | {row['total_mib'] / 1024:.2f} GiB | {note} |")
                results["memory"] = {"text_encoder_mib": B.text_mib, **out}
                save()
            say("\n".join(lines) + "\n")

    if a.memory:
        section("memory", memory)
    if a.compare and 1 in a.batch:
        section("compare, batch 1", compare, 1, a.budgets or ALL_BUDGETS)
    if a.schedules:
        section("schedules", schedules)
    if a.profile:
        section("profile", profile)
    if a.ablation:
        section("ablation", ablation)
    if a.compare:
        for bs in [x for x in a.batch if x != 1]:
            section(f"compare, batch {bs}", compare, bs, a.compare_budgets_large or a.budgets or ALL_BUDGETS)
    if a.variants:
        for bs in a.batch:
            section(f"variants, batch {bs}", variants, bs, a.budgets or ALL_BUDGETS)

    if not (a.profile or a.ablation or a.compare or a.schedules or a.memory or a.variants):
        st = FAST if a.fast else dict(PREVIOUS, reuse_layout=True, compile=a.compile, attention=a.attention, chunk=None,
                                      relayout_until="family")
        st = dict(st, **F.overrides(a))                              # (--no_fuse, --blas, --no_text_cache, ...)
        B.apply(st)
        budgets = a.budgets or ALL_BUDGETS
        for bs in a.batch:
            B.warm(budgets, bs)
            rows = {}
            for b in budgets:
                rows[f"{b:g}"] = B.run(b, bs)
                print(f"[bench] batch {bs}, {b:6.0%}: {rows[f'{b:g}']['seconds']:.3f} s/image", flush=True)
            desc = B.fast.describe(compile=st["compile"], attention=st["attention"], dynamic=B.dynamic(st),
                                   compile_layout=st["compile_layout"], preview=st["preview"], channels_last=st["channels_last"],
                                   deterministic=a.deterministic, fuse=st.get("fuse", False) and B.fam.arch == "flux3",
                                   blas=st.get("blas"))
            results[f"table/b{bs}"] = rows
            say(table(rows, budgets, f"### batch {bs}; {head}, relayout every {a.relayout_every} ({a.layout_source}); {desc}"))
    save()
    print(f"[bench] -> {a.out}/bench.md", flush=True)


if __name__ == "__main__":
    main()
