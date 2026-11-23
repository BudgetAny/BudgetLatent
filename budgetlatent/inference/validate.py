"""The fixed validation suite: 32 prompts at the 10 budgets with the family's sampler (FLUX.2 [klein] 4B Base: 50 Euler
steps, guidance 4.0, the layout re-derived at steps 0, 4, ..., 20; FLUX.2 [klein] 4B: 4 steps, no guidance, layouts
at steps 0 and 1, the second from a 256 px preview; FLUX 3 Image Klein Base: as FLUX.2 [klein] 4B Base with guidance
3.5; FLUX 3 Image Klein: 16 flow-map steps, no guidance, layouts at steps 0 and 8, the second from a 256 px preview),
seed 42.  Every image of a batch gets exactly N = N_MIN + 3k tokens (64 + 3k at 1024 x 1024), so nothing is padded.

  render              prompts x budgets -> one strip per prompt (images over their token layouts) + speed / memory
  write_index         index.html for a directory of strips
  validate_weights    what the trainers call at a snapshot: every rank renders its share with the EMA weights

scripts/inference/gallery.py is the command line around `render`.
"""
import html
import json
import os
import time

import torch

from ..tokenization import layout as L
from .sampling import FAMILY, encode_prompts, initial_noise, sample, sampler_defaults, to_uint8

BUDGETS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
PROMPTS = [
    "a red fox standing in a snowy forest at golden hour, detailed fur",
    "a corgi wearing a spacesuit on the moon, the earth in the black sky",
    "a watercolor painting of a lighthouse at dusk, soft washes of colour",
    "a portrait photograph of an elderly fisherman with a weathered face, shallow depth of field",
    "a bowl of ramen with a soft-boiled egg, steam rising, top-down food photography",
    "a hummingbird hovering next to a bright red hibiscus flower, macro shot",
    "a futuristic city skyline at night with neon signs reflected in wet streets",
    "a cozy reading nook by a rainy window, warm lamp light, stacked books",
    "an astronaut riding a horse on a beach at sunset, cinematic lighting",
    "a minimalist poster of a single green cactus on a plain pink background",
    "a medieval castle on a cliff above a stormy sea, dramatic clouds",
    "a close-up of a mechanical watch movement, gears and jewels, studio lighting",
    "a child's crayon drawing of a family of ducks crossing a road",
    "a snow leopard resting on a rocky ledge, mountains in the background",
    "a street market in Marrakech with colorful spices and woven baskets",
    "an oil painting of sunflowers in a blue vase, thick impasto brush strokes",
    "a glass of iced coffee on a wooden table, condensation droplets, morning light",
    "a wooden sailboat on a calm turquoise lagoon, aerial view",
    "a robot barista making latte art in a small cafe",
    "the word 'BUDGET' written in white chalk on a blackboard",
    "a field of lavender under a clear blue sky with a single tree on the horizon",
    "a steampunk owl made of brass and copper, intricate details",
    "a rainy Tokyo alley with glowing lanterns and a lone person holding an umbrella",
    "a plate of colorful macarons arranged in a spiral on white marble",
    "an isometric illustration of a tiny island with a house, a pond and a windmill",
    "a black and white photograph of an old bicycle leaning against a brick wall",
    "a dragon made of clouds flying over green mountains at sunrise",
    "a golden retriever puppy running through shallow water, splashes frozen in motion",
    "a stained glass window depicting a peacock, vivid colours, light shining through",
    "a desert road stretching to the horizon under a starry night sky with the milky way",
    "a pile of autumn leaves with one bright red maple leaf on top, macro",
    "a cross-section of a geode with purple amethyst crystals, studio photo",
]


@torch.no_grad()
def render(model, vae, text, prompts, out, budgets=BUDGETS, indices=None, steps=None, cfg=None, relayout_every=None,
           relayout_until=FAMILY, layout_source="pixel", seed=42, batch=8, thumb=256, amp=True, say=None, warm=True,
           offload_text=False, final_decode="fp32", shift=None, preview_px=None, blas=None, text_tokens=None, refs=None):
    """Writes <out>/<prompt index>.jpg for the prompts in `indices` (default: all): top row the samples, bottom row
    the token layouts (green 16 px, amber 32 px, blue 64 px, purple 128 px), one column per budget.  Image k always
    starts from the noise of seed + k, whatever the batching or the sharding.  steps / cfg / relayout_every / shift
    None, relayout_until FAMILY: the family's (sampling.sampler_defaults).
    warm: on a GPU, every budget and batch size first runs briefly untimed (fast.warmup), so that the times below
    are those of the budget, not of compiling or autotuning its first call; blas="timed": the warm-up also picks each
    budget's matmul library by timing (fast.tune_blas).
    Every prompt is encoded first; offload_text then moves the text encoder to the CPU (it is 5.6 GiB of the GPU's
    memory otherwise, at every budget).
    refs (models/refs.References): reference images, the same for every prompt (editing; dense, never budgeted).
    Returns {budget: tokens, seconds_per_image, peak_gib (weights included), activation_gib (the peak minus what was
    resident before the call), loop_gib / decode_gib (the activations of the denoising loop and of the final
    decode, each its own peak minus what was resident), resident_gib, mean_tokens, images}."""
    from PIL import Image
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    todo = list(range(len(prompts))) if indices is None else list(indices)
    os.makedirs(out, exist_ok=True)
    perf = {f"{b:g}": {"tokens": L.snap_tokens(b), "seconds": 0.0, "images": 0, "peak_gib": 0.0, "activation_gib": 0.0,
                       "loop_gib": 0.0, "decode_gib": 0.0, "resident_gib": 0.0, "mean_tokens": 0.0} for b in budgets}
    batches = [todo[lo:lo + batch] for lo in range(0, len(todo), batch)]
    steps, cfg, relayout_every, relayout_until = sampler_defaults(model, steps, cfg, relayout_every, relayout_until)
    states = [encode_prompts(text, model, [prompts[k] for k in idx], amp, uncond=cfg != 1.0, text_tokens=text_tokens)
              for idx in batches]
    if offload_text and cuda:
        text.to("cpu")
        torch.cuda.empty_cache()
    kw = dict(steps=steps, cfg=cfg, relayout_every=relayout_every, shift=shift, relayout_until=relayout_until,
              layout_source=layout_source, amp=amp, final_decode=final_decode, preview_px=preview_px)
    if refs is not None:
        kw["refs"] = refs
        blas = None                         # (with references the sampler keeps the global matmul library: no timed table)
    if warm and cuda and todo:
        from .fast import warmup
        for size in sorted({len(idx) for idx in batches}):
            j = next(j for j, idx in enumerate(batches) if len(idx) == size)
            noise = torch.cat([initial_noise(1, channels=model.in_channels, seed=seed, device=device, offset=k) for k in batches[j]])
            cond, uncond, lens = states[j]
            warmup(model, vae, cond, uncond, budgets, say=say, noise=noise, blas=blas, **lens,
                   **{k: v for k, v in kw.items() if k != "steps"})
    for idx, (cond, uncond, lens) in zip(batches, states):
        noise = torch.cat([initial_noise(1, channels=model.in_channels, seed=seed, device=device, offset=k) for k in idx])
        tiles = {k: ([], []) for k in idx}
        for b in budgets:
            mem = {}
            if cuda:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                resident = torch.cuda.memory_allocated(device)
            tic = time.time()
            res = sample(model, vae, cond, uncond, budget=b, noise=noise, memory=mem, **lens, **kw)
            if cuda:
                torch.cuda.synchronize(device)
            p = perf[f"{b:g}"]
            p["seconds"] += time.time() - tic
            p["images"] += len(idx)
            p["mean_tokens"] += res["tokens"] * len(idx)
            if cuda:
                peak = max(mem.values())
                p["peak_gib"] = max(p["peak_gib"], peak / 2 ** 30)
                p["activation_gib"] = max(p["activation_gib"], (peak - resident) / 2 ** 30)
                p["loop_gib"] = max(p["loop_gib"], (mem["loop"] - resident) / 2 ** 30)
                p["decode_gib"] = max(p["decode_gib"], (mem["decode"] - resident) / 2 ** 30)
                p["resident_gib"] = max(p["resident_gib"], resident / 2 ** 30)
            img = to_uint8(res["image"])
            over = L.overlay(res["image"], res["layout"] or L.dense_layout(len(idx), device)).cpu()
            for j, k in enumerate(idx):
                tiles[k][0].append(img[j])
                tiles[k][1].append(over[j])
        for k in idx:
            strip = torch.cat([torch.cat(tiles[k][0], dim=1), torch.cat(tiles[k][1], dim=1)], dim=0).numpy()
            im = Image.fromarray(strip)
            if thumb and thumb != im.height // 2:
                im = im.resize((thumb * len(budgets), thumb * 2), Image.LANCZOS)
            im.save(os.path.join(out, f"{k:03d}.jpg"), quality=92)
        if say:
            say(f"[gallery] prompts {idx[0]}..{idx[-1]} done")
    for p in perf.values():
        n = max(1, p["images"])
        p["seconds_per_image"], p["mean_tokens"] = p.pop("seconds") / n, p["mean_tokens"] / n
    return perf


def write_index(out, prompts, budgets, title, note="", base="FLUX.2 [klein] 4B Base"):
    head = "".join(f"<th>{round(100 * b)} %<br><small>{L.snap_tokens(b)} tokens</small></th>" for b in budgets)
    rows = "".join(f"<h3>{k:03d}. {html.escape(p)}</h3><img src='{k:03d}.jpg' loading='lazy'>" for k, p in enumerate(prompts))
    open(os.path.join(out, "index.html"), "w").write(
        f"<!doctype html><meta charset='utf-8'><title>BudgetLatent gallery: {html.escape(base)}</title>"
        "<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}img{max-width:100%}table{border-collapse:collapse}"
        "th{padding:4px 10px;font-weight:normal}</style>"
        f"<h2>BudgetLatent on {html.escape(base)} &mdash; {html.escape(title)}</h2>"
        f"<p>{html.escape(note)} Top row: samples, bottom row: token layout "
        "(green 16 px, amber 32 px, blue 64 px, purple 128 px). Columns, left to right:</p>"
        f"<table><tr>{head}</tr></table>{rows}")


def validate_weights(path, root, vae, text, out, rank=0, world=1, device="cuda", inference=None, limit=None, budgets=None, say=None,
                     family="flux2"):
    """Render the validation suite with the weights in `path` (an ema-*.safetensors of a trainer), this rank taking
    every world-th prompt.  The model is loaded in bf16 next to whatever the caller holds on the GPU and freed
    again; rank 0 writes the index and the timing table.  Nothing here synchronises the ranks.  inference: the
    config's `inference` section (num_steps, guidance_scale, relayout_every, relayout_until, time_shift; what it
    leaves out is the family's).  Returns this rank's perf dict."""
    from ..models.families import family_of
    from ..models.weights import load_model
    inference = dict(inference or {})
    fam = family_of(family)
    guidance = float(inference.get("guidance_scale", fam.guidance))
    num_steps = int(inference.get("num_steps", fam.steps))
    prompts = PROMPTS[:limit] if limit else PROMPTS
    budgets = tuple(budgets) if budgets else BUDGETS
    cuda = torch.device(device).type == "cuda"
    model = load_model(path, root, device, dtype=torch.bfloat16 if cuda else None, family=family).eval().requires_grad_(False)
    try:
        perf = render(model, vae, text, prompts, out, budgets, indices=[k for k in range(len(prompts)) if k % world == rank],
                      steps=num_steps, cfg=guidance, shift=inference.get("time_shift"),
                      relayout_every=int(inference.get("relayout_every", fam.relayout_every)),
                      relayout_until=inference.get("relayout_until", fam.relayout_until), amp=cuda, say=say)
    finally:
        del model
        if cuda:
            torch.cuda.empty_cache()
    json.dump({"prompts": prompts, "budgets": list(budgets), "perf": perf, "weights": path},
              open(os.path.join(out, f"manifest-{rank}of{world}.json"), "w"), indent=1)
    if rank == 0:
        write_index(out, prompts, budgets, os.path.basename(path), f"{num_steps} steps, guidance {guidance}, seed 42.",
                    base=fam.label)
    return perf
