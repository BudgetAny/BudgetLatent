"""Budget sweep gallery: every prompt at every budget, with the token layout drawn over each image.

    BP_GRID=64 python scripts/inference/gallery.py --ckpt BudgetAny/BudgetLatent-FLUX.2-klein-base-4B --out gallery_flux2 --fast
    BP_GRID=64 python scripts/inference/gallery.py --ckpt BudgetAny/BudgetLatent-FLUX.2-klein-4B --out gallery_flux2_distilled --fast
    python scripts/inference/gallery.py --ckpt runs/flux3_512/ema-final.safetensors --out gallery_flux3
    BP_GRID=64 python scripts/inference/gallery.py --family flux3 --out gallery_flux3_released --budgets 1.0       # the released model
    BP_GRID=64 python scripts/inference/gallery.py --ckpt BudgetAny/BudgetLatent-FLUX-3-Image-Klein --out gallery_flux3_distilled --fast
    BP_GRID=64 python scripts/inference/gallery.py --ckpt BudgetAny/BudgetLatent-FLUX-3-Image-Klein-Base --out gallery_edits --fast \
        --ref dog.png --prompts edits.txt                         # every edit instruction of edits.txt on one reference image

Writes <out>/<prompt index>.jpg (top row: images, bottom row: layouts; one column per budget), <out>/index.html and
<out>/manifest-<i>of<n>.json (tokens, seconds and peak memory per budget, of shard i of n).  Steps, guidance and
layout refreshes are the family's unless given (FLUX.2 [klein] 4B: 4 steps, no guidance, layouts at steps 0 and 1, the second from a 256 px
preview; FLUX 3 Image Klein: 16 flow-map steps, no guidance, layouts at steps 0 and 8, the second from a 256 px
preview).  --shard i/n takes every n-th prompt, for several GPUs.  Layout colours: green 16 px, amber 32 px, blue 64 px, purple 128 px tokens.
--ref (repeatable): reference images for every prompt, as in sample.py (the first is 1024 x 1024 at BP_GRID=64; never
budgeted).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))    # the repository root

import torch

from budgetlatent.inference.sampling import TEXT_TOKENS_HELP
from budgetlatent.models.families import FAMILIES


def until(v):
    """--relayout_until: an int, 'none' (to the end) or 'family' (the family's default)."""
    v = str(v).lower()
    return v if v == "family" else (None if v == "none" else int(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="a local .safetensors or a Hugging Face repo id (see sample.py)")
    ap.add_argument("--family", default=None, choices=sorted(FAMILIES), help="default: the checkpoint's (else flux2)")
    ap.add_argument("--weights", default=None, help="snapshot of the released base model (default: the HF cache)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=None, help="text file with one prompt per line (default: 32 built-in prompts)")
    ap.add_argument("--ref", action="append", default=[], metavar="IMAGE",
                    help="a reference image for every prompt (editing; repeat for multi-reference, up to 10): the first "
                         "sets the output size and must be 1024 x 1024 at BP_GRID=64; references are never budgeted")
    ap.add_argument("--budgets", type=float, nargs="+", default=None, help="default: the validation suite's, 100 %% down to 10 %%")
    ap.add_argument("--steps", type=int, default=None, help="default: the family's (50; FLUX.2 [klein] 4B: 4; FLUX 3 Image Klein: 16)")
    ap.add_argument("--cfg", type=float, default=None,
                    help="default: the family's (FLUX.2 [klein] 4B Base 4.0, FLUX.2 [klein] 4B 1.0 = none, FLUX 3 Image Klein Base 3.5, "
                         "FLUX 3 Image Klein 1.0 = none, the only value it takes)")
    ap.add_argument("--relayout_every", type=int, default=None, help="default: the family's (4; FLUX.2 [klein] 4B: 1; FLUX 3 Image Klein: 8)")
    ap.add_argument("--relayout_until", type=until, default="family",
                    help="last step at which the layout may change; 'none': every relayout_every steps to the end "
                         "(default: the family's, 20, FLUX.2 [klein] 4B 1, FLUX 3 Image Klein 8)")
    ap.add_argument("--layout_source", default="pixel", choices=["pixel", "latent"])
    ap.add_argument("--preview_px", type=int, default=None, help="side (px) of the decoded preview the refreshed layouts are computed from (default: the family's, 512 for the 50-step models, 256 for FLUX.2 [klein] 4B and FLUX 3 Image Klein; 1024 = the full image)")
    ap.add_argument("--template", default=None, choices=["bfl_nothink", "no_think", "flux3"], help="default: the checkpoint's own")
    ap.add_argument("--compile", action="store_true", help="torch.compile the blocks (fast.py)")
    ap.add_argument("--fast", action="store_true", help="compiled blocks, compiled layout builder and compiled bf16 "
                                                        "layout previews, NCHW VAE, fused FLUX 3 projections, each budget's "
                                                        "matmul library by timing at the warm-up (fast.py; a layout can "
                                                        "differ where two cell entropies were within rounding)")
    ap.add_argument("--deterministic", action="store_true",
                    help="kernels picked by heuristics, not by timing, and static graphs: the same image for a seed in "
                         "every run of the 50-step models, also with --fast (a few percent slower, a compilation per "
                         "token count; fast.py)")
    ap.add_argument("--final_decode", default=None, choices=["fp32", "bf16"],
                    help="default: bf16 with --fast (the bf16 decoder copy: a third of the fp32 decoder's memory, "
                         "60 dB from its images), else fp32")
    ap.add_argument("--keep_text", action="store_true", help="keep the text encoder on the GPU after encoding")
    ap.add_argument("--no_warmup", action="store_true", help="time each budget's first call too (compilation included)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--thumb", type=int, default=256, help="side of one tile in the strips")
    ap.add_argument("--text_tokens", default="512", help=TEXT_TOKENS_HELP)
    from budgetlatent.inference import fast as F
    F.add_arguments(ap)
    a = ap.parse_args()
    a.final_decode = a.final_decode or ("bf16" if a.fast else "fp32")
    from budgetlatent.encoders.text import text_encoder
    from budgetlatent.encoders.vae import Flux2VAE
    from budgetlatent.inference import validate as V
    from budgetlatent.inference.fast import apply_settings, settings
    from budgetlatent.models.families import family_of
    from budgetlatent.models.refs import encode_images, prepare_images
    from budgetlatent.models.weights import checkpoint_family, checkpoint_template, klein_dir, load_model, resolve_checkpoint
    from budgetlatent.tokenization.layout import CELL_PX, GRID
    try:                                                            # the references first: a wrong size fails before any loading
        from PIL import Image
        images = prepare_images([Image.open(f) for f in a.ref], CELL_PX * GRID) if a.ref else []
    except (OSError, ValueError) as e:
        ap.error(f"--ref: {e}")
    prompts = [p.strip() for p in open(a.prompts) if p.strip()] if a.prompts else V.PROMPTS
    prompts = prompts[: a.limit] if a.limit else prompts
    budgets = a.budgets or [b for b in V.BUDGETS if b >= 0.1]
    si, sn = (int(v) for v in a.shard.split("/"))
    dev = torch.device("cuda")
    ckpt = resolve_checkpoint(a.ckpt)
    family = a.family or checkpoint_family(ckpt)
    fam = family_of(family)
    if fam.flow_map and a.cfg not in (None, 1.0):
        ap.error(f"--cfg {a.cfg}: {fam.label} is a flow map and samples without guidance (cfg 1.0)")
    a.steps, a.cfg = a.steps or fam.steps, fam.guidance if a.cfg is None else a.cfg
    a.relayout_every = a.relayout_every or fam.relayout_every
    a.relayout_until = fam.relayout_until if a.relayout_until == "family" else a.relayout_until
    root = klein_dir(a.weights or fam.repo)
    text = text_encoder(family, root, dev, template=a.template or checkpoint_template(ckpt, family=family))
    trunk_cfg = {"image_cond": True} if images and fam.arch == "flux3" else None     # FLUX 3's reference branch
    model = load_model(ckpt, root, dev, dtype=torch.bfloat16, family=family, trunk_cfg=trunk_cfg).eval().requires_grad_(False)
    vae = Flux2VAE(root, dev)
    refs = encode_images(images, vae, family) if images else None
    blas = apply_settings(model, vae, settings(a))
    perf = V.render(model, vae, text, prompts, a.out, budgets, indices=[k for k in range(len(prompts)) if k % sn == si],
                    steps=a.steps, cfg=a.cfg, relayout_every=a.relayout_every, relayout_until=a.relayout_until,
                    layout_source=a.layout_source, preview_px=a.preview_px, seed=a.seed, batch=a.batch, thumb=a.thumb,
                    say=lambda m: print(m, flush=True),
                    warm=not a.no_warmup, offload_text=not a.keep_text, final_decode=a.final_decode,
                    blas=blas, text_tokens=a.text_tokens, refs=refs)
    json.dump({"prompts": prompts, "budgets": budgets, "perf": perf, "settings": vars(a), "gpu": torch.cuda.get_device_name(0)},
              open(os.path.join(a.out, f"manifest-{si}of{sn}.json"), "w"), indent=1)
    sched = f"layout refreshed every {a.relayout_every} steps" + (f" up to step {a.relayout_until}" if a.relayout_until is not None else "")
    if a.ref:
        sched += f", reference images {', '.join(os.path.basename(f) for f in a.ref)} (not budgeted)"
    V.write_index(a.out, prompts, budgets, os.path.basename(a.ckpt) if a.ckpt else "released weights",
                  f"{a.steps} steps, guidance {a.cfg}, seed {a.seed}, {sched}.", base=family_of(family).label)
    print(f"[gallery] {family}: {len(prompts[si::sn])} prompts x {len(budgets)} budgets -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
