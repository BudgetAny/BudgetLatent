"""Sample BudgetLatent (FLUX.2 [klein] 4B Base, FLUX.2 [klein] 4B, FLUX 3 Image Klein Base or FLUX 3 Image Klein) at a
token budget.

    BP_GRID=64 python scripts/inference/sample.py --ckpt BudgetAny/BudgetLatent-FLUX.2-klein-base-4B --prompts "a corgi in a spacesuit" --budget 1.0 0.5 0.1
    BP_GRID=64 python scripts/inference/sample.py --ckpt BudgetAny/BudgetLatent-FLUX.2-klein-4B --prompts "a corgi in a spacesuit" --budget 1.0 0.5 0.1
    python scripts/inference/sample.py --ckpt runs/flux3_512/ema-final.safetensors --prompts "..." --budget 0.5 --fast
    BP_GRID=64 python scripts/inference/sample.py --family flux3 --prompts "a red fox in a snowy forest"       # the released model
    BP_GRID=64 python scripts/inference/sample.py --ckpt BudgetAny/BudgetLatent-FLUX-3-Image-Klein --prompts "a corgi in a spacesuit" --budget 1.0 0.5 0.1
    BP_GRID=64 python scripts/inference/sample.py --ckpt BudgetAny/BudgetLatent-FLUX-3-Image-Klein-Base --prompts "put a red bandana around the dog's neck" \
        --ref dog.png --budget 1.0 0.25 --fast                                                     # an edit (dog.png 1024 x 1024)
    BP_GRID=64 python scripts/inference/sample.py --ckpt BudgetAny/BudgetLatent-FLUX.2-klein-base-4B --prompts "print the dog from the second image onto the mug" \
        --ref mug.png --ref dog.png --budget 1.0 0.25 --fast                                        # two references

--ckpt is a local file or a Hugging Face model repository id (its budgetlatent.safetensors is downloaded); the
family is the one the checkpoint records (else its tensor names tell; FLUX.2 [klein] 4B Base and FLUX.2 [klein] 4B
are told apart by the record only; FLUX 3 Image Klein Base and FLUX 3 Image Klein by its ref_time_in tensors too),
--family overrides it.  The family also sets the sampler: FLUX.2 [klein] 4B Base, 50 Euler steps with classifier-free guidance 4.0, one fused [cond; uncond] pass per step, the layout re-derived
at steps 4, 8, ..., 20; FLUX.2 [klein] 4B (step-distilled), 4 Euler steps without guidance (one row per image, no
unconditional text), the layout re-derived at step 1 only, from a 256 px preview; FLUX 3 Image Klein Base, as FLUX.2 [klein] 4B Base with
guidance 3.5; FLUX 3 Image Klein (step-distilled, a flow map), 16 steps on Klein Base's schedule, each told the time it
jumps to, without guidance (one row per image, no unconditional text; --cfg must stay 1), the layout re-derived once,
at step 8, from a 256 px preview.  With a budget below 1 the first layout comes from the noise itself (no dense
warm-up step), the later ones from the entropy of the decoded running estimate of the clean image.  Every image of a batch gets exactly the
same number of tokens, N = N_MIN + 3k (64 + 3k at 1024 x 1024), so there is no padding.  The text encoder is freed once the prompts are encoded,
and the VAE decodes one 1024 x 1024 image at a time.

--ref (repeatable, up to 10): reference images for editing and multi-reference generation, the same for every prompt,
taken as the released pipelines take them (models/refs.py).  The first reference sets the output size, so it must be
16 GRID x 16 GRID px (1024 x 1024 at BP_GRID=64); the others are fitted under 1 MP and cropped to multiples of 16 px.
The references are never budgeted: they keep all their tokens at every step, and only the generated image runs at
the budget.  FLUX 3 builds its reference branch (image_cond) only when --ref is given.
"""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))    # the repository root

import torch

from budgetlatent.encoders.text import TEMPLATES
from budgetlatent.inference.sampling import TEXT_TOKENS_HELP
from budgetlatent.models.families import FAMILIES


def until(v):
    """--relayout_until: an int, 'none' (to the end) or 'family' (the family's default)."""
    v = str(v).lower()
    return v if v == "family" else (None if v == "none" else int(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="BudgetLatent weights: a local .safetensors (the trainers' ema-*.safetensors, "
                                                 "the dit/ export) or a Hugging Face repo id; omit for the released model "
                                                 "with the added modules at zero")
    ap.add_argument("--family", default=None, choices=sorted(FAMILIES),
                    help="default: what the checkpoint records (else its tensor names), flux2 without --ckpt")
    ap.add_argument("--weights", default=None, help="snapshot of the released base model (default: the HF cache)")
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--ref", action="append", default=[], metavar="IMAGE",
                    help="a reference image (editing; repeat for multi-reference, up to 10): the first sets the output "
                         "size and must be 1024 x 1024 at BP_GRID=64; references are never budgeted")
    ap.add_argument("--budget", type=float, nargs="+", default=[1.0], help="fraction of the dense tokens (layout.N_DENSE)")
    ap.add_argument("--steps", type=int, default=None, help="default: the family's (50; FLUX.2 [klein] 4B: 4; FLUX 3 Image Klein: 16)")
    ap.add_argument("--cfg", type=float, default=None,
                    help="default: the family's (FLUX.2 [klein] 4B Base 4.0, FLUX.2 [klein] 4B 1.0 = none, FLUX 3 Image Klein Base 3.5, "
                         "FLUX 3 Image Klein 1.0 = none, the only value it takes)")
    ap.add_argument("--relayout_every", type=int, default=None, help="default: the family's (4; FLUX.2 [klein] 4B: 1; FLUX 3 Image Klein: 8)")
    ap.add_argument("--relayout_until", type=until, default="family",
                    help="last step at which the layout may change ('none': every relayout_every steps to the end; "
                         "default: the family's, 20, FLUX.2 [klein] 4B 1, FLUX 3 Image Klein 8)")
    ap.add_argument("--layout_source", default="pixel", choices=["pixel", "latent"])
    ap.add_argument("--preview_px", type=int, default=None, help="side (px) of the decoded preview the refreshed layouts are computed from (default: the family's, 512 for the 50-step models, 256 for FLUX.2 [klein] 4B and FLUX 3 Image Klein; 1024 = the full image)")
    ap.add_argument("--template", default=None, choices=list(TEMPLATES),
                    help="chat template of the prompt (default: the one the checkpoint was trained with, else the family's)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp32", action="store_true", help="fp32 weights and compute (exact, slow, 2x the memory)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the blocks: worth it for many images, not for one")
    ap.add_argument("--fast", action="store_true", help="--compile, plus a compiled layout builder, compiled bf16 layout "
                                                        "previews, an NCHW VAE, fused FLUX 3 projections and each budget's "
                                                        "matmul library by timing (fast.py): for many images; compiling takes minutes")
    ap.add_argument("--deterministic", action="store_true",
                    help="kernels picked by heuristics, not by timing, and static graphs: the same image for a seed in "
                         "every run of the 50-step models, also with --fast (a few percent slower, a compilation per "
                         "token count; fast.py)")
    ap.add_argument("--final_decode", default=None, choices=["fp32", "bf16"],
                    help="default: bf16 with --fast (the bf16 decoder copy: a third of the fp32 decoder's memory, "
                         "60 dB from its images), else fp32")
    ap.add_argument("--text_tokens", default="512", help=TEXT_TOKENS_HELP)
    ap.add_argument("--overlay", action="store_true", help="also write the token layout drawn over each image")
    ap.add_argument("--out", default="samples")
    from budgetlatent.inference import fast as F
    F.add_arguments(ap)
    a = ap.parse_args()
    a.final_decode = a.final_decode or ("bf16" if a.fast else "fp32")
    from PIL import Image

    from budgetlatent.encoders.text import text_encoder
    from budgetlatent.encoders.vae import Flux2VAE
    from budgetlatent.inference.fast import apply_settings, settings, warmup
    from budgetlatent.inference.sampling import encode_prompts, sample, to_uint8
    from budgetlatent.models.families import family_of
    from budgetlatent.models.refs import encode_images, prepare_images
    from budgetlatent.models.weights import checkpoint_family, checkpoint_template, klein_dir, load_model, resolve_checkpoint
    from budgetlatent.tokenization.layout import CELL_PX, GRID, dense_layout, overlay, snap_tokens
    try:                                                            # the references first: a wrong size fails before any loading
        images = prepare_images([Image.open(f) for f in a.ref], CELL_PX * GRID) if a.ref else []
    except (OSError, ValueError) as e:
        ap.error(f"--ref: {e}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not a.fp32 and device.type == "cuda"
    ckpt = resolve_checkpoint(a.ckpt)
    family = a.family or checkpoint_family(ckpt)
    if family_of(family).flow_map and a.cfg not in (None, 1.0):
        ap.error(f"--cfg {a.cfg}: {family_of(family).label} is a flow map and samples without guidance (cfg 1.0)")
    root = klein_dir(a.weights or family_of(family).repo)
    text = text_encoder(family, root, device, torch.bfloat16 if device.type == "cuda" else torch.float32,
                        template=a.template or checkpoint_template(ckpt, family=family))
    fam = family_of(family)
    trunk_cfg = {"image_cond": True} if images and fam.arch == "flux3" else None     # FLUX 3's reference branch
    model = load_model(ckpt, root, device, dtype=torch.bfloat16 if amp else None, family=family,
                       trunk_cfg=trunk_cfg).eval().requires_grad_(False)
    a.steps, a.cfg = a.steps or fam.steps, fam.guidance if a.cfg is None else a.cfg
    cond, uncond, lens = encode_prompts(text, model, a.prompts, amp, uncond=a.cfg != 1.0,   # (no guidance: no empty prompt)
                                        text_tokens=a.text_tokens)
    del text
    gc.collect()                                                    # (the encoder's modules hold reference cycles)
    torch.cuda.empty_cache()
    vae = Flux2VAE(root, device)
    refs = encode_images(images, vae, family) if images else None
    blas = apply_settings(model, vae, settings(a)) if device.type == "cuda" else None
    os.makedirs(a.out, exist_ok=True)
    kw = dict(cfg=a.cfg, relayout_until=a.relayout_until, layout_source=a.layout_source, preview_px=a.preview_px, seed=a.seed,
              amp=amp, final_decode=a.final_decode, **lens)
    if refs is not None:
        kw["refs"] = refs
        blas = None                         # (with references the sampler keeps the global matmul library: no timed table)
    if device.type == "cuda" and (a.fast or blas):                  # compile every budget first (and time its matmul library)
        warmup(model, vae, cond, uncond, a.budget, relayout_every=a.relayout_every, blas=blas, **kw)
    for budget in a.budget:
        res = sample(model, vae, cond, uncond, budget=budget, steps=a.steps, relayout_every=a.relayout_every, **kw)
        imgs = to_uint8(res["image"])
        over = overlay(res["image"], res["layout"] or dense_layout(len(a.prompts), device)).cpu() if a.overlay else None
        for k in range(len(a.prompts)):
            name = os.path.join(a.out, f"p{k:02d}_b{round(100 * budget):03d}")
            Image.fromarray(imgs[k].numpy()).save(name + ".png")
            if over is not None:
                Image.fromarray(over[k].numpy()).save(name + "_layout.png")
        with_refs = "" if refs is None else f", {len(images)} reference(s) of {refs.length} tokens in all (not budgeted)"
        print(f"[sample] {family}, {a.steps} steps, guidance {a.cfg:g}, budget {budget:.0%}: {snap_tokens(budget)} "
              f"tokens/image (mean over the trajectory {res['tokens']:.1f}){with_refs} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
