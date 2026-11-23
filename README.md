<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/budgetlatent_dark.svg">
    <img src="docs/assets/brand/budgetlatent_light.svg" height="92" alt="BudgetLatent">
  </picture>
</p>

<p align="center">
  🌐 <a href="https://budgetany.github.io/budgetlatent/"><b>Project page</b></a> &nbsp;|&nbsp;
  🎬 <a href="https://budgetany.github.io/budgetlatent/">Video</a> &nbsp;|&nbsp;
  📄 <a href="https://arxiv.org/abs/2610.12307">BudgetPix paper</a> &nbsp;|&nbsp;
  🤗 <a href="https://huggingface.co/BudgetAny">Models</a> &nbsp;|&nbsp;
  🧩 <a href="https://budgetany.github.io/">BudgetAny</a>: <a href="https://github.com/BudgetAny/BudgetPix">BudgetPix</a> · <a href="https://github.com/BudgetAny/Budget3D">Budget3D</a> · BudgetLatent
</p>

-----

**BudgetLatent** brings compute-adaptive tokenization to latent diffusion. It fine-tunes
[FLUX.2 [klein] 4B Base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B),
[FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B),
[FLUX 3 Image Klein Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) and
[FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) so that one checkpoint runs at any
token budget: a quadtree merges flat regions of the latent into coarser tokens, the transformer runs on the shorter
sequence, and small refiners restore the detail. Editing, multiple references and (FLUX 3) layout boxes work at every
budget too.

> **Unofficial, open-source implementation.** BudgetLatent is part of [BudgetAny](https://budgetany.github.io/) and is based
> on the method of [BudgetPix: Compute-Adaptive Tokenization for Pixel-Space Image Diffusion](https://arxiv.org/abs/2610.12307)
> (Kara et al., 2026), which it extends beyond the paper to latent diffusion. It is not an official release. Thanks to
> the BudgetPix authors for the method.

## 🔥 News

- **Nov 2026:** BudgetLatent is released: code and weights for FLUX.2 [klein] 4B Base, FLUX.2 [klein] 4B, FLUX 3 Image Klein Base and FLUX 3 Image Klein, each at any token budget, with editing, multiple references and layout boxes ([project page and video](https://budgetany.github.io/budgetlatent/), [models](https://huggingface.co/BudgetAny)).
- **Nov 2026:** BudgetLatent joins [BudgetAny](https://budgetany.github.io/), next to [BudgetPix](https://github.com/BudgetAny/BudgetPix) and [Budget3D](https://github.com/BudgetAny/Budget3D).

## 📦 Models

| Model | Base model | Sampler | License | Weights |
|---|---|---|---|---|
| BudgetLatent-FLUX.2-klein-base-4B | [FLUX.2 [klein] 4B Base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) | 50 steps, guidance 4.0 | Apache-2.0 | 🤗 [BudgetAny/BudgetLatent-FLUX.2-klein-base-4B](https://huggingface.co/BudgetAny/BudgetLatent-FLUX.2-klein-base-4B) |
| BudgetLatent-FLUX.2-klein-4B | [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (step-distilled) | 4 steps, no guidance | Apache-2.0 | 🤗 [BudgetAny/BudgetLatent-FLUX.2-klein-4B](https://huggingface.co/BudgetAny/BudgetLatent-FLUX.2-klein-4B) |
| BudgetLatent-FLUX-3-Image-Klein-Base | [FLUX 3 Image Klein Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) | 50 steps, guidance 3.5 | FLUX Kommunity License (non-commercial use of the model; see [its terms](https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/LICENSE.md)) | 🤗 [BudgetAny/BudgetLatent-FLUX-3-Image-Klein-Base](https://huggingface.co/BudgetAny/BudgetLatent-FLUX-3-Image-Klein-Base) |
| BudgetLatent-FLUX-3-Image-Klein | [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) (step-distilled) | 16 steps, no guidance | FLUX Kommunity License (non-commercial use of the model; see [its terms](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md)) | 🤗 [BudgetAny/BudgetLatent-FLUX-3-Image-Klein](https://huggingface.co/BudgetAny/BudgetLatent-FLUX-3-Image-Klein) |

Each repository holds one file, `budgetlatent.safetensors`; pass the repository id as `--ckpt`. The file records its
model family, so the scripts pick the right sampler. The token layout is chosen from the noise and refreshed during
sampling from a small decoded preview.

## 🚀 Usage

#### Installation

Python 3.10 - 3.13 (tested with 3.12) and a CUDA GPU.

```bash
git clone https://github.com/BudgetAny/BudgetLatent && cd BudgetLatent
pip install -r requirements.txt
python scripts/fetch_weights.py                            # FLUX.2 [klein] 4B Base
python scripts/fetch_weights.py --family flux2_distilled   # FLUX.2 [klein] 4B
python scripts/fetch_weights.py --family flux3             # FLUX 3 Image Klein Base (gated: accept its license, then `hf auth login`)
python scripts/fetch_weights.py --family flux3_distilled   # FLUX 3 Image Klein (gated, as FLUX 3 Image Klein Base)
```

#### Sample

- `--budget`: one or more fractions of the image tokens (1.0 = all 4,096 tokens at 1024 x 1024, 0.1 = 10 %).
- `--fast`: compiles the model and tunes its kernels for every budget; worth it for more than a few images.
- `BP_GRID=64`: needed for the 1024 x 1024 models.
- `--deterministic`: the same image for a seed in every run (50-step models), at a small cost in speed.

`python scripts/inference/sample.py --help` lists the other options.

```bash
BP_GRID=64 python scripts/inference/sample.py --ckpt BudgetAny/BudgetLatent-FLUX.2-klein-base-4B \
    --prompts "a red fox in a snowy forest" --budget 1.0 0.5 0.25 --overlay --fast
BP_GRID=64 python scripts/inference/sample.py --ckpt BudgetAny/BudgetLatent-FLUX.2-klein-4B \
    --prompts "an astronaut riding a horse on a beach at sunset, cinematic lighting" --budget 1.0 0.5 0.25 --overlay --fast
BP_GRID=64 python scripts/inference/sample.py --ckpt BudgetAny/BudgetLatent-FLUX-3-Image-Klein-Base \
    --prompts "a red fox in a snowy forest" --budget 1.0 0.5 0.25 --overlay --fast
BP_GRID=64 python scripts/inference/sample.py --ckpt BudgetAny/BudgetLatent-FLUX-3-Image-Klein \
    --prompts "a glowing jellyfish drifting in deep blue water" --budget 1.0 0.5 0.25 --overlay --fast
BP_GRID=64 python scripts/inference/sample.py --family flux3 --prompts "a red fox in a snowy forest"    # BFL's original model, no BudgetLatent weights
```

#### Editing and reference images

`--ref <image>` edits an image; repeat it to combine several references (up to 10). The first reference sets the output
size, so it must be 1024 x 1024. References keep all their tokens; only the generated image runs at the budget.

```bash
BP_GRID=64 python scripts/inference/sample.py --ckpt BudgetAny/BudgetLatent-FLUX.2-klein-base-4B \
    --prompts "put a red bandana around the dog's neck" --ref dog.png --budget 1.0 0.25 0.1 --overlay --fast
BP_GRID=64 python scripts/inference/sample.py --ckpt BudgetAny/BudgetLatent-FLUX.2-klein-4B \
    --prompts "print the dog from the second image onto the mug" --ref mug.png --ref dog.png --budget 1.0 0.25 --fast
BP_GRID=64 python scripts/inference/sample.py --ckpt BudgetAny/BudgetLatent-FLUX-3-Image-Klein-Base \
    --prompts "$(cat bandana_caption.txt)" --ref dog.png --budget 1.0 0.25 0.1 --overlay --fast
```

The images are `examples/inputs/3.png` (dog) and `examples/inputs/2.png` (mug) of Black Forest Labs'
[flux-3-image-klein](https://github.com/black-forest-labs/flux-3-image-klein) repository. The FLUX 3 models work best
with the captions of BFL's [prompt upsampler](https://github.com/black-forest-labs/flux-3-image-klein#prompt-upsampling)
(run in their repository); `bandana_caption.txt` is the `"prompt"` of their `examples/base/3_bandana.json`.

Since the references keep their tokens, edits speed up less than text to image. End-to-end speed-up at 10 % against
100 % on BFL's examples (prompt and references in, image out; `--fast`, one RTX PRO 6000, batch 1):

| | FLUX.2 [klein] 4B Base | FLUX.2 [klein] 4B | FLUX 3 Image Klein Base | FLUX 3 Image Klein |
|---|---|---|---|---|
| text to image | 4.37x | 2.57x | 7.18x | 4.38x |
| edit, one reference | 1.12x | 1.63x | 2.39x | 1.89x |
| two references | 1.58x | 1.41x | 1.82x | 1.50x |
| layout boxes |  |  | 5.71x | 3.88x |

#### Layout boxes (FLUX 3 Image Klein Base and FLUX 3 Image Klein)

Both FLUX 3 models place elements in boxes written into the prompt, in the format of BFL's
[bounding boxes](https://github.com/black-forest-labs/flux-3-image-klein#bounding-boxes) (`[y_min, x_min, y_max,
x_max]` on a 0-1000 canvas; their upsampler writes it from a prompt and a `boxes.json`). Boxes need no option and work
at every budget, also with `--ref`. FLUX.2 [klein] has no box mode.

```bash
BP_GRID=64 python scripts/inference/sample.py --ckpt BudgetAny/BudgetLatent-FLUX-3-Image-Klein \
    --prompts "$(cat balloons_caption.txt)" --budget 1.0 0.25 0.1 --overlay --fast
```

`balloons_caption.txt` is the `"prompt"` of BFL's `examples/distilled/balloons.json`.

#### Gallery and benchmark

```bash
BP_GRID=64 python scripts/inference/gallery.py --ckpt BudgetAny/BudgetLatent-FLUX.2-klein-base-4B --out gallery --fast   # every prompt at every budget, with token layouts
BP_GRID=64 python scripts/inference/bench.py --ckpt BudgetAny/BudgetLatent-FLUX.2-klein-base-4B --fast --out bench        # seconds, tokens and memory per budget
```

#### Train

Latents are encoded once per family; `scripts/data/` builds the training mix.

```bash
python scripts/data/encode_latents.py --family flux3 --data data/mix120k/chunks_512 --out data/mix120k/latents_flux3_512
torchrun --nproc_per_node=4 scripts/train/train.py --config configs/flux3_512.yaml \
    --data data/mix120k/latents_flux3_512 --out runs/flux3_512 --resume
```

#### Tests

```bash
bash tests/run_cpu.sh      # no GPU, no weights
bash tests/run_gpu.sh      # one GPU and the released weights of the two Base models
```

## 🗂️ Repository layout

```
budgetlatent/
  tokenization/layout.py        schedules, detail pyramids, threshold and exact-budget layouts, the training mixture
  models/flux2_trunk.py         the FLUX.2 [klein] MMDiT (BFL's names)
  models/flux3_trunk.py         the FLUX 3 Image Klein DiT (BFL's names)
  models/added_modules.py       cascade mixers and refiners, the 2-D position encoding of coarse tokens
  models/budgetpix.py           the adaptive-token model on either trunk
  models/refs.py                reference images (editing, multi-reference): files, latents, positions, kept dense
  models/families.py            the four families, their sampler defaults and the Hugging Face repositories of the weights
  models/weights.py             snapshots, checkpoints, loading and saving
  models/convert.py             key maps between weight formats
  encoders/text.py, vae.py      the frozen text encoders (Qwen3, Qwen3-VL) and the VAE
  inference/sampling.py         the sampler at a token budget (guided; the step-distilled models without guidance)
  inference/fast.py             the speed settings behind --fast
  inference/validate.py         the fixed validation gallery
  training/                     data chunks, a stage-1 step, the shared trainer pieces, the optional DMD2 + GAN stage
scripts/fetch_weights.py        the released weights into the Hugging Face cache
scripts/inference/              sample.py, gallery.py, bench.py
scripts/train/                  train.py (stage 1: FLUX.2 [klein] 4B Base and FLUX 3 Image Klein Base), train_dmd.py (stage 2, FLUX.2)
scripts/data/                   the training mix, its latents
configs/                        flux2_512.yaml, flux2_1024.yaml, flux3_512.yaml, flux3_1024.yaml
tests/                          CPU tests, GPU checks against the released weights
```

## 📚 Citation

If you use BudgetLatent, please cite the BudgetPix paper, whose method it builds on:

```bibtex
@article{kara2026budgetpix,
  title={BudgetPix: Compute-Adaptive Tokenization for Pixel-Space Image Diffusion},
  author={Ozgur Kara and Yujia Chen and Daniel Watson and David Forsyth and James Matthew Rehg and Wen-Sheng Chu and Du Tran},
  journal={arXiv preprint arXiv:2610.12307},
  year={2026}
}
```

## 📄 License

The code is released under the MIT License ([`LICENSE`](LICENSE)). The base models are not part of this repository and
come with their own licenses: [FLUX.2 [klein] 4B Base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) and
[FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), their VAE and their text encoder are
under Apache-2.0; [FLUX 3 Image Klein Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) and
[FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein)
are under Black Forest Labs' FLUX Kommunity License (non-commercial use of the model; see [its terms](https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/LICENSE.md)).
BudgetLatent weights derived from a base model follow that model's license.

BudgetLatent is not affiliated with or endorsed by Black Forest Labs.

## 🙏 Acknowledgements

BudgetLatent builds on [FLUX.2 [klein]](https://github.com/black-forest-labs/flux2) and [FLUX 3 Image Klein](https://github.com/black-forest-labs/flux-3-image-klein)
by Black Forest Labs, whose released weights (FLUX.2 [klein] 4B Base, FLUX.2 [klein] 4B, FLUX 3 Image Klein Base and
FLUX 3 Image Klein) are the starting point of every run, whose code is the reference of the parity checks, and whose
examples are the editing and layout cases above.

We acknowledge the compute support of the Health Care Engineering Systems Center within the Coordinated Science Lab at UIUC.
