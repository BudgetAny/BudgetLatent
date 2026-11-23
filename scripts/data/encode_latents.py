"""Encode the pixel chunks of the training mix to VAE latents, once.

    python scripts/data/encode_latents.py --data data/mix120k/chunks_512 --out data/mix120k/latents_512
    python scripts/data/encode_latents.py --family flux3 --data data/mix120k/chunks_512 --out data/mix120k/latents_flux3_512
    CUDA_VISIBLE_DEVICES=1 python scripts/data/encode_latents.py ... --shard 1/4          # one process per GPU

The training preprocessing is deterministic (centre crop, no flip), so the clean latent of an image never changes:
writing it once removes the VAE encoder from every training step.  Each `chunk_*.pt` of pixels becomes a
`chunk_*.pt` with {"latent": float32 (n, 32, 32, 128), "caption", "key"} -- the posterior mean, 2x2-packed and
normalised exactly as vae.Flux2VAE.encode returns it (fp32, like the trainer's own encode).  Resumable.

--family flux3 encodes with FLUX 3 Image Klein's VAE: the same architecture as FLUX.2's, but its own weights and
BatchNorm statistics (tests/check_flux3_release.py), so FLUX 3 needs latents of its own (latents_flux3_<res>).
The output directory records the family (.family); a directory of the other family is refused.

The VAE assertion is made here, on what was just encoded: |mean| < 0.15 and 0.75 < std < 1.25 over
this shard's latents (thousands of images, so these are sharp tests of the packing and the normalisation), and the
round trip of the first 16 images.  The recipe expects that above 30 dB; 16 images can fall a little short of
it on detailed pictures, so below 30 dB is reported and only below 25 dB (a wrong packing gives < 15) is a failure.
A failure stops the caller (the latents stay on disk); --no_check only reports (tiny or unusual image sets).
"""
import argparse
import os
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))    # the repository root

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory of pixel chunks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None, help="snapshot (default: the family's release in the HF cache)")
    ap.add_argument("--family", default="flux2", choices=["flux2", "flux3"], help="whose VAE encodes")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no_check", action="store_true", help="report the latent statistics and the round trip, do not assert them")
    a = ap.parse_args()
    from budgetlatent.encoders.vae import Flux2VAE
    from budgetlatent.models.families import family_of
    from budgetlatent.models.weights import klein_dir
    from budgetlatent.training.data import to_unit
    si, sn = (int(v) for v in a.shard.split("/"))
    files = sorted(Path(a.data).glob("chunk_*.pt"))[si::sn]
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.join(a.out, ".family")
    if os.path.exists(tag) and open(tag).read().strip() != a.family:
        raise SystemExit(f"{a.out} holds {open(tag).read().strip()} latents, not {a.family}")
    if not os.path.exists(tag) and any(Path(a.out).glob("chunk_*.pt")) and a.family != "flux2":
        raise SystemExit(f"{a.out} holds latents of an unknown family (older files are FLUX.2's); use another directory")
    if not os.path.exists(tag):
        open(tag, "w").write(a.family + "\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = Flux2VAE(klein_dir(a.weights or family_of(a.family).repo), device).optimize()
    t0, n, s1, s2, count, psnr = time.time(), 0, 0.0, 0.0, 0, None
    for f in files:
        dst = Path(a.out) / f.name
        if dst.exists():
            continue
        chunk = torch.load(f, map_location="cpu", weights_only=False)
        px = chunk["pixel_values"]
        lat = torch.cat([vae.encode(to_unit(px[i:i + a.batch], device)).cpu() for i in range(0, len(px), a.batch)])
        tmp = dst.with_suffix(f".tmp{si}")
        torch.save({"latent": lat, "caption": chunk["caption"], "key": chunk.get("key")}, tmp)
        os.replace(tmp, dst)
        n += len(px)
        s1, s2, count = s1 + float(lat.double().sum()), s2 + float(lat.double().pow(2).sum()), count + lat.numel()
        if psnr is None:
            x = to_unit(px[:16], device)
            psnr = float(10 * torch.log10(4.0 / (vae.decode(lat[:16].to(device)).clamp(-1, 1) - x).pow(2).mean()))
        print(f"[latents] {f.name}: {len(px)} images, mean {float(lat.mean()):+.3f} std {float(lat.std()):.3f} "
              f"({n / (time.time() - t0):.0f} img/s)", flush=True)
    print(f"[latents] shard {a.shard}: {n} images encoded -> {a.out}", flush=True)
    if count:
        mean = s1 / count
        std = (s2 / count - mean * mean) ** 0.5
        ok = abs(mean) < 0.15 and 0.75 < std < 1.25 and psnr > 25.0
        note = "ok" if ok and psnr > 30.0 else "ok (round trip below the 30 dB the recipe expects)" if ok else \
            "OUTSIDE the expected range (|mean| < 0.15, 0.75 < std < 1.25, round trip > 25 dB)"
        print(f"[latents] shard {a.shard}: latent mean {mean:+.4f}, std {std:.4f}; round trip of 16 images {psnr:.2f} dB -- {note}", flush=True)
        if not ok and not a.no_check:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
