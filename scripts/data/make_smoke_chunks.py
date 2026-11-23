"""A tiny chunk directory in the training data format, for smoke tests only (not for training).

    python scripts/data/make_smoke_chunks.py --out data/smoke_chunks

Takes every image under the given folders (default: the pictures shipped with a sibling BudgetPix checkout and the Idol demo of a sibling Budget3D one),
centre-crops and resizes them to 512 x 512, and pairs each with a generic caption.
"""
import argparse
import glob
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))    # the folder that holds the BudgetAny checkouts
CAPTIONS = ["a photograph", "a detailed picture of an object on a plain background", "an illustration", "a colorful image",
            "a studio photo of a small figurine", "a rendering of a toy", "a picture of food", "an image with fine texture"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", nargs="*", default=[os.path.join(PROJECT, "BudgetPix", "docs", "assets"),
                                                    os.path.join(PROJECT, "Budget3D", "third_party", "idol-sdk", "demo", "examples")])
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--per_chunk", type=int, default=32)
    a = ap.parse_args()
    import numpy as np
    from PIL import Image
    files = sorted(f for d in a.images for ext in ("jpg", "jpeg", "png", "webp") for f in glob.glob(os.path.join(d, f"*.{ext}")))
    if not files:
        raise SystemExit(f"no images under {a.images}")
    os.makedirs(a.out, exist_ok=True)
    k = 0
    for c in range(a.chunks):
        px, caps, keys = [], [], []
        for _ in range(a.per_chunk):
            im = Image.open(files[k % len(files)])
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(bg, im)
            im = im.convert("RGB")
            side = min(im.size)
            box = ((im.width - side) // 2, (im.height - side) // 2, (im.width + side) // 2, (im.height + side) // 2)
            im = im.crop(box).resize((512, 512), Image.BICUBIC)
            px.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1))
            caps.append(CAPTIONS[k % len(CAPTIONS)] + "\n")
            keys.append(f"{('blip3o', 'sharegpt4o', 'dalle3')[k % 3]}/{os.path.basename(files[k % len(files)])}")
            k += 1
        torch.save({"pixel_values": torch.stack(px), "caption": caps, "key": keys}, os.path.join(a.out, f"chunk_{c:06d}.pt"))
    print(f"[smoke-chunks] {a.chunks} x {a.per_chunk} samples from {len(files)} images -> {a.out}")


if __name__ == "__main__":
    main()
