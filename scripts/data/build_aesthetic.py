"""Aesthetic-Train-V2 as training chunks next to the 120K mix: real high-resolution photographs for the 1024 run.

    python scripts/data/build_aesthetic.py --out data/mix120k/chunks_1024 --cache data/aesthetic_v2 --procs 24

zhang0jhon/Aesthetic-Train-V2 (MIT; Diffusion-4K, CVPR 2025): ~105k photographs selected for fidelity and fine
detail (median ~4K on a side), one GPT-4o caption each (metadata.jsonl), in 106 tar shards of ~1,000.  The 120K mix
is all generated images (GPT-4o, DALL-E 3); this adds real photography at a resolution well above the 1024 it is used at.

Each shard becomes one chunk in the mix's format, <out>/chunk_aes_<shard>.pt = {"pixel_values": uint8 (n, 3, S, S),
"caption": [...], "key": ["aesthetic/<id>", ...]}: decoded at a reduced scale where the JPEG allows it (never below S),
EXIF orientation applied, short side resized to S (bicubic, antialiased), centre crop.  Dropped: images whose short side
is below S or whose aspect ratio exceeds --max_aspect (a square crop would cut away most of what the caption describes),
and images without a caption.  The shards are random in content, so a shard is a chunk; the loader shuffles within and
across chunks.  configs/flux2_1024.yaml and configs/flux3_1024.yaml need a positive source weight for "aesthetic".  Resumable; <out>/aesthetic_manifest.json
summarises the result.
"""
import argparse
import io
import json
import multiprocessing as mp
import os
import tarfile
import time
from pathlib import Path

REPO = "zhang0jhon/Aesthetic-Train-V2"
CAPTIONS = {}                                   # file_name -> caption, set in every worker by the pool initializer


def _init(captions):
    global CAPTIONS
    CAPTIONS = captions


def log(*a):
    print("[aesthetic]", time.strftime("%H:%M:%S"), *a, flush=True)


def prep(data, size, max_aspect):
    """Image bytes -> HxWx3 uint8 bytes at size^2, or a reason it is dropped."""
    from PIL import Image, ImageOps
    im = Image.open(io.BytesIO(data))
    w, h = im.size
    if min(w, h) < size:
        return None, "small"
    if max(w, h) / min(w, h) > max_aspect:
        return None, "aspect"
    if im.format == "JPEG":
        im.draft("RGB", (size * w // min(w, h), size * h // min(w, h)))       # DCT scaling, never below size
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    if min(w, h) < size:                                                       # (a rotated draft) decode in full
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC, reducing_gap=3.0)
    w, h = im.size
    left, top = (w - size) // 2, (h - size) // 2
    return im.crop((left, top, left + size, top + size)).tobytes(), None


def shard(task):
    tar_path, out, size, max_aspect = task
    import numpy as np
    import torch
    name = Path(tar_path).stem
    dst = Path(out) / f"chunk_aes_{name}.pt"
    if dst.exists():
        return name, "skip", 0, {}
    t0, bufs, caps, keys, dropped = time.time(), [], [], [], {}
    with tarfile.open(tar_path, "r|*") as tar:
        for m in tar:
            if not m.isfile() or m.name.rsplit(".", 1)[-1].lower() not in ("jpg", "jpeg", "png", "webp"):
                continue
            rel = m.name.lstrip("./")                                            # 0000/<id>.jpg
            cap = CAPTIONS.get(f"images/{rel}")
            if not cap:
                dropped["caption"] = dropped.get("caption", 0) + 1
                continue
            try:
                px, why = prep(tar.extractfile(m).read(), size, max_aspect)
            except Exception:                                                     # noqa: BLE001
                px, why = None, "decode"
            if px is None:
                dropped[why] = dropped.get(why, 0) + 1
                continue
            bufs.append(px)
            caps.append(cap.strip())
            keys.append(f"aesthetic/{Path(rel).stem}")
    arr = np.frombuffer(b"".join(bufs), dtype=np.uint8).reshape(len(bufs), size, size, 3)
    del bufs
    tmp = dst.with_suffix(".tmp")
    torch.save({"pixel_values": torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous(), "caption": caps, "key": keys}, tmp)
    os.replace(tmp, dst)
    return name, f"{time.time() - t0:.0f}s", len(caps), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the chunk directory of the mix at this size (e.g. chunks_1024)")
    ap.add_argument("--cache", required=True, help="where the dataset is downloaded")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--max_aspect", type=float, default=2.0)
    ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    t0 = time.time()
    root = Path(snapshot_download(REPO, repo_type="dataset", cache_dir=os.path.join(a.cache, "hf_cache"), max_workers=8))
    tars = sorted(root.glob("images/*.tar"))
    log(f"{REPO}: {len(tars)} shards in {root} ({time.time() - t0:.0f} s)")
    captions = {}
    for line in open(root / "metadata.jsonl"):
        r = json.loads(line)
        captions[r["file_name"]] = r["text"]
    log(f"{len(captions)} captions")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tasks = [(str(t), a.out, a.size, a.max_aspect) for t in tars]
    kept, dropped = 0, {}
    with mp.Pool(a.procs, initializer=_init, initargs=(captions,)) as pool:
        for name, dt, n, drop in pool.imap_unordered(shard, tasks):
            kept += n
            for k, v in drop.items():
                dropped[k] = dropped.get(k, 0) + v
            log(f"  shard {name}: {dt}, kept {n}, dropped {drop}")
    files = sorted(Path(a.out).glob("chunk_aes_*.pt"))
    json.dump({"source": REPO, "license": "MIT", "chunks": len(files), "kept_this_run": kept, "dropped_this_run": dropped,
               "size": a.size, "max_aspect": a.max_aspect,
               "preprocessing": f"PIL decode (JPEG draft scaling >= {a.size}), EXIF transpose, short side -> {a.size} bicubic, centre crop"},
              open(Path(a.out) / "aesthetic_manifest.json", "w"), indent=1)
    log(f"done: {len(files)} chunks in {a.out}; this run kept {kept}, dropped {dropped}")


if __name__ == "__main__":
    main()
