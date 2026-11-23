"""Training data: the MiniT2I 120K mix as tensor chunks, captions tokenised for Qwen3 on the fly.

Two chunk formats, both `<dir>/chunk_*.pt` with 1024 samples each:

    pixels   {"pixel_values": uint8 (n, 3, 512, 512), "caption": list[str], "key": list["<source>/<member>"], ...}
             written by scripts/data/build_mix.py: centre crop + bicubic
             resize to 512 x 512, sources below 512 px on a side dropped
    latents  {"latent": float32 (n, 32, 32, 128), "caption": ..., "key": ...}
             the same images through the frozen VAE (scripts/data/encode_latents.py); the trainer then skips the encoder

Chunks are split across ranks x workers and reshuffled every pass; inside a chunk the samples are drawn so that
the three sources (blip3o, sharegpt4o, dalle3) appear with the configured shares.  The stream is infinite.
With captions_only=True (stage 2) only the tokenised captions are yielded, from either format.

FLUX 3 (template "flux3"): the caption, stripped of surrounding whitespace (several sources end in a newline, which
prompts at sampling time do not), in Qwen3-VL's template, padded to max_tokens so that a batch stacks; the trainer
cuts the batch down to its longest text window and each row keeps its own (text.Qwen3VLText.lengths).
"""
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

from ..encoders.text import F3_MAX_TOKENS, load_tokenizer, tokenize, tokenize_flux3


class ChunkStream(IterableDataset):
    def __init__(self, root, tokenizer_path, template="bfl_nothink", seed=0, rank=0, world_size=1, source_weights=None,
                 captions_only=False, max_tokens=F3_MAX_TOKENS):
        self.files = sorted(Path(root).glob("chunk_*.pt"))
        if not self.files:
            raise SystemExit(f"no chunk_*.pt under {root}")
        self.tokenizer_path, self.template = tokenizer_path, template
        self.seed, self.rank, self.world_size = int(seed), rank, world_size
        self.source_weights = dict(source_weights or {})
        self.captions_only = bool(captions_only)
        self.max_tokens = int(max_tokens)
        self._tok = None

    def order(self, n, keys, rng):
        """Sample indices of one pass over a chunk: a shuffle, or a draw with the configured source shares."""
        if not self.source_weights or keys is None:
            idx = list(range(n))
            rng.shuffle(idx)
            return idx
        src = [str(k).split("/", 1)[0] for k in keys]
        count = {s: src.count(s) for s in set(src)}
        w = [self.source_weights.get(s, 0.0) / count[s] for s in src]
        if sum(w) <= 0:
            raise RuntimeError(f"no sample of {sorted(count)} has a positive source weight ({self.source_weights})")
        return rng.choices(range(n), weights=w, k=n)

    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        nw, wid = (w.num_workers, w.id) if w is not None else (1, 0)
        files = self.files[self.rank * nw + wid::self.world_size * nw] or self.files[wid::nw] or self.files
        rng = random.Random(self.seed + self.rank * 1000 + wid)
        if self._tok is None:
            self._tok = load_tokenizer(self.tokenizer_path)
        while True:
            order = list(files)
            rng.shuffle(order)
            for path in order:
                chunk = torch.load(path, map_location="cpu", weights_only=False)
                field = "latent" if "latent" in chunk else "pixel_values"
                values = None if self.captions_only else chunk[field]
                captions, keys = chunk["caption"], chunk.get("key")
                del chunk
                for i in self.order(len(captions), keys, rng):
                    if self.template == "flux3":
                        ids, mask = tokenize_flux3(self._tok, [str(captions[i]).strip()], self.max_tokens, pad_to=self.max_tokens)
                    else:
                        ids, mask = tokenize(self._tok, [captions[i]], self.template)
                    item = {"input_ids": ids[0], "attention_mask": mask[0]}
                    if values is not None:
                        item[field] = values[i]
                    yield item


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def make_loader(root, tokenizer_path, batch_size, num_workers=4, template="bfl_nothink", seed=0, rank=0, world_size=1,
                source_weights=None, captions_only=False, max_tokens=F3_MAX_TOKENS):
    ds = ChunkStream(root, tokenizer_path, template, seed, rank, world_size, source_weights, captions_only, max_tokens)
    num_workers = max(0, min(num_workers, len(ds.files)))
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate,
                      persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None,
                      drop_last=True, pin_memory=torch.cuda.is_available())


def to_unit(pixel_values, device):
    """uint8 (B, 3, H, W) -> fp32 in [-1, 1]: x / 127.5 - 1."""
    return pixel_values.to(device, non_blocking=True).float().div_(127.5).sub_(1.0)
