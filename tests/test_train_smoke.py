"""CPU smoke test of the stage-1 trainer, for both families, on tiny random trunks (no weights, no data, no GPU).

    python tests/test_train_smoke.py                 # flux2 and flux3
    python tests/test_train_smoke.py flux3

scripts/train/train.py's own main() runs, with its weight, data, text-encoder and VAE loaders swapped for tiny
stand-ins: a random trunk of the family's architecture serves as the "released" model (student and teacher start
from it), a stub text encoder gives deterministic states (FLUX 3: per-caption text windows), a stub VAE maps
latents to pixels.  Everything else is the real code: the start-up assertions (trainlib.preflight), the step
preparation (training/stage1.py: teacher targets, detail maps, the four layout modes), the student passes grouped
by length, loss, backward, gradient guard and clip, AdamW, the EMA, checkpoint-last, the EMA export with its
metadata, the validation gallery, and a resume that continues with every block checkpointed and one example per
pass.  Checked: finite losses, the files written, the exported weights (all tensors, the family they record and
the one their names tell).
"""
import contextlib
import importlib.util
import io
import math
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from budgetlatent.models import weights as W
from budgetlatent.models.budgetpix import BPConfig, add_modules, count_parameters
from budgetlatent.models.families import family_from_keys, family_of
from budgetlatent.tokenization import layout as L
from test_units import OK, TINY, check
from test_units_flux3 import TINY3

CONFIGS = {"flux2": TINY, "flux3": TINY3}
TEXT_LEN, WINDOW, MAX_TOKENS = 16, 8, 32          # FLUX.2's fixed text length; FLUX 3's window multiple and cap (tiny)


class StubText(nn.Module):
    """The text encoders' interface (encoders/text.py) with an embedding table instead of Qwen3 / Qwen3-VL."""

    def __init__(self, family, dim):
        super().__init__()
        self.family = family
        self.model = nn.Embedding(97, dim)
        nn.init.normal_(self.model.weight, std=1.0, generator=torch.Generator().manual_seed(7))
        self.model.requires_grad_(False)

    def tokenize(self, prompts, pad_to=None):
        rows = [[1 + (sum(map(ord, w)) % 95) for w in str(p).split()][: MAX_TOKENS - 2] for p in prompts]
        rows = [[96] + r + [96] for r in rows]                       # a template around the words
        if self.family == "flux2":
            length = TEXT_LEN
        else:
            length = pad_to or max(self.window(len(r)) for r in rows)
        ids = torch.zeros(len(prompts), length, dtype=torch.long)
        mask = torch.zeros(len(prompts), length, dtype=torch.long)
        for i, r in enumerate(rows):
            r = r[:length]
            ids[i, :len(r)], mask[i, :len(r)] = torch.tensor(r), 1
        if self.family == "flux2":
            mask[:, :] = 1                                           # FLUX.2 attends to its whole window
        return ids, mask

    @staticmethod
    def window(n):
        return min(-(-int(n) // WINDOW) * WINDOW, MAX_TOKENS)

    def lengths(self, mask):
        if self.family == "flux2":
            return None
        return torch.tensor([self.window(int(n)) for n in mask.long().sum(dim=1)], dtype=torch.long)

    def trim(self, ids, mask):
        if self.family == "flux2":
            return ids, mask
        n = int(self.lengths(mask).max())
        return ids[:, :n], mask[:, :n]

    @torch.no_grad()
    def encode_ids(self, ids, mask):
        return self.model(ids.to(self.model.weight.device)) * 0.5

    @torch.no_grad()
    def encode(self, prompts):
        ids, mask = self.tokenize(prompts)
        lens = self.lengths(mask)
        ids, mask = self.trim(ids, mask)
        return self.encode_ids(ids, mask), lens

    def empty_ids(self, batch=1, pad_to=None):
        ids, mask = self.tokenize([""], pad_to=pad_to)
        return ids.expand(batch, -1).clone(), mask.expand(batch, -1).clone()


class StubVAE:
    """(B, g, g, C) latents <-> (B, 3, 16 g, 16 g) pixels in [-1, 1]: enough for the detail maps and the gallery."""
    chunk = None

    def optimize(self, compile_decoder=False, channels_last=True):
        return self

    def bf16_preview(self, compile=False):
        return self

    def decode(self, z, bf16=False):
        return torch.tanh(F.interpolate(z[..., :3].float().permute(0, 3, 1, 2), scale_factor=16, mode="nearest"))


def released_state(family):
    """The tiny family trunk standing in for the release: the same random weights every time."""
    torch.manual_seed(1234)
    trunk = family_of(family).trunk_cls(CONFIGS[family])
    with torch.no_grad():
        for n, p in trunk.named_parameters():
            if n.endswith(".scale"):
                p.add_(0.1 * torch.randn(p.shape))
    return {k: v.detach().clone() for k, v in trunk.state_dict().items()}


def install_stubs(train, family):
    """Swap the loaders train.py (and the validation, through weights.load_model) uses for the tiny stand-ins."""
    fam = family_of(family)
    cfg = CONFIGS[family]
    released = released_state(family)

    def load_model(ckpt=None, weights=None, device="cpu", dtype=None, bp=None, family=family, trunk_cfg=None):
        bp = bp or BPConfig()
        model = fam.model_cls(cfg, bp)
        sd = dict(released)
        sd.update(add_modules(nn.Module(), cfg, bp).state_dict())   # the added modules' initialisation
        if ckpt:
            sd.update(W.read_state(ckpt)[0])
        model.load_state_dict({k: v.to(device, dtype or torch.float32) for k, v in sd.items()}, strict=True)
        return model.to(device).requires_grad_(True)

    def load_teacher(weights=None, device="cpu", dtype=torch.bfloat16, family=family, trunk_cfg=None):
        teacher = fam.trunk_cls(cfg)
        teacher.load_state_dict({k: v.to(device, dtype or torch.float32) for k, v in released.items()}, strict=True)
        return teacher.to(device).eval().requires_grad_(False)

    text = StubText(family, cfg.context_in_dim)

    def make_loader(root, tokenizer_path, batch_size, num_workers=4, template=None, seed=0, rank=0, world_size=1,
                    source_weights=None, captions_only=False, max_tokens=None):
        gen = torch.Generator().manual_seed(seed + 31 * rank)
        words = ["a", "red", "fox", "in", "the", "snow", "at", "dusk", "with", "detailed", "fur", "and", "a", "lighthouse"]
        while True:
            caps = [" ".join(words[: 1 + int(torch.randint(0, len(words), (1,), generator=gen))]) for _ in range(batch_size)]
            ids, mask = text.tokenize(caps, pad_to=MAX_TOKENS if family == "flux3" else None)
            yield {"latent": torch.randn(batch_size, L.GRID, L.GRID, cfg.in_channels, generator=gen),
                   "input_ids": ids, "attention_mask": mask}

    with torch.device("meta"):
        counts = count_parameters(fam.model_cls(cfg, BPConfig()))
    train.klein_dir = lambda repo=None: tempfile.gettempdir()
    train.load_model = W.load_model = load_model
    train.load_teacher = load_teacher
    train.text_encoder = lambda family, root, device="cpu", dtype=None, template=None, max_tokens=None: text.to(device)
    train.Flux2VAE = lambda root, device="cpu", chunk=None: StubVAE()
    train.make_loader = make_loader
    train.N_PARAMS = counts
    train.expected_counts = lambda family, root=None, bp=None, trunk_cfg=None: counts
    return counts


def run_family(family, tmp):
    print(f"[{family}: train.py on a tiny random trunk]")
    spec = importlib.util.spec_from_file_location(f"bl_train_{family}", os.path.join(ROOT, "scripts", "train", "train.py"))
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    counts = install_stubs(train, family)

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", f"{family}_512.yaml")))
    cfg["model"].update(resolution=L.GRID * L.CELL_PX, latent_grid=[L.GRID, L.GRID], latent_channels=CONFIGS[family].in_channels,
                        text_seq_len=TEXT_LEN if family == "flux2" else MAX_TOKENS)
    if family == "flux3":
        cfg["model"]["text_max_tokens"] = MAX_TOKENS
    cfg["optim"].update(warmup_steps=2, global_batch_size=2)
    cfg["inference"]["num_steps"] = 2
    out, data = os.path.join(tmp, f"run_{family}"), os.path.join(tmp, f"data_{family}")
    os.makedirs(data)
    torch.save({"latent": torch.zeros(1, L.GRID, L.GRID, CONFIGS[family].in_channels), "caption": ["x"], "key": ["k"]},
               os.path.join(data, "chunk_000000.pt"))
    open(os.path.join(data, ".family"), "w").write(family + "\n")    # what scripts/data/encode_latents.py writes
    path = os.path.join(tmp, f"{family}.yaml")
    yaml.safe_dump(cfg, open(path, "w"))
    common = ["--config", path, "--data", data, "--out", out, "--batch_per_gpu", "1", "--grad_accum", "2", "--prep_batch", "2",
              "--num_workers", "0", "--log_every", "1", "--save_every", "1", "--snapshot_every", "2", "--val_prompts", "2",
              "--val_budgets", "1.0", "0.3"]

    def main(*args):
        buf = io.StringIO()
        sys.argv = ["train.py", *common, *args]
        with contextlib.redirect_stdout(buf):
            train.main()
        text = buf.getvalue()
        print("\n".join("    " + line for line in text.splitlines() if line.startswith(("[preflight]", "step", "[train] resumed"))))
        return text

    log1 = main("--max_steps", "2")
    losses = [float(m) for m in re.findall(r"^step +\d+ +loss ([0-9.eE+-]+|nan|inf)", log1, re.M)]
    check(f"{family}: preflight", log1.count("[preflight]") >= 4, "dense layout == trunk, Wiener gain, schedule, parameter counts")
    check(f"{family}: two steps", len(losses) == 2 and all(math.isfinite(v) for v in losses), f"losses {losses}")
    files = sorted(os.listdir(out))
    check(f"{family}: checkpoint, EMA export, validation gallery",
          {"checkpoint-last", "ema-final.safetensors", "val-s2"} <= set(files)
          and os.path.exists(os.path.join(out, "val-s2", "index.html")), f"{files}")

    log2 = main("--max_steps", "3", "--resume", "--skip_preflight", "--max_micro", "1", "--ckpt_blocks", "100", "--val_every", "0")
    losses2 = [float(m) for m in re.findall(r"^step +\d+ +loss ([0-9.eE+-]+|nan|inf)", log2, re.M)]
    check(f"{family}: resume", "resumed from" in log2 and "at step 2" in log2 and len(losses2) == 1 and math.isfinite(losses2[0]),
          f"step 3 loss {losses2}")

    ema = os.path.join(out, "ema-final.safetensors")
    sd, meta = W.read_state(ema)
    from safetensors import safe_open
    with safe_open(ema, framework="pt") as f:
        names = list(f.keys())
    with torch.device("meta"):
        want = set(family_of(family).model_cls(CONFIGS[family], BPConfig()).state_dict())
    check(f"{family}: exported weights", set(sd) == want and meta.get("family") == family and meta.get("step") == "3"
          and W.checkpoint_family(ema) == family and family_from_keys(names) == family and sum(counts) == sum(v.numel() for v in sd.values()),
          f"{len(sd)} tensors, metadata family {meta.get('family')!r}, step {meta.get('step')}, template {meta.get('chat_template')!r}")


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    which = sys.argv[1:] or ["flux2", "flux3"]
    with tempfile.TemporaryDirectory() as tmp:
        for fam in which:
            run_family(fam, tmp)
    print(f"\n{len(OK)} checks passed (grid {L.GRID})")
