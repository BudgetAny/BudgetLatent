"""CPU checks of the FLUX 3 Image Klein Base release against this code (no GPU, no model forward; the released files
in the Hugging Face cache, scripts/fetch_weights.py --family flux3):

    python tests/check_flux3_release.py [--chunks data/mix120k/latents_512 ...]

  1  weights    the diffusers transformer (sharded), key-converted (convert.flux3_diffusers_to_bfl), is tensor for
                tensor BFL's single file model.safetensors -- which also fixes the order of the final modulation
  2  config     flux3_trunk.Flux3Config from config.json ("dit") == from transformer/config.json (diffusers)
  3  vae        the release's VAE has FLUX.2 [klein]'s architecture (the same tensor names and shapes, so
                encoders/vae.Flux2VAE runs it) but weights and BatchNorm statistics of its own: FLUX 3 needs latents
                of its own (scripts/data/encode_latents.py --family flux3)
  4  template   text.wrap_flux3 == Qwen3-VL's own chat template (apply_chat_template, generation prompt), the empty
                prompt's ids, the pad token
  5  windows    the text-window distribution of the training captions (FLUX 3: real tokens rounded up to 80)
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch

from budgetlatent.encoders.text import F3_EMPTY_PROMPT_IDS, PAD_ID, load_tokenizer, window, wrap_flux3
from budgetlatent.models import convert
from budgetlatent.models.families import FAMILIES
from budgetlatent.models.flux3_trunk import Flux3Config
from budgetlatent.models.weights import klein_dir

REPORT = {}


def note(name, ok, detail):
    REPORT[name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="*", default=[], help="chunk directories whose captions' text windows are counted (check 5)")
    ap.add_argument("--max_chunks", type=int, default=12, help="per directory (captions only are read)")
    ap.add_argument("--out", default=None, help="write the report here (json)")
    a = ap.parse_args()
    from safetensors.torch import load_file
    root = klein_dir(FAMILIES["flux3"].repo)
    print(f"[check] FLUX 3 snapshot {root}", flush=True)

    # 1 -----------------------------------------------------------------------------------------------------
    bfl = load_file(os.path.join(root, "model.safetensors"))
    index = json.load(open(os.path.join(root, "transformer", "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    dif = {}
    for shard in sorted(set(index.values())):
        dif.update(load_file(os.path.join(root, "transformer", shard)))
    conv = convert.flux3_diffusers_to_bfl(dif)
    same_keys = set(conv) == set(bfl)
    differ = [k for k in bfl if k in conv and not torch.equal(conv[k], bfl[k])]
    swapped = None
    k_fin = "final_layer.image.adaLN_modulation.1.weight"
    if k_fin in differ:                                              # which half order the BFL file has
        sc, sh = dif["norm_out.linear.weight"].chunk(2)
        swapped = torch.equal(torch.cat([sc, sh]), bfl[k_fin])
    n_par = sum(v.numel() for v in bfl.values())
    note("1 weights", same_keys and not differ,
         f"{len(bfl)} tensors, {n_par:,} parameters ({bfl[k_fin].dtype}); same keys {same_keys}, tensors that differ "
         f"{len(differ)} {differ[:3]}" + ("" if swapped is None else f"; final modulation unswapped would match: {swapped}"))
    del dif, conv

    # 2 -----------------------------------------------------------------------------------------------------
    c_bfl = Flux3Config.from_dict(json.load(open(os.path.join(root, "config.json"))))
    c_dif = Flux3Config.from_dict(json.load(open(os.path.join(root, "transformer", "config.json"))))
    note("2 config", c_bfl == c_dif and c_bfl == Flux3Config(), f"{c_bfl}")
    sampler = json.load(open(os.path.join(root, "config.json"))).get("sampler", {})
    te = json.load(open(os.path.join(root, "config.json"))).get("text_encoder", {})
    note("2b sampler / text recipe", sampler.get("alpha") == 6.93 and sampler.get("guidance") == 3.5 and sampler.get("steps") == 50
         and tuple(te.get("layers", ())) == (4, 8, 12, 16, 20, 24, 28, 32) and te.get("pad_multiple") == 80,
         f"sampler {sampler}; text encoder {te}")

    # 3 -----------------------------------------------------------------------------------------------------
    root2 = klein_dir(FAMILIES["flux2"].repo)
    v3 = load_file(os.path.join(root, "vae", "diffusion_pytorch_model.safetensors"))
    v2 = load_file(os.path.join(root2, "vae", "diffusion_pytorch_model.safetensors"))
    vdiff = [k for k in v3 if k not in v2 or not torch.equal(v3[k], v2[k])]
    shapes = set(v3) == set(v2) and all(v3[k].shape == v2[k].shape for k in v3)
    cfg3, cfg2 = (json.load(open(os.path.join(r, "vae", "config.json"))) for r in (root, root2))
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
    arch = {k: v for k, v in strip(cfg3).items() if k not in ("torch_dtype", "dtype")}
    same_arch = arch == {k: v for k, v in strip(cfg2).items() if k not in ("torch_dtype", "dtype")}
    note("3 vae", shapes, f"{len(v3)} tensors ({sum(v.numel() for v in v3.values()):,} parameters, BatchNorm running stats "
         f"included), the same names and shapes as FLUX.2's: {shapes}; tensors that differ {len(vdiff)} {vdiff[:3]} (FLUX 3 "
         f"latents are encoded with its own VAE); configs equal up to the dtype {same_arch}")

    # 4 -----------------------------------------------------------------------------------------------------
    tok = load_tokenizer(root)
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(os.path.join(root, "text_encoder"))
        ref = lambda p: proc.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    except Exception as e:                                           # noqa: BLE001
        print(f"[check] Qwen3-VL processor not loadable ({type(e).__name__}: {e}); comparing with the tokenizer's template")
        ref = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    prompts = ["", "a red fox in a snowy forest", "  two spaces in front, a newline after\n", "unicode: café, 東京, emoji 🦊"]
    mism = [p for p in prompts if ref(p) != wrap_flux3(p)]
    empty = tuple(tok(wrap_flux3(""), add_special_tokens=False)["input_ids"])
    note("4 template", not mism and empty == F3_EMPTY_PROMPT_IDS and tok.pad_token_id == PAD_ID,
         f"{len(prompts) - len(mism)} / {len(prompts)} prompts identical {mism[:1]}; empty prompt {len(empty)} tokens {empty}; "
         f"pad token {tok.pad_token!r} = {tok.pad_token_id}")

    # 5 -----------------------------------------------------------------------------------------------------
    for d in a.chunks:
        files = sorted(glob.glob(os.path.join(d, "chunk_*.pt")))
        if not files:
            continue
        pick = files[:: max(1, len(files) // a.max_chunks)][: a.max_chunks]
        aes = [f for f in files if os.path.basename(f).startswith("chunk_aes")]
        pick += [f for f in aes[:: max(1, len(aes) // 4)][:4] if f not in pick]
        lens, wins, by_src = [], Counter(), {}
        for f in pick:
            ch = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
            caps, keys = ch["caption"], ch.get("key") or [""] * len(ch["caption"])
            enc = tok([wrap_flux3(str(c).strip()) for c in caps], add_special_tokens=False)["input_ids"]
            for k, ids in zip(keys, enc):
                n = len(ids)
                lens.append(n)
                wins[window(n)] += 1
                by_src.setdefault(str(k).split("/", 1)[0], []).append(n)
            del ch
        lens.sort()
        q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]
        dist = {w: f"{100 * c / len(lens):.1f}%" for w, c in sorted(wins.items())}
        per = {s: f"median {sorted(v)[len(v) // 2]}, max {max(v)}" for s, v in by_src.items()}
        mean_w = sum(w * c for w, c in wins.items()) / len(lens)
        note(f"5 windows {os.path.basename(d.rstrip('/'))}", True,
             f"{len(lens)} captions from {len(pick)} chunks: tokens (template included) median {q(0.5)}, p90 {q(0.9)}, "
             f"p99 {q(0.99)}, max {lens[-1]}; windows {dist}; mean window {mean_w:.0f}; per source {per}; "
             f"text states per caption at the mean window: {mean_w * 20480 * 2 / 2 ** 20:.1f} MiB bf16")

    ok = all(r["ok"] for r in REPORT.values())
    print(f"[check] {sum(r['ok'] for r in REPORT.values())} / {len(REPORT)} checks passed", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(REPORT, open(a.out, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
