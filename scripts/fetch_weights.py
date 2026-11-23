"""Download the released files a family needs into the Hugging Face cache (and, optionally, BudgetLatent weights).

    python scripts/fetch_weights.py                    # FLUX.2 [klein] 4B Base, ~24 GB: BFL single file, diffusers transformer, Qwen3-4B, VAE
    python scripts/fetch_weights.py --family flux2_distilled   # FLUX.2 [klein] 4B (step-distilled), the same files of that release
    python scripts/fetch_weights.py --family flux3     # FLUX 3 Image Klein Base, ~32 GB (the fp8 / nvfp4 variants are skipped)
    python scripts/fetch_weights.py --family flux3_distilled   # FLUX 3 Image Klein (step-distilled), ~32 GB, the same files of that release
    python scripts/fetch_weights.py --family flux3 --budgetlatent      # + the BudgetLatent weights of the family

FLUX.2 [klein] 4B Base and 4B are public (Apache-2.0, not gated), so those downloads are anonymous.  FLUX 3 Image Klein Base and FLUX 3
Image Klein are gated under BFL's FLUX Kommunity License (non-commercial): accept it on each model page and log in
(`hf auth login`) first.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # the repository root

PATTERNS = {
    "flux2": ["flux-2-klein-base-4b.safetensors", "transformer/*", "text_encoder/*", "tokenizer/*", "vae/*",
              "scheduler/*", "model_index.json", "LICENSE.md"],
    "flux2_distilled": ["flux-2-klein-4b.safetensors", "transformer/*", "text_encoder/*", "tokenizer/*", "vae/*",
                        "scheduler/*", "model_index.json", "LICENSE.md"],
    "flux3": ["model.safetensors", "config.json", "manifest.json", "transformer/*", "text_encoder/*", "vae/*", "ae.safetensors",
              "modular_model_index.json", "README.md", "LICENSE.md", "NOTICE.md"],
}
PATTERNS["flux3_distilled"] = PATTERNS["flux3"]                  # the same layout (its variants/ are skipped too)
GATED = ("flux3", "flux3_distilled")


def size_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="flux2", choices=sorted(PATTERNS))
    ap.add_argument("--repo", default=None, help="default: the family's released repository")
    ap.add_argument("--skip_diffusers_transformer", action="store_true",
                    help="skip transformer/ (only the parity tests in tests/ read it, as the second reference)")
    ap.add_argument("--budgetlatent", action="store_true", help="also download the family's BudgetLatent weights")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download

    from budgetlatent.models.families import BUDGETLATENT_FILE, BUDGETLATENT_REPOS, FAMILIES
    repo = a.repo or FAMILIES[a.family].repo
    pats = [p for p in PATTERNS[a.family] if not (a.skip_diffusers_transformer and p.startswith("transformer/"))]
    path = snapshot_download(repo, allow_patterns=pats, token=None if a.family in GATED else False, max_workers=4)
    print(f"[fetch] {repo} -> {path}  ({size_gb(path):.2f} GB)", flush=True)
    if a.budgetlatent:
        f = hf_hub_download(BUDGETLATENT_REPOS[a.family], BUDGETLATENT_FILE)
        print(f"[fetch] {BUDGETLATENT_REPOS[a.family]} -> {f}  ({os.path.getsize(f) / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
