#!/usr/bin/env bash
# Every BudgetLatent test, in order, on one GPU with >= 48 GB and the released weights in the Hugging Face cache
# (scripts/fetch_weights.py, with --family flux3 for FLUX 3); stops at the first failure.
#   FAMILIES   which families' checks run (default "flux2 flux3")
#   CKPT       optional: BudgetLatent FLUX.2 weights (a file or a Hub repo id) for tests/sanity.py; the released model otherwise
#   OUT        where the outputs of the sanity checks go (runs/sanity)
#   SANITY_ARGS / SANITY3_ARGS   more arguments for tests/sanity.py / tests/sanity_flux3.py, e.g. SANITY_ARGS="--image photo.jpg"
#              (the VAE round trip of tests/sanity.py needs a photograph and is skipped without one)
#   bash tests/run_gpu.sh
# tests/reference_flux3.py (the official FLUX 3 code in an environment of its own) is optional and not run here.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=${OUT:-runs/sanity}
FAMILIES=${FAMILIES:-"flux2 flux3"}
step() { echo; echo "=== $1"; }
step "the CPU tests"; bash tests/run_cpu.sh
if [[ " $FAMILIES " == *" flux2 "* ]]; then
  step "FLUX.2 with the released weights: trunk, sampler, text encoder, VAE, sample images"
  $PY tests/sanity.py --out "$OUT/flux2" ${CKPT:+--ckpt "$CKPT"} ${SANITY_ARGS:-}
fi
if [[ " $FAMILIES " == *" flux3 "* ]]; then
  step "FLUX 3: the released files against this code (CPU)"; $PY tests/check_flux3_release.py
  step "FLUX 3 with the released weights: text features, dense parity, samples at budgets"
  $PY tests/sanity_flux3.py --out "$OUT/flux3" ${SANITY3_ARGS:-}
fi
echo; echo "all BudgetLatent tests passed"
