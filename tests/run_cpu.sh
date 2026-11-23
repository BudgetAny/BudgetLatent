#!/usr/bin/env bash
# The BudgetLatent tests that need no GPU and no weights, in order; stops at the first failure.
#   bash tests/run_cpu.sh
#   FLUX2_ROOT=/path/to/black-forest-labs/flux2 bash tests/run_cpu.sh              # also compares with BFL's FLUX.2 model.py
#   FLUX3_ROOT=/path/to/black-forest-labs/flux-3-image-klein bash tests/run_cpu.sh # ... and with BFL's FLUX 3 dit.py
# The checks with the released weights are tests/run_gpu.sh.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
step() { echo; echo "=== $1"; }
step "1/11 FLUX.2 and the shared parts: layouts, the trunk against its references, the model, conversions, stage 2"; $PY tests/test_units.py
step "2/11 FLUX 3: the trunk against its references, text windows, the model, sizes, the schedule"; $PY tests/test_units_flux3.py
step "3/11 FLUX.2 [klein] 4B (step-distilled): the family table, metadata detection, its 4-step schedule, the no-guidance sampler"; $PY tests/test_units_flux2_distilled.py
step "4/11 FLUX 3 Image Klein (step-distilled, a flow map): the family table, t_ref through the trunk and the model, conversion, detection, its 16-step schedule, the no-guidance sampler"; $PY tests/test_units_flux3_distilled.py
step "5/11 everything the grid size touches, every family, at 512 x 512 (32 x 32 cells)"; BP_GRID=32 $PY tests/test_grid.py
step "6/11 ... and at 1024 x 1024 (64 x 64 cells)"; BP_GRID=64 $PY tests/test_grid.py
step "7/11 the stage-1 trainer end to end on tiny FLUX.2 and FLUX 3 trunks"; BP_GRID=32 $PY tests/test_train_smoke.py
step "8/11 every script imports and parses its arguments"; $PY tests/test_scripts.py
step "9/11 the fast path: fused FLUX 3 projections, the matmul library per shape, FLUX 3 Image Klein Base's text-mode blocks once per sample"; $PY tests/test_units_fast.py
step "10/11 reference images (editing, multi-reference): both trunks against their references, the budget models, the sampler, the --ref files"; $PY tests/test_units_refs.py
step "11/11 the package imports without a GPU"; $PY -c "import budgetlatent.inference.sampling, budgetlatent.training.stage1, budgetlatent.models.weights; print('imports ok')"
echo; echo "all BudgetLatent CPU tests passed"
