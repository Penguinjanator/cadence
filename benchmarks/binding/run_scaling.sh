#!/usr/bin/env bash
# The scaling evaluation of the recall memory (SCALING_PROTOCOL.md) on one box.
#
# From the repository root, with Cadence's dependencies installed for `python`:
#   bash benchmarks/binding/run_scaling.sh OUT [WORKERS] [GRID]
# OUT is a directory outside the repository; WORKERS defaults to the box's cores, GRID to
# `trimmed`. The run resumes: a second call skips every life whose receipt is in OUT/lives.
# With PROBE=1 the box first measures the grid's cost classes (OUT/costs.json) and the plan
# prints its own estimate before the lives start.
set -euo pipefail

OUT=${1:?usage: run_scaling.sh OUT [WORKERS] [GRID]}
WORKERS=${2:-$(getconf _NPROCESSORS_ONLN)}
GRID=${3:-trimmed}
PYTHON=${PYTHON:-python}

export PYTHONPATH=src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export CADENCE_FUSED=0

mkdir -p "$OUT"
COSTS=()
if [ "${PROBE:-0}" = 1 ]; then
  [ -f "$OUT/costs.json" ] || "$PYTHON" benchmarks/binding/scaling.py probe --grid "$GRID" \
    --out "$OUT/costs.json" --workers "$WORKERS"
  COSTS=(--costs "$OUT/costs.json")
fi
[ -f "$OUT/jobs.json" ] || "$PYTHON" benchmarks/binding/scaling.py plan --grid "$GRID" \
  --out "$OUT/jobs.json" ${COSTS[@]+"${COSTS[@]}"}
"$PYTHON" benchmarks/binding/scaling.py run --jobs "$OUT/jobs.json" --out "$OUT" \
  --workers "$WORKERS" 2>&1 | tee -a "$OUT/run.log" || echo "lives failed: see $OUT/errors"
"$PYTHON" benchmarks/binding/scaling.py summary --out "$OUT" > /dev/null
echo "summary: $OUT/summary.md"
