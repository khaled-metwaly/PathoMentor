#!/usr/bin/env bash
# Train both arms x seeds 0-4 at one FOV, one process per run, then run the statistics.
# usage: scripts/run_all_seeds.sh <fov> <data_root> <teacher_cache.npy> [out_dir] [seeds...]
set -euo pipefail
FOV=${1:?fov (448 or 256)}; DATA=${2:?data root}; TEACHER=${3:?teacher cache .npy}; OUT=${4:-runs}
shift $(( $# < 4 ? $# : 4 )) || true
SEEDS=${*:-"0 1 2 3 4"}
for SEED in $SEEDS; do
  for ARM in baseline mentor; do
    if [ -f "$OUT/${ARM}_fov${FOV}_seed${SEED}/${ARM}_fov${FOV}_seed${SEED}_results.json" ]; then
      echo "skip ${ARM} fov${FOV} seed${SEED} (done)"; continue; fi
    EXTRA=(); [ "$ARM" = mentor ] && EXTRA=(--teacher-cache "$TEACHER")
    python -m pathomentor.train --arm "$ARM" --fov "$FOV" --seed "$SEED" --data-root "$DATA" --out-dir "$OUT" "${EXTRA[@]}"
  done
done
python -m pathomentor.compare --runs "$OUT" --out-dir "$OUT/compare"
