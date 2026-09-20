#!/usr/bin/env bash
# Dimension sweep driver for Linux hosts (Midas). Run from the repository root with the
# project's Python environment active:
#
#     nohup ./_run_dimension_sweep.sh 0 > dimension_sweep_run.log 2>&1 &     # argument = GPU id
#
# Steps (all resumable; completed (dimension, seed) pairs are skipped on re-run):
#   1. Preflight: report which checkpoints exist on THIS machine.
#   2. The thesis sweep: strict protocol, recipe width 1024, D = 16/64/256/768, seeds 42-44.
#      Each point is evaluated with the app half and on the screen vector alone (no_app).
#      D = 768 / seed 42 has the canonical artefact names: it is REUSED if
#      embeddings_seq_768_strict/ and lstm_predictor_768d_strict.pth exist here, otherwise
#      it is trained and written under those names. Never pass --force.
#   3. No-app backfill for the permissive sweeps already recorded (evaluation only). Only
#      runs whose checkpoints still exist qualify, i.e. normally the seed-42 points.
#   4. Re-execute the results notebook, if jupyter is available.
set -euo pipefail
GPU="${1:-0}"
PY="${PYTHON:-python}"

echo "== 1/4 preflight (GPU ${GPU}) =="
"$PY" - <<'PYEOF'
import glob, os
canon = ["embeddings_seq_768_strict", "lstm_predictor_768d_strict.pth"]
have = [os.path.exists(p) for p in canon]
if all(have):
    print("canonical strict checkpoints FOUND  -> D=768/seed 42 reuses the run behind the headline numbers")
else:
    print("canonical strict checkpoints MISSING -> D=768/seed 42 will be TRAINED here and written as", canon)
    print("   (it then differs slightly from the recorded headline run; trained_in_sweep=true marks it)")
for label, pat in (("permissive h1024", "lstm_predictor_*d.pth"), ("permissive h4096", "lstm_predictor_*d_h4096.pth")):
    print(f"{label}: {len(glob.glob(pat))} seed-42 checkpoint(s) available for the no-app backfill")
for need in ("trace_layout_index.csv", "embeddings_4096_last"):
    print(("ok      " if os.path.exists(need) else "MISSING ") + need)
PYEOF
[ -e trace_layout_index.csv ] || { echo "trace_layout_index.csv missing: run 0_build_trace_index.py first"; exit 1; }

echo "== 2/4 strict sweep =="
"$PY" 4_sweep_embedding_dimensions.py run --protocol strict --hidden_dim 1024 \
      --dims 16,64,256,768 --seeds 42,43,44 --gpu "$GPU"

echo "== 3/4 no-app backfill for the recorded permissive sweeps =="
"$PY" 4_sweep_embedding_dimensions.py noapp --protocol permissive --hidden_dim 1024 --gpu "$GPU"
"$PY" 4_sweep_embedding_dimensions.py noapp --protocol permissive --hidden_dim 4096 --gpu "$GPU"

echo "== 4/4 notebook =="
if "$PY" -m jupyter --version >/dev/null 2>&1; then
    "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
          --ExecutePreprocessor.timeout=1200 _results_dimension_sweep.ipynb
else
    echo "jupyter not available here: execute _results_dimension_sweep.ipynb elsewhere after pulling the results"
fi
echo "== done: commit dimension_sweep/*.json, *.csv, *.txt and the notebook =="
