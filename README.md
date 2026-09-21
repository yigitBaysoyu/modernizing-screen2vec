# screen2vec-revisited

Revisits *Screen2Vec* (Li et al., CHI 2021) with a frozen foundation model. Every RICO
screen — screenshot plus cleaned view-hierarchy JSON — is embedded by the released
`Qwen3-VL-Embedding-8B`, unchanged; small networks trained on top (a) reconstruct the
cleaned hierarchy from the embedding alone and (b) predict the next screen in an
interaction trace, the paper's Table 3 task.

- `SUMMARY.md` — what exists, what is finished, where the evidence lives (start here)
- `CLAUDE.md` — invariants and conventions for anyone editing the code
- `REPLICATION_AUDIT.md` — the Table 3 replication audit (trace-index bug, measurement
  problems in the original evaluation)
- `screen2vec-paper/` — the original paper (`screen2vec.txt`, HTML); `screen2vec-master/`
  — the original authors' code

## Setup

```powershell
.\.venv\Scripts\activate          # Windows dev box; analysis notebooks also run on macOS
pip install -r requirements.txt   # pulls the CUDA 12.4 torch index
python check_gpu.py
python -m unittest tests.function_tests -v   # from the repo root
```

RICO lives outside the repo at `../rico-data/`: `data/layouts/<id>.json|.jpg` (66,261
screens; global ids are not contiguous — they range over `0..72218` with gaps), `data/traces/filtered_traces/<app>/<trace>/`, and the
encoder at `models/Qwen3-VL-Embedding-8B`. Nothing large is committed.

## Pipeline

```
raw RICO ──utils.py──▶ cleaned JSON (+ screenshot)
   │
   ├─ 0_build_trace_index.py      trace ↔ layout id join (required by every trace stage)
   ├─ 1_qwen_generate_embeddings  frozen Qwen3-VL-Embedding-8B → 1 or N pooled 4096-d vectors
   │
   ├─ Reconstruction:  2_train_adapter ──▶ 3_json_reconstruction[_par] ──▶ repair_rescore
   └─ Sequence:        9_train_joint ──▶ 6_generate_app_context_embeddings ──▶ 4_evaluate_screen_prediction
                        (dimension sweep: 4_sweep_embedding_dimensions wraps the three)
```

**Cleaning** (`utils.py::process_json_file`): invisible/zero-area nodes dropped, inert
layout scaffolding unwrapped, visible containers kept, bounds binned to 1–100. The
minified string is both encoder input and reconstruction target — **changing the cleaner
invalidates every embedding, adapter and observability list.**

### Embeddings — several shapes in one pass

```bash
python 1_qwen_generate_embeddings.py job.json --pool-tokens 2 4 8 16 32 64 --pooling avg
# -> embeddings_2x4096_avg/ ... embeddings_64x4096_avg/   (embeddings_4096_last/ for N=1)
```

All widths are pooled from the same forward pass. They are *not* derivable from one
another afterwards: `adaptive_avg_pool1d` places bucket borders at `floor(i*L/N)` /
`ceil((i+1)*L/N)`, which only coincide between two N when the sequence length divides
evenly, and every bucket is L2-normalised separately. `--skip-existing` resumes a run
that hit its wall time. On the cluster, set `POOL_TOKENS` in `hpc/_hpc1_start_embedding.py`.

### Reconstruction

Training runs through accelerate (`2_train_adapter_amplitude.sh` / `_midas.sh`) or the
Slurm generators in `hpc/`: `_hpc5_start_adapter_and_reconstruction.py` (train →
eight-shard reconstruction → audit), `_hpc6_resume_*` (recover partial reconstruction
sets; `--only qwen3-1.7b-base` selects one), `_hpc7_qwen3_4b_64x4096_all_in_one.py`
(the 64×4096 run; excludes existing same-name directories so a rerun cannot reuse an old
checkpoint). All use `--dry-run` first. Single-screen reconstruction:

```bash
python 3_json_reconstruction.py --projector_model <dir>/best_projector.pth \
    --lora_model <dir>/best_lora_adapter/ --index 3 --embeddings_dir embeddings_4096_last/
```

Reports are written strictly at generation time (`<idx>_report.txt`); the repaired
columns come from `repair_rescore.py` and are never mixed with strict ones.

### Screen observability — whitelist / blacklist

Some screens are drawn by components the hierarchy cannot see (WebView, video/GL,
games, ads, maps): the cleaned JSON holds one opaque box where the screenshot holds a
page. `screen_observability.py` scores every layout by opaque coverage and blacklists at
≥ 0.5 or an empty cleaned screen (~20 s; writes `screen_observability.csv` + `.meta.json`
with the cleaner hash). `_results_screen_observability.ipynb` exports
`graphics/screen_observability/{whitelist,blacklist}_ids.txt` (55,220 / 11,041).

### Sequence prediction (Table 3)

```bash
python 0_build_trace_index.py                       # once, ~7 min
python 9_train_joint.py --proj_dim 768 [--strict | --nback K]
python 6_generate_app_context_embeddings.py --embeddings_dir embeddings_seq_768 --output_dir embeddings_seq_768_with_app_desc
python 4_evaluate_screen_prediction.py --embeddings_dir embeddings_seq_768_with_app_desc --lstm_model lstm_predictor_768d.pth [--strict | --nback K]
```

`_run_table3_pipeline2.ps1` runs this joint recipe end to end (the two-stage
`_pipeline1` variant via `5_` + `8_` is kept for reference). `--strict` reproduces
Screen2Vec's own evaluation population; `--nback K` predicts from exactly K preceding
screens; the two are mutually exclusive and their numbers are never comparable.

### Dimension sweep

```bash
# The thesis sweep: strict is the default protocol (Screen2Vec's own population).
# D = 768 / seed 42 at the recipe width REUSES the canonical strict run if its checkpoints exist.
python 4_sweep_embedding_dimensions.py run --hidden_dim 1024 --dims 16,64,256,768 --seeds 42,43,44 --gpu 0
python 4_sweep_embedding_dimensions.py run --dims 768 --dry_run                 # print the commands
python 4_sweep_embedding_dimensions.py plot --protocol permissive --hidden_dim 4096   # regenerate outputs
# No-app arm for already recorded runs whose checkpoints still exist (evaluation only):
python 4_sweep_embedding_dimensions.py noapp --protocol permissive --hidden_dim 1024 --gpu 0
```

Runs the joint recipe once per projection dimension D (and seed) and collects Top-1, Top-k%,
MRR and nRMSE into `dimension_sweep/results{_proto}{_hH}.json`, with a table, CSV and figure.
Each point is evaluated twice: with the app-description half (the Table 3 setting) and on the
**screen vector alone** (`no_app`), because with the app half Top-1 is largely a choice among
one app's screens. Non-default widths and seeds get `_h<H>` / `_s<S>` artefact tags and never
overwrite the canonical checkpoints; `--force` refuses to retrain the canonical point
(D = 768, recipe width, seed 42) unless `--force_canonical` is given.

**Which sweep is which.** `results_strict.json` (hidden 1024; D = 16, 64, 256, 768; 3 seeds) is
the thesis figure and the only one compared with the paper. `results.json` and
`results_h4096.json` are **permissive** sweeps (hidden 1024 and 4096, 3 seeds): valid within
their protocol — curve shape, hidden-width robustness — but their paper row is orientation
only. Seed SD is the resolution of every sweep: smaller differences are "no detectable
trend", not equality. `_run_dimension_sweep.sh` (Linux/Midas) and `_run_dimension_sweep.ps1`
run the strict sweep, the no-app backfill and the notebook.

## Results

The `_results_*.ipynb` notebooks are the record of every experiment (map and headline
numbers in `SUMMARY.md`). Reading rules: Top-1 and MRR are the informative retrieval
columns; Top-0.1% and above are saturated by the app-description half; normalised RMSE
is not comparable to the paper's; strict and repaired reconstruction scores are reported
separately. See `REPLICATION_AUDIT.md` §§2–3 for why.
