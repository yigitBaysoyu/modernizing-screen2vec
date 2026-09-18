# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research repo that revisits the CHI'21 paper *Screen2Vec* using Qwen foundation
models. RICO Android screens (screenshot + view hierarchy) are embedded with
`Qwen3-VL-Embedding-8B`, and small task networks are trained on top for two
downstream tasks:

1. **Screen-sequence prediction** (paper Table 3) — projector + LSTM, evaluated as
   nearest-neighbour retrieval of the next screen in an interaction trace.
2. **JSON UI reconstruction** — a LLaVA-style projector + LoRA decoder that decodes
   an embedding back into the cleaned view-hierarchy JSON. This task is *new*; the
   original paper has no comparable benchmark.

A third line (`10_`–`13_`) reproduces the paper's component-prediction task (Table 2).

Read `SUMMARY.md` for the full technical picture and the current defensible result
claims, and `REPLICATION_AUDIT.md` for the trace-indexing bug and the four measurement
problems in the paper's Table 3. `README.md` holds the user-facing pipeline recipes.
`screen2vec-master/` and `screen2vec-paper/` are the original authors' code and paper.

## Environment and data

- Windows dev box with `.venv/`; activate with `.\.venv\Scripts\activate`.
  Install: `pip install -r requirements.txt` (pulls the CUDA 12.4 torch index).
- `python check_gpu.py` verifies the CUDA setup.
- **RICO lives outside the repo** at `../rico-data/`:
  - `../rico-data/data/layouts/<id>.json|.jpg` — 66,261 screens, global ids `0..66260`
  - `../rico-data/data/traces/filtered_traces/<app>/<trace>/` — interaction traces
  - `../rico-data/models/Qwen3-VL-Embedding-8B` — the encoder, loaded from disk
- `data/rico-examples/` holds a handful of screens used by the tests.
- Nothing large is committed: `*.pth`, `embeddings*/`, `adapter_model_out*/`,
  `reconstructions*/`, `*.png`, `*.jpg` are gitignored. The many `embeddings_*` and
  `adapter_model_out_*` directories in the working tree are experiment artefacts.

## Running things

Tests (run from the repo root — the fixtures are relative paths):

```bash
python -m unittest tests.function_tests -v
python -m unittest tests.function_tests.TestUtils.test_clean_screen_shape   # single test
```

There is no linter or formatter configured.

Local end-to-end pipelines are the PowerShell drivers: `_run_table2_pipeline.ps1`
(component prediction), `_run_table3_pipeline1.ps1` (two-stage sequence model),
`_run_table3_pipeline2.ps1` (joint sequence model, preferred).

Multi-GPU adapter training goes through accelerate, not `python` directly:

```bash
./2_train_adapter_amplitude.sh --embeddings_dir embeddings_4x4096_avg   # 8 GPUs
./2_train_adapter_midas.sh    --embeddings_dir embeddings_4x4096_avg    # 2 GPUs
```

`hpc/_hpc*.py` are Slurm submit-script *generators* run from the dev box; they copy
the project to scratch and submit chained jobs. `_hpc5_start_adapter_and_reconstruction.py`
is the current entry point (train → sharded reconstruction → aggregate + mail) and
supersedes the `_hpc2` → download/edit → `_hpc3` dance. Their top-of-file settings
blocks (`LM_MODEL_ID`, `EMBEDDINGS`, `POOL_TOKENS`) are the only knobs meant to be edited.

## Architecture and invariants

**`utils.py` is the schema.** `process_json_file()` / `clean_screen()` turn raw RICO
JSON into `{"app", "activity", "components"}`: invisible/zero-area nodes dropped,
inert scaffolding (`DecorView`, most `*Layout`, `ScrollView`, `ViewPager`) unwrapped,
bounds binned to integers 1–100 of the device frame. This string is both the encoder
input and the reconstruction target, so **touching the cleaning logic invalidates every
embedding and every trained adapter** — they must be regenerated. `load_embeddings()`
is the shared loader; it reads whatever shape is in the `.npz` and searches
`../rico-data/data/layouts`, `data/rico-examples`, `rico/examples` for the raw files.

**Two RICO id namespaces, never conflate them.** `layouts/<id>.json` ids are global
(`0..66260`) and key the embeddings; `gestures.json` keys are per-trace and restart in
every trace. Treating them as one namespace was the bug that produced the now-obsolete
0.5368 Top-1 figure. `trace_index.py` + `0_build_trace_index.py` join them by SHA-1 of
canonical JSON (`sort_keys=True, separators=(",",":")`) into `trace_layout_index.csv`;
`activity.root` is a fallback, unmatched screens are dropped and logged, and assignment
is injective. Every trace-consuming stage (`4_`, `5_`, `6_`, `8_`, `9_`) resolves keys
through the index and refuses to run without it. Never reintroduce `int(trace_key)`
lookups against embeddings.

**Directory names encode the experiment, and the notebooks parse them.** Keep the
conventions:

- `embeddings_4096_{last,avg}` for N=1, `embeddings_{N}x4096_avg` for N>1
  (`1_qwen_generate_embeddings.embeddings_dir_for()`).
- `adapter_model_out_{embeddings_dir}_full_{model_slug}` where `model_slug` is
  `2_train_adapter.model_output_slug(lm_model_id)` — `main()` appends it if missing,
  so an `--output_dir` that already ends in the slug is not double-suffixed.
- `reconstructions_{embeddings}__via__{adapter_dir_without_prefix}`;
  `_results_reconstruction.ipynb` globs `reconstructions*/*` and uses everything after
  `reconstructions_` as its `dim` key.

Inside a reconstruction directory: `<idx>_reconstructed.json` (raw prediction) and
`<idx>_report.txt` (strict score) per screen.

**Pooling widths are not derivable from each other.** `adaptive_avg_pool1d` bucket
borders depend on the valid sequence length and each bucket is L2-normalized
separately, so a sweep must pass several `--pool-tokens` to one run (they share one
forward pass) rather than deriving 4× from 8×. `--skip-existing` resumes a walltime-
killed run.

**Scoring is reported twice.** `reconstruction_metrics.py` produces the *strict* score
(unparseable prediction = zero components, zero class accuracy, zero IoU) at generation
time; `repair_rescore.py` re-reads the raw predictions, runs `json_repair`, and rescores
with the same functions for the "repaired" column. Keep the two separate — repaired
validity is an operational number, not native generation quality. `repair_rescore.py` is
a module rather than a notebook cell because Windows `ProcessPoolExecutor` can only
unpickle importable worker functions.

**Evaluation protocols are mutually exclusive flags.** `4_evaluate_screen_prediction.py`
defaults to a permissive population; `--strict` reproduces Screen2Vec's own (traces with
≥ n_predictors+1 screens, exactly n_predictors context screens, restricted candidate
pool, `int()`-truncated top-k% thresholds); `--nback K` slides a fixed K-screen context.
Numbers from different protocols are not comparable and must be labelled as such.

**`--fast` in `2_train_adapter.py` is off by default on purpose.** The default path is
bit-for-bit the older code so earlier sweeps reproduce exactly; `--fast` is documented as
arithmetically equivalent in the `FAST_PATH_NOTES` block under `parse_args()`. Preserve
that property when editing the training loop.

`3_json_reconstruction.py` pins `CUDA_VISIBLE_DEVICES` from `utils.CUDA_VISIBLE_DEVICE`
(currently `"3"`, a dev-box GPU choice). The parallel variant sets the device per worker
*before* touching CUDA instead — don't copy the module-level pin into it.

## Reporting results

`SUMMARY.md` records which claims the evidence supports. In short: Top-1 is the only
meaningful paper-compatible retrieval metric (the concatenated app-description half makes
Top-0.01%…5% essentially measure app identity — an app-only oracle already hits 0.9968 at
Top-0.1%); normalized RMSE is not comparable to the paper's (they evaluate unnormalized
vectors); the original's within-trace order is `os.listdir()`-dependent and cannot be
reproduced; and reconstruction has no original baseline to beat.
