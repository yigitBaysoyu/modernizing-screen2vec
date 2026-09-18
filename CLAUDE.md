# CLAUDE.md

Guidance for Claude Code when working in this repository. Orientation and current state: `SUMMARY.md`. Table 3 replication audit:
`REPLICATION_AUDIT.md`. Pipeline recipes: `README.md`.

## What this is

A research repo that revisits the CHI'21 paper *Screen2Vec* with a frozen foundation
model. RICO screens (screenshot + cleaned view hierarchy) are embedded with the released
`Qwen3-VL-Embedding-8B`, unchanged, and small task networks are trained on top:

1. **Reconstruction** — LLaVA-style projector + LoRA decoder decodes an embedding back
   into the cleaned view-hierarchy JSON (new task; no counterpart in the paper).
2. **Screen-sequence prediction** (paper Table 3) — joint projector + LSTM, evaluated as
   nearest-neighbour retrieval of the next screen under Screen2Vec's population
   (`--strict`) or an n-back protocol.
3. **Component prediction** (paper Table 2, `10_`–`13_`) — unfinished and retired; not
   part of the thesis.

**Evidence freeze (2026-09):** no further training is executed. The only pending run is
the dimension sweep (`4_sweep_embedding_dimensions.py`). Everything else is recorded in
the `_results_*` notebooks — see the notebook map in `SUMMARY.md`.

## Environment and data

- Windows dev box with `.venv/` (`.\.venv\Scripts\activate`; `pip install -r
  requirements.txt` pulls the CUDA 12.4 torch index). Analysis notebooks also run on
  macOS/CPU. `python check_gpu.py` verifies CUDA.
- **RICO lives outside the repo** at `../rico-data/`: `data/layouts/<id>.json|.jpg`
  (66,261 screens, global ids `0..66260`), `data/traces/filtered_traces/<app>/<trace>/`,
  and `models/Qwen3-VL-Embedding-8B`. `data/rico-examples/` holds test screens.
- Nothing large is committed: `*.pth`, `embeddings*/`, `adapter_model_out*/`,
  `reconstructions*/`, `dimension_sweep/`, images. Experiment artefacts live on the
  machines that produced them; the notebooks' recorded outputs are the record.

## Running things

Tests, from the repo root (fixtures use relative paths):

```bash
python -m unittest tests.function_tests -v
```

No linter or formatter is configured. Local drivers: `_run_table3_pipeline2.ps1` (joint
sequence model, canonical), `_run_table3_pipeline1.ps1` (two-stage variant),
`_run_table2_pipeline.ps1` (retired Track C), `_run_dimension_sweep.ps1` (sweep).
Multi-GPU adapter training goes through accelerate (`2_train_adapter_amplitude.sh`,
`2_train_adapter_midas.sh`), not `python` directly. `hpc/_hpc*.py` generate Slurm
submissions from the dev box; `_hpc5` is train → sharded reconstruction → audit,
`_hpc6_*` resume partial reconstruction sets, `_hpc7_*` is the all-in-one 64×4096 run.
Their top-of-file settings blocks are the only knobs meant to be edited.

## Architecture and invariants

**`utils.py` is the schema.** `process_json_file()` / `clean_screen()` turn raw RICO
JSON into `{"app", "activity", "components"}`: invisible/zero-area nodes dropped, inert
scaffolding (`DecorView`, most `*Layout`, `ScrollView`, `ViewPager`) unwrapped, visible
containers kept as grouping nodes, bounds binned to 1–100. This string is both the
encoder input and the reconstruction target, so **touching the cleaning logic
invalidates every embedding, every adapter, and the observability lists**. The cleaner's
sha256 is recorded in `screen_observability.meta.json`; a mismatch means the lists must
be regenerated.

**Two RICO id namespaces, never conflate them.** `layouts/<id>` ids are global and key
the embeddings; `gestures.json` keys are per-trace and restart in every trace.
`trace_index.py` + `0_build_trace_index.py` join them by SHA-1 of canonical JSON into
`trace_layout_index.csv` (injective; unmatched screens dropped and logged). Every
trace-consuming stage (`4_`, `4_sweep`, `5_`, `6_`, `8_`, `9_`) resolves keys through the
index and refuses to run without it. Never reintroduce raw `int(key)` lookups against
embeddings; `deprecated_files/7_evaluate_weight_sweep.py` still has them and must not be
used. Numbers from before the 2026-08-21 fix are void.

**Directory names encode the experiment, and the notebooks parse them.**
- `embeddings_4096_{last,avg}` for N=1, `embeddings_{N}x4096_avg` for N>1
  (`1_qwen_generate_embeddings.embeddings_dir_for()`).
- `adapter_model_out_{embeddings_dir}_full_{model_slug}`, `model_slug` from
  `2_train_adapter.model_output_slug()`; not double-suffixed if already present.
- `reconstructions_{embeddings}__via__{adapter_dir_without_prefix}`; inside:
  `<idx>_reconstructed.json` (raw prediction) and `<idx>_report.txt` (strict score).
- Sequence artefacts: `embeddings_seq_{D}{proto}{tag}/`, `lstm_predictor_{D}d{proto}{tag}.pth`;
  `{proto}` ∈ {"", `_strict`, `_nback{K}`}; `{tag}` = `_h<H>` / `_s<S>` for non-default
  hidden width or seed (`9_train_joint.py --artifact_tag`), so sweep repeats never
  overwrite canonical checkpoints.

**Pooling widths are not derivable from each other.** `adaptive_avg_pool1d` bucket
borders depend on the valid sequence length and each bucket is L2-normalised, so a sweep
passes several `--pool-tokens` to one run (one forward pass). `--skip-existing` resumes.

**Scoring is reported twice.** `reconstruction_metrics.py` gives the *strict* score at
generation time (unparseable = zero); `repair_rescore.py` re-reads raw predictions, runs
`json_repair` and loop collapsing, and rescores with the same functions for the
*repaired* columns. Never mix the two. `repair_rescore.py` is a module (not a notebook
cell) because Windows `ProcessPoolExecutor` needs importable worker functions.

**Screen observability lists.** `screen_observability.py` → `screen_observability.csv`
+ `.meta.json`; the notebook exports `graphics/screen_observability/{whitelist,blacklist}_ids.txt`
(threshold 0.5 opaque coverage; 55,220 / 11,041 at the recorded cleaner hash). Any stage
restricted to a list must cite which list and which meta hash.

**Evaluation protocols are mutually exclusive flags.** `4_evaluate_screen_prediction.py`:
permissive default (superseded), `--strict` (Screen2Vec's own population, `int()`
thresholds), `--nback K`. Numbers from different protocols are never comparable and
must be labelled.

**`--fast` in `2_train_adapter.py` is off by default on purpose.** The default path is
bit-for-bit the older code; `--fast` is arithmetically equivalent (see `FAST_PATH_NOTES`).
Preserve that when editing the training loop.

`3_json_reconstruction.py` pins `CUDA_VISIBLE_DEVICES` from `utils.CUDA_VISIBLE_DEVICE`
(a dev-box choice); the parallel variant sets the device per worker before touching
CUDA — don't copy the module-level pin into it.

## Conventions

- Notebooks: `_results_*` are thesis evidence; `_paper_*`, `_category_*`, `_pixel*`,
  `_hybrid_ui*` are CHI-paper artefacts (reference only). New evidence notebooks take
  the `_results_` prefix.
- Retired code and documents go to `deprecated_files/`; never delete, never import
  from there.
- The reporting rules (which columns are informative, strict vs. repaired, void numbers)
  are in `SUMMARY.md` and apply to every table or figure produced from this repo.
