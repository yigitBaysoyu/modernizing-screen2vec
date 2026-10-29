# Thesis copy of screen2vec-revisited

This branch (`thesis-copy`) is a pruned copy of the `dev` branch at commit `cf61fc3`
(2026-10-29), made for writing and auditing the bachelor's thesis "Modernizing Screen2Vec".
It keeps only the files the thesis cites or runs. Every kept file is byte-identical to the
same path at `cf61fc3`, so file:line citations in the thesis's audits remain valid. The
evidence tag `results-freeze` is included.

The thesis repository expects this folder at `../screen2vec-revisited` relative to itself
(its CLAUDE.md, audit skills and code-verifier agent hardcode that path). On a new machine,
place or symlink this copy under that name.

## Removed in the prune commit

Nothing that the thesis text, its audits, or its results manifest references was removed.
The orientation documents (`README.md`, `SUMMARY.md`, `CLAUDE.md`, `REPLICATION_AUDIT.md`)
are unchanged and still mention some of these files.

- Component prediction (paper Table 2; retired, not part of the thesis):
  `10_extract_components.py`, `11_embed_component_vocab.py`, `12_train_component_predictor.py`,
  `13_evaluate_component_prediction.py`, `component_utils.py`, `_results_component_prediction.ipynb`,
  `_run_table2_pipeline.ps1`, `enrico_design_topics.csv`, `_category_embedding_projections*.ipynb`
  (three notebooks), `graphics/category_embedding_*` (two folders).
- Pixel experiments: `pixel.py`, `pixel_idle.py`, `_pixel.ipynb`, `run_pixel.ps1`.
- Paper figure notebooks and their exports: `_paper_*.ipynb` (five notebooks),
  `_hybrid_ui_reconstruction.ipynb`, `paper_embedding_analysis.py`, `graphics/paper_*` (five folders).
- Rendering helpers never cited: `render_layout.py`, `render_bbox_comparison.py`.
- Reference copies of the paper and a draft figure: `screen2vec-paper/`, `assets/`.
- Stray run logs: `generate_app_768_log.txt`, `lstm_train_768d_log.txt`, `projector_768d_log.txt`.
- `deprecated_files/` (excluded from thesis evidence by the audits; contains the void
  `7_evaluate_weight_sweep.py`).
- Environment helpers: `check_gpu.py`, `find_lora_targets.py`, `view_npz.py`,
  `transform_embeddings_pca64.py`.

## Kept on purpose

- The two-stage sequence variant (`5_train_sequence_projector.py`, `8_train_lstm_predictor.py`,
  `_run_table3_pipeline1.ps1`): not used for any thesis number, but the audited evaluator and
  joint trainer define their architectures by reference to these scripts.
- `hpc/` in full, including `mail.py`, because one audit cites `hpc/_hpc1_start_embedding.py`.
- `graphics/dimension_sweep`, `graphics/screen_observability` and the top-level `graphics/qwen3_*`
  exports of the frozen notebooks.
- `adapter_training_history/.gitkeep` (an audit records that the folder holds only this file).
- `data/rico-examples/` (used by `tests/function_tests.py` and `utils.py`).

## Not in git (same as the source repo)

RICO, the Qwen3-VL-Embedding-8B checkpoint, embeddings, adapters and reconstructions live
outside the repository; see `CLAUDE.md`. The thesis's own numbers come from the thesis
repository's `results/results.json`, not from this copy.

## Added from the run-machine archive (not in the source repo)

Four files the source repo's `.gitignore` excludes (`/graphics`), copied on 2026-10-29 from
`run-machine-data/archived-data/graphics/` and force-added here so the frozen notebooks can
read them under the paths they expect. SHA-256 of both `.gz` files matches the thesis's
`audits/evidence/run-machine/SHA256SUMS.txt`.

- `graphics/qwen3_repaired_boxplot_metrics.csv.gz` (+ `.meta.json`): the cache written by
  `_results_reconstructions_boxplots.ipynb`; 993,914 rows, one per screen and run for the 15
  Qwen3-Base runs (sha256 `6ce8d585…c109f7`).
- `graphics/qwen3_adapter_validation_ids.csv.gz` (+ `.meta.json`): the 66,261 screen ids with
  their adapter split (52,776 train, 13,192 validation, 290 excluded as too long, 3 excluded
  as shard tail) (sha256 `6af24670…13cd4`).
