# screen2vec-revisited: Technical Summary

Last regenerated: 2026-09-18, on branch `dev`. This is the repository's orientation
document: what the project is, what exists, what is finished, and where the evidence
lives. Operational invariants (ID namespaces, naming conventions, protocol flags) live in
`CLAUDE.md`; the Table 3 replication audit lives in `REPLICATION_AUDIT.md`.

## What this repository is

This repository revisits *Screen2Vec* (Li et al., CHI 2021) with a frozen foundation
model. Every RICO screen (screenshot + cleaned view-hierarchy JSON) is embedded by the
released **Qwen3-VL-Embedding-8B**, used unchanged; small task networks are trained on top
of the frozen vectors. Two experimental tracks are complete, one was attempted and
retired:

1. **Reconstruction** — a LLaVA-style adapter (MLP projector + LoRA-tuned decoder LM)
   decodes an embedding back into the cleaned view-hierarchy JSON. Reconstruction
   fidelity serves as an intrinsic measure of what the embedding retains. This task has
   no counterpart in the original paper.
2. **Screen-sequence prediction** (paper Table 3) — a jointly trained projector + LSTM
   predicts the next screen in an interaction trace; evaluation is nearest-neighbour
   retrieval under Screen2Vec's own population (`--strict`) and an n-back protocol.
3. **Component prediction** (paper Table 2) — attempted (`10_`–`13_`), never finished
   (the recorded run stops at epoch 11/120). **Out of the thesis**; kept as future work.

## Evidence freeze

**No further training will be executed.** The evidence base is what is recorded in the
`_results_*` notebooks, plus exactly one pending run: the embedding-dimension sweep for
sequence prediction (`4_sweep_embedding_dimensions.py`, running on a separate machine as
of 2026-09-17). When its results land in `_results_dimension_sweep.ipynb`, the evidence
is complete and a results manifest (`results.json`, extracted from the notebooks) becomes
the only citable source of numbers for the thesis.

## Notebook map

**Thesis evidence** (the record of the experiments; the manifest is extracted from these):

| Notebook | Contents |
|---|---|
| `_results_reconstruction.ipynb` | Reconstruction scores per configuration, strict and repaired, positional and order-free metrics |
| `_results_qwen3_correlation.ipynb` | The systematic 3×5 reconstruction grid — Qwen3-Base 0.6B/1.7B/4B × {4096_last, 4096_avg, 4×, 16×, 64×4096_avg} — rescored uniformly; primary metric: repaired unordered class F1; metric correlations |
| `_results_reconstructions_boxplots.ipynb` | Score distributions; **includes a validation-only analysis** that reproduces the adapter's held-out split and rescoring on it |
| `_results_reconstructions_viz.ipynb` | Wireframe renderings of best/median/worst reconstructions |
| `_results_screen_observability.ipynb` | The observability scan: whitelist/blacklist derivation, threshold sensitivity, covariates |
| `_results_sequence_prediction_s2v_strict.ipynb` | Table 3 under Screen2Vec's own population; no-app and with-app rows |
| `_results_sequence_prediction_nback{1,2,3}.ipynb` | N-back protocol, K = 1..3; no-app and with-app rows |
| `_results_dimension_sweep.ipynb` | Projection-dimension sweep D = 16…2048 (uniform hidden width 4096); **pending — awaiting the running sweep** |
| `_results_component_prediction.ipynb` | The unfinished Track C attempt; no results |
| `rico-metrics.ipynb` | Dataset statistics |

**Paper artifacts (Jonathan's)** — made for the CHI submission, not thesis evidence;
citable as reference material only: `_paper_*` (UMAP projections, cross-app nearest
neighbours), `_category_*` (category embedding projections), `_pixel.ipynb` (live-device
capture demo, with `pixel.py` / `pixel_idle.py` / `run_pixel.ps1`),
`_hybrid_ui_reconstruction.ipynb` (mixing two screens' embeddings and decoding the
mixture), `paper_embedding_analysis.py`.

## Data and preprocessing

RICO: 66,261 unique screens (global ids 0..66260) from 9,384 apps, plus filtered
interaction traces; lives outside the repo at `../rico-data/`. `utils.py` cleans each raw
hierarchy into `{"app", "activity", "components"}`: invisible and zero-area nodes
dropped, inert scaffolding unwrapped, visible containers kept as grouping nodes, bounds
binned to 1–100. The minified string is both the encoder input and the reconstruction
target; **touching the cleaner invalidates every embedding and adapter** (its sha256 is
recorded in `screen_observability.meta.json`).

Traces and layouts number screens in two different namespaces; they are joined by
content hash in `trace_layout_index.csv` (66,157 of 66,261 mapped, 99.84%; 104 dropped;
injective). Every trace-consuming stage refuses to run without the index. History and
evidence: `REPLICATION_AUDIT.md`.

## Screen observability (new since August)

`screen_observability.py` scans every screen for whether the cleaned hierarchy can
describe the screenshot at all. Screens dominated by components that render outside the
Android view system (WebView, video/GL surfaces, games, maps, ad containers) leave the
JSON holding an opaque box where the pixels hold a page. At the chosen threshold (0.5
opaque coverage): **whitelist 55,220 screens; blacklist 11,041** (10,872 opaque + 169
empty). Outputs and per-category analyses in `graphics/screen_observability/`;
reproducible via the recorded metadata. Whether the whitelist is the thesis's main
scoring population is a decision for the writing phase.

## Track A — Reconstruction

Pipeline: `1_qwen_generate_embeddings.py` (one forward pass per screen; pooling into 1
last-token or N average-pooled 4096-d vectors) → `2_train_adapter.py` (projector maps N
vectors to N prefix tokens, max 128; LoRA on the decoder; targets over 4,096 tokens
dropped, 0.44%) → `3_json_reconstruction[_par].py` (greedy decoding; strict scoring at
generation time) → `repair_rescore.py` (json-repair + loop collapse; repaired scoring,
kept separate from strict).

Two eras of runs exist:
- **The systematic grid** (current): Qwen3-Base decoders 0.6B / 1.7B / 4B × the five
  representations above, rescored uniformly in `_results_qwen3_correlation.ipynb`.
- **Earlier Qwen2.5/Qwen3.5-era runs** (recorded in `_results_reconstruction.ipynb`):
  1×/4×/8× full-corpus with Qwen2.5-1.5B and others, and a 128×4096 run scored on a
  ~50k subset. Kept for the capacity-vs-N story; the grid is the controlled comparison.

Scoring: JSON validity, sequence/Levenshtein similarity, app/activity match, positional
class accuracy and box IoU, and order-free class precision/recall/F1 plus matched IoU —
each strict and repaired. The boxplots notebook's validation-only section provides
held-out-only statistics. Which decoder, which population (whitelist vs. full corpus),
and which N-points form the thesis's headline curve are writing-phase decisions; the
candidate facts are all recorded.

## Track B — Screen-sequence prediction

Joint pipeline (`9_train_joint.py`): frozen 4096-d screen vectors → projector
(4096→1024→D, default D=768, L2-normalised) → LSTM, trained with InfoNCE-style negatives;
app-description vectors (`6_...py`: package name, name, category, scraped description for
4,155 of 9,384 apps; SBERT all-mpnet-base-v2) concatenated for ranking only. Protocols
are mutually exclusive and artefact-suffixed: permissive (superseded), `--strict`
(Screen2Vec's population), `--nback K`.

Recorded headline results (details and caveats in the notebooks and
`REPLICATION_AUDIT.md` §§2–3 — Top-1 and MRR are the informative columns):

| Protocol | Top-1 (with app) | MRR | Top-1 (no app) | Paper best |
|---|---:|---:|---:|---:|
| strict | 0.1182 | 0.2949 | 0.0697 | 0.061 |
| n-back 1 | 0.2793 | 0.4568 | 0.1780 | — |
| n-back 2 | 0.1650 | 0.3483 | 0.0890 | — |
| n-back 3 | 0.0972 | 0.2824 | 0.0483 | — |

**Pending:** the dimension sweep (D = 16…2048 at uniform hidden width 4096, seed-tagged
artefacts, strict/n-back aware) — the last run of the project.

## Reporting rules (carried over, still in force)

- Top-1 and MRR are the paper-comparable retrieval numbers; Top-0.01%–5% are saturated
  by the app-description half (app-only oracle: 0.9968 at Top-0.1%) and are reported,
  never headlined.
- Normalised RMSE is not comparable to the paper's (unit-norm vs. free-scale vectors)
  and is never claimed as a win.
- Strict and repaired reconstruction scores are never mixed; repaired validity is an
  operational number, not native generation quality.
- Numbers from before the 2026-08-21 trace-index fix are void.
- Reconstruction is currently scored corpus-wide including the adapter's training
  screens; the validation-only analysis exists and its use is a writing-phase decision.
