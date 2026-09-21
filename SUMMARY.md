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

**The evidence is complete (2026-09-21, tag `results-freeze`). No further runs are planned.**
The evidence base is what is recorded in the `_results_*` notebooks and in
`dimension_sweep/`. The last runs were the embedding-dimension sweeps: two under the
permissive protocol (2026-09-20; hidden 1024 and 4096, 3 seeds each) and the reduced
**strict** sweep (2026-09-21; hidden 1024; D = 16, 64, 256, 768; seeds 42–44), every point of
which is also evaluated on the screen vector alone (`no_app`). The thesis's results manifest
(`results.json`, extracted from these records) is the only citable source of numbers.

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
| `_results_dimension_sweep.ipynb` | Projection-dimension sweeps: permissive (hidden 1024, D = 16…1024; hidden 4096, D = 16…2048; 3 seeds) — within-protocol only, paper row is orientation; strict (hidden 1024, D = 16/64/256/768, 3 seeds) — the thesis figure, the only sweep compared with the paper. No-app arm: all 12 strict runs, and the 17 seed-42 permissive runs |
| `_results_component_prediction.ipynb` | The unfinished Track C attempt; no results |
| `rico-metrics.ipynb` | Dataset statistics |

**Paper artifacts (Jonathan's)** — made for the CHI submission, not thesis evidence;
citable as reference material only: `_paper_*` (UMAP projections, cross-app nearest
neighbours), `_category_*` (category embedding projections), `_pixel.ipynb` (live-device
capture demo, with `pixel.py` / `pixel_idle.py` / `run_pixel.ps1`),
`_hybrid_ui_reconstruction.ipynb` (mixing two screens' embeddings and decoding the
mixture), `paper_embedding_analysis.py`.

## Data and preprocessing

RICO: 66,261 unique screens (global ids within 0..72218, not contiguous) from 9,384 apps, plus filtered
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

**Dimension sweep.** Permissive, 3 seeds: mean Top-1 stays between 0.080 and 0.114 and MRR
between 0.270 and 0.295 from D = 16 to 2048, with seed SDs up to 0.029 — no detectable trend
within seed noise; the hidden width (1024 vs. 4096) makes no difference beyond that noise.
All 51 runs share one population (7,164 traces, 54,517 instances). These numbers are **not**
comparable with the paper's Table 3 or with the strict headline.

**Strict sweep** (3 seeds, trained on Midas at `92a3336`; mean ± sample SD):

| D | Top-1, with app | MRR, with app | Top-1, screen vector alone | MRR, screen vector alone |
|---:|---:|---:|---:|---:|
| 16 | 0.1084 ± 0.0164 | 0.2810 ± 0.0150 | 0.0245 ± 0.0094 | 0.0750 ± 0.0166 |
| 64 | 0.1368 ± 0.0389 | 0.3135 ± 0.0377 | 0.0752 ± 0.0248 | 0.1613 ± 0.0295 |
| 256 | 0.1866 ± 0.0159 | 0.3611 ± 0.0156 | 0.1275 ± 0.0195 | 0.2316 ± 0.0268 |
| 768 | 0.1318 ± 0.0279 | 0.3101 ± 0.0268 | 0.0834 ± 0.0232 | 0.1798 ± 0.0290 |

Every one of the 12 strict runs beats the paper's Top-1 of 0.061 with the app half (lowest
single run: 0.0898). The D = 768 seeds (0.1636, 0.1207, 0.1112) bracket the original single
run recorded in `_results_sequence_prediction_s2v_strict.ipynb` (0.1182), which was trained on
another machine; all three sweep seeds were trained fresh (`trained_in_sweep: true`). The
no-app arm shows what the app half hides: the screen vector alone depends strongly on D
(0.0245 at D = 16 vs. 0.1275 at D = 256). D = 256 is the best point in both arms; with three
seeds this is a finding to report, not a reason to move the headline off the recipe's D = 768.
The permissive no-app backfill (seed 42 only, single runs) shows the same collapse at small D.

One screen is missing on Midas, and only there: **layout id 2906** (a help dialog in
`com.peha.mapdistancemeter`) has no embedding in `embeddings_4096_last/`. Its cleaned JSON is
752,384 characters — one `TextView` carries a 735,440-character help text, against a typical
~1,900 characters per screen — and the 8B encoder ran out of memory on an 80 GB A100, twice
(2026-09-17). The screen is the first of six in its trace, so the trace still qualifies with
five screens and contributes one window fewer: Midas evaluates 34,139 strict instances
(pool 54,519) where the original machine evaluated 34,140 (pool 54,520), and 54,517
permissive instances instead of 54,518. Everything else is identical (trace index
byte-identical; no other screen skipped). The sweeps are therefore comparable with each other
exactly, and with the original notebooks up to this one instance. Investigated read-only on
2026-09-21; the embedding was deliberately not regenerated after the freeze. The same screen
is the likely reason some reconstruction runs record n = 66,260 rather than 66,261 (not
verified).

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
