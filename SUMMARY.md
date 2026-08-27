# screen2vec-revisited: Technical Repository Summary

## Plain-English overview

This repository revisits the 2021 CHI paper **Screen2Vec: Semantic Embedding of GUI Screens and GUI Components** using modern foundation models. The original Screen2Vec system built a numerical representation of an Android screen from its text, component types, layout, and app-store description, then learned from groups of screens occurring in the same RICO interaction trace. This project replaces much of that hand-built representation pipeline with representations extracted by Qwen vision-language/foundation models and trains smaller task-specific networks on top.

The project has two main goals:

1. **Interaction screen-sequence prediction:** given the screens already visited in a mobile interaction, predict which screen comes next. The predicted vector is compared with the corpus of RICO screen vectors, turning prediction into nearest-neighbor retrieval.
2. **JSON UI reconstruction:** compress a screenshot plus cleaned Android view hierarchy into an embedding, then decode that embedding back into a compact JSON description of the screen. The output should recover the screen identity, component types, text/metadata, hierarchy, and approximate component positions.

The repository contains a full experimental pipeline, local trained checkpoints and embeddings, result notebooks, the authors' original Screen2Vec source under `screen2vec-master/`, and a text/HTML copy of the original publication under `screen2vec-paper/`. The canonical encoder identified in `CLAUDE.md` is the locally stored **Qwen3-VL-Embedding-8B** model; the additional Qwen 2.5 and Qwen 3.5 directories are experiment variants.

## The central research idea

The original paper treats a GUI screen somewhat like a word: screens that appear in the same interaction context should receive related embeddings. Its architecture is explicitly hierarchical. At component level, Sentence-BERT text embeddings and learned Android class embeddings are trained through a CBOW-style component-prediction task. At screen level, component embeddings, a learned 64-dimensional layout embedding, and a Sentence-BERT app-description embedding are combined. The final screen representation has 1,536 dimensions: 768 learned screen dimensions concatenated with 768 app-description dimensions.

This repository keeps the useful self-supervised premise—interaction context provides supervision without manual labels—but replaces the narrow feature engineering with foundation-model representations of the complete cleaned UI. A Qwen model reads a multimodal prompt containing both the screenshot and compact view-hierarchy JSON. Hidden states from the foundation model become the screen embedding. Projectors, an LSTM, LoRA adapters, and language-model decoding then specialize those embeddings for the two downstream objectives.

In practical terms, the project asks whether a modern pretrained model can preserve a richer mixture of visual appearance, language, UI structure, and app semantics than the 2021 architecture, and whether that richer representation leads to more accurate next-screen retrieval and recoverable structured UI descriptions.

## Data and preprocessing

The work is built around RICO, which contains **66,261 unique Android screens from 9,384 apps** plus interaction traces. Each layout example normally has a screenshot (`.jpg`) and an Android view hierarchy (`.json`). The full dataset deliberately lives outside Git at `../rico-data/`: global layouts are under `data/layouts/`, filtered interaction traces under `data/traces/filtered_traces/`, and the local Qwen3-VL-Embedding-8B encoder under `models/`. Only a few small test examples live in `data/rico-examples/`.

`utils.py` converts raw RICO JSON into the stable compact object:

```json
{"app":"…","activity":"…","components":[…]}
```

The cleaner does more than remove whitespace:

- It extracts package and short activity names from `activity_name`.
- It discards invisible and zero-area nodes and prunes their subtrees.
- It unwraps inert structural scaffolding such as `DecorView`, most `*Layout` classes, `ScrollView`, and `ViewPager`, while retaining visually meaningful containers such as cards, toolbars, lists, recyclers, and grids.
- It removes empty bare `View`/`ViewGroup` spacer nodes.
- It retains useful fields in stable order: class, non-empty text, accessibility content description, non-generic resource ID, `clickable` when true, normalized bounds, and cleaned children.
- It maps pixel bounding boxes to integers from 1–100 relative to the device frame. This reduces token length and makes geometry resolution-independent.
- It emits minified JSON for embedding and generation.

Changing this preprocessing changes the meaning of the learned embedding space, so the README correctly notes that embeddings and adapter training must be regenerated whenever the cleaning logic changes.

## Foundation-model embedding pipeline

`1_qwen_generate_embeddings.py` is the main encoder stage. Its principal model is the local Qwen3-VL-Embedding-8B foundation model. It feeds the cleaned JSON and screenshot to the model and saves hidden-state representations as `.npz` files keyed by the global RICO layout ID. The repository also contains experiments with Qwen 2.5 VL and Qwen 3.5 variants, token pooling, and several representation sizes.

The pooling design is important. A screen can be represented by one 4,096-dimensional last-token or average-pooled vector, or by multiple pooled prefix/token buckets such as `4x4096`, `8x4096`, `16x4096`, up to `128x4096`. Each bucket is independently L2-normalized. Multiple pooling widths can be produced from one model forward pass, but cannot safely be derived from each other afterward because adaptive pooling boundaries depend on the valid sequence length.

This yields two useful kinds of representations:

- A compact single vector, convenient for sequence prediction.
- A sequence of soft-prefix vectors, which preserves more information for conditional JSON generation.

## Goal 1: interaction screen-sequence prediction

### Trace-to-screen indexing

The most important correctness work in this repository is `0_build_trace_index.py` and `trace_index.py`. RICO's layout corpus uses global IDs from 0 to 66,260, while trace view hierarchies use local per-trace IDs that restart in each trace. Earlier versions accidentally treated these as the same namespace. Because the integers existed in both places, the bug silently connected interaction traces to unrelated screens and produced an invalid Top-1 result of 0.5368.

The corrected join hashes canonicalized JSON content. The primary key is SHA-1 over the complete document serialized with sorted keys and compact separators; an `activity.root` hash is a limited fallback. Matching is injective, so two trace screens cannot silently point to one layout embedding. Recorded coverage is:

- 66,261 trace screens considered.
- 66,157 mapped (99.84%).
- 66,154 mapped through the full-document hash and 3 through the fallback.
- 104 dropped rather than guessed.
- Independent `activity_name` checks reached 1,200/1,200 agreement after the fix.

`trace_layout_index.csv`, its metadata JSON, and the dropped-screen audit make the mapping reproducible. Trace-consuming stages refuse to run without the index, preventing reintroduction of the silent aliasing bug.

### Learned sequence model

The main current route is `9_train_joint.py`. It jointly trains:

- A projector from the raw 4,096-dimensional Qwen screen representation through a 1,024-dimensional hidden layer to a normalized 768-dimensional screen vector.
- A unidirectional LSTM that consumes up to four previous screen vectors and predicts the next screen vector.
- A contrastive/InfoNCE-style objective using sampled negatives (recorded runs use temperature 0.07 and 128 negatives).

The default 768-dimensional choice gives a direct comparison point with the learned 768-dimensional screen half of Screen2Vec. Separate scripts support a two-stage variant: `5_train_sequence_projector.py` learns the contrastive projector, and `8_train_lstm_predictor.py` subsequently trains the LSTM.

`6_generate_app_context_embeddings.py` encodes Google Play descriptions using `sentence-transformers/all-mpnet-base-v2` and concatenates the normalized 768-dimensional app vector to the projected 768-dimensional screen vector. The result is 1,536-dimensional, matching the original paper's overall structure. Only the screen half is predicted by the LSTM; app context is added for retrieval. This avoids an avoidable norm mismatch and prevents the constant app vector from entering the recurrent prediction itself. The recorded metadata has full descriptions for 4,155 of 9,384 apps.

The `nback` variants explicitly measure how many prior screens are supplied. Their recorded corrected results show the closest previous screen is especially informative:

| Context | Prediction instances | Top-1 | Top-0.01% | Top-0.1% | Normalized RMSE |
|---|---:|---:|---:|---:|---:|
| Previous 1 screen | 55,881 | **0.2793** | 0.7713 | 0.9957 | 0.5626 |
| Previous 2 screens | 47,354 | 0.1650 | 0.6955 | 0.9931 | 0.5839 |
| Previous 3 screens | 40,190 | 0.0972 | 0.5811 | 0.9908 | 0.5710 |
| Up to 4 screens, permissive | 54,518 | 0.0891 | 0.6453 | 0.9856 | 0.5987 |
| Exactly 4, strict protocol | 34,140 | **0.1182** | 0.5004 | 0.9917 | 0.6028 |
| Original Screen2Vec best row | paper protocol | 0.0610 | 0.2580 | 0.9690 | 0.8530 |

The strict corrected run therefore raises headline Top-1 retrieval from **6.10% to 11.82%**, an absolute improvement of 5.72 points and roughly a **1.94× relative result**. The one-screen-context experiment reaches 27.93%, but it is a different prediction setting and should not be presented as an apples-to-apples Table 3 replacement.

### What the evaluation actually means

`4_evaluate_screen_prediction.py` performs nearest-neighbor retrieval against screen embeddings and reports Top-1, Top-0.01%, Top-0.1%, Top-1%, Top-5%, and normalized RMSE. The strict mode retains traces with at least five mapped screens, always uses exactly four context screens, restricts the candidate/negative pool to eligible trace screens, and uses the paper's integer truncation for percentage thresholds. The permissive mode admits traces with at least three screens and truncated 1–4-screen contexts.

There are several crucial qualifications:

- **Top-1 is the most meaningful paper-compatible metric.** Concatenated app descriptions effectively reveal app identity. Since a RICO app has only about 7.1 screens on average (median 5), app identity alone nearly saturates Top-0.1%, Top-1%, and Top-5%. A measured app-only oracle reaches 0.9968 at Top-0.1% and 1.0 at Top-1%. These columns say little about distinguishing screens within an app.
- **Normalized RMSE should not be used for a direct “better than Screen2Vec” claim.** The original evaluates unnormalized, freely scaled vectors; this project evaluates normalized vectors. In the current geometry, RMSE is largely a restatement of cosine distance and has a different bound and interpretation.
- **The original screen order cannot be exactly reproduced.** Released Screen2Vec preprocessing iterates an unsorted `os.listdir()` over numeric screen filenames. Its within-trace order is filesystem-dependent and not necessarily temporal. This project uses strictly ascending `gestures.json` order, which is the defensible next-screen task but differs from the original implementation.
- **The published/released baselines are not perfectly controlled.** Screen2Vec rows receive the app-description half, while TextOnly/LayoutOnly/VisualOnly baselines in the released evaluator do not. The released `modeltester.py` also contains syntax/argument defects. Comparisons should be described as comparisons with the published numbers, not a bit-for-bit reproduction.

The strongest future evaluation would add mean reciprocal rank, median rank, and especially **within-app rank/Top-k**, then split train/validation/test by app or trace to test generalization without app-identity shortcuts.

## Goal 2: reconstructed JSON UI prediction

### Model architecture

`2_train_adapter.py` trains a LLaVA-style bridge from saved encoder embeddings into a causal language model decoder. The bridge/projector translates one or more 4,096-dimensional Qwen encoder vectors into the decoder's token-embedding space. A LoRA adapter then specializes the decoder to emit the cleaned JSON representation without fully fine-tuning the foundation model. Its optional `--fast` path is off by default so older sweeps remain bit-for-bit reproducible; it is intended to be arithmetically equivalent, not a different model definition.

`3_json_reconstruction.py` and the parallel variant load the projector and LoRA weights, inject the projected screen representation as soft prefix embeddings, generate JSON autoregressively, and save both full outputs and extracted reconstructed objects. The single-process script currently pins the development-box GPU through `utils.CUDA_VISIBLE_DEVICE`; the parallel version instead selects each worker's device before initializing CUDA. The experiment directories show comparisons across pooling width and decoder family, including Qwen 2.5 1.5B and Qwen 3.5 2B/9B configurations.

The conceptual bottleneck is deliberate: reconstruction tests whether the embedding is not merely useful for retrieval but retains enough detailed information to recover structured screen content. Increasing pooled-token count increases representation capacity, while creating a storage/compute trade-off.

### Reconstruction metrics

`reconstruction_metrics.py` evaluates more than exact string matching and applies the strict interpretation during generation: an unparseable prediction receives zero components, zero class accuracy, and zero IoU. `repair_rescore.py` separately reparses the raw response with `json_repair` and runs the same scoring functions. It is a standalone module because Windows multiprocessing workers must be able to import their functions. Recorded fields include:

- Parseable JSON rate, plus parseability after `json-repair`.
- Gestalt sequence similarity and normalized Levenshtein similarity over minified JSON.
- Exact app and activity matches.
- Ground-truth and predicted component counts.
- Component class accuracy after pairing flattened components.
- Mean bounding-box intersection-over-union (IoU) for paired components.
- Execution time.

Repair is reported separately because syntactically repairing a language-model response is useful operationally but must not be confused with native valid generation. Likewise, character similarity can reward superficial serialization overlap, while class accuracy and bounding-box IoU more directly measure reconstructing the UI.

The aggregate notebook compares eight configurations. Some representative recorded means are:

| Configuration | Native valid JSON | Valid after repair | Class accuracy after repair | Bounds IoU after repair | Levenshtein similarity after repair |
|---|---:|---:|---:|---:|---:|
| 1×4096 average, Qwen 2.5 1.5B | 0.6136 | 0.9729 | 0.2935 | 0.1953 | 0.4212 |
| 4×4096 average, Qwen 2.5 1.5B | **0.8218** | 0.9735 | 0.3332 | 0.2252 | 0.4864 |
| 4×4096 average, Qwen 3.5 9B | 0.7756 | 0.9917 | 0.2939 | 0.1868 | 0.4754 |
| 8×4096 average, Qwen 2.5 1.5B | 0.7374 | 0.9740 | 0.3415 | 0.2255 | 0.4846 |
| 128×4096 average, 50k subset | 0.6300 | **0.9983** | **0.4642** | **0.2674** | **0.5622** |

These results show that reconstruction is functioning at corpus scale, but the objectives are not solved. The best structural figures still imply that more than half of paired classes are wrong and average box overlap is only about 0.27. The 128-prefix result is also based on 46,927/roughly 50k examples rather than the complete 66,261-screen run, so it should not be treated as a perfectly controlled winner. More prefix capacity improves detailed recovery in this experiment, while native JSON validity does not improve monotonically.

There is no directly comparable JSON-reconstruction benchmark in the original Screen2Vec paper: the paper evaluates embeddings through component prediction, screen prediction/retrieval, composability, and task representations rather than decoding complete view-hierarchy JSON. Reconstruction is therefore a new downstream task and needs its own baselines, ablations, held-out test protocol, and human/semantic evaluation rather than a claim of beating an original reconstruction score.

## Component-level reproduction

Scripts `10_extract_components.py` through `13_evaluate_component_prediction.py` reproduce the spirit of the paper's Table 2 task using a modern text encoder and Transformer context predictor. The extracted corpus contains 66,092 usable screens, 1,518,766 components, 613,261 text-bearing components, and 198,263 unique component texts. `all-MiniLM-L6-v2` provides 384-dimensional vocabulary embeddings. Context consists of the 16 spatially or hierarchically closest other components; spatial features can include bounding boxes, sizes, and target-relative offsets.

This implementation intentionally removes self-leakage found in the original code for screens with at most 16 components and uses a held-out validation split by default. Consequently, its scores are not automatically comparable to the paper's easier all-data evaluation. The checked-in notebook records an incomplete CPU training run, so this part appears less mature than the screen-sequence and reconstruction experiments.

## Repository map

- `0_build_trace_index.py`, `trace_index.py`: safe, reproducible trace/local-ID to layout/global-ID join.
- `1_qwen_generate_embeddings.py`: multimodal Qwen embedding extraction and token pooling.
- `2_train_adapter.py`: encoder-to-decoder projector plus LoRA reconstruction training.
- `3_json_reconstruction.py`, `3_json_reconstruction_par.py`: single/parallel JSON generation.
- `4_evaluate_screen_prediction.py`, parallel variant: retrieval-based next-screen evaluation.
- `5_train_sequence_projector.py`: standalone contrastive screen projector.
- `6_generate_app_context_embeddings.py`: SBERT app-description encoding and concatenation.
- `7_evaluate_weight_sweep.py`: screen/app weighting experiments.
- `8_train_lstm_predictor.py`: standalone recurrent sequence predictor.
- `9_train_joint.py`: preferred joint projector/LSTM sequence training.
- `10_`–`13_`, `component_utils.py`: component-context prediction/Table 2 work.
- `utils.py`: canonical RICO cleaning and shared model/data utilities.
- `reconstruction_metrics.py`, `repair_rescore.py`: structural/textual reconstruction scoring and repair-aware rescoring.
- `_results_*.ipynb`: recorded experiments and comparison tables.
- `hpc/` and PowerShell pipeline files: cluster and local orchestration.
- `screen2vec-master/`, `screen2vec-paper/`: original implementation and publication.

## Environment, execution, and artifact conventions

The development environment is Windows with a repository-local `.venv`; dependencies come from `requirements.txt`, including PyTorch from the CUDA 12.4 wheel index. `python check_gpu.py` checks CUDA availability. Tests are standard-library `unittest` tests and must be launched from the repository root because their fixtures use relative paths:

```powershell
python -m unittest tests.function_tests -v
```

There is currently no configured linter or formatter. The local PowerShell entry points are `_run_table2_pipeline.ps1` for component prediction, `_run_table3_pipeline1.ps1` for the two-stage sequence model, and `_run_table3_pipeline2.ps1` for the preferred joint sequence model. Multi-GPU adapter training uses the `accelerate`-based shell drivers rather than invoking `2_train_adapter.py` directly. The files under `hpc/` generate Slurm scripts from the development machine; `hpc/_hpc5_start_adapter_and_reconstruction.py` is the current chained train, sharded reconstruction, aggregation, and notification entry point.

Large artifacts are intentionally ignored by Git. The numerous embedding, adapter, reconstruction, checkpoint, and image directories visible locally are experimental outputs rather than source-controlled inputs. Their names form part of the experiment metadata and should not be changed casually:

- A single pooled vector uses `embeddings_4096_last` or `embeddings_4096_avg`; multiple vectors use `embeddings_<N>x4096_avg`.
- Adapter output uses `adapter_model_out_<embedding-directory>_full_<model-slug>`.
- Reconstruction output uses `reconstructions_<embeddings>__via__<adapter-name>`.
- Each reconstruction directory contains `<idx>_reconstructed.json` and `<idx>_report.txt`, and `_results_reconstruction.ipynb` parses the directory name to identify the experiment.

These conventions matter operationally: changing them can make a valid run invisible to the result notebooks or accidentally create a second, misleading experiment label.

## Current state and recommended research claims

The defensible headline is:

> Using corrected RICO trace-to-layout alignment and true temporal interaction order, Qwen-derived screen representations with a jointly trained projector and LSTM achieve 11.82% Top-1 next-screen retrieval under the repository's strict Screen2Vec-like population, compared with 6.10% reported by Screen2Vec. A one-previous-screen variant reaches 27.93% in a different, explicitly labeled setting.

The reconstruction headline should be framed independently:

> A LLaVA-style projector and LoRA decoder can reconstruct compact JSON UI descriptions from Qwen screen embeddings across tens of thousands of RICO screens. The strongest recorded high-capacity run reaches 99.83% repaired JSON validity, 46.42% repaired component-class accuracy, and 26.74% repaired mean bounding-box IoU, leaving substantial room for improved structural fidelity.

Claims to avoid are that every Table 3 metric is a clean semantic win, that normalized RMSE is directly comparable, that the obsolete 53.68% result is valid, or that JSON reconstruction beats an original Screen2Vec reconstruction benchmark. The repository's own analysis establishes why those claims would overstate the evidence.

## Best next technical steps

1. Establish an immutable train/validation/test split at the app level and publish the exact trace IDs, layout IDs, and hashes.
2. Make within-app MRR, median rank, and Recall@k the primary sequence metrics; report global Top-1 only as a paper bridge.
3. Compare temporal LSTM prediction with simple baselines: previous-screen nearest neighbor, transition-frequency lookup, app-only retrieval, GRU/Transformer sequence models, and frozen-foundation-model prompting.
4. Ablate screenshot versus JSON input, foundation-model family/size, pooling width, app-description context, projector dimension, number of previous screens, and joint versus two-stage training.
5. Strengthen JSON evaluation with tree-edit distance, key/value precision-recall, component matching via Hungarian assignment, per-class accuracy, text/content-description recovery, and IoU at multiple thresholds.
6. Separate syntax from semantics by constraining decoding to the JSON schema or using grammar-guided generation, then report native validity, repaired validity, and semantic metrics independently.
7. Add retrieval-then-generation and nearest-neighbor-copy reconstruction baselines. These reveal how much the decoder genuinely recovers from the embedding rather than memorizing common RICO templates.
8. Complete the component-prediction experiment on held-out data and clearly distinguish it from the paper's self-leaking/all-data protocol.
9. Record model revisions, seeds, checkpoint hashes, hardware, wall-clock time, peak memory, and energy use so the foundation-model accuracy improvement can be weighed against computational cost.

Overall, this is an ambitious and technically substantive modernization of Screen2Vec. Its strongest contribution is not merely swapping in a newer model: it combines multimodal foundation embeddings, corrects a subtle but severe RICO indexing failure, defines true temporal prediction, exposes weaknesses in the original evaluation, and adds structured UI reconstruction as a demanding new test of representation quality.
