# Screen2Vec Replication Audit

> **What this file is.** The audit from replicating Screen2Vec's Table 3: one bug in our
> pipeline (found and fixed — status below) and four measurement problems in the original
> paper's evaluation. It is **not** this project's results — those live in the
> `_results_*.ipynb` notebooks and the results manifest.
>
> **Status (2026-09-18).** Finding #1 is fixed (commit 90a6b8b) and re-verified against the
> current code: all trace-consuming stages resolve gesture keys through `trace_index.py`.
> One exception: `7_evaluate_weight_sweep.py` predates the fix, still contains the old
> lookup, and must not be used. Findings #2–5 describe the original paper's evaluation and
> remain current.

Review of our GUI screen prediction pipeline against Screen2Vec (Li et al., CHI 2021),
its released source, and the RICO dataset.

Two classes of finding: **one bug in our pipeline** (found and fixed) and **four
measurement problems in the paper's Table 3** that affect how any replication should
report its numbers.

---

## TL;DR

| # | Finding | Status |
|---|---|---|
| 1 | Trace screen IDs and embedding IDs were different namespaces — every trace pointed at unrelated screens | **Fixed** (content-hash index) |
| 2 | Table 3's Top-0.01% / 0.1% / 1% / 5% columns measure *app identity*, not screen semantics | Report accordingly |
| 3 | Normalized RMSE is not comparable between their setup and ours | Replace or restate |
| 4 | The original's traces are not in temporal order | Cannot be reproduced |
| 5 | The released `modeltester.py` does not run; the Table 3 baselines lack the app-description half | Note when citing |

Everything below is measured, with the method stated. All numbers come from the RICO
release in `../rico-data/`.

---

## 1. Screen-ID namespace mismatch (our bug)

### What was wrong

Two RICO downloads number screens independently, and the pipeline treated them as one
namespace.

| | source | numbering |
|---|---|---|
| Embeddings | `data/layouts/<id>.json` — the "UI Screenshots and View Hierarchies" corpus | **global**, `0…66260`, one counter across the whole crawl |
| Traces | `filtered_traces/<app>/<trace>/gestures.json` | **per-trace** UI indices, restarting in every trace |

`1_qwen_generate_embeddings.py` embeds the layouts corpus and names each `.npz` after its
source file. `4_`, `5_`, `6_`, `8_`, `9_` then read a trace key with `int(key)` and looked
it up among those embeddings. Both are small integers, so nothing ever errored — the
lookup just returned an unrelated screen.

### Evidence

- **Content check.** For 731 random trace screens, comparing
  `filtered_traces/<app>/<trace>/view_hierarchies/<k>.json` against `layouts/<k>.json` by
  `activity_name`: **0 matched (0.0%)**. Example — `com.eat24.app/trace_0`, screen 138:

  | source | activity |
  |---|---|
  | trace file | `com.eat24.app/…RestaurantListTrendingActivity` |
  | `layouts/138.json` | `com.despdev.weight_loss_calculator/…AdActivity` |

- **Collisions.** Screen id `1` appears in 480 distinct traces; `2` in 475; `3` in 314.
  Only **5,178 distinct ids** were reachable this way, out of 66,261 embeddings — the
  other ~61k could never be a retrieval target.

- **App attribution.** `6_generate_app_context_embeddings.py` built `screen_to_app` from
  the same aliased keys, so for each colliding id whichever app sorted last won. Only ids
  the traces never mentioned fell back to the correct package.

### Why the old numbers still looked plausible

The aliasing is deterministic, and training and evaluation shared it. But local trace keys
ascend, so each "trace" became an *ascending run of global ids* — and RICO assigns global
ids per app crawl, so neighbouring ids usually share an app. The model learned "given ids
near X, predict a nearby id", which is easy and artificial. That is what our previously
reported Top-1 of 0.5368 measured.

### The fix

The two corpora contain the *same screens*, only renumbered, so they can be joined on
content. `0_build_trace_index.py` builds a persistent join, `trace_index.py` holds the
logic, and every trace-consuming stage now resolves keys through it.

**Join key** — SHA-1 of the canonical JSON of the entire document:

```python
json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

Hash scope was chosen by measurement, over 3,999 sampled trace screens:

| scope | distinct hashes / 66,261 | unique join | ambiguous | miss |
|---|---|---|---|---|
| raw file bytes | 66,261 | **0.0%** | 0.0% | 100.0% |
| **full document, canonical JSON** | **66,261** | **99.7%** | **0.0%** | 0.3% |
| document minus `request_id` | 66,143 | 99.7% | 0.1% | 0.2% |
| `activity.root` subtree only | 66,138 | 99.7% | 0.1% | 0.2% |

Raw bytes fail completely — the two corpora serialize the same JSON with different key
order and whitespace, so canonicalization is required. The full document is the only scope
with **zero collisions on the layouts side and zero ambiguous joins**; narrowing to
`activity.root` discards `request_id` (which survives the renumbering for 99.9% of
screens) and collapses 212 screens into duplicate groups. `activity.root` is kept as a
fallback for the remainder. Anything still unmatched is dropped and logged, never guessed.

Assignment is injective by construction: a layouts id is consumed at most once, so two
trace screens can never alias onto one embedding.

### Result

```
trace gesture screens       : 66,261
mapped                      : 66,157  (99.84%)
  via full-document hash    : 66,154
  via activity.root hash    :      3
dropped                     :    104   (99 no match, 5 unhashable)
distinct layout ids used    : 66,157   (injective)
```

Acceptance tests:

| check | before | after |
|---|---|---|
| `activity_name` agreement on paired screens | 0 / 1,162 (0.00%) | **1,200 / 1,200 (100.00%)** |
| index injective | — | pass |
| no trace repeats a layouts id | — | pass |
| adjacent-screen cosine (raw 4096-d embeddings) | 0.8241 | **0.8970** |
| distinct screens reachable from traces | 5,178 | **66,157** |

`activity_name` lives outside the hashed subtree for the fallback matches, so its
agreement is independent evidence rather than a tautology.

**No re-embedding was needed** — `embeddings_4096_*` is keyed by layouts id and remains
valid. Everything trained on traces does have to be rebuilt.

---

## 2. Table 3's top-k% columns measure app identity

Screen2Vec's screen vector is `[screen_embedding ‖ app_description_embedding]`
(`modeltester.py:102-104,126-127`). Every screen in a trace shares an app, so the
description half is a constant match for all same-app candidates — it narrows the
candidate set to one app before screen content is considered.

RICO app sizes, computed from the corrected index (9,372 apps, 66,157 screens):

```
screens per app:  mean 7.1   median 5   p90 16   max 76
```

So an oracle that knows **only the app** and nothing about the screen achieves:

| column | app-identity oracle | paper: Screen2Vec-EUCLIDEAN-spatial |
|---|---|---|
| Top-0.01% (k=6) | **0.2616** | 0.258 |
| Top-0.1% (k=66) | **0.9968** | 0.969 |
| Top-1% (k=661) | **1.0000** | 0.998 |

The paper's model does not beat an app-name lookup on any of these columns. Top-0.1%,
Top-1% and Top-5% are saturated and carry no information about screen representation
quality; Top-0.01% is marginally *below* the oracle.

**Top-1 is the only column that requires distinguishing screens within an app.** Report it
as the headline; report the others as saturated, or replace them with a within-app rank.

---

## 3. Normalized RMSE is not comparable

Both implementations compute
`sqrt(Σ‖pred − target‖² / n) / mean‖target‖` — summed over all dimensions, not divided by
`D`, and measured against the *correct* screen rather than the retrieved one
(`modeltester.py:134-137,171`). Our implementation matches theirs exactly.

The quantity still is not comparable:

- **Theirs** is over unnormalized vectors with a free-scale predictor, so values above 1
  are possible (TextOnly 1.241, no-spatial 1.723).
- **Ours** has unit-norm halves and an L2-normalized prediction, so
  `‖p − t‖² = 2(1 − cos)` exactly and nRMSE reduces to `sqrt(1 − mean cos)`, bounded in
  `[0, √2]`.

Ours is a monotone restatement of mean cosine similarity on a different scale. Any "we beat
their RMSE" claim is unsupported. Prefer MRR or median rank, or state the identity
explicitly.

Related trap: if the LSTM's input dimension equals the full embedding width while the
embedding is a concatenation of two unit vectors, the prediction is normalized to 1 while
targets have norm √2, putting a floor of 0.293 on nRMSE. Keep the app half outside the
LSTM.

---

## 4. The original's traces are not in temporal order

`UI_embedding/precompute_embeddings.py:72` builds each trace with:

```python
for view_hierarchy_json in os.listdir(.../view_hierarchies):
```

No sort. Filenames are numeric, so even an alphabetical listing would be wrong
(`"124" < "35"`). Under any filesystem the within-trace order is arbitrary.

Screen2Vec's "screen prediction" is therefore *predict one screen from four arbitrary
other screens of the same trace* — the LSTM's recurrence carries no temporal meaning.

Our pipeline uses `gestures.json` key order. Verified across all 10,292 traces: **keys are
strictly ascending in every one**, i.e. true temporal order. This is a better-defined task,
but it is not their task, and it cannot be reproduced — `os.listdir` order is not portable.

---

## 5. The released code and the Table 3 baselines

- **`modeltester.py` does not run as released.** Line 39 concatenates two
  `add_argument` calls on one line (syntax error), and lines 69/83/88/95/96 reference
  `args.version`, which is never defined — the flag is `--net_version`. The README points
  at `modeltester_screen.py`, which does not exist. Exact replication requires
  interpretation; this is worth a sentence in any write-up.

- **The baselines never get the app description.** `baseline/modeltester_baseline.py`
  builds TextOnly / LayoutOnly / VisualOnly pools from `*_eval.json` with no description
  concatenation, while the Screen2Vec rows have it. Given finding #2, a large part of
  Table 3's headline gap is the app-identity shortcut rather than representation quality.

---

## 6. Evaluation protocol: ours vs theirs

Our default was more permissive than `modeltester.py`. Both are now available.

| | permissive (default) | `--strict` | Screen2Vec |
|---|---|---|---|
| traces kept | ≥ 3 screens | ≥ `n_predictors+1` (5) | ≥ 5 (`dataset.py:28,67`) |
| context per prediction | 1–4 (truncated near trace start) | exactly 4 | exactly 4 |
| candidate pool | every embedding in the directory | screens of eligible traces | screens of eligible traces (`vocab.py:74-94`) |
| top-k% rounding | `ceil()` | `int()` | `int()` (`modeltester.py:141`) |
| training negatives | any screen | eligible screens | `ScreenVocab` = filtered dataset |
| screen order | temporal | temporal | arbitrary (finding #4) |
| traces of apps without a Play Store description | kept (fallback text or zero vector) | kept (fallback text or zero vector) | **dropped**: a trace is kept only if `d != ""` (`dataset.py:67,166`); the scraper stored `''` on failure (`UI_embedding/precompute_embeddings.py:59-61`) |

Under the permissive setting, 37.4% of evaluation instances ran on a truncated context the
original never evaluates, and training zero-padded those short contexts while evaluation
did not — a train/test mismatch that `--strict` removes by construction.

`--strict` is off by default in the stage scripts (`4_`, `5_`, `8_`, `9_`); their default path is
byte-identical to before it was added. The dimension sweep (`4_sweep_embedding_dimensions.py`)
defaults to strict, because only strict numbers may be set next to the paper's Table 3.

The description filter is the one part of Screen2Vec's population that `--strict` does not
reproduce, and it cannot be: which apps had an empty description depends on the authors' 2020
Play Store scrape. Our strict population is therefore a superset of theirs by an unknown
number of traces. Like the screen order, this is a documented difference, not a fixable one.

---

## 7. What changed in this repo

| file | change |
|---|---|
| `trace_index.py` | **new** — join logic, index build and load |
| `0_build_trace_index.py` | **new** — CLI builder, writes index + meta + dropped audit |
| `4_`, `4_par`, `5_`, `8_`, `9_` | resolve trace keys through the index; `--trace_index`; `--strict` |
| `6_` | app attribution rebuilt from the index instead of aliased trace keys |
| `hpc/_hpc4_*` | `STRICT` setting threaded into generated SLURM commands |

The scripts now **refuse to run without the index** rather than falling back — the old
behaviour is silently wrong, so a fallback would reintroduce the bug.

The index is reproducible: `trace_layout_index.meta.json` records the hash scheme,
canonicalization, source paths, per-method counts and coverage. `0_build_trace_index.py`
fails if coverage drops below `--min_coverage` (default 0.99), so a mismatched RICO copy
cannot silently degrade the join.

---

## 8. Reproducing

```bash
# once, ~7 min
python 0_build_trace_index.py

# retrain — previous artefacts were fitted on mis-joined traces
python 9_train_joint.py --proj_dim 768
python 6_generate_app_context_embeddings.py \
    --embeddings_dir embeddings_seq_768 \
    --output_dir embeddings_seq_768_with_app_desc
python 4_evaluate_screen_prediction.py \
    --embeddings_dir embeddings_seq_768_with_app_desc \
    --lstm_model lstm_predictor_768d.pth

# same, under Screen2Vec's own evaluation population
python 9_train_joint.py --proj_dim 768 --strict
python 6_generate_app_context_embeddings.py \
    --embeddings_dir embeddings_seq_768_strict \
    --output_dir embeddings_seq_768_strict_with_app_desc
python 4_evaluate_screen_prediction.py \
    --embeddings_dir embeddings_seq_768_strict_with_app_desc \
    --lstm_model lstm_predictor_768d_strict.pth --strict
```

`embeddings_4096_*` does **not** need regenerating.

---

## 9. Open items

- **All previously reported screen-prediction numbers are void.** The 0.5368 Top-1 was
  produced on mis-joined traces. A plumbing smoke test with the stale model on correctly
  joined traces already drops it to 0.1475 — that figure is meaningless too (model and
  data disagree), but it shows the fix changes what it should. Real numbers require the
  retraining above.
- **Report a within-app rank** alongside Top-1. Given finding #2 it is the only measure
  that isolates screen-level discrimination from app identity.
- **Report the app-identity oracle as a baseline row** (0.2616 / 0.9968 / 1.0000). It is
  the honest floor for any model carrying an app-description half.
- **104 trace screens (0.16%) are unjoined**, listed in `trace_layout_index_dropped.csv`.
  Recovering them would mean embedding the trace view hierarchies directly, which is what
  Screen2Vec does — a full re-embedding run for 0.16% of the data.
