"""
4_evaluate_screen_prediction_par.py

Parallelized version of 4_evaluate_screen_prediction.py for HPC execution.

Replicates the Screen2Vec paper's Table 3: GUI Screen Prediction evaluation.

Supports three execution modes:
  1. Standalone (no extra flags): identical behaviour to the original script
  2. Worker   (--job_json):  evaluates a subset of traces, writes partial results
  3. Aggregate (--aggregate): reads all partial results, computes final metrics

Usage:
    # Standalone (same as original)
    python 4_evaluate_screen_prediction_par.py --embeddings_dir embeddings_seq_768_with_app \\
        --lstm_model lstm_predictor_768d.pth

    # Worker mode (called by HPC array job — reads batch info from job.json)
    python 4_evaluate_screen_prediction_par.py --embeddings_dir embeddings_seq_768_with_app \\
        --lstm_model lstm_predictor_768d.pth --job_json job.json

    # Aggregation mode (called by HPC dependency job)
    python 4_evaluate_screen_prediction_par.py --aggregate \\
        --results_dir /path/to/shared/results

Two evaluation protocols (see 4_evaluate_screen_prediction.py for details):
  - Permissive (default): traces >= 3 screens, context of 1..n_predictors,
    pool = every embedding in --embeddings_dir, ceil()-rounded thresholds.
  - Strict (--strict): Screen2Vec's own population — traces >= n_predictors+1,
    exactly n_predictors context screens, pool restricted to the screens of
    those traces, int()-truncated thresholds.
    In worker mode --strict must be passed to *every* worker: the pool (and
    therefore the top-k%% thresholds) is derived per worker, and
    aggregate_results assumes all workers agree.
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from joblib import Parallel, delayed

import trace_index


# ============================================================================
# LSTM Predictor (must match 8_train_lstm_predictor.py exactly)
# ============================================================================
class LSTMPredictor(nn.Module):
    """
    Replicates Screen2Vec's TracePredictor LSTM combiner.
    Loaded from a .pth checkpoint produced by 8_train_lstm_predictor.py.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.combiner = nn.LSTM(embed_dim, embed_dim, batch_first=True)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (1, T, embed_dim) context screen embeddings
        Returns:
            (1, embed_dim) prediction vector, L2-normalized
        """
        _, (h, _) = self.combiner(context)
        pred = h[0]  # (1, embed_dim)
        return F.normalize(pred, p=2, dim=-1)


def load_lstm_predictor(lstm_model_path: str):
    """
    Load a trained LSTMPredictor from a .pth checkpoint.
    Returns the model in eval mode (on CPU).
    """
    checkpoint = torch.load(lstm_model_path, map_location="cpu", weights_only=False)
    embed_dim = checkpoint["embed_dim"]
    model = LSTMPredictor(embed_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"  Loaded LSTM predictor (embed_dim={embed_dim}, "
          f"best_val_loss={checkpoint.get('best_val_loss', 'N/A'):.4f})")
    return model


# ============================================================================
# Paper's Table 3 reference numbers
# ============================================================================
PAPER_TABLE3 = {
    "Screen2Vec-EUCLIDEAN-spatial": {
        "top1": 0.061, "top001pct": 0.258, "top01pct": 0.969,
        "top1pct": 0.998, "top5pct": 1.00, "nrmse": 0.853,
    },
    "Screen2Vec-EUCLIDEAN-no spatial": {
        "top1": 0.0065, "top001pct": 0.116, "top01pct": 0.896,
        "top1pct": 0.986, "top5pct": 0.999, "nrmse": 1.723,
    },
    "TextOnly": {
        "top1": 0.012, "top001pct": 0.055, "top01pct": 0.196,
        "top1pct": 0.439, "top5pct": 0.643, "nrmse": 1.241,
    },
    "LayoutOnly": {
        "top1": 0.0041, "top001pct": 0.024, "top01pct": 0.091,
        "top1pct": 0.222, "top5pct": 0.395, "nrmse": 1.135,
    },
    "VisualOnly": {
        "top1": 0.0060, "top001pct": 0.026, "top01pct": 0.121,
        "top1pct": 0.252, "top5pct": 0.603, "nrmse": 1.543,
    },
}


# ============================================================================
# 1. Load Embeddings
# ============================================================================
def load_all_embeddings(embeddings_dir: str):
    """
    Load all .npz files from embeddings_dir into:
      - id_to_idx: {screen_id (int): index (int)} mapping
      - embedding_matrix: (N, D) float32 numpy array, L2-normalized
    """
    npz_files = sorted(Path(embeddings_dir).glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files found in '{embeddings_dir}'. "
            "Run 1_qwen_generate_embeddings.py first."
        )

    print(f"Loading {len(npz_files)} embeddings from '{embeddings_dir}' in parallel...")

    def load_single(npz_path):
        screen_id = int(npz_path.stem)
        with np.load(npz_path) as f:
            emb = f["embedding"].astype(np.float32)
        return screen_id, emb

    # Load npz files in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(load_single, npz_files),
            total=len(npz_files),
            desc="Loading embeddings"
        ))

    id_to_idx = {}
    embeddings = []
    for idx, (screen_id, emb) in enumerate(results):
        id_to_idx[screen_id] = idx
        embeddings.append(emb)

    embedding_matrix = np.stack(embeddings, axis=0)  # (N, D)

    # L2-normalize each embedding for cosine similarity via dot product
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid division by zero
    embedding_matrix_normed = embedding_matrix / norms

    print(f"  Loaded matrix shape: {embedding_matrix.shape}")
    return id_to_idx, embedding_matrix, embedding_matrix_normed, norms.squeeze()


# ============================================================================
# 2. Parse RICO Interaction Traces
# ============================================================================
def parse_traces(traces_dir: str, available_ids: set, index: dict,
                 min_trace_len: int = 3):
    """
    Walk filtered_traces/<app>/trace_N/gestures.json to extract ordered
    screen sequences. Only keeps screens that have embeddings available.

    index: the trace->layout index from trace_index.load_index(). A trace's
    gestures.json keys are per-trace indices, NOT the global layouts ids the
    embeddings are keyed by, so every key must be resolved through it. See
    trace_index.py for why.

    min_trace_len: traces with fewer screens are skipped. Default 3 keeps the
    permissive behaviour; the strict protocol passes n_predictors + 1, which
    is what Screen2Vec's RicoDataset requires (dataset.py:28,67).

    Returns list of traces, where each trace is a list of layouts ids (int).
    """
    traces_root = Path(traces_dir)
    if not traces_root.exists():
        raise FileNotFoundError(f"Traces directory not found: {traces_dir}")

    traces = []
    skipped_short = 0
    total_screens_in_traces = 0
    unresolved_screens = 0

    app_dirs = sorted(traces_root.iterdir())
    for app_dir in tqdm(app_dirs, desc="Parsing traces"):
        if not app_dir.is_dir():
            continue
        for trace_dir in sorted(app_dir.iterdir()):
            if not trace_dir.is_dir() or not trace_dir.name.startswith("trace_"):
                continue

            gestures_path = trace_dir / "gestures.json"
            if not gestures_path.exists():
                continue

            try:
                with open(gestures_path, "r", encoding="utf-8") as f:
                    gestures = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            # Keys of the gestures dict are per-trace screen indices in trace
            # order (Python 3.7+ dicts preserve insertion order). They must be
            # resolved to global layouts ids before touching the embeddings.
            screen_ids = []
            for key in gestures.keys():
                try:
                    screen_key = int(key)
                except ValueError:
                    continue
                layout_id = index.get((app_dir.name, trace_dir.name, screen_key))
                if layout_id is None:
                    unresolved_screens += 1
                    continue
                if layout_id in available_ids:
                    screen_ids.append(layout_id)

            if len(screen_ids) < min_trace_len:
                skipped_short += 1
                continue

            traces.append(screen_ids)
            total_screens_in_traces += len(screen_ids)

    print(f"  Parsed {len(traces)} traces ({total_screens_in_traces} screens)")
    print(f"  Skipped {skipped_short} traces with < {min_trace_len} screens")
    if unresolved_screens:
        print(f"  Skipped {unresolved_screens} screens absent from the "
              f"trace->layout index")
    return traces


def restrict_pool_to_traces(traces, id_to_idx, embedding_matrix,
                            embedding_matrix_normed, norms):
    """
    Restrict the retrieval pool to the screens contained in `traces`.

    Screen2Vec builds its candidate pool from the *filtered* dataset
    (vocab.py:74-94 iterates dataset.traces, which RicoDataset has already
    filtered), so screens that live only in ineligible traces are never
    retrieval candidates. The permissive default keeps every embedding in the
    directory, which makes the pool larger and the ranking harder.

    In worker mode this MUST be called on the full trace list, before the
    shard slice, so that every worker derives an identical pool and identical
    top-k% thresholds (aggregate_results assumes they match).

    Returns freshly indexed (id_to_idx, matrix, normed_matrix, norms).
    """
    keep_ids = sorted({sid for trace in traces for sid in trace})
    keep_idx = np.array([id_to_idx[sid] for sid in keep_ids], dtype=np.int64)

    new_id_to_idx = {sid: i for i, sid in enumerate(keep_ids)}
    return (
        new_id_to_idx,
        embedding_matrix[keep_idx],
        embedding_matrix_normed[keep_idx],
        norms[keep_idx],
    )


# ============================================================================
# 3. Evaluate Screen Prediction
# ============================================================================
def eval_single_trace(
    trace,
    id_to_idx,
    embedding_matrix,
    embedding_matrix_normed,
    k_001pct,
    k_01pct,
    k_1pct,
    k_5pct,
    lstm_predictor=None,
    n_predictors=4,
    strict=False,
):
    """Evaluates prediction metrics for a single RICO interaction trace.

    Instead of leaving one out from the whole trace (which gives future context),
    this strictly evaluates predicting the *next* screen given the *previous*
    screens (up to n_predictors), matching Screen2Vec's modeltester.py and the
    causal nature of the LSTM.

    strict=False (default): predicts screens 1..T-1, so the first n_predictors
    predictions run on a truncated context of 1..n_predictors-1 screens.
    strict=True: only full windows, i.e. predicts screens n_predictors..T-1
    with exactly n_predictors context screens each, which is what
    TesterRicoDataset.get_item_part (dataset.py:34-45) yields.
    """
    trace_indices = [id_to_idx[sid] for sid in trace]
    trace_normed = embedding_matrix_normed[trace_indices]  # (T, D)
    trace_raw = embedding_matrix[trace_indices]            # (T, D)

    T = len(trace)
    if T < 2:
        return T, 0.0, 0, 0, 0, 0, 0

    # Permissive: evaluate predictions for screens i = 1 to T-1.
    # Strict: only full windows, i.e. i = n_predictors to T-1.
    first_target = n_predictors if strict else 1

    sq_err_sum = 0.0
    n_instances = 0
    hits_top1 = 0
    hits_top001pct = 0
    hits_top01pct = 0
    hits_top1pct = 0
    hits_top5pct = 0

    for i in range(first_target, T):
        n_instances += 1
        # Context is up to n_predictors previous screens
        # (exactly n_predictors when strict, since i >= n_predictors there)
        start_idx = max(0, i - n_predictors)
        ctx_raw = trace_raw[start_idx:i]         # (ctx_len, D)
        ctx_normed = trace_normed[start_idx:i]   # (ctx_len, D)
        target_raw = trace_raw[i]                # (D,)
        target_normed = trace_normed[i]          # (D,)
        target_global_idx = trace_indices[i]

        if lstm_predictor is not None:
            # LSTM prediction
            lstm_dim = lstm_predictor.combiner.input_size
            with torch.no_grad():
                # Extract the screen part for the LSTM and normalize it
                ctx_screen = ctx_raw[:, :lstm_dim]
                ctx_screen_norm = np.linalg.norm(ctx_screen, axis=1, keepdims=True)
                ctx_screen_normed = ctx_screen / np.maximum(ctx_screen_norm, 1e-12)

                ctx_tensor = torch.from_numpy(ctx_screen_normed).unsqueeze(0)  # (1, ctx_len, D)
                out = lstm_predictor(ctx_tensor)                        # (1, D)
                pred_screen = out.squeeze(0).numpy()                    # (D,)
                
                # If there are extra dimensions (e.g., app description), concatenate them!
                if ctx_raw.shape[1] > lstm_dim:
                    app_desc = target_raw[lstm_dim:]
                    pred_raw = np.concatenate([pred_screen, app_desc])
                else:
                    pred_raw = pred_screen

                pred_norm = np.linalg.norm(pred_raw)
                pred_normed = pred_raw / max(pred_norm, 1e-12)
        else:
            # Baseline: mean of context
            pred_raw = ctx_raw.mean(axis=0)                             # (D,)
            pred_norm = np.linalg.norm(pred_raw)
            pred_normed = pred_raw / max(pred_norm, 1e-12)              # (D,)

        # RMSE
        sq_err_sum += float(np.sum((pred_raw - target_raw) ** 2))

        # Ranking (cosine similarity)
        # Cosine similarity against ALL screen embeddings
        similarities = cosine_similarity(pred_normed.reshape(1, -1), embedding_matrix_normed)[0]
        
        target_sim = similarities[target_global_idx]
        rank = np.sum(similarities >= target_sim)

        if rank <= 1: hits_top1 += 1
        if rank <= k_001pct: hits_top001pct += 1
        if rank <= k_01pct: hits_top01pct += 1
        if rank <= k_1pct: hits_top1pct += 1
        if rank <= k_5pct: hits_top5pct += 1

    return n_instances, sq_err_sum, hits_top1, hits_top001pct, hits_top01pct, hits_top1pct, hits_top5pct


def evaluate_screen_prediction(
    traces: list,
    id_to_idx: dict,
    embedding_matrix: np.ndarray,
    embedding_matrix_normed: np.ndarray,
    norms: np.ndarray,
    n_jobs: int = -1,
    lstm_predictor=None,
    strict: bool = False,
    n_predictors: int = 4,
):
    """
    For each prediction instance in each trace, run the preceding screens
    through the predictor and find the nearest neighbour in the candidate
    pool. Record the rank of the true screen.

    strict=True switches to Screen2Vec's own top-k% truncation and full-window
    instances; the caller is responsible for having restricted the pool.

    Returns:
        metrics: dict with computed percentages (top1, top001pct, ..., nrmse)
        raw:     dict with raw accumulators for partial-result aggregation
    """
    N = embedding_matrix.shape[0]

    # Top-k% thresholds. Screen2Vec truncates when slicing the candidate list
    # (modeltester.py:141-146: temp[:int(0.0001 * comp_len)]); the permissive
    # default rounds up instead.
    round_k = int if strict else math.ceil
    k_001pct = max(1, round_k(0.0001 * N))  # ~7
    k_01pct = max(1, round_k(0.001 * N))    # ~67
    k_1pct = max(1, round_k(0.01 * N))      # ~663
    k_5pct = max(1, round_k(0.05 * N))      # ~3314

    # Announced only under --strict, so the default output stays byte-identical
    # to the pre---strict version of this script.
    if strict:
        print(f"\n  Protocol: STRICT (Screen2Vec modeltester.py population, "
              f"exactly {n_predictors} context screens per prediction)")
    print(f"\n  Total screens in embedding index: {N}")
    print(f"  Top-0.01% threshold: k={k_001pct}")
    print(f"  Top-0.1%  threshold: k={k_01pct}")
    print(f"  Top-1%    threshold: k={k_1pct}")
    print(f"  Top-5%    threshold: k={k_5pct}")

    # For Normalized RMSE: mean embedding norm
    mean_norm = float(np.mean(norms))
    print(f"  Mean embedding L2 norm: {mean_norm:.4f}")

    if lstm_predictor is not None:
        print("  Prediction mode: LSTM (Screen2Vec-equivalent)")
        print(f"  Evaluating traces in parallel via threads (n_jobs={n_jobs})...")
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(eval_single_trace)(
                trace, id_to_idx, embedding_matrix, embedding_matrix_normed,
                k_001pct, k_01pct, k_1pct, k_5pct, lstm_predictor=lstm_predictor,
                n_predictors=n_predictors, strict=strict,
            )
            for trace in tqdm(traces, desc="Evaluating traces (LSTM)")
        )
    else:
        print(f"  Prediction mode: mean-of-context (baseline)")
        print(f"  Evaluating traces in parallel (n_jobs={n_jobs})...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_single_trace)(
                trace, id_to_idx, embedding_matrix, embedding_matrix_normed,
                k_001pct, k_01pct, k_1pct, k_5pct, lstm_predictor=None,
                n_predictors=n_predictors, strict=strict,
            )
            for trace in tqdm(traces, desc="Evaluating traces")
        )

    # Accumulators
    total_predictions = 0
    hits_top1 = 0
    hits_top001pct = 0
    hits_top01pct = 0
    hits_top1pct = 0
    hits_top5pct = 0
    sum_sq_error = 0.0

    for res in results:
        T, sq_err_sum, h1, h001, h01, h1pct, h5pct = res
        total_predictions += T
        sum_sq_error += sq_err_sum
        hits_top1 += h1
        hits_top001pct += h001
        hits_top01pct += h01
        hits_top1pct += h1pct
        hits_top5pct += h5pct

    # Raw accumulators (for partial-result saving in parallel HPC mode)
    raw = {
        "total_predictions": int(total_predictions),
        "sum_sq_error": float(sum_sq_error),
        "hits_top1": int(hits_top1),
        "hits_top001pct": int(hits_top001pct),
        "hits_top01pct": int(hits_top01pct),
        "hits_top1pct": int(hits_top1pct),
        "hits_top5pct": int(hits_top5pct),
        "mean_norm": float(mean_norm),
        "N": int(N),
        "strict": bool(strict),
        "n_predictors": int(n_predictors),
    }

    # Compute final metrics
    metrics = {
        "top1": hits_top1 / total_predictions,
        "top001pct": hits_top001pct / total_predictions,
        "top01pct": hits_top01pct / total_predictions,
        "top1pct": hits_top1pct / total_predictions,
        "top5pct": hits_top5pct / total_predictions,
        "nrmse": math.sqrt(sum_sq_error / total_predictions) / mean_norm,
    }

    print(f"\n  Total prediction instances: {total_predictions}")
    return metrics, raw


# ============================================================================
# 4. Partial Results (for HPC worker mode)
# ============================================================================
def save_partial_results(results_dir: str, batch_index: int, raw: dict):
    """Save raw accumulators as JSON for later aggregation by the dedicated job."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"partial_results_{batch_index}.json")
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"  Saved partial results to '{path}'")


# ============================================================================
# 5. Aggregation (for HPC dependency job)
# ============================================================================
def aggregate_results(results_dir: str, label: str = None, lstm_mode: bool = False):
    """
    Read all partial_results_*.json from the shared results directory,
    aggregate the raw accumulators, compute final metrics, and print the
    comparison table. Writes final_metrics.json and final_metrics.csv to CWD
    (for mail.py to pick up).
    """
    from glob import glob as glob_fn

    partial_files = sorted(glob_fn(os.path.join(results_dir, "partial_results_*.json")))
    if not partial_files:
        raise FileNotFoundError(f"No partial_results_*.json found in '{results_dir}'")

    print(f"\n{'=' * 70}")
    print(f"  AGGREGATION: Found {len(partial_files)} partial result files")
    print(f"  Results dir: {results_dir}")
    print(f"{'=' * 70}")

    # Read and sum accumulators
    total_predictions = 0
    sum_sq_error = 0.0
    hits_top1 = 0
    hits_top001pct = 0
    hits_top01pct = 0
    hits_top1pct = 0
    hits_top5pct = 0
    mean_norm = None
    N = None
    total_traces = 0
    num_batches_expected = None
    strict = False
    protocol_mismatch = False

    for pf in partial_files:
        with open(pf, "r") as f:
            p = json.load(f)
        print(f"  Reading {os.path.basename(pf)}: "
              f"batch {p['batch_index']}, "
              f"{p['num_traces']} traces, "
              f"{p['total_predictions']} predictions")
        total_predictions += p["total_predictions"]
        sum_sq_error += p["sum_sq_error"]
        hits_top1 += p["hits_top1"]
        hits_top001pct += p["hits_top001pct"]
        hits_top01pct += p["hits_top01pct"]
        hits_top1pct += p["hits_top1pct"]
        hits_top5pct += p["hits_top5pct"]
        total_traces += p["num_traces"]
        # mean_norm and N are identical across all workers (same embedding set)
        if mean_norm is None:
            mean_norm = p["mean_norm"]
            N = p["N"]
            num_batches_expected = p["num_batches"]
            strict = p.get("strict", False)
        elif (p["N"] != N or p["mean_norm"] != mean_norm
                or p.get("strict", False) != strict):
            # Workers disagree on the candidate pool or the protocol, so the
            # top-k% thresholds they applied were not the same. Summing their
            # hit counts would be meaningless.
            protocol_mismatch = True

    if len(partial_files) != num_batches_expected:
        print(f"  WARNING: Expected {num_batches_expected} partial files, "
              f"found {len(partial_files)}")

    if protocol_mismatch:
        print("  WARNING: partial results disagree on N / mean_norm / strict. "
              "The workers used different candidate pools or protocols, so the "
              "aggregated top-k% numbers are NOT valid. Re-run all workers "
              "with identical flags.")

    if total_predictions == 0:
        raise ValueError("Total predictions is 0 — no data to aggregate.")

    # Compute final metrics
    metrics = {
        "top1": hits_top1 / total_predictions,
        "top001pct": hits_top001pct / total_predictions,
        "top01pct": hits_top01pct / total_predictions,
        "top1pct": hits_top1pct / total_predictions,
        "top5pct": hits_top5pct / total_predictions,
        "nrmse": math.sqrt(sum_sq_error / total_predictions) / mean_norm,
    }

    # Set label
    suffix = " [strict]" if strict else ""
    if label:
        metrics["label"] = label
    elif lstm_mode:
        metrics["label"] = f">>> Qwen3-VL + LSTM (Ours){suffix} <<<"
    else:
        metrics["label"] = f">>> Qwen3-VL-Embedding (Ours){suffix} <<<"

    if strict:
        print(f"\n  Protocol: STRICT")
    print(f"\n  Aggregated: {total_traces} traces, {total_predictions} predictions")

    # Print comparison table
    print_comparison_table(metrics)

    # Write final_metrics.json to CWD
    final_json_path = os.path.join(os.getcwd(), "final_metrics.json")
    metrics_for_file = dict(metrics)
    metrics_for_file["total_traces"] = total_traces
    metrics_for_file["total_predictions"] = total_predictions
    metrics_for_file["N"] = N
    metrics_for_file["strict"] = strict
    with open(final_json_path, "w") as f:
        json.dump(metrics_for_file, f, indent=2)
    print(f"\n  Saved final metrics (JSON) to '{final_json_path}'")

    # Write final_metrics.csv for mail.py to pick up and attach
    csv_path = os.path.join(os.getcwd(), "final_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        for k, v in metrics_for_file.items():
            f.write(f"{k},{v}\n")
    print(f"  Saved final metrics (CSV) to '{csv_path}'")

    return metrics


# ============================================================================
# 6. Print Comparison Table
# ============================================================================
def print_comparison_table(our_metrics: dict):
    """Print a formatted table comparing our metrics to the paper's Table 3."""

    header = (
        f"{'Model':<45} {'Top-1':>8} {'Top-0.01%':>10} "
        f"{'Top-0.1%':>10} {'Top-1%':>8} {'Top-5%':>8} {'Norm.RMSE':>10}"
    )
    separator = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("  GUI Screen Prediction — Comparison with Paper Table 3")
    print(f"{'=' * len(header)}")
    print(header)
    print(separator)

    # Print our result first (highlighted)
    label = our_metrics.get("label", ">>> Qwen3-VL-Embedding (Ours) <<<")
    print(
        f"{label:<45} "
        f"{our_metrics['top1']:>8.4f} {our_metrics['top001pct']:>10.4f} "
        f"{our_metrics['top01pct']:>10.4f} {our_metrics['top1pct']:>8.4f} "
        f"{our_metrics['top5pct']:>8.4f} {our_metrics['nrmse']:>10.4f}"
    )
    print(separator)

    # Print paper's numbers
    for model_name, m in PAPER_TABLE3.items():
        print(
            f"{model_name:<45} "
            f"{m['top1']:>8.4f} {m['top001pct']:>10.4f} "
            f"{m['top01pct']:>10.4f} {m['top1pct']:>8.4f} "
            f"{m['top5pct']:>8.4f} {m['nrmse']:>10.4f}"
        )

    print(f"{'=' * len(header)}\n")

    # Print delta vs. best paper result (Screen2Vec-EUCLIDEAN-spatial)
    best = PAPER_TABLE3["Screen2Vec-EUCLIDEAN-spatial"]
    print("  Delta vs. Screen2Vec-EUCLIDEAN-spatial (best paper model):")
    for key, label in [
        ("top1", "Top-1"), ("top001pct", "Top-0.01%"),
        ("top01pct", "Top-0.1%"), ("top1pct", "Top-1%"),
        ("top5pct", "Top-5%"),
    ]:
        delta = our_metrics[key] - best[key]
        sign = "+" if delta >= 0 else ""
        status = "BETTER" if delta > 0 else ("SAME" if delta == 0 else "WORSE")
        print(f"    {label:<12}: {sign}{delta:.4f}  ({status})")

    rmse_delta = our_metrics["nrmse"] - best["nrmse"]
    sign = "+" if rmse_delta >= 0 else ""
    status = "BETTER" if rmse_delta < 0 else ("SAME" if rmse_delta == 0 else "WORSE")
    print(f"    {'Norm.RMSE':<12}: {sign}{rmse_delta:.4f}  ({status})")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Replicate Screen2Vec paper Table 3: GUI Screen Prediction "
                    "(parallelized version with HPC worker/aggregate modes)"
    )
    parser.add_argument(
        "--embeddings_dir", type=str, default="embeddings_4096_last",
        help="Directory containing .npz embedding files (default: embeddings_4096_last)",
    )
    parser.add_argument(
        "--traces_dir", type=str,
        default="../rico-data/data/traces/filtered_traces",
        help="Path to RICO filtered_traces directory",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1,
        help="Number of parallel jobs for evaluation (default: -1, use all cores)",
    )
    parser.add_argument(
        "--lstm_model", type=str, default=None,
        help="Path to trained LSTM predictor .pth file (from 8_train_lstm_predictor.py). "
             "When provided, uses LSTM prediction instead of mean-of-context, "
             "matching Screen2Vec's exact evaluation protocol.",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Custom label for the 'Ours' row in the output table.",
    )
    parser.add_argument(
        "--sample_fraction", type=float, default=None,
        help="Sample a fraction of RICO traces for evaluation (e.g., 0.1 for 10%% sample)",
    )
    parser.add_argument(
        "--n_predictors", type=int, default=4,
        help="Number of context screens used per prediction (Screen2Vec uses 4)",
    )
    parser.add_argument(
        "--trace_index", type=str, default=trace_index.DEFAULT_INDEX_PATH,
        help="Trace->layout index CSV from 0_build_trace_index.py. Required: "
             "trace gestures.json keys are per-trace indices, not the global "
             "layouts ids the embeddings are keyed by.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Use Screen2Vec's own evaluation population: traces with "
             ">= n_predictors+1 screens, exactly n_predictors context screens "
             "per prediction, candidate pool restricted to the screens of "
             "those traces, and int()-truncated top-k%% thresholds. "
             "Off by default; the default behaviour is unchanged. "
             "In worker mode, pass it to every worker.",
    )
    # ── Parallel HPC arguments ──────────────────────────────────────────
    parser.add_argument(
        "--job_json", type=str, default=None,
        help="Path to job.json (HPC worker mode). Contains batch_index, "
             "num_batches, and results_dir for trace splitting.",
    )
    parser.add_argument(
        "--aggregate", action="store_true",
        help="Aggregation mode: read partial results and compute final metrics.",
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Shared directory for partial results (required with --aggregate).",
    )
    args = parser.parse_args()

    # ══════════════════════════════════════════════════════════════════════
    # MODE A: Aggregation — read partial results, compute final metrics
    # ══════════════════════════════════════════════════════════════════════
    if args.aggregate:
        if not args.results_dir:
            parser.error("--results_dir is required when using --aggregate")
        aggregate_results(
            results_dir=args.results_dir,
            label=args.label,
            lstm_mode=args.lstm_model is not None,
        )
        return

    # ══════════════════════════════════════════════════════════════════════
    # Common setup for standalone and worker modes
    # ══════════════════════════════════════════════════════════════════════
    start = time.time()

    # Load job config if in worker mode
    job_config = None
    if args.job_json:
        with open(args.job_json, "r") as f:
            job_config = json.load(f)
        print(f"\n[Worker mode] Loaded job config from '{args.job_json}'")
        print(f"  batch_index = {job_config['batch_index']}")
        print(f"  num_batches = {job_config['num_batches']}")
        print(f"  results_dir = {job_config['results_dir']}")

    # 0. Load LSTM predictor (if provided)
    lstm_predictor = None
    if args.lstm_model:
        print(f"\n[Step 0/3] Loading LSTM predictor from '{args.lstm_model}'...")
        lstm_predictor = load_lstm_predictor(args.lstm_model)

    # 1. Load embeddings (all workers load the full set — needed for
    #    global cosine similarity ranking)
    print("\n[Step 1/3] Loading embeddings...")
    id_to_idx, emb_matrix, emb_normed, norms = load_all_embeddings(
        args.embeddings_dir
    )
    available_ids = set(id_to_idx.keys())
    print(f"  {len(available_ids)} screen embeddings available")

    # 2. Parse traces
    print("\n[Step 2/3] Parsing RICO interaction traces...")
    index = trace_index.load_index(args.trace_index)
    print(f"  Loaded trace->layout index: {len(index)} screens "
          f"('{args.trace_index}')")
    min_trace_len = (args.n_predictors + 1) if args.strict else 3
    traces = parse_traces(args.traces_dir, available_ids, index, min_trace_len)

    # Strict: the retrieval pool is the screens of the eligible traces only.
    # This runs on the FULL trace list, before the worker shard slice below,
    # so every worker derives an identical pool and identical thresholds.
    if args.strict:
        id_to_idx, emb_matrix, emb_normed, norms = restrict_pool_to_traces(
            traces, id_to_idx, emb_matrix, emb_normed, norms
        )
        print(f"  Strict pool: restricted to {len(id_to_idx)} screens "
              f"from eligible traces (was {len(available_ids)})")

    # ── Worker mode: deterministic contiguous trace split ────────────────
    if job_config:
        batch_index = job_config["batch_index"]
        num_batches = job_config["num_batches"]
        results_dir = job_config["results_dir"]

        total_traces = len(traces)
        chunk_size = math.ceil(total_traces / num_batches)
        start_idx = batch_index * chunk_size
        end_idx = min(start_idx + chunk_size, total_traces)
        traces = traces[start_idx:end_idx]
        print(f"\n  [Worker {batch_index}/{num_batches}] "
              f"Evaluating traces {start_idx}–{end_idx - 1} "
              f"({len(traces)} of {total_traces} traces)")

    # Optional sampling
    if args.sample_fraction is not None:
        np.random.seed(42)
        n_sample = max(1, int(len(traces) * args.sample_fraction))
        sample_indices = np.random.choice(len(traces), size=n_sample, replace=False)
        traces = [traces[i] for i in sample_indices]
        print(f"  Sampled {len(traces)} traces ({args.sample_fraction*100:.1f}%) for evaluation.")

    # 3. Evaluate
    print("\n[Step 3/3] Running screen prediction evaluation...")
    metrics, raw = evaluate_screen_prediction(
        traces, id_to_idx, emb_matrix, emb_normed, norms,
        n_jobs=args.n_jobs, lstm_predictor=lstm_predictor,
        strict=args.strict, n_predictors=args.n_predictors,
    )

    elapsed = time.time() - start

    # ══════════════════════════════════════════════════════════════════════
    # MODE B: Worker — save partial results and exit
    # ══════════════════════════════════════════════════════════════════════
    if job_config:
        raw["batch_index"] = batch_index
        raw["num_batches"] = num_batches
        raw["num_traces"] = len(traces)
        save_partial_results(results_dir, batch_index, raw)
        print(f"\nWorker {batch_index} finished in {elapsed:.1f}s")
        return

    # ══════════════════════════════════════════════════════════════════════
    # MODE C: Standalone — print results (same as original script)
    # ══════════════════════════════════════════════════════════════════════
    suffix = " [strict]" if args.strict else ""
    if args.label:
        metrics["label"] = args.label
    elif args.lstm_model:
        metrics["label"] = f">>> Qwen3-VL + LSTM (Ours){suffix} <<<"
    else:
        metrics["label"] = f">>> Qwen3-VL-Embedding (Ours){suffix} <<<"

    print_comparison_table(metrics)
    print(f"Total evaluation time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
