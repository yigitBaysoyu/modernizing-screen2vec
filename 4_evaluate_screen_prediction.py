"""
4_evaluate_screen_prediction.py

Replicates the Screen2Vec paper's Table 3: GUI Screen Prediction evaluation.

For each screen in each RICO interaction trace, uses the other screens in the
trace as context to form a prediction vector, then retrieves the nearest
neighbour from ALL screen embeddings. Computes Top-k accuracy and Normalized
RMSE, and prints a comparison table against the paper.

Two prediction modes:
  - Default (mean-of-context): averages the context screen embeddings
  - LSTM (--lstm_model): runs context through a trained LSTM predictor,
    matching Screen2Vec's exact evaluation protocol

Three evaluation protocols:
  - Permissive (default, neither flag): traces with >= 3 screens, context
    of 1..n_predictors screens, candidate pool = every embedding in
    --embeddings_dir, top-k%% thresholds rounded up.
  - N-back (--nback K): our own alternative protocol. Predicts a screen from
    exactly the K screens before it, sliding over the trace, so the target is
    always the *end* of the context chain:
      * only traces with >= K + 1 screens
      * only full windows: exactly K context screens per prediction
      * candidate pool restricted to the screens of those eligible traces
      * top-k%% thresholds rounded up, as in the permissive default
    Note that the eligible-trace population — and therefore the candidate pool
    and the top-k%% thresholds — shrinks as K grows, so numbers from different
    K are not directly comparable to each other.
  - Strict (--strict): reproduces the evaluation population of Screen2Vec's
    own modeltester.py / dataset.py:
      * only traces with >= n_predictors + 1 screens   (dataset.py:28,67)
      * only full windows: exactly n_predictors context screens per
        prediction, sliding over the trace                (dataset.py:29-37)
      * candidate pool restricted to the screens of those eligible traces
        (vocab.py:74-94 builds the pool from the filtered dataset)
      * top-k%% thresholds truncated with int(), not ceil (modeltester.py:141)
    Note that --strict cannot reproduce Screen2Vec's *screen ordering*: the
    original builds each trace from an unsorted os.listdir() of the
    view_hierarchies directory (UI_embedding/precompute_embeddings.py:72),
    which is not temporal order. This script always uses gestures.json order,
    which is true temporal order.

Usage:
    python 4_evaluate_screen_prediction.py [--embeddings_dir DIR] [--traces_dir DIR]
    python 4_evaluate_screen_prediction.py --embeddings_dir embeddings_seq_768 \\
        --lstm_model lstm_predictor_768d.pth
    python 4_evaluate_screen_prediction.py --embeddings_dir embeddings_seq_768_with_app_strict \\
        --lstm_model lstm_predictor_768d_strict.pth --strict
    python 4_evaluate_screen_prediction.py \\
        --embeddings_dir embeddings_seq_768_nback2_with_app_desc \\
        --lstm_model lstm_predictor_768d_nback2.pth --nback 2

--nback and --strict are mutually exclusive: they are two different protocols,
and --strict must keep reproducing Screen2Vec's exactly.
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
    full_windows=False,
):
    """Evaluates prediction metrics for a single RICO interaction trace.

    Instead of leaving one out from the whole trace (which gives future context),
    this strictly evaluates predicting the *next* screen given the *previous*
    screens (up to n_predictors), matching Screen2Vec's modeltester.py and the
    causal nature of the LSTM.

    full_windows=False (permissive default): predicts screens 1..T-1, so the
    first n_predictors predictions run on a truncated context of
    1..n_predictors-1 screens.
    full_windows=True (set by --strict and by --nback): only full windows, i.e.
    predicts screens n_predictors..T-1 with exactly n_predictors context screens
    each, which is what TesterRicoDataset.get_item_part (dataset.py:34-45)
    yields. Under --nback, n_predictors is the requested n-back depth.
    """
    trace_indices = [id_to_idx[sid] for sid in trace]
    trace_normed = embedding_matrix_normed[trace_indices]  # (T, D)
    trace_raw = embedding_matrix[trace_indices]            # (T, D)

    T = len(trace)
    if T < 2:
        return T, 0.0, 0, 0, 0, 0, 0, 0.0

    # Permissive: evaluate predictions for screens i = 1 to T-1.
    # Full windows (--strict / --nback): only i = n_predictors to T-1.
    first_target = n_predictors if full_windows else 1

    sq_err_sum = 0.0
    sum_rr = 0.0
    n_instances = 0
    hits_top1 = 0
    hits_top001pct = 0
    hits_top01pct = 0
    hits_top1pct = 0
    hits_top5pct = 0

    for i in range(first_target, T):
        n_instances += 1
        # Context is up to n_predictors previous screens (exactly n_predictors
        # under full_windows, since i >= n_predictors there)
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

        # Reciprocal rank. `rank` is the *pessimistic* rank under ties -- every
        # candidate at least as similar as the target counts -- which is the
        # same convention the top-k% hits above already use, so MRR does not
        # quietly grade on a different curve than the columns beside it.
        sum_rr += 1.0 / rank

    return (n_instances, sq_err_sum, hits_top1, hits_top001pct,
            hits_top01pct, hits_top1pct, hits_top5pct, sum_rr)


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
    nback: int = None,
):
    """
    For each prediction instance in each trace, run the preceding screens
    through the predictor and find the nearest neighbour in the candidate
    pool. Record the rank of the true screen.

    strict=True switches to Screen2Vec's own top-k%% truncation and full-window
    instances; the caller is responsible for having restricted the pool.

    nback=K restricts to full windows too (with n_predictors == K), but keeps
    the permissive ceil() thresholds: the int() truncation is part of
    reproducing modeltester.py, not of this repo's own protocol.
    """
    N = embedding_matrix.shape[0]

    # Only --strict predicts on full windows for Screen2Vec-compatibility
    # reasons; --nback does so because the n-back protocol is defined that way.
    full_windows = strict or nback is not None

    # Top-k% thresholds. Screen2Vec truncates when slicing the candidate list
    # (modeltester.py:141-146: temp[:int(0.0001 * comp_len)]); the permissive
    # default and --nback round up instead.
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
    elif nback is not None:
        print(f"\n  Protocol: N-BACK (predicting the end of the chain from "
              f"exactly {nback} preceding screens)")
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
                n_predictors=n_predictors, full_windows=full_windows,
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
                n_predictors=n_predictors, full_windows=full_windows,
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
    sum_reciprocal_rank = 0.0

    for res in results:
        T, sq_err_sum, h1, h001, h01, h1pct, h5pct, rr = res
        total_predictions += T
        sum_sq_error += sq_err_sum
        sum_reciprocal_rank += rr
        hits_top1 += h1
        hits_top001pct += h001
        hits_top01pct += h01
        hits_top1pct += h1pct
        hits_top5pct += h5pct

    # Compute final metrics
    metrics = {
        "top1": hits_top1 / total_predictions,
        "top001pct": hits_top001pct / total_predictions,
        "top01pct": hits_top01pct / total_predictions,
        "top1pct": hits_top1pct / total_predictions,
        "top5pct": hits_top5pct / total_predictions,
        "nrmse": math.sqrt(sum_sq_error / total_predictions) / mean_norm,
        # Mean Reciprocal Rank. Not part of the paper's Table 3 -- it is
        # reported alongside, never inside, the comparison table. It needs no
        # top-k% threshold, so unlike the Top-0.01%..5% columns it is not
        # sensitive to where those cut-offs happen to fall, while still being
        # dominated by the head of the ranking (1/rank decays fast). Read it
        # with the same caveat as Top-1: on an embeddings_*_with_app_desc run
        # the concatenated app half makes near-ranks mostly app identity, so
        # the *_wo_app variants are where this number says something about the
        # screen model.
        "mrr": float(sum_reciprocal_rank / total_predictions),
        # Population actually evaluated under this protocol, carried into
        # the metrics dict so print_comparison_table() can report it as the
        # last line of the run. --strict and --nback drop traces shorter
        # than n_predictors + 1, so this is smaller than the permissive
        # default and shrinks further as K grows.
        "n_traces": len(traces),
        "n_screens_in_traces": sum(len(t) for t in traces),
        "n_instances": total_predictions,
    }

    print(f"\n  Total prediction instances: {total_predictions}")
    return metrics


# ============================================================================
# 4. Print Comparison Table
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

    # MRR has no counterpart in Table 3, so it is printed beside the table
    # rather than as a column full of "n/a" for every paper row.
    print(f"  Supplementary (not in Table 3): MRR {our_metrics['mrr']:.4f}")

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

    # Printed last so the notebooks show the evaluated trace population at
    # the very bottom of their output, next to the numbers it produced.
    if "n_traces" in our_metrics:
        print()
        print(f"  Eligible traces evaluated: {our_metrics['n_traces']} "
              f"({our_metrics['n_screens_in_traces']} screens, "
              f"{our_metrics['n_instances']} prediction instances)")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Replicate Screen2Vec paper Table 3: GUI Screen Prediction"
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
        help="Sample a fraction of RICO traces for evaluation (e.g., 0.1 for 10% sample)",
    )
    parser.add_argument(
        "--n_predictors", type=int, default=None,
        help="Number of context screens used per prediction "
             "(default: 4, as in Screen2Vec)",
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
             "Off by default; the default behaviour is unchanged.",
    )
    parser.add_argument(
        "--nback", type=int, default=None, metavar="K",
        help="N-back protocol (alternative to --strict, mutually exclusive "
             "with it): predict a screen from exactly K preceding screens, "
             "sliding over the trace, so the target is always the end of the "
             "context chain. Keeps only traces with >= K+1 screens and "
             "restricts the candidate pool to their screens. Sets "
             "n_predictors = K.",
    )
    args = parser.parse_args()

    # --nback and --strict are different protocols; --strict must keep
    # reproducing Screen2Vec's exactly, so they cannot be combined.
    if args.nback is not None:
        if args.strict:
            parser.error("--nback and --strict are mutually exclusive")
        if args.nback < 1:
            parser.error("--nback must be >= 1")
        if args.n_predictors is not None and args.n_predictors != args.nback:
            parser.error(
                f"--n_predictors {args.n_predictors} contradicts --nback "
                f"{args.nback}; --nback already sets the context length"
            )
        args.n_predictors = args.nback
    elif args.n_predictors is None:
        args.n_predictors = 4

    start = time.time()

    # 0. Load LSTM predictor (if provided)
    lstm_predictor = None
    if args.lstm_model:
        print(f"\n[Step 0/3] Loading LSTM predictor from '{args.lstm_model}'...")
        lstm_predictor = load_lstm_predictor(args.lstm_model)

    # 1. Load embeddings
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
    full_windows = args.strict or args.nback is not None
    min_trace_len = (args.n_predictors + 1) if full_windows else 3
    traces = parse_traces(args.traces_dir, available_ids, index, min_trace_len)

    # Strict and n-back: the retrieval pool is the screens of the eligible
    # traces only. Must happen before any sampling so the pool (and therefore
    # the top-k% thresholds) is a pure function of the protocol, not of the
    # trace subset. Under --nback the pool shrinks as K grows, so numbers from
    # different K are not directly comparable.
    if full_windows:
        id_to_idx, emb_matrix, emb_normed, norms = restrict_pool_to_traces(
            traces, id_to_idx, emb_matrix, emb_normed, norms
        )
        label = "Strict" if args.strict else f"N-back (K={args.nback})"
        print(f"  {label} pool: restricted to {len(id_to_idx)} screens "
              f"from eligible traces (was {len(available_ids)})")

    if args.sample_fraction is not None:
        np.random.seed(42)
        n_sample = max(1, int(len(traces) * args.sample_fraction))
        sample_indices = np.random.choice(len(traces), size=n_sample, replace=False)
        traces = [traces[i] for i in sample_indices]
        print(f"  Sampled {len(traces)} traces ({args.sample_fraction*100:.1f}%) for evaluation.")

    # 3. Evaluate
    print("\n[Step 3/3] Running screen prediction evaluation...")
    metrics = evaluate_screen_prediction(
        traces, id_to_idx, emb_matrix, emb_normed, norms,
        n_jobs=args.n_jobs, lstm_predictor=lstm_predictor,
        strict=args.strict, n_predictors=args.n_predictors,
        nback=args.nback,
    )

    # Set label for output table
    if args.strict:
        suffix = " [strict]"
    elif args.nback is not None:
        suffix = f" [nback={args.nback}]"
    else:
        suffix = ""
    if args.label:
        metrics["label"] = args.label
    elif args.lstm_model:
        metrics["label"] = f">>> Qwen3-VL + LSTM (Ours){suffix} <<<"
    else:
        metrics["label"] = f">>> Qwen3-VL-Embedding (Ours){suffix} <<<"

    # 4. Print results
    print_comparison_table(metrics)

    elapsed = time.time() - start
    print(f"Total evaluation time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
