"""
4_evaluate_screen_prediction.py

Replicates the Screen2Vec paper's Table 3: GUI Screen Prediction evaluation.

For each screen in each RICO interaction trace, uses the other screens in the
trace as context, averages their embeddings to form a prediction vector, then
retrieves the nearest neighbour from ALL screen embeddings. Computes Top-k
accuracy and Normalized RMSE, and prints a comparison table against the paper.

Usage:
    python 4_evaluate_screen_prediction.py [--embeddings_dir DIR] [--traces_dir DIR]
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


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

    print(f"Loading {len(npz_files)} embeddings from '{embeddings_dir}'...")

    id_to_idx = {}
    embeddings = []

    for npz_path in tqdm(npz_files, desc="Loading embeddings"):
        screen_id = int(npz_path.stem)
        with np.load(npz_path) as f:
            emb = f["embedding"].astype(np.float32)
        id_to_idx[screen_id] = len(embeddings)
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
def parse_traces(traces_dir: str, available_ids: set):
    """
    Walk filtered_traces/<app>/trace_N/gestures.json to extract ordered
    screen sequences. Only keeps screens that have embeddings available.

    Returns list of traces, where each trace is a list of screen_id (int).
    """
    traces_root = Path(traces_dir)
    if not traces_root.exists():
        raise FileNotFoundError(f"Traces directory not found: {traces_dir}")

    traces = []
    skipped_short = 0
    total_screens_in_traces = 0

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

            # Keys of the gestures dict are screen IDs in trace order
            # (Python 3.7+ dicts preserve insertion order)
            screen_ids = []
            for key in gestures.keys():
                try:
                    sid = int(key)
                except ValueError:
                    continue
                if sid in available_ids:
                    screen_ids.append(sid)

            if len(screen_ids) < 3:
                skipped_short += 1
                continue

            traces.append(screen_ids)
            total_screens_in_traces += len(screen_ids)

    print(f"  Parsed {len(traces)} traces ({total_screens_in_traces} screens)")
    print(f"  Skipped {skipped_short} traces with < 3 screens")
    return traces


# ============================================================================
# 3. Evaluate Screen Prediction
# ============================================================================
def evaluate_screen_prediction(
    traces: list,
    id_to_idx: dict,
    embedding_matrix: np.ndarray,
    embedding_matrix_normed: np.ndarray,
    norms: np.ndarray,
):
    """
    For each screen in each trace, use all other screens in the trace as
    context, average their (normalized) embeddings, and find the nearest
    neighbour in the full embedding set. Record the rank of the true screen.
    """
    N = embedding_matrix.shape[0]

    # Top-k% thresholds (matching the paper)
    k_001pct = max(1, math.ceil(0.0001 * N))  # ~7
    k_01pct = max(1, math.ceil(0.001 * N))    # ~67
    k_1pct = max(1, math.ceil(0.01 * N))      # ~663
    k_5pct = max(1, math.ceil(0.05 * N))      # ~3314

    print(f"\n  Total screens in embedding index: {N}")
    print(f"  Top-0.01% threshold: k={k_001pct}")
    print(f"  Top-0.1%  threshold: k={k_01pct}")
    print(f"  Top-1%    threshold: k={k_1pct}")
    print(f"  Top-5%    threshold: k={k_5pct}")

    # For Normalized RMSE: mean embedding norm
    mean_norm = float(np.mean(norms))
    print(f"  Mean embedding L2 norm: {mean_norm:.4f}")

    # Accumulators
    total_predictions = 0
    hits_top1 = 0
    hits_top001pct = 0
    hits_top01pct = 0
    hits_top1pct = 0
    hits_top5pct = 0
    sum_sq_error = 0.0

    for trace in tqdm(traces, desc="Evaluating traces"):
        # Pre-fetch the normalized embeddings for this trace
        trace_indices = [id_to_idx[sid] for sid in trace]
        trace_normed = embedding_matrix_normed[trace_indices]  # (T, D)
        trace_raw = embedding_matrix[trace_indices]            # (T, D)

        T = len(trace)

        # RMSE: raw (unnormalized) average of other screens in trace
        trace_raw_sum = trace_raw.sum(axis=0, keepdims=True)  # (1, D)
        pred_raw = (trace_raw_sum - trace_raw) / (T - 1)      # (T, D)
        sq_errs = np.sum((pred_raw - trace_raw) ** 2, axis=1) # (T,)
        sum_sq_error += float(np.sum(sq_errs))

        # Prediction: mean of normalized other screens, then L2-normalize
        trace_normed_sum = trace_normed.sum(axis=0, keepdims=True)  # (1, D)
        pred = (trace_normed_sum - trace_normed) / (T - 1)          # (T, D)
        pred_norm = np.linalg.norm(pred, axis=1, keepdims=True)     # (T, 1)
        pred_normed = pred / np.maximum(pred_norm, 1e-12)           # (T, D)

        # Cosine similarity against ALL screen embeddings
        similarities = cosine_similarity(pred_normed, embedding_matrix_normed)

        # Rank of target similarity: count screens with similarity >= target's similarity
        target_sims = similarities[np.arange(T), trace_indices]     # (T,)
        ranks = np.sum(similarities >= target_sims[:, None], axis=1) # (T,)

        # Accumulate top-k hits
        hits_top1 += int(np.sum(ranks <= 1))
        hits_top001pct += int(np.sum(ranks <= k_001pct))
        hits_top01pct += int(np.sum(ranks <= k_01pct))
        hits_top1pct += int(np.sum(ranks <= k_1pct))
        hits_top5pct += int(np.sum(ranks <= k_5pct))

        total_predictions += T

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
    print(
        f"{'>>> Qwen3-VL-Embedding (Ours) <<<':<45} "
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
        description="Replicate Screen2Vec paper Table 3: GUI Screen Prediction"
    )
    parser.add_argument(
        "--embeddings_dir", type=str, default="embeddings_4096",
        help="Directory containing .npz embedding files (default: embeddings_4096)",
    )
    parser.add_argument(
        "--traces_dir", type=str,
        default="../rico-data/data/traces/filtered_traces",
        help="Path to RICO filtered_traces directory",
    )
    args = parser.parse_args()

    start = time.time()

    # 1. Load embeddings
    print("\n[Step 1/3] Loading embeddings...")
    id_to_idx, emb_matrix, emb_normed, norms = load_all_embeddings(
        args.embeddings_dir
    )
    available_ids = set(id_to_idx.keys())
    print(f"  {len(available_ids)} screen embeddings available")

    # 2. Parse traces
    print("\n[Step 2/3] Parsing RICO interaction traces...")
    traces = parse_traces(args.traces_dir, available_ids)

    # 3. Evaluate
    print("\n[Step 3/3] Running screen prediction evaluation...")
    metrics = evaluate_screen_prediction(
        traces, id_to_idx, emb_matrix, emb_normed, norms
    )

    # 4. Print results
    print_comparison_table(metrics)

    elapsed = time.time() - start
    print(f"Total evaluation time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
