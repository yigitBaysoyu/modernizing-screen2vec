"""
13_evaluate_component_prediction.py

Replicates the Screen2Vec paper's Table 2: GUI Component Prediction evaluation.
The component-level counterpart of 4_evaluate_screen_prediction.py.

For each GUI component that carries text, the n components closest to it on the
same screen are fed to the trained predictor, and the predicted text vector is
ranked against the embeddings of *all* component texts in RICO -- "the model
tries to predict the text content for each GUI component in all the GUI screens
in the Rico dataset using its context (the other GUI components around it) among
the collection of all the GUI components in the Rico dataset" (paper, Sec. 2.5).

Reported, exactly as in Table 2: Top-1, Top-0.01%, Top-0.1%, Top-1%, Top-5% and
Top-10% accuracy, plus two supplementary numbers the paper's own modeltester.py
prints (GUI class accuracy and normalized RMSE).

Prediction modes:
  --model <checkpoint>   trained ComponentPredictor from 12_train_...py
  --model none           mean-of-context baseline (no training), the component
                         analogue of the mean-of-context mode in script 4

Usage:
    python 13_evaluate_component_prediction.py --data_dir component_data \\
        --model component_predictor_euclidean.pth
    python 13_evaluate_component_prediction.py --data_dir component_data \\
        --model component_predictor_hierarchical.pth --label "Ours-HIERARCHICAL"
"""

import argparse
import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from component_utils import (
    COMP_CLS,
    COMP_TEXT,
    apply_vocab_pca,
    build_context_tensors,
    build_predictor_from_checkpoint,
    fit_vocab_pca,
    load_component_data,
)
from importlib import import_module

# Sample construction is shared with the trainer (module name starts with a
# digit, so it cannot be imported with the usual syntax).
_trainer = import_module("12_train_component_predictor")
build_samples = _trainer.build_samples
screen_component_arrays = _trainer.screen_component_arrays


# ============================================================================
# Paper's Table 2 reference numbers
# ============================================================================
PAPER_TABLE2 = {
    "Screen2Vec-EUCLIDEAN-text": {
        "top1": 0.443, "top001pct": 0.619, "top01pct": 0.783,
        "top1pct": 0.856, "top5pct": 0.885, "top10pct": 0.901,
    },
    "Screen2Vec-HIERARCHICAL-text": {
        "top1": 0.588, "top001pct": 0.687, "top01pct": 0.798,
        "top1pct": 0.849, "top5pct": 0.878, "top10pct": 0.894,
    },
}


# ============================================================================
# 1. Collect the evaluation targets
# ============================================================================
def collect_targets(screen_ids, per_screen, targets_mode, rng):
    """Flatten the (screen, target component) pairs to evaluate.

    "all"            : every text-bearing component on every screen
    "one_per_screen" : one random component per screen, matching the sampling
                       in Screen2Vec's own ScreenDataset
    """
    picks = []
    for screen_idx in screen_ids:
        entry = per_screen[screen_idx]
        if entry is None:
            continue
        n_targets = len(entry[0])
        if targets_mode == "one_per_screen":
            picks.append((screen_idx, rng.integers(n_targets)))
        else:
            picks.extend((screen_idx, slot) for slot in range(n_targets))
    return picks


def materialise(picks, per_screen, comp_arrays):
    """Turn (screen, slot) picks into samples + their text/class labels."""
    samples = []
    text_labels = np.empty(len(picks), dtype=np.int64)
    class_labels = np.empty(len(picks), dtype=np.int64)
    for i, (screen_idx, slot) in enumerate(picks):
        targets, contexts = per_screen[screen_idx]
        comps = comp_arrays[screen_idx]
        ctx_pos = contexts[slot]
        ctx_pos = ctx_pos[ctx_pos >= 0]
        target_row = comps[targets[slot]]
        samples.append((comps[ctx_pos], target_row))
        text_labels[i] = target_row[COMP_TEXT]
        class_labels[i] = target_row[COMP_CLS]
    return samples, text_labels, class_labels


# ============================================================================
# 2. Prediction
# ============================================================================
def predict_batch(model, samples, vocab_feats, max_context, use_spatial, device):
    """Predicted text vectors (B, P) and class predictions (B,) for one batch."""
    ctx_text, ctx_cls, ctx_spatial, ctx_mask = build_context_tensors(
        samples, vocab_feats, max_context, use_spatial
    )
    if model is None:
        # Baseline: the mean of the context text embeddings. Rows without text
        # sit at the origin (see 11_embed_component_vocab.py), so they drop out.
        ctx = ctx_text.numpy()
        pred = ctx.sum(axis=1)
        norms = np.linalg.norm(pred, axis=1, keepdims=True)
        pred = pred / np.maximum(norms, 1e-12)
        return pred.astype(np.float32), np.zeros(len(samples), dtype=np.int64)

    with torch.no_grad():
        text_pred, class_logits = model(
            ctx_text.to(device), ctx_cls.to(device),
            ctx_spatial.to(device), ctx_mask.to(device),
        )
    return text_pred.cpu().numpy(), class_logits.argmax(dim=-1).cpu().numpy()


# ============================================================================
# 3. Ranking against the whole vocabulary
# ============================================================================
def rank_batch(preds, text_labels, candidates, device):
    """Rank of the correct text among all vocabulary entries, 1 = best.

    `candidates` holds the vocabulary without the empty-text row (index 0): the
    empty string is not a GUI component text, it only marks components that have
    none. Also returns the squared prediction error per instance, for the
    normalized RMSE that Screen2Vec's modeltester.py reports.
    """
    block = torch.from_numpy(preds).to(device)
    sims = block @ candidates.T                                      # (b, V-1)
    target_idx = torch.from_numpy(text_labels - 1).to(device)
    target_sim = sims.gather(1, target_idx.unsqueeze(1))
    ranks = (sims >= target_sim).sum(dim=1).cpu().numpy()
    sq_err = ((block - candidates[target_idx]) ** 2).sum(dim=1).cpu().numpy()
    return ranks, sq_err


def metrics_from_ranks(ranks, sq_err, vocab_candidates, class_correct, n_total):
    """Table 2's accuracies plus the two supplementary numbers."""
    thresholds = {
        "top1": 1,
        "top001pct": max(1, math.ceil(0.0001 * vocab_candidates)),
        "top01pct": max(1, math.ceil(0.001 * vocab_candidates)),
        "top1pct": max(1, math.ceil(0.01 * vocab_candidates)),
        "top5pct": max(1, math.ceil(0.05 * vocab_candidates)),
        "top10pct": max(1, math.ceil(0.10 * vocab_candidates)),
    }
    out = {key: float(np.mean(ranks <= k)) for key, k in thresholds.items()}
    # Normalized RMSE: prediction error over the mean length of the true
    # vectors, which is 1.0 here because the candidates are L2-normalised.
    out["nrmse"] = float(math.sqrt(sq_err.mean()))
    out["class_top1"] = float(class_correct / max(1, n_total))
    out["_thresholds"] = thresholds
    return out


# ============================================================================
# 4. Print comparison table
# ============================================================================
def print_comparison_table(our_metrics, label):
    header = (
        f"{'Model':<45} {'Top-1':>8} {'Top-0.01%':>10} {'Top-0.1%':>10} "
        f"{'Top-1%':>8} {'Top-5%':>8} {'Top-10%':>9}"
    )
    separator = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("  GUI Component Prediction — Comparison with Paper Table 2")
    print(f"{'=' * len(header)}")
    print(header)
    print(separator)
    print(
        f"{label:<45} "
        f"{our_metrics['top1']:>8.4f} {our_metrics['top001pct']:>10.4f} "
        f"{our_metrics['top01pct']:>10.4f} {our_metrics['top1pct']:>8.4f} "
        f"{our_metrics['top5pct']:>8.4f} {our_metrics['top10pct']:>9.4f}"
    )
    print(separator)
    for model_name, m in PAPER_TABLE2.items():
        print(
            f"{model_name:<45} "
            f"{m['top1']:>8.4f} {m['top001pct']:>10.4f} "
            f"{m['top01pct']:>10.4f} {m['top1pct']:>8.4f} "
            f"{m['top5pct']:>8.4f} {m['top10pct']:>9.4f}"
        )
    print(f"{'=' * len(header)}\n")

    print(f"  Supplementary: GUI class Top-1 {our_metrics['class_top1']:.4f}  |  "
          f"Normalized RMSE {our_metrics['nrmse']:.4f}")

    best_name = max(PAPER_TABLE2, key=lambda k: PAPER_TABLE2[k]["top1"])
    best = PAPER_TABLE2[best_name]
    print(f"\n  Delta vs. {best_name} (best paper model):")
    for key, name in [
        ("top1", "Top-1"), ("top001pct", "Top-0.01%"), ("top01pct", "Top-0.1%"),
        ("top1pct", "Top-1%"), ("top5pct", "Top-5%"), ("top10pct", "Top-10%"),
    ]:
        delta = our_metrics[key] - best[key]
        sign = "+" if delta >= 0 else ""
        status = "BETTER" if delta > 0 else ("SAME" if delta == 0 else "WORSE")
        print(f"    {name:<12}: {sign}{delta:.4f}  ({status})")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Replicate Screen2Vec paper Table 2: GUI Component Prediction"
    )
    parser.add_argument("--data_dir", type=str, default="component_data",
                        help="Directory written by 10_extract_components.py")
    parser.add_argument("--model", type=str, default="component_predictor_euclidean.pth",
                        help="Trained predictor .pth, or 'none' for the "
                             "mean-of-context baseline")
    parser.add_argument("--vocab_emb", type=str, default=None,
                        help="Vocabulary embeddings (default: <data_dir>/vocab_emb.npy)")
    parser.add_argument("--distance", choices=("euclidean", "hierarchical"), default=None,
                        help="Context distance (default: whatever the model was trained with)")
    parser.add_argument("--n_context", type=int, default=None,
                        help="Context size (default: the model's training value)")
    parser.add_argument("--pred_dim", type=int, default=512,
                        help="PCA dimensionality for the baseline mode; a trained "
                             "model brings its own PCA")
    parser.add_argument("--eval_split", choices=("val", "all"), default="val",
                        help="'val' uses the screens held out during training, "
                             "'all' scores every screen in the dataset")
    parser.add_argument("--targets", choices=("all", "one_per_screen"), default="all",
                        help="Evaluate every text-bearing component, or one per "
                             "screen as Screen2Vec's ScreenDataset samples")
    parser.add_argument("--max_targets", type=int, default=None,
                        help="Randomly subsample this many prediction instances")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--rank_chunk", type=int, default=1024,
                        help="Predictions ranked per matmul against the vocabulary")
    parser.add_argument("--limit_screens", type=int, default=None,
                        help="Only use the first N screens (smoke tests)")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--label", type=str, default=None,
                        help="Label for the 'Ours' row in the output table")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    start = time.time()

    # ---------------------------------------------------------------
    # 1. Model
    # ---------------------------------------------------------------
    use_model = args.model.lower() not in ("none", "")
    checkpoint = None
    model = None
    if use_model:
        print(f"\n[Step 1/4] Loading predictor from '{args.model}'...")
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
        model = build_predictor_from_checkpoint(checkpoint, device)
        print(f"  feat_dim={checkpoint['feat_dim']} | distance={checkpoint['distance']} "
              f"| n_context={checkpoint['n_context']} "
              f"| spatial={'on' if checkpoint['use_spatial'] else 'off'} "
              f"| best_val_loss={checkpoint['best_val_loss']:.4f}")
    else:
        print("\n[Step 1/4] No model — mean-of-context baseline")

    distance = args.distance or (checkpoint["distance"] if checkpoint else "euclidean")
    n_context = args.n_context or (checkpoint["n_context"] if checkpoint else 16)
    use_spatial = checkpoint["use_spatial"] if checkpoint else False

    # ---------------------------------------------------------------
    # 2. Data
    # ---------------------------------------------------------------
    print(f"\n[Step 2/4] Loading component data from '{args.data_dir}'...")
    screens, vocab, meta = load_component_data(args.data_dir, limit=args.limit_screens)
    vocab_emb_path = args.vocab_emb or (
        checkpoint["vocab_emb"] if checkpoint else os.path.join(args.data_dir, "vocab_emb.npy")
    )
    vocab_emb = np.load(vocab_emb_path)
    if vocab_emb.shape[0] != len(vocab):
        raise ValueError(
            f"'{vocab_emb_path}' has {vocab_emb.shape[0]} rows but the vocabulary "
            f"has {len(vocab)} entries — re-run 11_embed_component_vocab.py"
        )

    if checkpoint is not None:
        pca_mean, pca_components = checkpoint["pca_mean"], checkpoint["pca_components"]
    else:
        pca_mean, pca_components = fit_vocab_pca(vocab_emb, args.pred_dim, seed=args.seed)
    vocab_feats = apply_vocab_pca(vocab_emb, pca_mean, pca_components)
    vocab_feats[0] = 0.0
    print(f"  Screens: {len(screens)}  |  Vocabulary: {len(vocab)} "
          f"({vocab_emb.shape[1]}d -> {vocab_feats.shape[1]}d)")

    print(f"\n  Building {distance} contexts (n={n_context})...")
    comp_arrays = screen_component_arrays(screens)
    per_screen = build_samples(comp_arrays, n_context, distance, n_jobs=args.n_jobs)

    # ---------------------------------------------------------------
    # 3. Targets
    # ---------------------------------------------------------------
    print(f"\n[Step 3/4] Selecting evaluation targets (split={args.eval_split}, "
          f"targets={args.targets})...")
    if args.eval_split == "val" and checkpoint is not None:
        held_out = set(checkpoint["val_screen_ids"])
        screen_ids = [
            i for i, s in enumerate(screens)
            if int(s["id"]) in held_out and per_screen[i] is not None
        ]
        print(f"  Using the {len(screen_ids)} screens held out during training")
    else:
        screen_ids = [i for i, s in enumerate(per_screen) if s is not None]
        if args.eval_split == "val":
            print("  No checkpoint to take a split from — scoring every screen")
        print(f"  Using all {len(screen_ids)} usable screens")

    picks = collect_targets(screen_ids, per_screen, args.targets, rng)
    if args.max_targets and len(picks) > args.max_targets:
        chosen = rng.choice(len(picks), size=args.max_targets, replace=False)
        picks = [picks[i] for i in np.sort(chosen)]
        print(f"  Subsampled to {len(picks)} prediction instances")
    print(f"  Prediction instances: {len(picks)}")

    # ---------------------------------------------------------------
    # 4. Predict + rank
    # ---------------------------------------------------------------
    print("\n[Step 4/4] Running component prediction evaluation...")
    vocab_candidates = vocab_feats.shape[0] - 1  # the empty-text row is not a candidate
    print(f"  Ranked against {vocab_candidates} component texts")

    candidates = torch.from_numpy(vocab_feats[1:]).to(device)  # (V-1, P)
    ranks = np.empty(len(picks), dtype=np.int64)
    sq_err = np.empty(len(picks), dtype=np.float64)
    class_correct = 0

    # Predict and rank one batch at a time: with every text-bearing component as
    # a target there are hundreds of thousands of instances, so nothing bigger
    # than a batch of prediction vectors is ever held at once.
    for start_idx in tqdm(range(0, len(picks), args.batch_size), desc="Predicting"):
        batch_picks = picks[start_idx:start_idx + args.batch_size]
        samples, batch_text, batch_class = materialise(batch_picks, per_screen, comp_arrays)
        pred, class_pred = predict_batch(
            model, samples, vocab_feats, n_context, use_spatial, device
        )
        for chunk_start in range(0, len(batch_picks), args.rank_chunk):
            chunk_stop = min(chunk_start + args.rank_chunk, len(batch_picks))
            r, e = rank_batch(
                pred[chunk_start:chunk_stop], batch_text[chunk_start:chunk_stop],
                candidates, device,
            )
            ranks[start_idx + chunk_start:start_idx + chunk_stop] = r
            sq_err[start_idx + chunk_start:start_idx + chunk_stop] = e
        if model is not None:
            class_correct += int((class_pred == batch_class).sum())

    our_metrics = metrics_from_ranks(
        ranks, sq_err, vocab_candidates, class_correct, len(picks)
    )

    thresholds = our_metrics.pop("_thresholds")
    print(f"\n  Top-0.01% threshold: k={thresholds['top001pct']}")
    print(f"  Top-0.1%  threshold: k={thresholds['top01pct']}")
    print(f"  Top-1%    threshold: k={thresholds['top1pct']}")
    print(f"  Top-5%    threshold: k={thresholds['top5pct']}")
    print(f"  Top-10%   threshold: k={thresholds['top10pct']}")
    print(f"  Median rank of the correct text: {int(np.median(ranks))}")

    label = args.label or (
        f">>> Qwen3-VL + Transformer ({distance[:4].upper()}, Ours) <<<"
        if model is not None else ">>> Mean-of-context baseline (Ours) <<<"
    )
    print_comparison_table(our_metrics, label)

    print(f"Total evaluation time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
