"""
9_train_joint.py

Joint end-to-end training of SequenceProjector + LSTMPredictor.

This script replaces the two-step pipeline of:
  5_train_sequence_projector.py  (projector only, InfoNCE loss)
  8_train_lstm_predictor.py      (LSTM only, frozen projector output)

with a single training loop where gradients from the prediction loss flow
through the LSTM combiner AND the projector, so the projector learns to
produce embeddings specifically optimized for the next-screen prediction task.

This mirrors Screen2Vec's original training where Screen2Vec (the embedder)
and TracePredictor (the LSTM) are trained jointly with a single optimizer.

Architecture:
  raw 4096-dim embedding → SequenceProjector → 768-dim → LSTM → prediction
                                                                   ↓
                                                          CE + negative sampling loss
                                                                   ↓
                                                     backprop through both models

Outputs:
  - Projected embeddings in embeddings_seq_{dim}/ (same format as step 5)
  - LSTM model checkpoint lstm_predictor_{dim}d.pth (same format as step 8)
  - Projector model checkpoint sequence_projector_{dim}d.pth

Usage:
    python 9_train_joint.py --proj_dim 768

    # Then continue with the existing pipeline:
    python 6_generate_app_context_embeddings.py \\
        --embeddings_dir embeddings_seq_768 \\
        --output_dir embeddings_seq_768_with_app

    python 4_evaluate_screen_prediction.py \\
        --embeddings_dir embeddings_seq_768_with_app \\
        --lstm_model lstm_predictor_768d.pth

    # Screen2Vec's own protocol end to end. --strict must be used for BOTH
    # training and evaluation; every artefact gets a _strict suffix so it
    # cannot clobber a permissive run:
    python 9_train_joint.py --proj_dim 768 --strict
    python 6_generate_app_context_embeddings.py \\
        --embeddings_dir embeddings_seq_768_strict \\
        --output_dir embeddings_seq_768_strict_with_app
    python 4_evaluate_screen_prediction.py \\
        --embeddings_dir embeddings_seq_768_strict_with_app \\
        --lstm_model lstm_predictor_768d_strict.pth --strict
"""

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import trace_index


# ============================================================================
# Models (same architectures as steps 5 and 8)
# ============================================================================
class SequenceProjector(nn.Module):
    """
    Projects pre-computed Qwen3-VL embeddings (4096-dim) into a lower-dimensional
    space. Same architecture as 5_train_sequence_projector.py.
    """

    def __init__(self, input_dim: int = 4096, proj_dim: int = 768, hidden_dim: int = 1024):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(input_dim, proj_dim, bias=False)
            nn.init.orthogonal_(self.net.weight)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=False),
                nn.ReLU(),
                nn.Linear(hidden_dim, proj_dim, bias=False),
            )
            nn.init.orthogonal_(self.net[0].weight)
            nn.init.orthogonal_(self.net[2].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class LSTMPredictor(nn.Module):
    """
    Replicates Screen2Vec's TracePredictor LSTM combiner.
    Same architecture as 8_train_lstm_predictor.py.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.combiner = nn.LSTM(embed_dim, embed_dim, batch_first=True)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.combiner(context)
        pred = h[0]  # (B, embed_dim)
        return F.normalize(pred, p=2, dim=-1)


# ============================================================================
# Data Loading (reused from step 8)
# ============================================================================
def load_all_embeddings(embeddings_dir: str):
    """Load all .npz embeddings in parallel into a matrix and id mapping."""
    npz_files = sorted(Path(embeddings_dir).glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in '{embeddings_dir}'")

    print(f"Loading {len(npz_files)} embeddings from '{embeddings_dir}' in parallel...")

    def load_single(npz_path):
        screen_id = int(npz_path.stem)
        with np.load(npz_path) as f:
            emb = f["embedding"].astype(np.float32)
        return screen_id, emb

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(load_single, npz_files),
            total=len(npz_files),
            desc="Loading embeddings",
        ))

    id_to_idx = {}
    embeddings = []
    for idx, (screen_id, emb) in enumerate(results):
        id_to_idx[screen_id] = idx
        embeddings.append(emb)

    matrix = np.stack(embeddings, axis=0)  # (N, D)
    print(f"  Loaded matrix shape: {matrix.shape}")
    return id_to_idx, matrix


def parse_traces(traces_dir: str, available_ids: set, index: dict,
                 min_trace_len: int = 3):
    """Parse RICO interaction traces in parallel, returning layouts-id lists.

    index: the trace->layout index from trace_index.load_index(). A trace's
    gestures.json keys are per-trace indices, NOT the global layouts ids the
    embeddings are keyed by, so every key must be resolved through it. See
    trace_index.py for why.

    min_trace_len: traces with fewer screens are skipped. Default 3 keeps the
    permissive behaviour; --strict passes n_predictors + 1, matching
    Screen2Vec's RicoDataset (dataset.py:28,166). With that filter every
    training window has exactly n_predictors context screens, so the
    zero-padding branch in compute_joint_loss never fires and training matches
    what 4_evaluate_screen_prediction.py --strict feeds the model at test time.
    """
    traces_root = Path(traces_dir)
    if not traces_root.exists():
        raise FileNotFoundError(f"Traces directory not found: {traces_dir}")

    trace_dirs = []
    app_dirs = sorted(traces_root.iterdir())
    for app_dir in app_dirs:
        if not app_dir.is_dir():
            continue
        for trace_dir in sorted(app_dir.iterdir()):
            if trace_dir.is_dir() and trace_dir.name.startswith("trace_"):
                trace_dirs.append(trace_dir)

    def parse_single_trace(trace_dir):
        gestures_path = trace_dir / "gestures.json"
        if not gestures_path.exists():
            return None
        try:
            with open(gestures_path, "r", encoding="utf-8") as f:
                gestures = json.load(f)
        except Exception:
            return None
        app_name = trace_dir.parent.name
        screen_ids = []
        for key in gestures.keys():
            try:
                screen_key = int(key)
            except ValueError:
                continue
            layout_id = index.get((app_name, trace_dir.name, screen_key))
            if layout_id is None:
                continue
            if layout_id in available_ids:
                screen_ids.append(layout_id)
        if len(screen_ids) < min_trace_len:
            return None
        return screen_ids

    print(f"Parsing {len(trace_dirs)} traces in parallel...")
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(parse_single_trace, trace_dirs),
            total=len(trace_dirs),
            desc="Parsing traces",
        ))

    traces = [t for t in results if t is not None]
    print(f"  Parsed {len(traces)} traces")
    return traces


# ============================================================================
# Joint Loss: CrossEntropy with Negative Sampling
# ============================================================================
# Sentinel used to pad the ragged hard-negative block. 0.0 is an ordinary
# similarity value, so padding with it injects fake negatives whose count
# scales with trace length; this value is mapped to zero weight by softmax.
HARD_NEG_PAD = -1e4


def select_trace_negatives(trace, id_to_idx, target_idx, ctx_indices, mode):
    """
    Pick the same-trace screens that go into the softmax denominator.

    "context" reproduces Screen2Vec: pretrainer.py:106-107 computes
    `neg_self_dot_products` against `context`, and prediction.py:38 defines
    `context` as exactly the n_predictors screens that were fed to the LSTM.
    Those screens remain in the evaluation candidate pool, so training against
    them is what stops the predictor from simply reproducing the last context
    screen.

    "other" is the behaviour this repo used before 2026-08-22: the remaining
    trace screens, with context excluded. "both" is the union.
    """
    if mode == "context":
        return list(dict.fromkeys(i for i in ctx_indices if i != target_idx))
    rest = sorted(
        set(id_to_idx[sid] for sid in trace) - {target_idx} - set(ctx_indices)
    )
    if mode == "other":
        return rest
    if mode == "both":
        return list(dict.fromkeys(i for i in ctx_indices if i != target_idx)) + rest
    raise ValueError("unknown trace_negatives mode: %r" % (mode,))


def compute_joint_loss(
    projector: SequenceProjector,
    predictor: LSTMPredictor,
    raw_emb_tensor: torch.Tensor,
    batch_traces: list,
    id_to_idx: dict,
    neg_sample_size: int,
    device: torch.device,
    n_predictors: int = 4,
    temperature: float = 0.07,
    neg_pool_indices: torch.Tensor = None,
    trace_negatives: str = "context",
) -> torch.Tensor:
    """
    Joint loss that projects raw embeddings through the projector on-the-fly,
    then predicts through the LSTM. Gradients flow through both models.

    This matches Screen2Vec's training protocol (pretrainer.py) where:
      c, result, context = self.predictor(UIs, descr, trace_screen_lengths, layouts)
      prediction_loss.backward()  # gradients flow through TracePredictor AND Screen2Vec

    Key difference from step 8: Instead of looking up pre-computed embeddings
    (which has no gradient), we forward-pass through the projector each time.

    neg_pool_indices: if given, random negatives are drawn only from these
    embedding rows. Screen2Vec samples negatives from its ScreenVocab, which
    is built from the *filtered* dataset (vocab.py:12-20), so screens outside
    the eligible traces are never negatives. Default None keeps sampling from
    the full embedding matrix.

    trace_negatives: which same-trace screens enter the softmax denominator as
    hard negatives. "context" (default) uses the window's own context screens,
    as Screen2Vec does (pretrainer.py:106-107 over prediction.py:38); "other"
    uses the remaining trace screens and excludes context, the behaviour before
    2026-08-22; "both" is the union. See select_trace_negatives.
    """
    B = len(batch_traces)

    context_list = []       # list of (T_ctx, D) tensors
    target_list = []        # list of (D,) tensors
    same_trace_list = []    # list of (n_same, D) tensors or None

    for trace in batch_traces:
        n = min(n_predictors + 1, len(trace))
        start = random.randint(0, len(trace) - n)
        window = trace[start : start + n]

        target_idx = id_to_idx[window[-1]]
        ctx_indices = [id_to_idx[sid] for sid in window[:-1]]

        # *** Key change: project raw embeddings through projector ***
        # This creates a computation graph that connects to projector weights
        raw_target = raw_emb_tensor[target_idx].unsqueeze(0).to(device)  # (1, 4096)
        proj_target = projector(raw_target).squeeze(0)                    # (D,)
        target_list.append(proj_target)

        raw_ctx = raw_emb_tensor[ctx_indices].to(device)                  # (T_ctx, 4096)
        proj_ctx = projector(raw_ctx)                                     # (T_ctx, D)
        context_list.append(proj_ctx)

        # Same-trace hard negatives (also projected through projector)
        hard_neg_indices = select_trace_negatives(
            trace, id_to_idx, target_idx, ctx_indices, trace_negatives
        )
        if hard_neg_indices:
            raw_hn = raw_emb_tensor[hard_neg_indices].to(device)          # (n_hn, 4096)
            proj_hn = projector(raw_hn)                                    # (n_hn, D)
            same_trace_list.append(proj_hn)
        else:
            same_trace_list.append(None)

    # Pad context sequences for LSTM batch processing
    proj_dim = target_list[0].shape[0]
    max_ctx_len = max(c.shape[0] for c in context_list)
    ctx_padded = torch.zeros(B, max_ctx_len, proj_dim, device=device)
    for i, ctx in enumerate(context_list):
        ctx_padded[i, : ctx.shape[0]] = ctx

    targets = torch.stack(target_list)  # (B, D)

    # LSTM forward pass
    predictions = predictor(ctx_padded)  # (B, D)

    # --- Dot products (matches Screen2Vec's pretrainer.py) ---

    # 1. Positive: prediction vs all targets → (B, B), diagonal = positive
    pos_dot = torch.mm(predictions, targets.T)  # (B, B)

    # 2. Random negative sample from the vocab (also projected)
    if neg_pool_indices is None:
        neg_indices = torch.randint(0, raw_emb_tensor.shape[0], (neg_sample_size,))
    else:
        neg_indices = neg_pool_indices[
            torch.randint(0, neg_pool_indices.shape[0], (neg_sample_size,))
        ]
    raw_neg = raw_emb_tensor[neg_indices].to(device)    # (neg_sample_size, 4096)
    proj_neg = projector(raw_neg)                        # (neg_sample_size, D)
    neg_dot = torch.mm(predictions, proj_neg.T)          # (B, neg_sample_size)

    # 3. Hard negatives: same-trace screens (Screen2Vec pretrainer.py:106-107)
    # Row i of the block must belong to batch item i: a trace can yield no hard
    # negatives (in "other" mode every trace of exactly n_predictors+1 screens
    # does), and compacting those rows away would file one item's logits under
    # another item's label.
    max_hn = max((0 if hn is None else hn.shape[0]) for hn in same_trace_list)
    if max_hn > 0:
        hard_padded = torch.full(
            (B, max_hn), HARD_NEG_PAD, device=device, dtype=pos_dot.dtype
        )
        for i, hn in enumerate(same_trace_list):
            if hn is not None:
                hard_padded[i, : hn.shape[0]] = torch.mv(hn, predictions[i])
        dot_products = torch.cat([pos_dot, neg_dot, hard_padded], dim=1)
    else:
        dot_products = torch.cat([pos_dot, neg_dot], dim=1)

    # Correct label: index i in row i (diagonal of pos_dot)
    labels = torch.arange(B, device=device)
    loss = F.cross_entropy(dot_products / temperature, labels, reduction="mean")
    return loss


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Joint end-to-end training of SequenceProjector + LSTMPredictor"
    )
    parser.add_argument(
        "--raw_embeddings_dir", type=str, default="embeddings_4096_last",
        help="Directory with raw 4096-dim Qwen embeddings (default: embeddings_4096_last)",
    )
    parser.add_argument(
        "--traces_dir", type=str,
        default="../rico-data/data/traces/filtered_traces",
        help="RICO filtered_traces directory",
    )
    parser.add_argument(
        "--proj_dim", type=int, default=768,
        help="Projector output dimension (default: 768)",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=1024,
        help="Projector hidden dimension (default: 1024)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output dir for projected embeddings (default: embeddings_seq_{proj_dim})",
    )
    parser.add_argument("--epochs", type=int, default=120,
                        help="Max epochs (Screen2Vec uses 80-120)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Traces per batch (Screen2Vec uses 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (Screen2Vec uses 0.001)")
    parser.add_argument("--neg_samp", type=int, default=128,
                        help="Negative sample size (Screen2Vec uses 128)")
    parser.add_argument("--n_predictors", type=int, default=4,
                        help="Context window size (Screen2Vec uses 4)")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of traces held out for validation")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="InfoNCE temperature parameter")
    parser.add_argument("--trace_negatives", type=str, default="context",
                        choices=["context", "other", "both"],
                        help="Same-trace hard negatives. 'context': the "
                             "window's own context screens, as in Screen2Vec's "
                             "pretrainer.py:106-107 (default). 'other': the "
                             "remaining trace screens, excluding context "
                             "(behaviour before 2026-08-22). 'both': union.")
    parser.add_argument("--trace_index", type=str,
                        default=trace_index.DEFAULT_INDEX_PATH,
                        help="Trace->layout index CSV from "
                             "0_build_trace_index.py. Required: trace "
                             "gestures.json keys are per-trace indices, not the "
                             "global layouts ids the embeddings are keyed by.")
    parser.add_argument("--strict", action="store_true",
                        help="Train on Screen2Vec's own population: only traces "
                             "with >= n_predictors+1 screens (so every window has "
                             "exactly n_predictors context screens and no "
                             "zero-padding is needed), and draw random negatives "
                             "only from the screens of those traces. Artefacts get "
                             "a _strict suffix so they cannot clobber a permissive "
                             "run. Off by default; default behaviour is unchanged.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Suffix strict artefacts so a strict run cannot overwrite a permissive one.
    strict_suffix = "_strict" if args.strict else ""

    if args.output_dir is None:
        args.output_dir = f"embeddings_seq_{args.proj_dim}{strict_suffix}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Joint training: Projector (4096 → {args.hidden_dim} → {args.proj_dim}) + LSTM ({args.proj_dim})")

    # ------------------------------------------------------------------
    # 1. Load raw 4096-dim embeddings
    # ------------------------------------------------------------------
    id_to_idx, raw_matrix = load_all_embeddings(args.raw_embeddings_dir)
    available_ids = set(id_to_idx.keys())
    index = trace_index.load_index(args.trace_index)
    print(f"  Loaded trace->layout index: {len(index)} screens "
          f"('{args.trace_index}')")
    min_trace_len = (args.n_predictors + 1) if args.strict else 3
    traces = parse_traces(args.traces_dir, available_ids, index, min_trace_len)

    # Strict: negatives come only from the screens of the eligible traces,
    # matching Screen2Vec's ScreenVocab (vocab.py:12-20).
    neg_pool_indices = None
    if args.strict:
        eligible = sorted({id_to_idx[sid] for trace in traces for sid in trace})
        neg_pool_indices = torch.tensor(eligible, dtype=torch.long)
        print(f"  STRICT: traces >= {min_trace_len} screens, "
              f"negative pool = {len(eligible)} of {len(id_to_idx)} screens")

    input_dim = raw_matrix.shape[1]
    raw_emb_tensor = torch.from_numpy(raw_matrix)  # (N, 4096) — stays on CPU

    print(f"  Input dim: {input_dim}  |  Traces: {len(traces)}  |  Screens: {len(id_to_idx)}")

    # ------------------------------------------------------------------
    # 2. Train / val split
    # ------------------------------------------------------------------
    random.shuffle(traces)
    n_val = max(1, int(len(traces) * args.val_fraction))
    val_traces = traces[:n_val]
    train_traces = traces[n_val:]
    print(f"  Train traces: {len(train_traces)}  |  Val traces: {len(val_traces)}")

    # ------------------------------------------------------------------
    # 3. Create both models + single optimizer
    # ------------------------------------------------------------------
    projector = SequenceProjector(
        input_dim=input_dim,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    predictor = LSTMPredictor(args.proj_dim).to(device)

    proj_params = sum(p.numel() for p in projector.parameters())
    lstm_params = sum(p.numel() for p in predictor.parameters())
    print(f"  Projector parameters: {proj_params:,}")
    print(f"  LSTM parameters:      {lstm_params:,}")
    print(f"  Total parameters:     {proj_params + lstm_params:,}")

    # Single optimizer for BOTH models — this is the key difference
    all_params = list(projector.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.lr)

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_proj_state = None
    best_lstm_state = None
    patience_counter = 0
    history = {"train": [], "val": []}

    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  Joint Training: {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"  neg_samp={args.neg_samp}, n_predictors={args.n_predictors}")
    print(f"  temperature={args.temperature}")
    print(f"  trace_negatives={args.trace_negatives}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        # --- Train ---
        projector.train()
        predictor.train()
        random.shuffle(train_traces)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_traces), args.batch_size):
            batch = train_traces[start : start + args.batch_size]
            if len(batch) < 2:
                continue

            loss = compute_joint_loss(
                projector, predictor, raw_emb_tensor, batch, id_to_idx,
                args.neg_samp, device, args.n_predictors, args.temperature,
                neg_pool_indices=neg_pool_indices,
                trace_negatives=args.trace_negatives,
            )

            optimizer.zero_grad()
            loss.backward()
            # Clip gradients for both models
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(1, n_batches)
        history["train"].append(avg_train)

        # --- Validation ---
        projector.eval()
        predictor.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for start in range(0, len(val_traces), args.batch_size):
                batch = val_traces[start : start + args.batch_size]
                if len(batch) < 2:
                    continue
                loss = compute_joint_loss(
                    projector, predictor, raw_emb_tensor, batch, id_to_idx,
                    args.neg_samp // 8, device, args.n_predictors, args.temperature,
                    neg_pool_indices=neg_pool_indices,
                    trace_negatives=args.trace_negatives,
                )
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val = val_loss_sum / max(1, val_batches)
        history["val"].append(avg_val)

        status = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_proj_state = {k: v.cpu().clone() for k, v in projector.state_dict().items()}
            best_lstm_state = {k: v.cpu().clone() for k, v in predictor.state_dict().items()}
            patience_counter = 0
            status = " * best"
        else:
            patience_counter += 1
            status = f" (patience {patience_counter}/{args.patience})"

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train: {avg_train:.4f} | Val: {avg_val:.4f}{status}"
        )

        if patience_counter >= args.patience:
            print(f"\n--- Early stopping at epoch {epoch+1} ---")
            break

    # ------------------------------------------------------------------
    # 5. Restore best models
    # ------------------------------------------------------------------
    if best_proj_state is not None:
        projector.load_state_dict(best_proj_state)
        projector.to(device)
    if best_lstm_state is not None:
        predictor.load_state_dict(best_lstm_state)
        predictor.to(device)

    print(f"\nRestored best models (val loss: {best_val_loss:.4f})")

    # ------------------------------------------------------------------
    # 6. Project ALL embeddings and save (same format as step 5)
    # ------------------------------------------------------------------
    projector.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Projecting {len(id_to_idx)} embeddings → '{args.output_dir}'...")

    chunk_size = 4096
    all_projected = []
    with torch.no_grad():
        for start in tqdm(
            range(0, raw_emb_tensor.shape[0], chunk_size),
            desc="Projecting",
        ):
            chunk = raw_emb_tensor[start : start + chunk_size].to(device)
            proj = projector(chunk).cpu().numpy()
            all_projected.append(proj)

    all_projected = np.concatenate(all_projected, axis=0)  # (N, proj_dim)
    print(f"  Projected shape: {all_projected.shape}")

    idx_to_id = {v: k for k, v in id_to_idx.items()}
    for idx in tqdm(range(len(all_projected)), desc="Saving embeddings"):
        screen_id = idx_to_id[idx]
        out_path = os.path.join(args.output_dir, f"{screen_id}.npz")
        np.savez_compressed(
            out_path, embedding=all_projected[idx].astype(np.float32)
        )

    # ------------------------------------------------------------------
    # 7. Save model checkpoints
    # ------------------------------------------------------------------
    # LSTM (same format as step 8, compatible with 4_evaluate_screen_prediction.py)
    lstm_path = f"lstm_predictor_{args.proj_dim}d{strict_suffix}.pth"
    torch.save(
        {
            "state_dict": predictor.state_dict(),
            "embed_dim": args.proj_dim,
            "args": vars(args),
            "best_val_loss": best_val_loss,
            "history": history,
        },
        lstm_path,
    )
    print(f"Saved LSTM predictor: {lstm_path}")

    # Projector
    proj_path = f"sequence_projector_{args.proj_dim}d{strict_suffix}.pth"
    torch.save(
        {
            "state_dict": projector.state_dict(),
            "args": vars(args),
            "best_val_loss": best_val_loss,
            "history": history,
        },
        proj_path,
    )
    print(f"Saved projector: {proj_path}")

    # ------------------------------------------------------------------
    # 8. Plot training history
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history["train"], label="Train Loss", marker="o", markersize=3)
    plt.plot(history["val"], label="Val Loss", marker="x", markersize=3)
    plt.title(f"Joint Training (4096 → {args.proj_dim} + LSTM)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = f"joint_training_{args.proj_dim}d{strict_suffix}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved training plot: {plot_path}")

    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed:.1f}s")

    print(f"\nNext steps:")
    print(f"  python 6_generate_app_context_embeddings.py \\")
    print(f"    --embeddings_dir {args.output_dir} \\")
    print(f"    --output_dir {args.output_dir}_with_app")
    print(f"")
    print(f"  python 4_evaluate_screen_prediction.py \\")
    print(f"    --embeddings_dir {args.output_dir}_with_app \\")
    print(f"    --lstm_model {lstm_path}" + (" \\" if args.strict else ""))
    if args.strict:
        print(f"    --strict")


if __name__ == "__main__":
    main()
