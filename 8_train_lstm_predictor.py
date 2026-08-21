"""
8_train_lstm_predictor.py

Trains an LSTM-based sequence predictor on top of pre-computed (projected)
screen embeddings, exactly replicating Screen2Vec's TracePredictor training
procedure from the CHI '21 paper.

Architecture:
  - Input: pre-computed embeddings from embeddings_seq_768/ (or any dir)
  - LSTM combiner: takes context screens → predicts target screen
  - Loss: CrossEntropy with negative sampling (matches Screen2Vec exactly)
    Negatives: 128 random screens + other batch targets + same-trace screens

After training, this predictor is used at evaluation time in
4_evaluate_screen_prediction.py via --lstm_model flag.

Usage:
    # Train on 768-dim projected embeddings
    python 8_train_lstm_predictor.py --embeddings_dir embeddings_seq_768

    # Train on 128+384 concatenated embeddings
    python 8_train_lstm_predictor.py --embeddings_dir embeddings_seq_128_with_app

    # Then evaluate with LSTM:
    python 4_evaluate_screen_prediction.py \\
        --embeddings_dir embeddings_seq_768 \\
        --lstm_model lstm_predictor_768d.pth

    # Screen2Vec's own protocol end to end. --strict must be used for BOTH
    # training and evaluation; the checkpoint is suffixed _strict so it
    # cannot clobber a permissive run:
    python 8_train_lstm_predictor.py --embeddings_dir embeddings_seq_768 --strict
    python 4_evaluate_screen_prediction.py \\
        --embeddings_dir embeddings_seq_768 \\
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import trace_index


# ============================================================================
# LSTM Predictor Model
# ============================================================================
class LSTMPredictor(nn.Module):
    """
    Replicates Screen2Vec's TracePredictor architecture exactly.

    Takes a sequence of context screen embeddings (all screens in the trace
    except the target), runs them through an LSTM, and produces a prediction
    vector that should be close to the target screen's embedding.

    From Screen2Vec source (prediction.py):
      self.combiner = nn.LSTM(self.bert_size, self.bert_size, batch_first=True)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Matches Screen2Vec's combiner exactly
        self.combiner = nn.LSTM(embed_dim, embed_dim, batch_first=True)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (B, T, embed_dim) context screen embeddings (variable T)
        Returns:
            (B, embed_dim) prediction vector, L2-normalized
        """
        _, (h, _) = self.combiner(context)
        pred = h[0]  # (B, embed_dim) — last hidden state
        return F.normalize(pred, p=2, dim=-1)


# ============================================================================
# Data Loading
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
            desc="Loading embeddings"
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
    zero-padding branch in compute_loss_batch never fires and training matches
    what 4_evaluate_screen_prediction.py --strict feeds the model at test time.
    """
    traces_root = Path(traces_dir)
    if not traces_root.exists():
        raise FileNotFoundError(f"Traces directory not found: {traces_dir}")

    # Collect all candidate trace directories
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

    # Parse trace files in parallel
    print(f"Parsing {len(trace_dirs)} traces in parallel...")
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(parse_single_trace, trace_dirs),
            total=len(trace_dirs),
            desc="Parsing traces"
        ))

    traces = [t for t in results if t is not None]
    print(f"  Parsed {len(traces)} traces")
    return traces


# ============================================================================
# Loss: CrossEntropy with Negative Sampling (matches Screen2Vec exactly)
# ============================================================================
def compute_loss_batch(
    predictor: LSTMPredictor,
    emb_tensor: torch.Tensor,
    batch_traces: list,
    id_to_idx: dict,
    neg_sample_size: int,
    device: torch.device,
    n_predictors: int = 4,
    temperature: float = 0.07,
    neg_pool_indices: torch.Tensor = None,
) -> torch.Tensor:
    """
    Screen2Vec's training loss:
      For each trace in the batch:
        - Pick a random window of n_predictors context screens + 1 target
        - Run context through LSTM → prediction c
        - Compute dot products vs:
            * target (positive)
            * neg_sample_size random screens from the vocab
            * other screens in the same batch (negatives)
            * other screens in the same trace (hard negatives)
        - CrossEntropy: the correct index is 0 (the target)

    This exactly mirrors Screen2Vec's pretrainer.py iteration() method.

    neg_pool_indices: if given, random negatives are drawn only from these
    embedding rows. Screen2Vec samples negatives from its ScreenVocab, which
    is built from the *filtered* dataset (vocab.py:12-20), so screens outside
    the eligible traces are never negatives. Default None keeps sampling from
    the full embedding matrix.
    """
    B = len(batch_traces)
    embed_dim = emb_tensor.shape[1]

    context_list = []   # (B, T_ctx, D) — variable T_ctx
    target_list = []    # (B, D)
    same_trace_list = []  # list of tensors (n_same, D) per batch item

    for trace in batch_traces:
        n = min(n_predictors + 1, len(trace))
        start = random.randint(0, len(trace) - n)
        window = trace[start : start + n]

        target_idx = id_to_idx[window[-1]]
        ctx_indices = [id_to_idx[sid] for sid in window[:-1]]

        target_list.append(emb_tensor[target_idx])
        context_list.append(emb_tensor[ctx_indices])  # (T_ctx, D)

        # Other screens in the trace (hard negatives, excluding ctx and target)
        all_trace_indices = set(id_to_idx[sid] for sid in trace)
        hard_neg_indices = list(all_trace_indices - {target_idx} - set(ctx_indices))
        if hard_neg_indices:
            same_trace_list.append(emb_tensor[hard_neg_indices])
        else:
            same_trace_list.append(None)

    # Pad context sequences for LSTM batch processing
    max_ctx_len = max(c.shape[0] for c in context_list)
    ctx_padded = torch.zeros(B, max_ctx_len, embed_dim, device=device)
    for i, ctx in enumerate(context_list):
        ctx_padded[i, : ctx.shape[0]] = ctx.to(device)

    targets = torch.stack(target_list).to(device)       # (B, D)

    # Forward pass
    predictions = predictor(ctx_padded)                  # (B, D)

    # --- Dot products (Screen2Vec uses raw dot products, not cosine sim) ---

    # 1. Positive: prediction vs target  →  shape (B, B)
    pos_dot = torch.mm(predictions, targets.T)           # (B, B)
    # diagonal is the correct (positive) pair

    # 2. Random negative sample from the vocab
    if neg_pool_indices is None:
        neg_indices = torch.randint(0, emb_tensor.shape[0], (neg_sample_size,))
    else:
        neg_indices = neg_pool_indices[
            torch.randint(0, neg_pool_indices.shape[0], (neg_sample_size,))
        ]
    neg_embs = emb_tensor[neg_indices].to(device)        # (neg_sample_size, D)
    neg_dot = torch.mm(predictions, neg_embs.T)          # (B, neg_sample_size)

    # 3. Hard negatives: same-trace screens
    hard_dots = []
    for i, hn in enumerate(same_trace_list):
        if hn is not None:
            hd = torch.mv(hn.to(device), predictions[i])  # (n_same,)
            hard_dots.append(hd.unsqueeze(0))             # (1, n_same)
    if hard_dots:
        # Pad to same length
        max_hn = max(h.shape[1] for h in hard_dots)
        hard_padded = torch.zeros(B, max_hn, device=device)
        for i, hd in enumerate(hard_dots):
            hard_padded[i, : hd.shape[1]] = hd
        dot_products = torch.cat([pos_dot, neg_dot, hard_padded], dim=1)
    else:
        dot_products = torch.cat([pos_dot, neg_dot], dim=1)

    # Correct label is always the diagonal of pos_dot, i.e., index i in row i
    labels = torch.arange(B, device=device)
    loss = F.cross_entropy(dot_products / temperature, labels, reduction="mean")
    return loss


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train an LSTM sequence predictor matching Screen2Vec's approach"
    )
    parser.add_argument(
        "--embeddings_dir", type=str, default="embeddings_seq_768",
        help="Input embeddings directory (default: embeddings_seq_768)",
    )
    parser.add_argument(
        "--traces_dir", type=str,
        default="../rico-data/data/traces/filtered_traces",
        help="RICO filtered_traces directory",
    )
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of epochs (Screen2Vec uses 80-120)")
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
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pth path (default: lstm_predictor_{dim}d.pth, "
                             "or lstm_predictor_{dim}d_strict.pth with --strict)")
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
                             "only from the screens of those traces. Off by "
                             "default; the default behaviour is unchanged.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    id_to_idx, emb_matrix = load_all_embeddings(args.embeddings_dir)
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

    embed_dim = emb_matrix.shape[1]
    emb_tensor = torch.from_numpy(emb_matrix)  # (N, D) on CPU

    # L2-normalize the stored embeddings (they may already be normalized)
    norms = torch.linalg.norm(emb_tensor, dim=1, keepdim=True)
    emb_tensor = emb_tensor / torch.clamp(norms, min=1e-12)

    print(f"  Embed dim: {embed_dim}  |  Traces: {len(traces)}  |  Screens: {len(id_to_idx)}")

    if args.output is None:
        # Suffix the strict artefact so it cannot clobber a permissive run.
        args.output = (f"lstm_predictor_{embed_dim}d_strict.pth" if args.strict
                       else f"lstm_predictor_{embed_dim}d.pth")

    # ---------------------------------------------------------------
    # 2. Train / val split
    # ---------------------------------------------------------------
    random.shuffle(traces)
    n_val = max(1, int(len(traces) * args.val_fraction))
    val_traces = traces[:n_val]
    train_traces = traces[n_val:]
    print(f"  Train traces: {len(train_traces)}  |  Val traces: {len(val_traces)}")

    # ---------------------------------------------------------------
    # 3. Model + optimizer  (Screen2Vec uses Adam lr=0.001)
    # ---------------------------------------------------------------
    predictor = LSTMPredictor(embed_dim).to(device)
    param_count = sum(p.numel() for p in predictor.parameters())
    print(f"  LSTM parameters: {param_count:,}")

    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

    # ---------------------------------------------------------------
    # 4. Training loop
    # ---------------------------------------------------------------
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train": [], "val": []}

    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  Training: {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"  neg_samp={args.neg_samp}, n_predictors={args.n_predictors}")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        # --- Train ---
        predictor.train()
        random.shuffle(train_traces)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_traces), args.batch_size):
            batch = train_traces[start : start + args.batch_size]
            if len(batch) < 2:
                continue

            loss = compute_loss_batch(
                predictor, emb_tensor, batch, id_to_idx,
                args.neg_samp, device, args.n_predictors, args.temperature,
                neg_pool_indices=neg_pool_indices,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(1, n_batches)
        history["train"].append(avg_train)

        # --- Validation ---
        predictor.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for start in range(0, len(val_traces), args.batch_size):
                batch = val_traces[start : start + args.batch_size]
                if len(batch) < 2:
                    continue
                loss = compute_loss_batch(
                    predictor, emb_tensor, batch, id_to_idx,
                    args.neg_samp // 8, device, args.n_predictors, args.temperature,
                    neg_pool_indices=neg_pool_indices,
                )
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val = val_loss_sum / max(1, val_batches)
        history["val"].append(avg_val)

        status = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in predictor.state_dict().items()}
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

    # ---------------------------------------------------------------
    # 5. Save best model
    # ---------------------------------------------------------------
    if best_state is not None:
        predictor.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": predictor.state_dict(),
            "embed_dim": embed_dim,
            "args": vars(args),
            "best_val_loss": best_val_loss,
            "history": history,
        },
        args.output,
    )

    elapsed = time.time() - start_time
    print(f"\nSaved LSTM predictor: {args.output}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Total training time: {elapsed:.1f}s")
    print(f"\nNext step — evaluate with:")
    print(f"  python 4_evaluate_screen_prediction.py \\")
    print(f"    --embeddings_dir {args.embeddings_dir} \\")
    print(f"    --lstm_model {args.output}" + (" \\" if args.strict else ""))
    if args.strict:
        print(f"    --strict")


if __name__ == "__main__":
    main()
