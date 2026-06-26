"""
5_train_sequence_projector.py

Trains a lightweight MLP projector to produce sequence-aware embeddings
from pre-computed Qwen2-VL embeddings using symmetric InfoNCE (contrastive)
loss on RICO interaction traces.

The projected embeddings can then be evaluated with 4_evaluate_screen_prediction.py.

Usage:
    # Default: project 4096 -> 128
    python 5_train_sequence_projector.py

    # Project 4096 -> 64 (matches Screen2Vec paper dimensionality)
    python 5_train_sequence_projector.py --proj_dim 64

    # Project 4096 -> 4096 (restructure space without compression)
    python 5_train_sequence_projector.py --proj_dim 4096 --hidden_dim 2048

    # Then evaluate:
    python 4_evaluate_screen_prediction.py --embeddings_dir embeddings_seq_128
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ============================================================================
# Model
# ============================================================================
class SequenceProjector(nn.Module):
    """
    Projects pre-computed Qwen2-VL embeddings into a lower-dimensional
    space optimized for sequence prediction using a Linear mapping.
    
    A simpler Linear mapping often generalizes much better than an MLP 
    for finding a CBOW subspace on already dense, frozen embeddings.
    """

    def __init__(self, input_dim: int = 4096, proj_dim: int = 128, hidden_dim: int = 1024):
        # hidden_dim is ignored but kept for compatibility with argparse
        super().__init__()
        self.net = nn.Linear(input_dim, proj_dim, bias=False)
        
        # Initialize orthogonally to preserve as much variance as possible at start
        nn.init.orthogonal_(self.net.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) raw embeddings
        Returns:
            (B, proj_dim) L2-normalized projected embeddings
        """
        return F.normalize(self.net(x), p=2, dim=-1)


# ============================================================================
# Data Loading (reuses patterns from 4_evaluate_screen_prediction.py)
# ============================================================================
def load_all_embeddings(embeddings_dir: str):
    """Load all .npz embeddings into a matrix and id mapping."""
    npz_files = sorted(Path(embeddings_dir).glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files in '{embeddings_dir}'")

    print(f"Loading {len(npz_files)} embeddings from '{embeddings_dir}'...")
    id_to_idx = {}
    embeddings = []

    for npz_path in tqdm(npz_files, desc="Loading embeddings"):
        screen_id = int(npz_path.stem)
        with np.load(npz_path) as f:
            emb = f["embedding"].astype(np.float32)
        id_to_idx[screen_id] = len(embeddings)
        embeddings.append(emb)

    matrix = np.stack(embeddings, axis=0)  # (N, D)
    print(f"  Loaded matrix shape: {matrix.shape}")
    return id_to_idx, matrix


def parse_traces(traces_dir: str, available_ids: set):
    """Parse RICO interaction traces, returning list of screen-id lists."""
    traces_root = Path(traces_dir)
    if not traces_root.exists():
        raise FileNotFoundError(f"Traces directory not found: {traces_dir}")

    traces = []
    skipped = 0

    for app_dir in tqdm(sorted(traces_root.iterdir()), desc="Parsing traces"):
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

            screen_ids = []
            for key in gestures.keys():
                try:
                    sid = int(key)
                except ValueError:
                    continue
                if sid in available_ids:
                    screen_ids.append(sid)

            if len(screen_ids) < 3:
                skipped += 1
                continue
            traces.append(screen_ids)

    print(f"  {len(traces)} traces, skipped {skipped} with < 3 screens")
    return traces


# ============================================================================
# Loss Computation
# ============================================================================
def compute_infonce_loss(
    projector: SequenceProjector,
    emb_tensor: torch.Tensor,
    batch_traces: list,
    id_to_idx: dict,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute symmetric InfoNCE loss for a batch of traces.

    For each trace, randomly pick one screen as the target; the rest are
    context.  Context embeddings are projected, averaged, and L2-normalized
    to form the prediction vector.  The loss encourages each prediction to
    be most similar to its own target (and dissimilar to all other targets
    in the batch).
    """
    B = len(batch_traces)

    target_indices = []       # global embedding indices for each target
    all_ctx_indices = []      # flat list of all context embedding indices
    ctx_group_ids = []        # which trace each context embedding belongs to
    ctx_counts = []           # number of context screens per trace

    for i, trace in enumerate(batch_traces):
        t = random.randint(0, len(trace) - 1)
        target_indices.append(id_to_idx[trace[t]])

        ctx = [id_to_idx[trace[j]] for j in range(len(trace)) if j != t]
        all_ctx_indices.extend(ctx)
        ctx_group_ids.extend([i] * len(ctx))
        ctx_counts.append(len(ctx))

    # --- Single forward pass for all embeddings in the batch ---
    combined_indices = target_indices + all_ctx_indices
    combined_embs = emb_tensor[combined_indices].to(device)
    combined_proj = projector(combined_embs)        # (B + total_ctx, proj_dim)

    target_proj = combined_proj[:B]                 # (B, proj_dim)
    ctx_proj = combined_proj[B:]                    # (total_ctx, proj_dim)

    # --- Compute per-trace context means via scatter_add ---
    proj_dim = target_proj.shape[1]
    ctx_group_tensor = torch.tensor(ctx_group_ids, device=device, dtype=torch.long)

    context_sums = torch.zeros(B, proj_dim, device=device)
    context_sums.scatter_add_(
        0,
        ctx_group_tensor.unsqueeze(1).expand(-1, proj_dim),
        ctx_proj,
    )
    ctx_count_tensor = torch.tensor(
        ctx_counts, device=device, dtype=torch.float32
    ).unsqueeze(1)
    context_means = F.normalize(context_sums / ctx_count_tensor, p=2, dim=1)

    # --- InfoNCE against ALL screens in the batch (targets + contexts) ---
    # Logits: each prediction vs ALL elements in combined_proj
    logits = context_means @ combined_proj.T / temperature   # (B, B + total_ctx)
    
    # We want to mask out any negative that is actually the exact same screen 
    # as the target screen (in case the target screen ID appears multiple times).
    combined_indices_tensor = torch.tensor(combined_indices, device=device)
    target_indices_tensor = torch.tensor(target_indices, device=device)
    
    # Shape of mask: (B, B + total_ctx). True if it's the exact same screen ID.
    mask = (target_indices_tensor.unsqueeze(1) == combined_indices_tensor.unsqueeze(0))
    
    # Unmask the actual diagonal (where i == j) so that the correct positive is preserved
    diagonal_mask = torch.eye(B, B + len(all_ctx_indices), device=device, dtype=torch.bool)
    mask = mask & ~diagonal_mask
    
    # Set logits for identical screens to -inf
    logits.masked_fill_(mask, float('-inf'))
    
    # Labels: the correct positive for row i is index i
    labels = torch.arange(B, device=device)
    loss = F.cross_entropy(logits, labels)

    # Symmetric loss for targets
    target_logits = logits[:, :B]
    loss = (loss + F.cross_entropy(target_logits.T, labels)) / 2.0

    return loss


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train a contrastive sequence projector on RICO traces"
    )
    parser.add_argument(
        "--proj_dim", type=int, default=128,
        help="Output embedding dimensionality (default: 128). "
             "Use 64 to match Screen2Vec, 4096 to keep original dim.",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=1024,
        help="Hidden layer dimensionality (default: 1024)",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="InfoNCE temperature parameter")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                        help="Fraction of traces sampled each epoch for val loss")
    parser.add_argument(
        "--embeddings_dir", type=str, default="embeddings_4096",
        help="Input embeddings directory",
    )
    parser.add_argument(
        "--traces_dir", type=str,
        default="../rico-data/data/traces/filtered_traces",
        help="RICO filtered_traces directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for projected embeddings "
             "(default: embeddings_seq_{proj_dim})",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"embeddings_seq_{args.proj_dim}"

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Projection: 4096 -> {args.hidden_dim} -> {args.proj_dim}")

    start_time = time.time()

    # -----------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------
    id_to_idx, emb_matrix = load_all_embeddings(args.embeddings_dir)
    available_ids = set(id_to_idx.keys())
    traces = parse_traces(args.traces_dir, available_ids)

    input_dim = emb_matrix.shape[1]
    emb_tensor = torch.from_numpy(emb_matrix)  # keep on CPU, slice to device

    print(f"  {len(traces)} traces, {len(id_to_idx)} embeddings ({input_dim}-dim)")

    # -----------------------------------------------------------------
    # 2. Model + Optimizer
    # -----------------------------------------------------------------
    projector = SequenceProjector(
        input_dim, args.proj_dim, args.hidden_dim
    ).to(device)

    param_count = sum(p.numel() for p in projector.parameters())
    print(f"  Projector parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        projector.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # -----------------------------------------------------------------
    # 3. Training loop
    # -----------------------------------------------------------------
    # Deterministic train/val split
    random.shuffle(traces)
    n_val = max(1, int(len(traces) * args.val_fraction))
    val_traces = traces[:n_val]
    train_traces = traces[n_val:]

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(
        f"\n{'='*60}\n"
        f"  Training: {args.epochs} epochs, {len(train_traces)} train traces, {len(val_traces)} val traces, "
        f"batch_size={args.batch_size}\n"
        f"{'='*60}\n"
    )

    for epoch in range(args.epochs):
        # --- Train ---
        projector.train()
        random.shuffle(train_traces)

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_traces), args.batch_size):
            batch = train_traces[start : start + args.batch_size]
            if len(batch) < 4:
                continue

            loss = compute_infonce_loss(
                projector, emb_tensor, batch, id_to_idx,
                args.temperature, device,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(1, n_batches)
        history["train_loss"].append(avg_train)

        # --- Val (fixed split) ---
        projector.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for start in range(0, len(val_traces), args.batch_size):
                batch = val_traces[start : start + args.batch_size]
                if len(batch) < 4:
                    continue
                loss = compute_infonce_loss(
                    projector, emb_tensor, batch, id_to_idx,
                    args.temperature, device,
                )
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val = val_loss_sum / max(1, val_batches)
        history["val_loss"].append(avg_val)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        status = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {
                k: v.cpu().clone()
                for k, v in projector.state_dict().items()
            }
            patience_counter = 0
            status = " * best"
        else:
            patience_counter += 1
            status = f" (patience {patience_counter}/{args.patience})"

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
            f"LR: {lr:.6f}{status}"
        )

        if patience_counter >= args.patience:
            print(f"\n--- Early stopping at epoch {epoch+1} ---")
            break

    # -----------------------------------------------------------------
    # 4. Restore best model
    # -----------------------------------------------------------------
    if best_state is not None:
        projector.load_state_dict(best_state)
        projector.to(device)
        print(f"\nRestored best model (val loss: {best_val_loss:.4f})")

    # -----------------------------------------------------------------
    # 5. Project all embeddings and save
    # -----------------------------------------------------------------
    projector.eval()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Projecting {len(id_to_idx)} embeddings -> '{args.output_dir}'...")

    chunk_size = 4096
    all_projected = []
    with torch.no_grad():
        for start in tqdm(
            range(0, emb_tensor.shape[0], chunk_size),
            desc="Projecting",
        ):
            chunk = emb_tensor[start : start + chunk_size].to(device)
            proj = projector(chunk).cpu().numpy()
            all_projected.append(proj)

    all_projected = np.concatenate(all_projected, axis=0)  # (N, proj_dim)
    print(f"  Projected shape: {all_projected.shape}")

    # Save individual .npz files (matching the format expected by
    # 4_evaluate_screen_prediction.py)
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    for idx in tqdm(range(len(all_projected)), desc="Saving embeddings"):
        screen_id = idx_to_id[idx]
        out_path = os.path.join(args.output_dir, f"{screen_id}.npz")
        np.savez_compressed(
            out_path, embedding=all_projected[idx].astype(np.float32)
        )

    # -----------------------------------------------------------------
    # 6. Plot training history
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", marker="o", markersize=3)
    plt.plot(
        history["val_loss"], label="Val Loss (fixed split)",
        marker="x", markersize=3,
    )
    plt.title(f"Sequence Projector Training (4096 -> {args.proj_dim})")
    plt.xlabel("Epoch")
    plt.ylabel("InfoNCE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = f"projector_training_{args.proj_dim}d.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved training plot: {plot_path}")

    # -----------------------------------------------------------------
    # 7. Save model checkpoint
    # -----------------------------------------------------------------
    model_path = f"sequence_projector_{args.proj_dim}d.pth"
    torch.save(
        {
            "state_dict": projector.state_dict(),
            "args": vars(args),
            "best_val_loss": best_val_loss,
            "history": history,
        },
        model_path,
    )
    print(f"Saved model checkpoint: {model_path}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\nNext step - evaluate with:")
    print(f"  python 4_evaluate_screen_prediction.py --embeddings_dir {args.output_dir}")


if __name__ == "__main__":
    main()
