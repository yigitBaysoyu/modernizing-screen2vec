"""
12_train_component_predictor.py

Step 3 of the GUI component prediction pipeline: trains the model that predicts
a GUI component's text from the components around it -- Screen2Vec's GUI
component embedding model (paper Section 2.1 / Table 2), retrained on top of the
component text embeddings from 11_embed_component_vocab.py.

Training protocol follows the paper:
  - one randomly hidden component per screen per epoch
    (screen2vec-master/UI_embedding/dataset/dataset.py::ScreenDataset)
  - context = the n=16 components closest to the hidden one, EUCLIDEAN or
    HIERARCHICAL (the two rows of Table 2)
  - loss = CrossEntropy of the text prediction against *every* vocabulary entry
    + CrossEntropy of the GUI class prediction against all 26 classes
  - Adam, lr 0.001, batch 256

Prediction and ranking happen in a PCA-reduced version of the embedding space
(--pred_dim, 512 by default), which is what keeps the full-vocabulary softmax
affordable at Qwen's 4096 dimensions. The PCA is stored in the checkpoint so
13_evaluate_component_prediction.py ranks in exactly the same space.

Usage:
    python 12_train_component_predictor.py --data_dir component_data
    python 12_train_component_predictor.py --data_dir component_data \\
        --distance hierarchical --output component_predictor_hier.pth
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from tqdm import tqdm

from component_utils import (
    COMP_CLS,
    COMP_PARENT,
    COMP_TEXT,
    COMP_X1,
    COMP_X2,
    COMP_Y1,
    COMP_Y2,
    NUM_CLASSES,
    ComponentPredictor,
    apply_vocab_pca,
    build_context_tensors,
    closest_context_indices,
    fit_vocab_pca,
    load_component_data,
)


# ============================================================================
# 1. Sample construction
# ============================================================================
def build_samples_for_screens(comp_chunk, n_context, distance):
    """Every (screen, text-bearing component) pair with its context indices.

    Contexts depend only on geometry / tree structure, so they are computed once
    here instead of per epoch. Returns (target_pos, ctx_pos) per screen, both as
    positions into that screen's own component list (-1 pads the context).
    """
    out = []
    for comp_array in comp_chunk:
        comps = comp_array.tolist()
        if len(comps) < 2:
            out.append(None)
            continue
        bounds = [(c[COMP_X1], c[COMP_Y1], c[COMP_X2], c[COMP_Y2]) for c in comps]
        parents = [c[COMP_PARENT] for c in comps]

        targets = []
        contexts = []
        for i, comp in enumerate(comps):
            if comp[COMP_TEXT] == 0:  # no text -> context only, never a target
                continue
            ctx = closest_context_indices(bounds, parents, i, n_context, distance)
            if not ctx:
                continue
            targets.append(i)
            contexts.append(ctx + [-1] * (n_context - len(ctx)))
        if not targets:
            out.append(None)
            continue
        out.append((
            np.asarray(targets, dtype=np.int32),
            np.asarray(contexts, dtype=np.int32),
        ))
    return out


def build_samples(comp_arrays, n_context, distance, n_jobs=-1, chunk_size=512):
    """Parallel wrapper around build_samples_for_screens()."""
    chunks = [comp_arrays[i:i + chunk_size] for i in range(0, len(comp_arrays), chunk_size)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(build_samples_for_screens)(chunk, n_context, distance)
        for chunk in tqdm(chunks, desc=f"Building {distance} contexts")
    )
    flat = []
    for chunk_result in results:
        flat.extend(chunk_result)
    return flat


def screen_component_arrays(screens, free=True):
    """Per-screen (n_components, 7) int32 arrays for fast row lookup.

    With `free`, the nested Python lists are dropped from `screens` afterwards:
    at RICO scale they are ~1.5M seven-element lists, several times the size of
    the equivalent arrays, and nothing downstream needs them again.
    """
    arrays = [np.asarray(s["c"], dtype=np.int32) for s in screens]
    if free:
        for s in screens:
            s.pop("c", None)
    return arrays


def sample_batch_rows(batch, comp_arrays, per_screen):
    """Materialise (context_rows, target_row) pairs for a batch of screen ids."""
    samples = []
    labels = []
    for screen_idx, target_slot in batch:
        targets, contexts = per_screen[screen_idx]
        comps = comp_arrays[screen_idx]
        target_pos = targets[target_slot]
        ctx_pos = contexts[target_slot]
        ctx_pos = ctx_pos[ctx_pos >= 0]
        target_row = comps[target_pos]
        samples.append((comps[ctx_pos], target_row))
        labels.append((int(target_row[COMP_TEXT]), int(target_row[COMP_CLS])))
    return samples, labels


def draw_epoch_batches(screen_ids, per_screen, batch_size, rng, targets_per_screen=1):
    """One random hidden component per screen per epoch, shuffled into batches."""
    picks = []
    for screen_idx in screen_ids:
        n_targets = len(per_screen[screen_idx][0])
        for _ in range(targets_per_screen):
            picks.append((screen_idx, rng.randrange(n_targets)))
    rng.shuffle(picks)
    return [picks[i:i + batch_size] for i in range(0, len(picks), batch_size)]


# ============================================================================
# 2. Loss (paper: text CE over the full vocabulary + class CE)
# ============================================================================
def compute_loss(model, batch, comp_arrays, per_screen, vocab_feats, vocab_feats_t,
                 device, max_context, use_spatial, temperature, neg_vocab):
    """One forward pass: text CE over the vocabulary + class CE over 26 classes."""
    samples, labels = sample_batch_rows(batch, comp_arrays, per_screen)
    ctx_text, ctx_cls, ctx_spatial, ctx_mask = build_context_tensors(
        samples, vocab_feats, max_context, use_spatial
    )
    text_targets = torch.tensor([l[0] for l in labels], dtype=torch.long, device=device)
    class_targets = torch.tensor([l[1] for l in labels], dtype=torch.long, device=device)

    text_pred, class_logits = model(
        ctx_text.to(device), ctx_cls.to(device), ctx_spatial.to(device), ctx_mask.to(device)
    )

    if neg_vocab and neg_vocab < vocab_feats_t.shape[0]:
        # Sampled softmax: the true text first, then `neg_vocab` random entries.
        neg = torch.randint(0, vocab_feats_t.shape[0], (neg_vocab,), device=device)
        cand = torch.cat([text_targets, neg])
        logits = text_pred @ vocab_feats_t[cand].T / temperature
        text_labels = torch.arange(len(text_targets), device=device)
    else:
        # Paper: compare each prediction to every embedding in the vocabulary.
        logits = text_pred @ vocab_feats_t.T / temperature
        text_labels = text_targets

    text_loss = F.cross_entropy(logits, text_labels)
    class_loss = F.cross_entropy(class_logits, class_targets)
    return text_loss + class_loss, text_loss.item(), class_loss.item()


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train the GUI component predictor (Screen2Vec Table 2 task)"
    )
    parser.add_argument("--data_dir", type=str, default="component_data",
                        help="Directory written by 10_extract_components.py")
    parser.add_argument("--vocab_emb", type=str, default=None,
                        help="Vocabulary embeddings (default: <data_dir>/vocab_emb.npy)")
    parser.add_argument("--distance", choices=("euclidean", "hierarchical"),
                        default="euclidean",
                        help="Context distance measure (the two rows of Table 2)")
    parser.add_argument("--n_context", type=int, default=16,
                        help="Context components per target (Screen2Vec uses 16)")
    parser.add_argument("--no_spatial", action="store_true",
                        help="Drop the spatial features, leaving only the closeness "
                             "ordering Screen2Vec itself uses")
    parser.add_argument("--pred_dim", type=int, default=512,
                        help="PCA target dimensionality of the prediction space "
                             "(0 = keep the encoder's full dimensionality)")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=120,
                        help="Screen2Vec trains the component model for ~120 epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Screens per batch (Screen2Vec uses 256)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (Screen2Vec uses 0.001)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Softmax temperature on the cosine logits")
    parser.add_argument("--neg_vocab", type=int, default=0,
                        help="Sampled-softmax negatives instead of the full "
                             "vocabulary (0 = full vocabulary, as in the paper)")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of screens held out (Screen2Vec uses 0.1)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--targets_per_screen", type=int, default=1,
                        help="Hidden components drawn per screen per epoch")
    parser.add_argument("--limit_screens", type=int, default=None,
                        help="Only use the first N screens (smoke tests)")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pth (default: component_predictor_<distance>.pth)")
    parser.add_argument("--plot", action="store_true",
                        help="Write a loss curve next to the checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab_emb_path = args.vocab_emb or os.path.join(args.data_dir, "vocab_emb.npy")
    if args.output is None:
        args.output = f"component_predictor_{args.distance}.pth"

    start_time = time.time()

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    print(f"\n[Step 1/5] Loading component data from '{args.data_dir}'...")
    screens, vocab, meta = load_component_data(args.data_dir, limit=args.limit_screens)
    print(f"  Screens: {len(screens)}  |  Vocabulary: {len(vocab)}")

    vocab_emb = np.load(vocab_emb_path)
    if vocab_emb.shape[0] != len(vocab):
        raise ValueError(
            f"'{vocab_emb_path}' has {vocab_emb.shape[0]} rows but the vocabulary "
            f"has {len(vocab)} entries — re-run 11_embed_component_vocab.py"
        )
    print(f"  Vocabulary embeddings: {vocab_emb.shape} from '{vocab_emb_path}'")

    # ---------------------------------------------------------------
    # 2. Prediction space (PCA)
    # ---------------------------------------------------------------
    print(f"\n[Step 2/5] Preparing the prediction space (pred_dim={args.pred_dim})...")
    pca_mean, pca_components = fit_vocab_pca(vocab_emb, args.pred_dim, seed=args.seed)
    vocab_feats = apply_vocab_pca(vocab_emb, pca_mean, pca_components)
    vocab_feats[0] = 0.0  # the "no text" row stays at the origin
    feat_dim = vocab_feats.shape[1]
    print(f"  Prediction space: {vocab_emb.shape[1]}d -> {feat_dim}d"
          f"{' (PCA)' if pca_components is not None else ' (unchanged)'}")
    vocab_feats_t = torch.from_numpy(vocab_feats).to(device)

    # ---------------------------------------------------------------
    # 3. Contexts + split
    # ---------------------------------------------------------------
    print(f"\n[Step 3/5] Building {args.distance} contexts (n={args.n_context})...")
    comp_arrays = screen_component_arrays(screens)
    per_screen = build_samples(comp_arrays, args.n_context, args.distance, n_jobs=args.n_jobs)
    usable = [i for i, s in enumerate(per_screen) if s is not None]
    n_targets = sum(len(per_screen[i][0]) for i in usable)
    print(f"  Usable screens: {len(usable)}  |  Predictable components: {n_targets}")

    rng.shuffle(usable)
    n_val = max(1, int(len(usable) * args.val_fraction))
    val_ids, train_ids = usable[:n_val], usable[n_val:]
    print(f"  Train screens: {len(train_ids)}  |  Val screens: {len(val_ids)}")

    # ---------------------------------------------------------------
    # 4. Model
    # ---------------------------------------------------------------
    model_config = dict(
        feat_dim=feat_dim, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, num_classes=NUM_CLASSES, max_context=args.n_context,
        use_spatial=not args.no_spatial, dropout=args.dropout,
    )
    model = ComponentPredictor(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Step 4/5] Model: {n_params:,} parameters "
          f"(spatial={'off' if args.no_spatial else 'on'})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---------------------------------------------------------------
    # 5. Training loop
    # ---------------------------------------------------------------
    print(f"\n[Step 5/5] Training: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"lr={args.lr}, "
          f"{'full-vocabulary' if not args.neg_vocab else f'{args.neg_vocab}-negative'} softmax")

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train": [], "val": []}
    max_context = args.n_context
    use_spatial = not args.no_spatial

    for epoch in range(args.epochs):
        model.train()
        batches = draw_epoch_batches(
            train_ids, per_screen, args.batch_size, rng, args.targets_per_screen
        )
        epoch_loss = epoch_text = epoch_class = 0.0
        n_batches = 0
        for batch in tqdm(batches, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            loss, text_l, class_l = compute_loss(
                model, batch, comp_arrays, per_screen, vocab_feats,
                vocab_feats_t, device, max_context, use_spatial,
                args.temperature, args.neg_vocab,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_text += text_l
            epoch_class += class_l
            n_batches += 1
        avg_train = epoch_loss / max(1, n_batches)
        avg_text = epoch_text / max(1, n_batches)
        avg_class = epoch_class / max(1, n_batches)
        history["train"].append(avg_train)

        model.eval()
        val_batches = draw_epoch_batches(val_ids, per_screen, args.batch_size, rng)
        val_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_batches:
                loss, _, _ = compute_loss(
                    model, batch, comp_arrays, per_screen, vocab_feats,
                    vocab_feats_t, device, max_context, use_spatial,
                    args.temperature, args.neg_vocab,
                )
                val_sum += loss.item()
                n_val_batches += 1
        avg_val = val_sum / max(1, n_val_batches)
        history["val"].append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            status = " * best"
        else:
            patience_counter += 1
            status = f" (patience {patience_counter}/{args.patience})"

        print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {avg_train:.4f} "
              f"(text {avg_text:.4f} / class {avg_class:.4f}) | Val: {avg_val:.4f}{status}")

        if patience_counter >= args.patience:
            print(f"\n--- Early stopping at epoch {epoch+1} ---")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model_config,
            "pca_mean": pca_mean,
            "pca_components": pca_components,
            "vocab_size": len(vocab),
            "feat_dim": feat_dim,
            "distance": args.distance,
            "n_context": args.n_context,
            "use_spatial": use_spatial,
            "temperature": args.temperature,
            "data_dir": args.data_dir,
            "vocab_emb": vocab_emb_path,
            # RICO screen ids, not list positions, so the evaluator can find
            # the same held-out screens no matter how the data is loaded.
            "val_screen_ids": [int(screens[i]["id"]) for i in val_ids],
            "args": vars(args),
            "best_val_loss": best_val_loss,
            "history": history,
            "meta": meta,
        },
        args.output,
    )

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(7, 4))
            plt.plot(history["train"], label="train")
            plt.plot(history["val"], label="val")
            plt.xlabel("epoch")
            plt.ylabel("loss (text CE + class CE)")
            plt.title(f"Component predictor ({args.distance}, {feat_dim}d)")
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.splitext(args.output)[0] + ".png"
            plt.savefig(plot_path, dpi=120)
            print(f"Wrote loss curve: {plot_path}")
        except ImportError:
            print("matplotlib not available — skipping the loss curve")

    elapsed = time.time() - start_time
    print(f"\nSaved component predictor: {args.output}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Total training time: {elapsed:.1f}s")
    print("\nNext step — evaluate with:")
    print("  python 13_evaluate_component_prediction.py \\")
    print(f"    --data_dir {args.data_dir} --model {args.output}")


if __name__ == "__main__":
    main()
