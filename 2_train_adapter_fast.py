# Faster drop-in variant of 2_train_adapter.py — same training mechanism.
#
# Speedups over the original (see git history / discussion):
#   A1: targets are cleaned + tokenized ONCE up front and cached in memory
#       (the original re-reads and re-cleans every JSON every epoch).
#       Invalid JSONs are filtered out deterministically before sharding,
#       so ranks can never desynchronize on per-sample skips.
#   A3: validation runs in batches instead of sample-by-sample.
#   B1: training runs in batches (default batch_size=4, grad_accum_steps=1,
#       i.e. the SAME effective batch of 4 as the original's 1x4). The loss
#       is averaged per sample (each sample's token-mean, then mean over the
#       batch), matching the original's accumulated gradient exactly up to
#       float ordering.
#
# Additionally, per-rank shards are trimmed to equal length so every rank
# performs the same number of backward passes per epoch. This fixes the
# NCCL collective mismatch that deadlocks the original script whenever the
# training set does not divide evenly by the number of processes.
#
# Mechanism kept identical to 2_train_adapter.py:
#   - same AdapterConfig (model, LoRA setup, learning rates, patience,
#     max_target_len, epoch budget)
#   - same 80/20 positional split on the numerically sorted embeddings
#   - same per-sample loss semantics and effective batch size
#   - same epoch-level early stopping on mean validation loss
#   - no shuffling (the original iterates in dataset order)
#
# Usage:
#   accelerate launch --multi_gpu --gpu_ids 0,2 --num_processes 2 \
#       2_train_adapter_fast.py
#
#   # Verification run on a small subset:
#   python 2_train_adapter_fast.py --max_samples 64 --num_epochs 2 \
#       --seed 42 --output_dir adapter_model_out_smoke

import os
import json
import time
import argparse
import importlib
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator

from utils import load_embeddings, process_json_file

# Reuse the projector and config from the original trainer (leading digit in
# the filename -> importlib), so both scripts share one definition.
_orig = importlib.import_module("2_train_adapter")
EmbeddingProjector = _orig.EmbeddingProjector
AdapterConfig = _orig.AdapterConfig


# =============================================================================
# Data preparation (A1: one-time clean + tokenize + embedding cache)
# =============================================================================
def prepare_split(df, tokenizer, cfg, desc, is_main):
    """Clean, tokenize and cache one split. Returns a list of dicts with
    'input_ids' (np.int32, EOS appended, truncated) and 'embedding'
    (np.float32 [4096])."""
    samples = []
    skipped = 0
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Preparing {desc}",
        disable=not is_main,
    ):
        try:
            target_json = process_json_file(row["json-path"], minify=True)
        except Exception:
            skipped += 1
            continue
        if not target_json or len(target_json) < 5:
            skipped += 1
            continue

        enc = tokenizer(
            target_json + tokenizer.eos_token,
            truncation=True,
            max_length=cfg.max_target_len,
            padding=False,
        )
        with np.load(row["embedding_path"]) as f_data:
            embedding = f_data["embedding"].astype(np.float32)

        samples.append({
            "input_ids": np.asarray(enc["input_ids"], dtype=np.int32),
            "embedding": embedding,
        })

    if is_main and skipped:
        print(f"[*] {desc}: skipped {skipped} samples with invalid JSON.")
    return samples


def make_batches(samples, batch_size):
    """Consecutive batches in dataset order (the original does not shuffle)."""
    return [
        samples[i: i + batch_size]
        for i in range(0, len(samples), batch_size)
    ]


# =============================================================================
# Batched loss (B1: per-sample token-mean, then mean over the batch — the
# same quantity the original accumulates one sample at a time)
# =============================================================================
def batch_forward(batch, lm_model, projector, tokenizer, accelerator,
                  cfg, device, model_dtype):
    """Returns (loss, per_sample_losses) for one batch."""
    B = len(batch)
    max_len = max(len(s["input_ids"]) for s in batch)
    pad_id = tokenizer.pad_token_id

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels_tgt = torch.full((B, max_len), -100, dtype=torch.long)
    attn_tgt = torch.zeros((B, max_len), dtype=torch.long)
    for i, s in enumerate(batch):
        ids = torch.from_numpy(s["input_ids"].astype(np.int64))
        input_ids[i, : len(ids)] = ids
        labels_tgt[i, : len(ids)] = ids
        attn_tgt[i, : len(ids)] = 1

    input_ids = input_ids.to(device)
    labels_tgt = labels_tgt.to(device)
    attn_tgt = attn_tgt.to(device)

    emb = torch.from_numpy(
        np.stack([s["embedding"] for s in batch])
    ).to(device=device, dtype=model_dtype)
    emb = F.normalize(emb, p=2, dim=-1)
    prefix_embeds = projector(emb)  # [B, N, D]

    target_embeds = accelerator.unwrap_model(
        lm_model
    ).get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)

    prefix_attn = torch.ones(
        (B, cfg.num_prefix_tokens), dtype=torch.long, device=device
    )
    attn_mask = torch.cat([prefix_attn, attn_tgt], dim=1)

    prefix_labels = torch.full(
        (B, cfg.num_prefix_tokens), -100, dtype=torch.long, device=device
    )
    labels = torch.cat([prefix_labels, labels_tgt], dim=1)

    outputs = lm_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
    )

    # Shifted causal LM loss, unreduced, then per-sample token-mean
    shift_logits = outputs.logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, -1)
    valid_mask = (shift_labels != -100).float()
    per_sample = (token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)
    return per_sample.mean(), per_sample


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Batched, cached variant of 2_train_adapter.py "
                    "(same training mechanism, much faster)."
    )
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Samples per forward pass (default 4 = original "
                             "effective batch)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Batches to accumulate per optimizer step "
                             "(effective batch = batch_size * this)")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override cfg.num_epochs (for smoke tests)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Use only the first N samples of each split "
                             "(for smoke tests)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed model init for reproducible comparisons")
    parser.add_argument("--lora_dropout", type=float, default=None,
                        help="Override cfg.lora_dropout (testing only: set 0 "
                             "to compare batch configurations exactly)")
    parser.add_argument("--gradient_checkpointing",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Recompute activations in backward (identical "
                             "gradients, ~10x less activation memory; needed "
                             "for batch_size > 1 with 2048-token targets)")
    parser.add_argument("--output_dir", type=str, default="adapter_model_out",
                        help="Checkpoint directory")
    parser.add_argument("--embeddings_dir", type=str, default="embeddings")
    args = parser.parse_args()

    cfg = AdapterConfig()
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.lora_dropout is not None:
        cfg.lora_dropout = args.lora_dropout
    cfg.grad_accum_steps = args.grad_accum_steps

    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum_steps)
    device = accelerator.device
    is_main = accelerator.is_local_main_process
    if is_main:
        print(f"Using device: {device} "
              f"(Distributed across {accelerator.num_processes} processes)")
        print(f"Batch size: {args.batch_size} x accum {cfg.grad_accum_steps} "
              f"= effective batch {args.batch_size * cfg.grad_accum_steps}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # -------------------------------------------------------------------------
    # 1. Load metadata and split (same positional 80/20 as the original)
    # -------------------------------------------------------------------------
    all_embeddings = load_embeddings(args.embeddings_dir, load_data=False)
    train_size = int(len(all_embeddings) * 0.8)
    train_df = all_embeddings.iloc[:train_size]
    val_df = all_embeddings.iloc[train_size:]
    if args.max_samples is not None:
        train_df = train_df.iloc[: args.max_samples]
        val_df = val_df.iloc[: max(args.max_samples // 4, args.batch_size)]
    if is_main:
        print(f"Using {len(train_df)} samples for training "
              f"and {len(val_df)} for validation.")

    # -------------------------------------------------------------------------
    # 2. Tokenizer (needed for caching), then one-time data preparation
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_samples = prepare_split(train_df, tokenizer, cfg, "train", is_main)
    val_samples = prepare_split(val_df, tokenizer, cfg, "val", is_main)

    # Shard, trimming train shards to equal length so all ranks execute the
    # same number of collectives per epoch (drops < num_processes samples).
    n_proc, rank = accelerator.num_processes, accelerator.process_index
    shard_len = len(train_samples) // n_proc
    train_shard = train_samples[rank::n_proc][:shard_len]
    val_shard = val_samples[rank::n_proc]
    if len(train_samples) % n_proc and is_main:
        print(f"[*] Trimmed train shards to {shard_len} samples each "
              f"({len(train_samples) % n_proc} dropped for rank parity).")

    train_batches = make_batches(train_shard, args.batch_size)
    val_batches = make_batches(val_shard, args.batch_size)

    # -------------------------------------------------------------------------
    # 3. Model + LoRA + projector + optimizer (identical to the original)
    # -------------------------------------------------------------------------
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    with accelerator.main_process_first():
        lm_model = AutoModelForCausalLM.from_pretrained(
            cfg.lm_model_id,
            torch_dtype=model_dtype,
        )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
    )
    lm_model = get_peft_model(lm_model, lora_config)
    if args.gradient_checkpointing:
        lm_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if is_main:
        lm_model.print_trainable_parameters()

    model_dtype = next(lm_model.parameters()).dtype
    lm_hidden_dim = lm_model.config.hidden_size
    projector = EmbeddingProjector(
        cfg.qwen_embed_dim, lm_hidden_dim, cfg.num_prefix_tokens
    ).to(model_dtype)

    optimizer = torch.optim.AdamW([
        {"params": projector.parameters(), "lr": cfg.lr_projector},
        {
            "params": [p for p in lm_model.parameters() if p.requires_grad],
            "lr": cfg.lr_lora,
        },
    ])

    lm_model, projector, optimizer = accelerator.prepare(
        lm_model, projector, optimizer
    )

    # -------------------------------------------------------------------------
    # 4. Training loop
    # -------------------------------------------------------------------------
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    avg_loss = -1.0

    if is_main:
        print("\n--- Starting LLaVA-Style Adapter Training (fast) ---\n")
    run_start = time.time()

    for epoch in range(cfg.num_epochs):
        lm_model.train()
        projector.train()
        epoch_loss = 0.0
        n_samples = 0

        for batch in tqdm(
            train_batches,
            disable=not is_main,
            desc=f"Epoch {epoch + 1}/{cfg.num_epochs} | Loss: {avg_loss:.6f}",
        ):
            with accelerator.accumulate(lm_model, projector):
                loss, per_sample = batch_forward(
                    batch, lm_model, projector, tokenizer, accelerator,
                    cfg, device, model_dtype,
                )
                accelerator.backward(loss)
                epoch_loss += per_sample.detach().sum().item()
                n_samples += len(batch)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(projector.parameters())
                        + [p for p in lm_model.parameters() if p.requires_grad],
                        max_norm=1.0,
                    )
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, n_samples)
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss = accelerator.reduce(avg_loss_tensor, reduction="mean").item()
        if is_main:
            history["train_loss"].append(avg_loss)

        # --- VALIDATION (A3: batched, forward-only) ---
        lm_model.eval()
        projector.eval()
        val_loss_total = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in tqdm(
                val_batches, desc="Validating", leave=False,
                disable=not is_main,
            ):
                _, per_sample = batch_forward(
                    batch, lm_model, projector, tokenizer, accelerator,
                    cfg, device, model_dtype,
                )
                val_loss_total += per_sample.sum().item()
                val_n += len(batch)

        avg_val_loss = val_loss_total / max(1, val_n)
        avg_val_tensor = torch.tensor(avg_val_loss, device=device)
        avg_val_loss = accelerator.reduce(avg_val_tensor, reduction="mean").item()

        if is_main:
            history["val_loss"].append(avg_val_loss)
            print(f" -> Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # --- EARLY STOPPING + BEST CHECKPOINT (same policy as original) ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if is_main:
                print("    [*] New best validation loss!")
                os.makedirs(args.output_dir, exist_ok=True)
                unwrapped_projector = accelerator.unwrap_model(projector)
                unwrapped_lm = accelerator.unwrap_model(lm_model)
                torch.save(
                    unwrapped_projector.state_dict(),
                    os.path.join(args.output_dir, "best_projector.pth"),
                )
                unwrapped_lm.save_pretrained(
                    os.path.join(args.output_dir, "best_lora_adapter")
                )
                print(f"    [*] Saved best model checkpoint to {args.output_dir}")
        else:
            patience_counter += 1
            if is_main:
                print(f"    [*] No improvement "
                      f"(patience {patience_counter}/{cfg.patience})")
            if patience_counter >= cfg.patience:
                if is_main:
                    print(f"\n--- Early stopping at epoch {epoch + 1} ---")
                break

    # -------------------------------------------------------------------------
    # 5. Final timestamped save + history plot (main process only)
    # -------------------------------------------------------------------------
    accelerator.wait_for_everyone()
    if is_main:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(args.output_dir, exist_ok=True)
        unwrapped_projector = accelerator.unwrap_model(projector)
        unwrapped_lm = accelerator.unwrap_model(lm_model)
        torch.save(
            unwrapped_projector.state_dict(),
            os.path.join(args.output_dir, f"projector_{timestamp}.pth"),
        )
        unwrapped_lm.save_pretrained(
            os.path.join(args.output_dir, f"lora_adapter_{timestamp}")
        )

        os.makedirs("adapter_training_history", exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_loss"], label="train")
        plt.plot(history["val_loss"], label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title(f"Adapter training (fast) {timestamp}")
        plot_path = os.path.join(
            "adapter_training_history", f"training_history_{timestamp}.png"
        )
        plt.savefig(plot_path)

        total_min = (time.time() - run_start) / 60
        print(f"\nBest val loss: {best_val_loss:.6f}")
        print(f"Total training time: {total_min:.1f} min")
        print(f"History plot: {plot_path}")
        with open(os.path.join(args.output_dir, f"history_{timestamp}.json"),
                  "w") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
