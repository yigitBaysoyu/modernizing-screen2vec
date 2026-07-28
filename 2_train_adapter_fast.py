# Two-stage LLaVA-style adapter trainer (fast variant).
#
# Replaces the single-stage joint training that mode-collapsed (see the
# EmbeddingProjector docstring in 2_train_adapter.py for the 2026-07-10
# diagnosis). Follows the LLaVA / LLaVA-1.5 recipe:
#
#   Stage 1 — feature alignment ("training a compatible visual tokenizer
#             for the frozen LLM"): base LM fully frozen, NO LoRA; only the
#             projector trains, at a high LR (cfg.lr_projector_stage1),
#             weight decay 0. Saves stage1_projector.pth.
#   Stage 2 — joint fine-tune: LoRA on the LM + projector (warm-started
#             from stage 1), both at ~2e-5. Saves best_projector.pth and
#             best_lora_adapter/ as before.
#
# Built-in conditioning diagnostic: every epoch, validation runs twice —
# once with the true embeddings and once with embeddings shuffled across
# samples. The gap (shuffled - true) measures how much the decoder actually
# uses the screen embedding. In stage 1, if the gap has not opened after
# --kill_epochs epochs, the run aborts (kill criterion: a collapsed
# projector never recovers, so fail fast instead of burning GPU-days).
#
# Speedups kept from the previous fast variant:
#   A1: targets cleaned + tokenized once up front and cached in memory.
#   A3: batched validation.  B1: batched training (per-sample token-mean
#   loss, then mean over the batch).  Per-rank shards trimmed to equal
#   length so multi-GPU runs cannot deadlock on uneven collectives.
# New: per-epoch seeded shuffling of the training set (dataset order is
# app-alphabetical, so consecutive same-app batches hurt; disable with
# --no-shuffle to reproduce old runs).
#
# The projector runs in float32 (stage-1 LR is too hot for bf16 params);
# its output is cast to the LM dtype. Checkpoints are saved in fp32 and
# load fine into the bf16 copies used by the reconstruction scripts.
#
# Usage:
#   # Stage 1 (alignment; cheap — no LoRA grads, larger batches fit):
#   accelerate launch --multi_gpu --gpu_ids 0,2 --num_processes 2 \
#       2_train_adapter_fast.py --stage 1 --batch_size 8 --grad_accum_steps 4
#
#   # Stage 2 (joint fine-tune, warm-started from stage1_projector.pth):
#   accelerate launch --multi_gpu --gpu_ids 0,2 --num_processes 2 \
#       2_train_adapter_fast.py --stage 2
#
#   # Smoke test on a small subset:
#   python 2_train_adapter_fast.py --stage 1 --max_samples 64 \
#       --num_epochs 2 --seed 42 --output_dir adapter_model_out_smoke

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
    return [
        samples[i: i + batch_size]
        for i in range(0, len(samples), batch_size)
    ]


# =============================================================================
# Batched loss (per-sample token-mean, then mean over the batch)
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

    # Projector runs in fp32; only its output is cast to the LM dtype.
    emb = torch.from_numpy(
        np.stack([s["embedding"] for s in batch])
    ).to(device=device, dtype=torch.float32)
    emb = F.normalize(emb, p=2, dim=-1)
    prefix_embeds = projector(emb).to(model_dtype)  # [B, N, D]

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
        description="Two-stage LLaVA-style adapter trainer "
                    "(stage 1: projector alignment, stage 2: joint LoRA)."
    )
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                        help="1 = projector-only alignment (LM frozen), "
                             "2 = joint LoRA fine-tune (warm-started)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Override cfg.num_epochs (for smoke tests)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Use only the first N samples of each split")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed init + shuffling for reproducibility")
    parser.add_argument("--lora_dropout", type=float, default=None,
                        help="Override cfg.lora_dropout (stage 2 only)")
    parser.add_argument("--lr_projector", type=float, default=None,
                        help="Override the stage-default projector LR")
    parser.add_argument("--lr_lora", type=float, default=None,
                        help="Override cfg.lr_lora (stage 2 only)")
    parser.add_argument("--init_projector", type=str, default=None,
                        help="Projector checkpoint to warm-start from. "
                             "Stage 2 defaults to "
                             "<output_dir>/stage1_projector.pth; pass 'none' "
                             "to (re-)initialise fresh.")
    parser.add_argument("--kill_gap", type=float, default=0.005,
                        help="Stage 1: abort if (shuffled - true) val loss "
                             "gap is below this after --kill_epochs epochs. "
                             "<= 0 disables the check.")
    parser.add_argument("--kill_epochs", type=int, default=2,
                        help="Stage 1: epochs the conditioning gap gets to "
                             "open before the kill criterion applies")
    parser.add_argument("--shuffle",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Reshuffle training samples each epoch (seeded)")
    parser.add_argument("--gradient_checkpointing",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", type=str, default="adapter_model_out")
    parser.add_argument("--embeddings_dir", type=str, default="embeddings")
    args = parser.parse_args()

    cfg = AdapterConfig()
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.lora_dropout is not None:
        cfg.lora_dropout = args.lora_dropout
    cfg.grad_accum_steps = args.grad_accum_steps
    stage = args.stage

    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum_steps)
    device = accelerator.device
    is_main = accelerator.is_local_main_process
    if is_main:
        print(f"=== Stage {stage} "
              f"({'projector alignment, LM frozen' if stage == 1 else 'joint LoRA fine-tune'}) ===")
        print(f"Using device: {device} "
              f"(Distributed across {accelerator.num_processes} processes)")
        print(f"Batch size: {args.batch_size} x accum {cfg.grad_accum_steps} "
              f"= effective batch {args.batch_size * cfg.grad_accum_steps}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # -------------------------------------------------------------------------
    # 1. Load metadata and split (same positional 80/20 as always)
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
    # 2. Tokenizer, then one-time data preparation
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_samples = prepare_split(train_df, tokenizer, cfg, "train", is_main)
    val_samples = prepare_split(val_df, tokenizer, cfg, "val", is_main)

    # Calibration embeddings for out_scale: taken from the FULL train list
    # (before sharding) so every rank computes the identical value.
    calib_embs = np.stack(
        [s["embedding"] for s in train_samples[: min(64, len(train_samples))]]
    )

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

    # Conditioning diagnostic: same val targets, embeddings rotated by one
    # sample (a cheap derangement). If the decoder uses the embedding, loss
    # on this set must be clearly higher than on the true pairing.
    val_shard_shuf = [
        dict(s, embedding=val_shard[(i + 1) % len(val_shard)]["embedding"])
        for i, s in enumerate(val_shard)
    ]
    val_batches_shuf = make_batches(val_shard_shuf, args.batch_size)

    # -------------------------------------------------------------------------
    # 3. LM (frozen in stage 1, LoRA in stage 2) + projector + optimizer
    # -------------------------------------------------------------------------
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    with accelerator.main_process_first():
        lm_model = AutoModelForCausalLM.from_pretrained(
            cfg.lm_model_id,
            torch_dtype=model_dtype,
        )

    if stage == 2:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
        )
        lm_model = get_peft_model(lm_model, lora_config)
        if is_main:
            lm_model.print_trainable_parameters()
    else:
        lm_model.requires_grad_(False)
        if is_main:
            print("[*] Stage 1: base LM fully frozen (no LoRA); "
                  "training projector only.")

    if args.gradient_checkpointing:
        lm_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    model_dtype = next(lm_model.parameters()).dtype
    lm_hidden_dim = lm_model.config.hidden_size

    # Projector stays in float32 (see header note).
    projector = EmbeddingProjector(
        cfg.qwen_embed_dim, lm_hidden_dim, cfg.num_prefix_tokens,
        cfg.proj_hidden_dim,
    )

    init_path = args.init_projector
    if stage == 2 and init_path is None:
        init_path = os.path.join(args.output_dir, "stage1_projector.pth")
    if init_path and init_path.lower() != "none":
        if not os.path.exists(init_path):
            raise SystemExit(
                f"[!] Projector init checkpoint not found: {init_path}\n"
                f"    Run stage 1 first, or pass --init_projector none "
                f"to start fresh (not recommended for stage 2)."
            )
        projector.load_state_dict(torch.load(init_path, map_location="cpu"))
        if is_main:
            print(f"[*] Warm-started projector from {init_path}")
    else:
        # Fresh init: calibrate out_scale to the LM token-embedding norm.
        with torch.no_grad():
            emb_table = accelerator.unwrap_model(
                lm_model
            ).get_input_embeddings().weight
            target_norm = emb_table.float().norm(dim=-1).mean().item()
        calib = F.normalize(torch.from_numpy(calib_embs).float(), dim=-1)
        projector.calibrate_out_scale(calib, target_norm)
        if is_main:
            print(f"[*] Calibrated projector out_scale="
                  f"{projector.out_scale.item():.4e} "
                  f"(LM token-embedding mean norm {target_norm:.4f})")

    if is_main:
        n_params = sum(p.numel() for p in projector.parameters())
        print(f"Projector parameters: {n_params:,} "
              f"({cfg.num_prefix_tokens} prefix tokens x {lm_hidden_dim} dims)")

    # Optimizer: never any weight decay on the projector.
    if stage == 1:
        lr_p = args.lr_projector or cfg.lr_projector_stage1
        optimizer = torch.optim.AdamW(
            projector.parameters(), lr=lr_p, weight_decay=0.0
        )
        if is_main:
            print(f"Stage 1 LRs: projector {lr_p:g}")
    else:
        lr_p = args.lr_projector or cfg.lr_projector_stage2
        lr_l = args.lr_lora or cfg.lr_lora
        optimizer = torch.optim.AdamW([
            {"params": projector.parameters(), "lr": lr_p,
             "weight_decay": 0.0},
            {"params": [p for p in lm_model.parameters() if p.requires_grad],
             "lr": lr_l},
        ])
        if is_main:
            print(f"Stage 2 LRs: projector {lr_p:g}, LoRA {lr_l:g}")

    # Stage 1: the frozen LM is NOT wrapped by accelerate (DDP rejects
    # modules without trainable params, and there is nothing to sync).
    if stage == 2:
        lm_model, projector, optimizer = accelerator.prepare(
            lm_model, projector, optimizer
        )
    else:
        lm_model = lm_model.to(device)
        projector, optimizer = accelerator.prepare(projector, optimizer)

    def trainable_params():
        params = list(projector.parameters())
        if stage == 2:
            params += [p for p in lm_model.parameters() if p.requires_grad]
        return params

    def evaluate(batches, desc):
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(batches, desc=desc, leave=False,
                              disable=not is_main):
                _, per_sample = batch_forward(
                    batch, lm_model, projector, tokenizer, accelerator,
                    cfg, device, model_dtype,
                )
                total += per_sample.sum().item()
                n += len(batch)
        t = torch.tensor([total, float(n)], dtype=torch.float64, device=device)
        t = accelerator.reduce(t, reduction="sum")
        return (t[0] / torch.clamp(t[1], min=1.0)).item()

    # -------------------------------------------------------------------------
    # 4. Training loop
    # -------------------------------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "val_loss_shuffled": [],
               "cond_gap": []}
    best_val_loss = float("inf")
    patience_counter = 0
    avg_loss = -1.0
    conditioning_failed = False
    rng = np.random.default_rng(args.seed if args.seed is not None else 0)
    acc_models = (projector,) if stage == 1 else (lm_model, projector)

    if is_main:
        print(f"\n--- Starting Adapter Training (stage {stage}) ---\n")
    run_start = time.time()

    for epoch in range(cfg.num_epochs):
        lm_model.train()  # base model has no dropout; needed so gradient
        projector.train()  # checkpointing engages in stage 1 too

        if args.shuffle:
            order = rng.permutation(len(train_shard))
            train_batches = make_batches(
                [train_shard[i] for i in order], args.batch_size
            )

        epoch_loss = 0.0
        n_samples = 0
        for batch in tqdm(
            train_batches,
            disable=not is_main,
            desc=f"Epoch {epoch + 1}/{cfg.num_epochs} | Loss: {avg_loss:.6f}",
        ):
            with accelerator.accumulate(*acc_models):
                loss, per_sample = batch_forward(
                    batch, lm_model, projector, tokenizer, accelerator,
                    cfg, device, model_dtype,
                )
                accelerator.backward(loss)
                epoch_loss += per_sample.detach().sum().item()
                n_samples += len(batch)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params(),
                                                max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        t = torch.tensor([epoch_loss, float(n_samples)],
                         dtype=torch.float64, device=device)
        t = accelerator.reduce(t, reduction="sum")
        avg_loss = (t[0] / torch.clamp(t[1], min=1.0)).item()
        if is_main:
            history["train_loss"].append(avg_loss)

        # --- VALIDATION (true pairing + shuffled-embedding diagnostic) ---
        lm_model.eval()
        projector.eval()
        avg_val_loss = evaluate(val_batches, "Validating")
        avg_val_shuf = evaluate(val_batches_shuf, "Validating (shuffled emb)")
        cond_gap = avg_val_shuf - avg_val_loss

        if is_main:
            history["val_loss"].append(avg_val_loss)
            history["val_loss_shuffled"].append(avg_val_shuf)
            history["cond_gap"].append(cond_gap)
            print(f" -> Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                  f" | shuffled-emb Val: {avg_val_shuf:.6f}"
                  f" | conditioning gap: {cond_gap:+.6f}")
            if cond_gap <= 0.001:
                print("    [!] Gap ~ 0: the decoder is currently NOT using "
                      "the screen embedding.")

        # --- KILL CRITERION (stage 1 only) ---
        if (stage == 1 and args.kill_gap > 0
                and (epoch + 1) >= args.kill_epochs
                and cond_gap < args.kill_gap):
            conditioning_failed = True
            if is_main:
                print(f"\n[!!!] KILL CRITERION: conditioning gap "
                      f"{cond_gap:+.6f} < {args.kill_gap} after "
                      f"{epoch + 1} epochs — the projector is not learning "
                      f"a usable conditioning signal. Aborting so no "
                      f"further GPU time is wasted. (Raise --kill_epochs "
                      f"or lower --kill_gap to override.)")
            break

        # --- EARLY STOPPING + BEST CHECKPOINT (on true-pairing val loss) ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            if is_main:
                print("    [*] New best validation loss!")
                os.makedirs(args.output_dir, exist_ok=True)
                unwrapped_projector = accelerator.unwrap_model(projector)
                if stage == 1:
                    torch.save(
                        unwrapped_projector.state_dict(),
                        os.path.join(args.output_dir, "stage1_projector.pth"),
                    )
                else:
                    torch.save(
                        unwrapped_projector.state_dict(),
                        os.path.join(args.output_dir, "best_projector.pth"),
                    )
                    accelerator.unwrap_model(lm_model).save_pretrained(
                        os.path.join(args.output_dir, "best_lora_adapter")
                    )
                print(f"    [*] Saved best checkpoint to {args.output_dir}")
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
        if stage == 1:
            torch.save(
                unwrapped_projector.state_dict(),
                os.path.join(args.output_dir,
                             f"stage1_projector_{timestamp}.pth"),
            )
        else:
            torch.save(
                unwrapped_projector.state_dict(),
                os.path.join(args.output_dir, f"projector_{timestamp}.pth"),
            )
            accelerator.unwrap_model(lm_model).save_pretrained(
                os.path.join(args.output_dir, f"lora_adapter_{timestamp}")
            )

        os.makedirs("adapter_training_history", exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_loss"], label="train")
        plt.plot(history["val_loss"], label="val (true emb)")
        plt.plot(history["val_loss_shuffled"], label="val (shuffled emb)",
                 linestyle="--")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title(f"Adapter stage {stage} training {timestamp}")
        plot_path = os.path.join(
            "adapter_training_history",
            f"training_history_stage{stage}_{timestamp}.png"
        )
        plt.savefig(plot_path)

        total_min = (time.time() - run_start) / 60
        print(f"\nBest val loss: {best_val_loss:.6f}")
        if history["cond_gap"]:
            print(f"Final conditioning gap: {history['cond_gap'][-1]:+.6f}")
        if conditioning_failed:
            print("RESULT: run ABORTED by the kill criterion — do not use "
                  "these checkpoints.")
        print(f"Total training time: {total_min:.1f} min")
        print(f"History plot: {plot_path}")
        history["stage"] = stage
        history["conditioning_failed"] = conditioning_failed
        with open(os.path.join(args.output_dir,
                               f"history_stage{stage}_{timestamp}.json"),
                  "w") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
