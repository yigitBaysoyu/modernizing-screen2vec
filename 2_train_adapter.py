# Trains the LLaVA-style adapter to map Qwen3-VL embedding space to
# cleaned JSON via a LoRA-finetuned small language model.
#
# Architecture:
#   Pre-computed Qwen3-VL embedding (4096-dim)
#     -> EmbeddingProjector (MLP): maps to num_prefix_tokens x lm_hidden_dim
#     -> Concatenated as prefix to target JSON token embeddings
#     -> Qwen2.5-0.5B (LoRA fine-tuned) learns to generate cleaned JSON
#
# Target: minified cleaned JSON string from process_json_file()

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from contextlib import ExitStack
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator

from utils import load_embeddings, process_json_file


# =============================================================================
# Configuration
# =============================================================================
class AdapterConfig:
    """Configuration for the LLaVA-style adapter training."""

    # Qwen3-VL embedding dimension (from step 1_qwen_generate_embeddings.py)
    qwen_embed_dim = 4096

    # Projector: maps Qwen embedding -> virtual prefix tokens for the LM
    num_prefix_tokens = 128

    # Base LM for training and evaluation. Adapter must be retrained
    # when this changes — LoRA weights are architecture-specific.
    # lm_model_id = "Qwen/Qwen3.5-2B"
    lm_model_id = "Qwen/Qwen2.5-1.5B"  # sanity check: state of 926391f (2026-06-23)

    # LoRA fine-tuning parameters updated for newer Qwen structural compliance
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    # lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]  # sanity check: attention only, as in 926391f

    # Training hyperparameters
    lr_projector = 1e-4  # Learning rate for the MLP projector
    lr_lora = 2e-5  # Learning rate for LoRA parameters
    num_epochs = 100
    # patience = 5  # Early stopping patience
    patience = 3  # sanity check: as in 926391f
    # Max tokens for the target JSON string.  Targets that do not fit are
    # dropped by build_target_column() rather than truncated -- see the note
    # there on why truncating them poisons the stop behaviour.
    #
    # Measured over all 66,261 cleaned Rico ground truths (Qwen2.5 tokenizer):
    # p50 617, p90 1788, p95 2268, p99 3360, p99.9 6301, max 158,896 tokens.
    # 4096 keeps 99.56% of the dataset and costs ~1.03x the training compute of
    # 2048, because only 7% of the samples are anywhere near the limit.
    max_target_len = 4096
    grad_accum_steps = 4  # Gradient accumulation steps

    # Throughput optimisations, off by default so the default path reproduces
    # earlier sweeps exactly. Overridden from --fast in main(). See
    # FAST_PATH_NOTES under parse_args() for what it does and does not change.
    fast = False


EMBEDDINGS_DIR = "embeddings_4096_last"
OUTPUT_DIR = "adapter_model_out"


def model_output_slug(lm_model_id):
    """"Qwen/Qwen2.5-1.5B" -> "qwen2.5-1.5b", appended to output_dir so
    checkpoints trained against different base LMs never collide."""
    return lm_model_id.split("/")[-1].lower()


def str2bool(value):
    """argparse type for a flag that also accepts an explicit true/false value."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if value.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the LLaVA-style adapter (EmbeddingProjector + LoRA) that "
            "maps Qwen3-VL embeddings to cleaned JSON."
        )
    )
    parser.add_argument(
        "--embeddings_dir", type=str, default=EMBEDDINGS_DIR,
        help=f"Directory of .npz embeddings to train on (default: {EMBEDDINGS_DIR}). "
             "Accepts both (4096,) and (N, 4096) embeddings.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=OUTPUT_DIR,
        help=f"Where projector and LoRA checkpoints are written "
             f"(default: {OUTPUT_DIR}). Give each sweep run its own directory "
             "so the checkpoints do not overwrite each other.",
    )
    parser.add_argument(
        "--fast", type=str2bool, nargs="?", const=True, default=False,
        help=(
            "Enable the throughput optimisations (default: off). They are "
            "arithmetically equivalent to the default path -- they only remove "
            "redundant work -- but the default stays bit-for-bit the old code "
            "so a sweep can be reproduced exactly. See FAST_PATH_NOTES."
        ),
    )
    return parser.parse_args()


# =============================================================================
# FAST_PATH_NOTES -- what --fast changes, and why it does not change results
# =============================================================================
# Everything below is gated on cfg.fast and leaves the --fast=false path
# untouched.  Three sources of avoidable overhead, none of which touches the
# gradients, the sample order, or the number of optimizer steps:
#
# 1.  Gradient all-reduce on every micro-step.  Accelerator() is constructed
#     with gradient_accumulation_steps, but the training loop never enters the
#     accelerator.accumulate() context, so accelerator.sync_gradients stays True
#     and every accelerator.backward() triggers a full all-reduce over all
#     ranks -- for the LoRA parameters *and* the projector, grad_accum_steps
#     times more often than needed.  The fast path wraps the non-boundary
#     micro-steps in accelerator.no_sync(), which is the pattern PyTorch
#     documents for DDP gradient accumulation: the local .grad tensors keep
#     accumulating, and the reducer on the boundary step all-reduces the
#     accumulated total.  Averaging a sum equals summing the averages, so the
#     resulting gradient is the same up to floating-point associativity.
#
#     The boundary must include the *last* sample of the epoch even when it
#     falls into a short trailing group, otherwise the flush below would step
#     on gradients that were never all-reduced and the ranks would silently
#     drift apart.  shard_for_process() gives every rank the same number of
#     rows, so every rank picks the same boundaries without communicating.
#
# 2.  A .item() per sample.  Reading the loss back on the host synchronises the
#     CPU with the GPU on every single step, so the CPU can never queue the
#     next forward while the current one is still running.  The fast path
#     accumulates the loss in a float64 GPU scalar instead and reads it once
#     per epoch.  float64 rather than the loss's own float32 because the value
#     feeds the early-stopping comparison, and a float32 running sum over
#     thousands of samples would drift away from the Python float accumulation
#     of the default path far enough to matter at the margin.
#
# 3.  Re-tokenizing every target in every epoch.  build_target_column() already
#     caches the cleaned JSON, but the tokenizer still ran over all of it once
#     per epoch, single-threaded, in the critical path of the training loop.
#     The fast path tokenizes once, up front, with the exact same call the loop
#     used, and stores the ids as int32 (~150 MB for the full Rico set, against
#     ~1 GB for Python lists).  The token budget filter above it is left
#     completely alone, so --fast trains on exactly the same set of samples.


# =============================================================================
# Embedding Projector (The LLaVA-style MLP adapter)
# =============================================================================
class EmbeddingProjector(nn.Module):
    """
    Projects a single Qwen3-VL embedding vector into a sequence of
    *unique* virtual prefix tokens in the language model's embedding space.

    This is the core "adapter" component, analogous to LLaVA's visual
    projection layer.  It translates the embedding model's representation
    into a conditioning signal that the language model can attend to.

    The full 4096-dim embedding is broadcast to each of the
    `num_prefix_tokens` positions, so every prefix token sees the
    complete representation.  A learned positional embedding is added
    per token to differentiate them.

    Example:
        projector = EmbeddingProjector(qwen_dim=4096, lm_hidden_dim=896, num_prefix_tokens=128)

        qwen_embedding = torch.randn(1, 4096)        # one pre-computed Qwen3-VL embedding
        prefix = projector(qwen_embedding)            # -> [1, 128, 896]
        # These 128 virtual tokens are prepended to the LM's input sequence.
        # The LM then generates the cleaned JSON conditioned on them.

    Input:  [B, 4096]  (Qwen3-VL embedding)
    Output: [B, num_prefix_tokens, lm_hidden_dim]  (prefix for the LM)
    """

    def __init__(self, qwen_dim, lm_hidden_dim, num_prefix_tokens, use_prefix_pos=False):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.lm_hidden_dim = lm_hidden_dim

        self.projector = nn.Sequential(
            nn.Linear(qwen_dim, lm_hidden_dim * 2),
            nn.GELU(),
            nn.Linear(lm_hidden_dim * 2, lm_hidden_dim),
        )
        # Added in 8453eda because b161f33 had collapsed embeddings to (4096,),
        # where expand() produces num_prefix_tokens identical copies. The table
        # is what makes those tokens distinguishable again, so it belongs to the
        # (4096,) case and only to it: with (N, 4096) input the tokens already
        # differ and 926391f trained fine without one. main() therefore derives
        # the flag from the embedding shape instead of fixing it here.
        # 3_json_reconstruction.py detects it from the checkpoint either way.
        self.use_prefix_pos = use_prefix_pos
        if use_prefix_pos:
            self.prefix_pos = nn.Embedding(num_prefix_tokens, lm_hidden_dim)

    def forward(self, qwen_embedding):
        """
        Args:
            qwen_embedding: [B, qwen_dim] or [B, N, qwen_dim] pre-computed
                Qwen3-VL embeddings, where N <= num_prefix_tokens.
        Returns:
            [B, P, lm_hidden_dim] virtual prefix token embeddings, where P is
            N for an [B, N, qwen_dim] input and num_prefix_tokens for a
            [B, qwen_dim] one.
        """
        if qwen_embedding.dim() == 2:
            # Expand [B, 4096] -> [B, 128, 4096]: each token sees the full embedding
            qwen_embedding = qwen_embedding.unsqueeze(1).expand(-1, self.num_prefix_tokens, -1)

        # An [B, N, qwen_dim] embedding already carries N distinct tokens, so it
        # maps to N prefix tokens directly. The positional table is sized for
        # num_prefix_tokens, so any N <= that shares one checkpoint layout.
        seq_len = qwen_embedding.size(1)
        if seq_len > self.num_prefix_tokens:
            raise ValueError(
                f"Embedding carries {seq_len} tokens but the projector was built "
                f"for at most {self.num_prefix_tokens}. Raise "
                f"AdapterConfig.num_prefix_tokens to at least {seq_len}."
            )

        projected = self.projector(qwen_embedding)  # [B, seq_len, lm_hidden_dim]
        if self.use_prefix_pos:
            positions = torch.arange(seq_len, device=projected.device)
            return projected + self.prefix_pos(positions)
        return projected  # sanity check: as in 926391f


# =============================================================================
# Autoregressive Generation with KV-Cache
# =============================================================================
def generate_from_prefix(
    model, tokenizer, prefix_embeds, max_new_tokens=512, eos_check_every=16
):
    """
    Autoregressively generate tokens conditioned on prefix embeddings.
    Uses KV-caching for efficient O(n) generation.

    Args:
        model:           The (LoRA-wrapped) causal LM.
        tokenizer:       The LM's tokenizer (for EOS detection).
        prefix_embeds:   [B, num_prefix, hidden_dim] projected prefix embeddings.
        max_new_tokens:  Maximum number of tokens to generate.
        eos_check_every: How many decode steps to run between EOS checks.  See
                         below -- this is a throughput knob, not a semantic one.

    Returns:
        generated_ids: [B, generated_len] tensor of generated token IDs,
                       truncated after the first position where every sequence
                       emitted EOS.

    Reading the sampled token back on the host is the single most expensive
    thing this loop can do.  `next_token` lives on the GPU, so testing it in a
    Python `if` forces a device synchronisation, and a sync every step means
    the CPU can never queue the next step's kernels while the current one runs.
    At batch size 1 a 1.5B model launches far more kernels than it does real
    arithmetic, so that serialisation -- not the matmuls -- sets the pace.

    Checking every `eos_check_every` steps instead lets the launch queue stay
    ahead of the device.  The cost is that generation can overshoot EOS by up
    to `eos_check_every - 1` tokens; the truncation at the end removes them, so
    the returned tensor is identical to what a per-step check would produce.
    """
    eos_token_id = tokenizer.eos_token_id

    # First forward pass: process all prefix embeddings, populate KV cache
    outputs = model(inputs_embeds=prefix_embeds, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    generated = [next_token]
    steps_since_check = 1

    # Subsequent passes: single token at a time with cached KV
    for _ in range(max_new_tokens - 1):
        if steps_since_check >= eos_check_every:
            # One sync per `eos_check_every` steps instead of one per step.
            window = torch.cat(generated[-steps_since_check:], dim=1)
            if (window == eos_token_id).all(dim=0).any():
                break
            steps_since_check = 0

        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated.append(next_token)
        steps_since_check += 1

    generated_ids = torch.cat(generated, dim=1)

    # Drop whatever was generated past the first all-sequences-EOS position.
    # `.all(dim=0)` mirrors the batch semantics of the original per-step check.
    eos_positions = (generated_ids == eos_token_id).all(dim=0).nonzero()
    if eos_positions.numel() > 0:
        generated_ids = generated_ids[:, : eos_positions[0].item() + 1]

    return generated_ids


# =============================================================================
# Dataset preparation (target caching + distributed sharding)
# =============================================================================
def fits_token_budget(target_json, tokenizer, max_target_len):
    """
    True if `target_json` plus its EOS token fits within `max_target_len`.

    A string of N characters can never tokenize to more than N tokens, so a
    target shorter than the budget fits without asking the tokenizer at all.
    That short-circuit skips roughly half the dataset (median target is ~2.1k
    characters against a 4096-token budget) and keeps the one-off cost of this
    pass down.
    """
    if len(target_json) + 1 <= max_target_len:
        return True
    n_tokens = len(tokenizer(target_json, add_special_tokens=False).input_ids)
    return n_tokens + 1 <= max_target_len


def build_target_column(embeddings, accelerator, tokenizer, max_target_len, fast=False):
    """
    Pre-computes the cleaned target JSON for every row and drops the rows that
    cannot serve as a training target -- either because the JSON is unusable,
    or because it does not fit in `max_target_len` tokens.

    With `fast`, also pre-computes the target token ids in a "target-ids"
    column, so the training loop does not re-tokenize the whole dataset once
    per epoch.  This runs *after* the filter and never feeds back into it, so
    the set of surviving rows is identical either way.

    Dropping the over-long ones matters more than it looks.  The training loop
    tokenizes `target_json + eos_token` with truncation, and the tokenizer
    truncates from the right, so an over-long target loses exactly the EOS it
    was supposed to teach.  Every such sample is a worked example of "generate
    this far and then keep going", which is precisely the degenerate behaviour
    that makes reconstruction runs hit the token ceiling and emit invalid JSON.
    At max_target_len=2048 this affected 4,609 of 66,261 samples (6.96%);
    at 4096 it would be 290, and dropping them instead removes the failure mode
    entirely at a *lower* cost than truncating (the longest sequences are also
    the most expensive ones to train on).

    Runs before sharding and identically on every rank, for two reasons:

    1.  Correctness under DDP.  Skipping a row inside the training loop makes
        the number of backward() calls rank-dependent, and every backward()
        triggers a gradient all-reduce.  A rank that skips one sample more than
        its peers leaves them blocked in a collective that never completes,
        which surfaces ten minutes later as an NCCL watchdog timeout rather
        than as an error at the point of cause.
    2.  Speed.  process_json_file() re-read and re-parsed each target once per
        epoch; caching turns num_epochs passes over the dataset into one.

    process_json_file() is deterministic, so every rank derives the same set of
    surviving rows without any communication.
    """
    targets = []
    keep = []
    n_unusable = 0
    n_too_long = 0
    for _, row in tqdm(
            embeddings.iterrows(),
            total=len(embeddings),
            desc="Preparing targets",
            disable=not accelerator.is_local_main_process,
    ):
        try:
            target_json = process_json_file(row["json-path"], minify=True)
        except Exception:
            target_json = None

        usable = bool(target_json) and len(target_json) >= 5
        if not usable:
            n_unusable += 1
        elif not fits_token_budget(target_json, tokenizer, max_target_len):
            usable = False
            n_too_long += 1

        keep.append(usable)
        targets.append(target_json if usable else "")

    prepared = embeddings.copy()
    prepared["target-json"] = targets
    prepared = prepared[keep]

    if accelerator.is_local_main_process:
        if n_unusable:
            print(f"Dropped {n_unusable} samples without a usable JSON target.")
        if n_too_long:
            print(
                f"Dropped {n_too_long} samples longer than "
                f"{max_target_len} tokens (truncating them would strip the EOS)."
            )
        print(f"Kept {len(prepared)} of {len(embeddings)} samples.")

    if fast:
        # Exactly the call the training loop used to make per sample and per
        # epoch, so the ids are the ones it would have produced. Stored as
        # int32 because the Python lists the tokenizer returns cost roughly
        # seven times the memory, on every rank.
        prepared["target-ids"] = [
            np.asarray(
                tokenizer(
                    target_json + tokenizer.eos_token,
                    truncation=True,
                    max_length=max_target_len,
                    padding=False,
                ).input_ids,
                dtype=np.int32,
            )
            for target_json in tqdm(
                prepared["target-json"],
                total=len(prepared),
                desc="Tokenizing targets",
                disable=not accelerator.is_local_main_process,
            )
        ]

    return prepared


def shard_for_process(embeddings, accelerator):
    """
    Strided shard of `embeddings` for the current rank, with *equal* length on
    every rank.

    The length is first truncated down to a multiple of num_processes, so all
    ranks run exactly the same number of forward/backward steps and their
    gradient all-reduces stay in lockstep.  At most num_processes - 1 samples
    are dropped per epoch.

    Without this, an uneven split deadlocks: with 40543 training samples over
    8 ranks, ranks 0-6 get 5068 rows and rank 7 gets 5067, so rank 7 reaches
    the epoch-end loss reduction while the others are still in the gradient
    all-reduce of their extra sample.

    Equal shards also make accelerator.reduce(..., "mean") an exact mean over
    samples instead of an unweighted mean of differently sized per-rank means.
    """
    world_size = accelerator.num_processes
    usable = (len(embeddings) // world_size) * world_size
    return embeddings.iloc[:usable].iloc[accelerator.process_index::world_size]


def encode_target_ids(row, tokenizer, cfg, device):
    """
    [1, seq_len] target token ids for one row, with EOS appended so the model
    learns when to stop.

    With cfg.fast the ids come from the column build_target_column() filled in
    once; otherwise they are produced here, per sample and per epoch, by the
    call the training loop has always made. Both paths yield the same ids --
    see FAST_PATH_NOTES (3).
    """
    if cfg.fast:
        return torch.from_numpy(row["target-ids"]).long().unsqueeze(0).to(device)

    target_enc = tokenizer(
        row["target-json"] + tokenizer.eos_token,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_target_len,
        padding=False,
    ).to(device)
    return target_enc.input_ids  # [1, seq_len]


# =============================================================================
# Main Training Pipeline
# =============================================================================
def main():
    args = parse_args()
    cfg = AdapterConfig()
    cfg.fast = args.fast

    # Tag the output dir with the base LM so checkpoints trained against
    # different models never collide, e.g. "..._full" -> "..._full_qwen2.5-1.5b".
    model_slug = model_output_slug(cfg.lm_model_id)
    if not args.output_dir.endswith(model_slug):
        args.output_dir = f"{args.output_dir}_{model_slug}"

    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum_steps)
    device = accelerator.device
    if accelerator.is_local_main_process:
        print(f"Using device: {device} (Distributed across {accelerator.num_processes} processes)")
        print(f"Embeddings: {args.embeddings_dir} | Output: {args.output_dir}")
        if cfg.fast:
            print(
                "Fast path: on (cached target ids, deferred loss readback, "
                "all-reduce only on optimizer steps)"
            )
        else:
            print("Fast path: off (unchanged reference path)")

    # -------------------------------------------------------------------------
    # 1. Load pre-computed embeddings
    # -------------------------------------------------------------------------
    try:
        all_embeddings = load_embeddings(embeddings_dir=args.embeddings_dir)
    except Exception as e:
        print(f"Critical error: {e}")
        return

    # -------------------------------------------------------------------------
    # 2. Load small LM + tokenizer
    # -------------------------------------------------------------------------
    # Loaded before the targets are prepared because build_target_column() needs
    # the tokenizer to decide which targets fit the token budget.
    #
    # Load model with bfloat16 if GPU is available to save memory and match native training precision
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    with accelerator.main_process_first():
        if accelerator.is_local_main_process:
            import shutil
            from pathlib import Path
            model_id = cfg.lm_model_id
            try:
                print(f"Verifying cache for {model_id}...")
                AutoTokenizer.from_pretrained(model_id)
                AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=model_dtype)
                print(f"Cache for {model_id} is valid.")
            except Exception as e:
                print(f"Failed to load {model_id} due to possible cache corruption: {e}")
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
                if cache_dir.exists():
                    print(f"Deleting corrupted cache directory: {cache_dir}")
                    shutil.rmtree(cache_dir, ignore_errors=True)
                print("Retrying clean download...")
                AutoTokenizer.from_pretrained(model_id, force_download=True)
                AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=model_dtype, force_download=True)
                print(f"{model_id} downloaded successfully after cache clean!")

        # local_files_only=True: the main process has just guaranteed the cache
        # is fully populated above. Without this flag, every rank still calls
        # into huggingface_hub's remote-resolution/locking path, and concurrent
        # lock attempts on a networked filesystem (NFS/Lustre home dirs) can
        # spuriously fail with "does not appear to have a file named ...".
        # Forcing local-only reads here skips that path entirely and removes
        # the race between ranks.
        tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model_id, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lm_model = AutoModelForCausalLM.from_pretrained(
            cfg.lm_model_id,
            torch_dtype=model_dtype,
            local_files_only=True,
        )

    # -------------------------------------------------------------------------
    # 3. Prepare targets, split, and shard
    # -------------------------------------------------------------------------
    # Resolve and cache the target JSON up front, so the per-rank row counts
    # below cannot diverge later inside the training loop.
    all_embeddings = build_target_column(
        all_embeddings, accelerator, tokenizer, cfg.max_target_len, fast=cfg.fast
    )

    # Without this the run would "succeed" doing nothing: every loop below is
    # driven by the DataFrame, so an empty one trains zero steps and still
    # writes a checkpoint of the untouched initial weights.
    if len(all_embeddings) == 0:
        print(
            f"No usable embedding/JSON pairs in '{args.embeddings_dir}'. "
            "Nothing to train on."
        )
        return

    train_size = int(len(all_embeddings) * 0.8)
    train_embeddings = all_embeddings.iloc[:train_size]
    val_embeddings = all_embeddings.iloc[train_size:]
    if accelerator.is_local_main_process:
        print(
            f"Using {len(train_embeddings)} samples for training "
            f"and {len(val_embeddings)} for validation."
        )

    # Shard the datasets manually for distributed processes, equal length per rank
    train_embeddings = shard_for_process(train_embeddings, accelerator)
    val_embeddings = shard_for_process(val_embeddings, accelerator)
    if accelerator.is_local_main_process:
        print(
            f"Per-rank shard: {len(train_embeddings)} train / "
            f"{len(val_embeddings)} val across {accelerator.num_processes} processes."
        )

    # -------------------------------------------------------------------------
    # 4. Apply LoRA
    # -------------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
    )
    lm_model = get_peft_model(lm_model, lora_config)
    if accelerator.is_local_main_process:
        lm_model.print_trainable_parameters()

    # -------------------------------------------------------------------------
    # 5. Create projector
    # -------------------------------------------------------------------------
    # Ensure all components use the model's actual loaded precision
    model_dtype = next(lm_model.parameters()).dtype
    if accelerator.is_local_main_process:
        print(f"Model loaded with dtype: {model_dtype}")

    lm_hidden_dim = lm_model.config.hidden_size
    # A (4096,) embedding is expanded into num_prefix_tokens identical copies by
    # the projector, so it needs the positional table to tell them apart again.
    # An (N, 4096) one carries N distinct tokens already and trains without it,
    # as in 926391f. Deriving the flag from the shape reproduces the 2k sweep
    # exactly: its two (4096,) checkpoints carry prefix_pos.weight, the
    # (N, 4096) ones do not.
    use_prefix_pos = all_embeddings["embedding"].iloc[0].ndim == 1
    projector = EmbeddingProjector(
        cfg.qwen_embed_dim, lm_hidden_dim, cfg.num_prefix_tokens,
        use_prefix_pos=use_prefix_pos,
    ).to(model_dtype)
    if accelerator.is_local_main_process:
        print(
            f"Embedding shape: {all_embeddings['embedding'].iloc[0].shape} | "
            f"prefix_pos table: {'on' if use_prefix_pos else 'off'}"
        )

    proj_params = sum(p.numel() for p in projector.parameters())
    if accelerator.is_local_main_process:
        print(f"Projector parameters: {proj_params:,}")

    # -------------------------------------------------------------------------
    # 6. Optimizer with separate learning rates
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW([
        {"params": projector.parameters(), "lr": cfg.lr_projector},
        {
            "params": [p for p in lm_model.parameters() if p.requires_grad],
            "lr": cfg.lr_lora,
        },
    ])

    # Prepare model, projector, and optimizer with accelerator
    lm_model, projector, optimizer = accelerator.prepare(lm_model, projector, optimizer)

    # -------------------------------------------------------------------------
    # 7. Training loop
    # -------------------------------------------------------------------------
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_projector_state = None
    best_lora_params = None

    if accelerator.is_local_main_process:
        print("\n--- Starting LLaVA-Style Adapter Training ---\n")
    avg_loss = -1.0  # sentinel so the first progress-bar print is readable

    for epoch in range(cfg.num_epochs):
        lm_model.train()
        projector.train()

        epoch_loss = 0.0
        # See FAST_PATH_NOTES (2): the running loss lives on the GPU so the
        # loop never has to synchronise with the host to read it.
        epoch_loss_gpu = (
            torch.zeros((), device=device, dtype=torch.float64) if cfg.fast else None
        )
        valid_samples = 0
        # Same on every rank (shard_for_process equalises the shards), so the
        # gradient-sync boundaries below are picked identically without
        # communicating.
        n_train = len(train_embeddings)
        optimizer.zero_grad()

        # --- TRAINING LOOP ---
        for _, row in tqdm(
                train_embeddings.iterrows(),
                total=len(train_embeddings),
                disable=not accelerator.is_local_main_process,
                desc=(
                        f"Epoch {epoch + 1}/{cfg.num_epochs} "
                        f"| Loss: {avg_loss:.6f}"
                ),
        ):
            # See FAST_PATH_NOTES (1). The trailing group at the end of the
            # epoch is a boundary too: the flush below steps on whatever has
            # accumulated, so those gradients must have been all-reduced.
            # DDP decides this in forward(), so the context has to cover the
            # forward pass as well, not just the backward.
            is_sync_step = (
                (valid_samples + 1) % cfg.grad_accum_steps == 0
                or (valid_samples + 1) == n_train
            )

            with ExitStack() as no_sync_stack:
                if cfg.fast and not is_sync_step:
                    no_sync_stack.enter_context(accelerator.no_sync(lm_model))
                    no_sync_stack.enter_context(accelerator.no_sync(projector))

                # Target: cleaned, minified JSON, resolved once by build_target_column().
                # Never skip a row here — that would desynchronise the ranks.
                input_ids = encode_target_ids(row, tokenizer, cfg, device)

                # Project Qwen embedding -> prefix tokens
                emb = torch.tensor(
                    row["embedding"], dtype=model_dtype
                ).unsqueeze(0).to(device)
                emb = F.normalize(emb, p=2, dim=-1)
                prefix_embeds = projector(emb)  # [1, N, D]

                # Target token embeddings from the LM's own embedding table
                target_embeds = accelerator.unwrap_model(lm_model).get_input_embeddings()(input_ids)

                # Concatenate: [prefix | target]
                inputs_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)

                # Attention mask: every position is attended to
                attn_mask = torch.ones(
                    1, inputs_embeds.size(1), device=device, dtype=torch.long
                )

                # Labels: -100 for prefix (no loss), real IDs for target tokens.
                # The prefix length follows the projector output, which depends on
                # the embedding shape, so it must not be hardcoded to num_prefix_tokens.
                prefix_labels = torch.full(
                    (1, prefix_embeds.size(1)), -100, device=device, dtype=torch.long
                )
                labels = torch.cat([prefix_labels, input_ids], dim=1)

                # Forward pass through the LoRA-finetuned LM
                outputs = lm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    labels=labels,
                )

                loss = outputs.loss / cfg.grad_accum_steps
                accelerator.backward(loss)

            valid_samples += 1
            if cfg.fast:
                epoch_loss_gpu += outputs.loss.detach().double()
            else:
                epoch_loss += outputs.loss.item()

            # Gradient accumulation step
            if valid_samples % cfg.grad_accum_steps == 0:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(projector.parameters())
                        + [p for p in lm_model.parameters() if p.requires_grad],
                        max_norm=1.0,
                    )
                optimizer.step()
                optimizer.zero_grad()

        # Flush remaining accumulated gradients
        if valid_samples % cfg.grad_accum_steps != 0:
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    list(projector.parameters())
                    + [p for p in lm_model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
            optimizer.step()
            optimizer.zero_grad()

        if cfg.fast:
            # The single host synchronisation of the epoch.
            epoch_loss = epoch_loss_gpu.item()

        avg_loss = epoch_loss / max(1, valid_samples)
        # Reduce loss across all processes
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss_tensor = accelerator.reduce(avg_loss_tensor, reduction="mean")
        avg_loss = avg_loss_tensor.item()
        if accelerator.is_local_main_process:
            history["train_loss"].append(avg_loss)

        # --- VALIDATION LOOP ---
        lm_model.eval()
        projector.eval()
        val_loss_total = 0.0
        val_loss_gpu = (
            torch.zeros((), device=device, dtype=torch.float64) if cfg.fast else None
        )
        val_valid = 0

        with torch.no_grad():
            for _, row in tqdm(
                    val_embeddings.iterrows(),
                    total=len(val_embeddings),
                    desc="Validating",
                    leave=False,
                    disable=not accelerator.is_local_main_process,
            ):
                input_ids = encode_target_ids(row, tokenizer, cfg, device)

                emb = torch.tensor(
                    row["embedding"], dtype=model_dtype
                ).unsqueeze(0).to(device)
                emb = F.normalize(emb, p=2, dim=-1)
                prefix_embeds = projector(emb)

                target_embeds = accelerator.unwrap_model(lm_model).get_input_embeddings()(input_ids)
                inputs_embeds = torch.cat(
                    [prefix_embeds, target_embeds], dim=1
                )
                attn_mask = torch.ones(
                    1, inputs_embeds.size(1), device=device, dtype=torch.long
                )

                prefix_labels = torch.full(
                    (1, prefix_embeds.size(1)),
                    -100,
                    device=device,
                    dtype=torch.long,
                )
                labels = torch.cat([prefix_labels, input_ids], dim=1)

                outputs = lm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    labels=labels,
                )
                if cfg.fast:
                    val_loss_gpu += outputs.loss.detach().double()
                else:
                    val_loss_total += outputs.loss.item()
                val_valid += 1

        if cfg.fast:
            val_loss_total = val_loss_gpu.item()

        avg_val_loss = val_loss_total / max(1, val_valid)
        # Reduce loss across all processes
        avg_val_loss_tensor = torch.tensor(avg_val_loss, device=device)
        avg_val_loss_tensor = accelerator.reduce(avg_val_loss_tensor, reduction="mean")
        avg_val_loss = avg_val_loss_tensor.item()

        if accelerator.is_local_main_process:
            history["val_loss"].append(avg_val_loss)
            print(
                f" -> Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )

        # --- EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_projector_state = {
                k: v.cpu().clone()
                for k, v in accelerator.unwrap_model(projector).state_dict().items()
            }
            best_lora_params = {
                name: param.data.cpu().clone()
                for name, param in accelerator.unwrap_model(lm_model).named_parameters()
                if param.requires_grad
            }
            patience_counter = 0
            if accelerator.is_local_main_process:
                print("    [*] New best validation loss! Weights cached in memory.")

                # Automatically save the best model directly to disk
                os.makedirs(args.output_dir, exist_ok=True)
                best_projector_path = os.path.join(args.output_dir, "best_projector.pth")
                best_lora_path = os.path.join(args.output_dir, "best_lora_adapter")

                unwrapped_projector = accelerator.unwrap_model(projector)
                unwrapped_lm = accelerator.unwrap_model(lm_model)

                # Save to disk (overwriting the previous best)
                torch.save(unwrapped_projector.state_dict(), best_projector_path)
                unwrapped_lm.save_pretrained(best_lora_path)
                print(f"    [*] Saved best model checkpoint to disk: {best_projector_path} and {best_lora_path}")
        else:
            patience_counter += 1
            if accelerator.is_local_main_process:
                print(
                    f"    [!] No improvement. "
                    f"Patience: {patience_counter}/{cfg.patience}"
                )

        if patience_counter >= cfg.patience:
            if accelerator.is_local_main_process:
                print("\n--- Early Stopping Triggered! ---")
            break

    # =========================================================================
    # End of Training: Restore Best, Save, Plot
    # =========================================================================
    if accelerator.is_local_main_process:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        unwrapped_lm = accelerator.unwrap_model(lm_model)
        unwrapped_projector = accelerator.unwrap_model(projector)

        # Restore best projector weights
        if best_projector_state:
            unwrapped_projector.load_state_dict(best_projector_state)
            print("Restored best projector weights.")

        # Restore best LoRA weights
        if best_lora_params:
            for name, param in unwrapped_lm.named_parameters():
                if name in best_lora_params:
                    param.data.copy_(best_lora_params[name].to(device))
            print("Restored best LoRA weights.")

        # Save projector
        os.makedirs(args.output_dir, exist_ok=True)
        projector_path = os.path.join(args.output_dir, f"projector_{timestamp}.pth")
        torch.save(unwrapped_projector.state_dict(), projector_path)
        print(f"Saved projector to: {projector_path}")

        # Save LoRA adapter (PEFT handles extracting only the LoRA delta weights)
        lora_path = os.path.join(args.output_dir, f"lora_adapter_{timestamp}")
        unwrapped_lm.save_pretrained(lora_path)
        print(f"Saved LoRA adapter to: {lora_path}")

        # Plot training history
        os.makedirs("adapter_training_history", exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_loss"], label="Train Loss", marker="o")
        plt.plot(history["val_loss"], label="Val Loss", marker="x")
        plt.title("LLaVA-Style Adapter: Train vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plot_path = f"adapter_training_history/training_history_{timestamp}.png"
        plt.savefig(plot_path)
        print(f"Saved training history plot to: {plot_path}")

        # =========================================================================
        # Inference Demo
        # =========================================================================
        print("\n--- Inference Demo (Last Validation Sample) ---")
        unwrapped_lm.eval()
        unwrapped_projector.eval()

        with torch.no_grad():
            last_row = val_embeddings.iloc[-1]

            emb = torch.tensor(
                last_row["embedding"], dtype=model_dtype
            ).unsqueeze(0).to(device)
            emb = F.normalize(emb, p=2, dim=-1)
            prefix_embeds = unwrapped_projector(emb)

            print(f"Source: {last_row['json-path']}")

            generated_ids = generate_from_prefix(
                unwrapped_lm, tokenizer, prefix_embeds,
                max_new_tokens=cfg.max_target_len,
            )
            generated_text = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )

            print(f"\nGenerated JSON ({len(generated_ids[0])} tokens):")
            # Pretty-print if it is valid JSON
            try:
                parsed = json.loads(generated_text)
                print(json.dumps(parsed, indent=2))
            except json.JSONDecodeError:
                print(generated_text)
                print("\n[!] Warning: Generated text is not valid JSON.")

    # Wait for all processes to complete
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()