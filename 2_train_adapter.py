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
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
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

    # RECOMMENDED: Qwen3.5-4B for complex structural layouts (or Qwen3.5-2B)
    lm_model_id = "Qwen/Qwen3.5-4B"

    # LoRA fine-tuning parameters updated for newer Qwen structural compliance
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Training hyperparameters
    lr_projector = 1e-4       # Learning rate for the MLP projector
    lr_lora = 2e-5            # Learning rate for LoRA parameters
    num_epochs = 100
    patience = 3              # Early stopping patience
    max_target_len = 2048     # Max tokens for the target JSON string
    grad_accum_steps = 4      # Gradient accumulation steps


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

    The 4096-dim embedding is reshaped into `num_prefix_tokens` chunks
    (e.g. 128 × 32), so each prefix token receives a *different* slice
    of the original representation.  A learned positional embedding is
    added per token to further differentiate them.

    Example:
        projector = EmbeddingProjector(qwen_dim=4096, lm_hidden_dim=896, num_prefix_tokens=128)

        qwen_embedding = torch.randn(1, 4096)        # one pre-computed Qwen3-VL embedding
        prefix = projector(qwen_embedding)            # -> [1, 128, 896]
        # These 128 virtual tokens are prepended to the LM's input sequence.
        # The LM then generates the cleaned JSON conditioned on them.

    Input:  [B, 4096]  (Qwen3-VL embedding)
    Output: [B, num_prefix_tokens, lm_hidden_dim]  (prefix for the LM)
    """

    def __init__(self, qwen_dim, lm_hidden_dim, num_prefix_tokens):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.lm_hidden_dim = lm_hidden_dim

        assert qwen_dim % num_prefix_tokens == 0, (
            f"qwen_dim ({qwen_dim}) must be divisible by "
            f"num_prefix_tokens ({num_prefix_tokens})"
        )
        chunk_dim = qwen_dim // num_prefix_tokens  # e.g. 4096 / 128 = 32

        self.projector = nn.Sequential(
            nn.Linear(chunk_dim, lm_hidden_dim * 2),
            nn.GELU(),
            nn.Linear(lm_hidden_dim * 2, lm_hidden_dim),
        )
        self.prefix_pos = nn.Embedding(num_prefix_tokens, lm_hidden_dim)

    def forward(self, qwen_embedding):
        """
        Args:
            qwen_embedding: [B, qwen_dim] pre-computed Qwen3-VL embeddings.
        Returns:
            [B, num_prefix_tokens, lm_hidden_dim] virtual prefix token embeddings.
        """
        B = qwen_embedding.size(0)
        if qwen_embedding.dim() == 2:
            # Reshape [B, 4096] -> [B, 128, 32]: each token gets a unique slice
            qwen_embedding = qwen_embedding.view(B, self.num_prefix_tokens, -1)

        projected = self.projector(qwen_embedding)  # [B, 128, lm_hidden_dim]
        positions = torch.arange(self.num_prefix_tokens, device=projected.device)
        return projected + self.prefix_pos(positions)


# =============================================================================
# Autoregressive Generation with KV-Cache
# =============================================================================
def generate_from_prefix(model, tokenizer, prefix_embeds, max_new_tokens=512):
    """
    Autoregressively generate tokens conditioned on prefix embeddings.
    Uses KV-caching for efficient O(n) generation.

    Args:
        model:          The (LoRA-wrapped) causal LM.
        tokenizer:      The LM's tokenizer (for EOS detection).
        prefix_embeds:  [B, num_prefix, hidden_dim] projected prefix embeddings.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        generated_ids: [B, generated_len] tensor of generated token IDs.
    """
    device = prefix_embeds.device

    # First forward pass: process all prefix embeddings, populate KV cache
    outputs = model(inputs_embeds=prefix_embeds, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    generated = [next_token]

    # Subsequent passes: single token at a time with cached KV
    for _ in range(max_new_tokens - 1):
        if (next_token == tokenizer.eos_token_id).all():
            break

        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated.append(next_token)

    return torch.cat(generated, dim=1)


# =============================================================================
# Main Training Pipeline
# =============================================================================
def main():
    cfg = AdapterConfig()
    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum_steps)
    device = accelerator.device
    if accelerator.is_local_main_process:
        print(f"Using device: {device} (Distributed across {accelerator.num_processes} processes)")

    # -------------------------------------------------------------------------
    # 1. Load pre-computed embeddings
    # -------------------------------------------------------------------------
    try:
        all_embeddings = load_embeddings("embeddings")
    except Exception as e:
        print(f"Critical error: {e}")
        return

    train_size = int(len(all_embeddings) * 0.8)
    train_embeddings = all_embeddings.iloc[:train_size]
    val_embeddings = all_embeddings.iloc[train_size:]
    if accelerator.is_local_main_process:
        print(
            f"Using {len(train_embeddings)} samples for training "
            f"and {len(val_embeddings)} for validation."
        )

    # Shard the datasets manually for distributed processes
    train_embeddings = train_embeddings.iloc[accelerator.process_index::accelerator.num_processes]
    val_embeddings = val_embeddings.iloc[accelerator.process_index::accelerator.num_processes]

    # -------------------------------------------------------------------------
    # 2. Load small LM + tokenizer
    # -------------------------------------------------------------------------
    # Load model with bfloat16 if GPU is available to save memory and match native training precision
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if accelerator.is_local_main_process:
        print(f"Loading language model: {cfg.lm_model_id} ...")

    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lm_model = AutoModelForCausalLM.from_pretrained(
            cfg.lm_model_id,
            torch_dtype=model_dtype,
        )

    # -------------------------------------------------------------------------
    # 3. Apply LoRA
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
    # 4. Create projector
    # -------------------------------------------------------------------------
    # Ensure all components use the model's actual loaded precision
    model_dtype = next(lm_model.parameters()).dtype
    if accelerator.is_local_main_process:
        print(f"Model loaded with dtype: {model_dtype}")

    lm_hidden_dim = lm_model.config.hidden_size
    projector = EmbeddingProjector(
        cfg.qwen_embed_dim, lm_hidden_dim, cfg.num_prefix_tokens
    ).to(model_dtype)

    proj_params = sum(p.numel() for p in projector.parameters())
    if accelerator.is_local_main_process:
        print(f"Projector parameters: {proj_params:,}")

    # -------------------------------------------------------------------------
    # 5. Optimizer with separate learning rates
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
    # 6. Training loop
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
        valid_samples = 0
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
            # Target: cleaned, minified JSON from the Rico cleaning pipeline
            try:
                target_json = process_json_file(row["json-path"], minify=True)
            except Exception:
                continue
            if not target_json or len(target_json) < 5:
                continue

            # Tokenize target (append EOS so the model learns when to stop)
            target_enc = tokenizer(
                target_json + tokenizer.eos_token,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.max_target_len,
                padding=False,
            ).to(device)
            input_ids = target_enc.input_ids  # [1, seq_len]

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

            # Labels: -100 for prefix (no loss), real IDs for target tokens
            prefix_labels = torch.full(
                (1, cfg.num_prefix_tokens), -100, device=device, dtype=torch.long
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
        val_valid = 0

        with torch.no_grad():
            for _, row in tqdm(
                val_embeddings.iterrows(),
                total=len(val_embeddings),
                desc="Validating",
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):
                try:
                    target_json = process_json_file(
                        row["json-path"], minify=True
                    )
                except Exception:
                    continue
                if not target_json or len(target_json) < 5:
                    continue

                target_enc = tokenizer(
                    target_json + tokenizer.eos_token,
                    return_tensors="pt",
                    truncation=True,
                    max_length=cfg.max_target_len,
                    padding=False,
                ).to(device)
                input_ids = target_enc.input_ids

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
                    (1, cfg.num_prefix_tokens),
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
                val_loss_total += outputs.loss.item()
                val_valid += 1

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
                os.makedirs("adapter_model_out", exist_ok=True)
                best_projector_path = "adapter_model_out/best_projector.pth"
                best_lora_path = "adapter_model_out/best_lora_adapter"
                
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
        os.makedirs("adapter_model_out", exist_ok=True)
        projector_path = f"adapter_model_out/projector_{timestamp}.pth"
        torch.save(unwrapped_projector.state_dict(), projector_path)
        print(f"Saved projector to: {projector_path}")

        # Save LoRA adapter (PEFT handles extracting only the LoRA delta weights)
        lora_path = f"adapter_model_out/lora_adapter_{timestamp}"
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