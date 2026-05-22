# Trains the adapter to adapt embedding space to reconstruction space.

import os
import json
import orjson
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from utils import (Config, LayoutDecoder, StructuralDecoder, parse_rico_json,
                   decode_layout_tokens, decode_json_tokens, load_embeddings_from_csv, CUDA_VISIBLE_DEVICE)

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICE

# =============================================================================
# Configuration Initialization
# =============================================================================
cfg = Config()


# =============================================================================
# Main Pipeline execution logic
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        all_embeddings = load_embeddings_from_csv("../rico-data/embeddings/embeddings_qwen.csv")
    except Exception as e:
        print(f"Critical error: {e}")
        return

    # Split dataset: 80% train, 20% validation
    train_size = int(len(all_embeddings) * 0.8)
    train_embeddings = all_embeddings.iloc[:train_size]
    val_embeddings = all_embeddings.iloc[train_size:]
    print(f"Using {len(train_embeddings)} samples for training and {len(val_embeddings)} for validation.")

    layout_decoder = LayoutDecoder(cfg).to(device)
    structural_decoder = StructuralDecoder(cfg).to(device)

    opt_layout = torch.optim.Adam(layout_decoder.parameters(), lr=1e-4)
    opt_struct = torch.optim.Adam(structural_decoder.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)

    # Tracking History & Early Stopping
    history = {'train_layout': [], 'train_struct': [], 'val_layout': [], 'val_struct': []}
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    best_layout_state = None
    best_struct_state = None

    print("\n--- Starting Training Phase on CSV Data ---")

    num_epochs = 50
    # Set to -1.0 so the very first print shows exactly -1.000000
    avg_layout_loss, avg_struct_loss = -1.0, -1.0

    for epoch in range(num_epochs):
        layout_decoder.train()
        structural_decoder.train()

        layout_loss_total, struct_loss_total = 0, 0
        valid_samples = 0

        # --- TRAINING LOOP ---
        for i, row in tqdm(train_embeddings.iterrows(), total=len(train_embeddings),
                           desc=f"Epoch {epoch + 1}/{num_epochs} | Train Layout Loss: {avg_layout_loss:.6f} | Train Structure Loss: {avg_struct_loss:.6f}"):

            tgt_layout, tgt_json = parse_rico_json(row["json-path"], cfg)
            if tgt_layout is None or tgt_json is None or len(tgt_layout) <= 1 or len(tgt_json) <= 1:
                continue

            tgt_layout = torch.tensor(tgt_layout, dtype=torch.long).unsqueeze(0).to(device)
            tgt_json = torch.tensor(tgt_json, dtype=torch.long).unsqueeze(0).to(device)

            latent_vec = torch.tensor(row["embedding"], dtype=torch.float32).unsqueeze(0).to(
                device) if not torch.is_tensor(row["embedding"]) else row["embedding"].unsqueeze(0).to(device)
            latent_vec = F.normalize(latent_vec, p=2, dim=-1)
            valid_samples += 1

            opt_layout.zero_grad()
            layout_mask = nn.Transformer.generate_square_subsequent_mask(tgt_layout.size(1)).to(device)
            layout_logits = layout_decoder(latent_vec, tgt_layout, tgt_mask=layout_mask)
            loss_layout = criterion(layout_logits[:, :-1, :].reshape(-1, cfg.layout_vocab_size),
                                    tgt_layout[:, 1:].reshape(-1))
            loss_layout.backward()
            torch.nn.utils.clip_grad_norm_(layout_decoder.parameters(), max_norm=1.0)
            opt_layout.step()
            layout_loss_total += loss_layout.item()

            opt_struct.zero_grad()
            json_mask = nn.Transformer.generate_square_subsequent_mask(tgt_json.size(1)).to(device)
            json_logits = structural_decoder(latent_vec, tgt_json, tgt_mask=json_mask)
            loss_struct = criterion(json_logits[:, :-1, :].reshape(-1, cfg.json_vocab_size),
                                    tgt_json[:, 1:].reshape(-1))
            loss_struct.backward()
            torch.nn.utils.clip_grad_norm_(structural_decoder.parameters(), max_norm=1.0)
            opt_struct.step()
            struct_loss_total += loss_struct.item()

        avg_layout_loss = layout_loss_total / max(1, valid_samples)
        avg_struct_loss = struct_loss_total / max(1, valid_samples)
        history['train_layout'].append(avg_layout_loss)
        history['train_struct'].append(avg_struct_loss)

        # --- VALIDATION LOOP ---
        layout_decoder.eval()
        structural_decoder.eval()
        val_layout_loss_total, val_struct_loss_total = 0, 0
        val_valid_samples = 0

        with torch.no_grad():
            for i, row in tqdm(val_embeddings.iterrows(), total=len(val_embeddings), desc="Validating", leave=False):
                tgt_layout, tgt_json = parse_rico_json(row["json-path"], cfg)
                if tgt_layout is None or tgt_json is None or len(tgt_layout) <= 1 or len(tgt_json) <= 1:
                    continue

                tgt_layout = torch.tensor(tgt_layout, dtype=torch.long).unsqueeze(0).to(device)
                tgt_json = torch.tensor(tgt_json, dtype=torch.long).unsqueeze(0).to(device)

                latent_vec = torch.tensor(row["embedding"], dtype=torch.float32).unsqueeze(0).to(
                    device) if not torch.is_tensor(row["embedding"]) else row["embedding"].unsqueeze(0).to(device)
                latent_vec = F.normalize(latent_vec, p=2, dim=-1)
                val_valid_samples += 1

                layout_mask = nn.Transformer.generate_square_subsequent_mask(tgt_layout.size(1)).to(device)
                layout_logits = layout_decoder(latent_vec, tgt_layout, tgt_mask=layout_mask)
                loss_layout = criterion(layout_logits[:, :-1, :].reshape(-1, cfg.layout_vocab_size),
                                        tgt_layout[:, 1:].reshape(-1))
                val_layout_loss_total += loss_layout.item()

                json_mask = nn.Transformer.generate_square_subsequent_mask(tgt_json.size(1)).to(device)
                json_logits = structural_decoder(latent_vec, tgt_json, tgt_mask=json_mask)
                loss_struct = criterion(json_logits[:, :-1, :].reshape(-1, cfg.json_vocab_size),
                                        tgt_json[:, 1:].reshape(-1))
                val_struct_loss_total += loss_struct.item()

        avg_val_layout = val_layout_loss_total / max(1, val_valid_samples)
        avg_val_struct = val_struct_loss_total / max(1, val_valid_samples)
        history['val_layout'].append(avg_val_layout)
        history['val_struct'].append(avg_val_struct)

        combined_val_loss = avg_val_layout + avg_val_struct
        print(f" -> Validation Layout Loss: {avg_val_layout:.6f}, Validation Structure Loss: {avg_val_struct:.6f}")

        # --- EARLY STOPPING LOGIC ---
        if combined_val_loss < best_val_loss:
            best_val_loss = combined_val_loss
            # Save a copy of the best weights to RAM
            best_layout_state = {k: v.cpu().clone() for k, v in layout_decoder.state_dict().items()}
            best_struct_state = {k: v.cpu().clone() for k, v in structural_decoder.state_dict().items()}
            patience_counter = 0
            print(f"    [*] New best validation loss! Models cached in memory.")
        else:
            patience_counter += 1
            print(f"    [!] Validation did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("\n--- Early Stopping Triggered! ---")
            break

    # =============================================================================
    # End of Training: Plotting & Saving
    # =============================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load the best weights back into the models
    if best_layout_state and best_struct_state:
        layout_decoder.load_state_dict(best_layout_state)
        structural_decoder.load_state_dict(best_struct_state)
        print("Restored best model weights from memory.")

    layout_save_path = f"layout_decoder_best_{timestamp}.pth"
    struct_save_path = f"structural_decoder_best_{timestamp}.pth"

    torch.save(layout_decoder.state_dict(), layout_save_path)
    torch.save(structural_decoder.state_dict(), struct_save_path)
    print(f"\nSaved models as:\n - {layout_save_path}\n - {struct_save_path}")

    # Plot History
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_layout'], label='Train Layout Loss', marker='o')
    plt.plot(history['val_layout'], label='Val Layout Loss', marker='x')
    plt.title('Layout Decoder: Train vs Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_struct'], label='Train Structure Loss', marker='o')
    plt.plot(history['val_struct'], label='Val Structure Loss', marker='x')
    plt.title('Structural Decoder: Train vs Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plot_filename = f"training_history_{timestamp}.png"
    plt.savefig(plot_filename)
    print(f"Saved training history plot to '{plot_filename}'")

    print("\n--- Starting Phase 4: Inference Pipeline (Using Last Loaded Validation Embedding) ---")
    layout_decoder.eval()
    structural_decoder.eval()

    with torch.no_grad():
        last_row = val_embeddings.iloc[-1]

        if not torch.is_tensor(last_row["embedding"]):
            vec_input = torch.tensor(last_row["embedding"], dtype=torch.float32).unsqueeze(0).to(device)
        else:
            vec_input = last_row["embedding"].unsqueeze(0).to(device)

        vec_input = F.normalize(vec_input, p=2, dim=-1)

        print(f"Generating for: {last_row['json-path']}")

        gen_layout = layout_decoder.generate(vec_input, max_length=20)
        gen_json = structural_decoder.generate(vec_input, max_length=30)

        layout_tokens_list = gen_layout[0].tolist()
        json_tokens_list = gen_json[0].tolist()

        print("\nDecoded Layout:")
        decoded_layout = decode_layout_tokens(layout_tokens_list, cfg)
        print(json.dumps(decoded_layout, indent=2))

        print("\nDecoded JSON:")
        decoded_json = decode_json_tokens(json_tokens_list, cfg)
        print(decoded_json)


if __name__ == "__main__":
    main()