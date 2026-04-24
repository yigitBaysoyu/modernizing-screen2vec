import os
import json
import orjson
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Configuration & Hyperparameters
# =============================================================================
class Config:
    # Qwen dimensions
    qwen_embed_dim = 4096

    # Layout Decoder Specs
    layout_vocab_size = 1100  # 1000 bins for coordinates + ~100 classes/special tokens
    layout_hidden_dim = 512
    layout_num_layers = 4
    layout_num_heads = 8
    max_layout_seq_len = 500

    # Structural Decoder Specs (Compact Custom Transformer for JSON)
    json_vocab_size = 2000  # Simplified vocab for JSON structure
    json_hidden_dim = 512
    json_num_layers = 4
    json_num_heads = 8
    max_json_seq_len = 1000

    # Special Tokens
    pad_token_id = 0
    sos_token_id = 1
    eos_token_id = 2


cfg = Config()


# =============================================================================
# Phase 1: Data Prep & Latent Space Loading
# =============================================================================
def load_embeddings_from_csv(filepath):
    """Loads all pre-computed Qwen embeddings from the CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The required file '{filepath}' was not found. Execution aborted.")

    df = pd.read_csv(filepath, names=["json-path", "embedding"], header=None)

    # Convert string representation of list back to an actual Python list
    df["embedding"] = [orjson.loads(x) for x in df["embedding"]]

    df["json-path"] = df["json-path"].apply(lambda x: x.replace("/lustre/scratch/sgjolieb-screen2vec", ".."))

    print(f"Successfully loaded {len(df)} embeddings.")
    return df


def parse_rico_json(filepath, config):
    """Reads actual RICO JSON and converts layout and structure into token sequences."""

    # FIXME this needs to be adapted, a lot

    if not os.path.exists(filepath):
        return None, None

    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return None, None

    layout_tokens = [config.sos_token_id]
    json_tokens = [config.sos_token_id]

    def traverse(node):
        if not isinstance(node, dict):
            return

        # Layout extraction: Extract bounds if present
        if 'bounds' in node:
            b = node['bounds']
            # Bounding boxes in RICO are usually [x1, y1, x2, y2]
            # We clamp them between 0 and 999 to fit our 1000 bins, and offset by 3 (reserving 0,1,2)
            for val in b:
                bin_val = max(0, min(999, int(val))) + 3
                if len(layout_tokens) < config.max_layout_seq_len - 2:
                    layout_tokens.append(bin_val)

            # Simple hash for UI class ID mapping to 1003-1099 range
            ui_class = node.get('class', 'Unknown')
            class_id = (hash(ui_class) % 96) + 1003
            if len(layout_tokens) < config.max_layout_seq_len - 2:
                layout_tokens.append(class_id)

        # Structure extraction: Simple tokenization for JSON structure
        node_type = node.get('class', 'Unknown')
        token_hash = (hash(node_type) % (config.json_vocab_size - 3)) + 3
        if len(json_tokens) < config.max_json_seq_len - 2:
            json_tokens.append(token_hash)

        if 'children' in node:
            for child in node['children']:
                traverse(child)

    traverse(data)

    layout_tokens.append(config.eos_token_id)
    json_tokens.append(config.eos_token_id)

    # Convert to tensor and add batch dimension
    tgt_layout = torch.tensor(layout_tokens, dtype=torch.long).unsqueeze(0)
    tgt_json = torch.tensor(json_tokens, dtype=torch.long).unsqueeze(0)

    return tgt_layout, tgt_json


# =============================================================================
# Phase 2: Layout Decoder (Autoregressive Bounding Boxes)
# =============================================================================
class LayoutDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Project 4096d Qwen Vector -> Layout Hidden Dim (Conditioning)
        self.condition_proj = nn.Linear(config.qwen_embed_dim, config.layout_hidden_dim)

        self.embedding = nn.Embedding(config.layout_vocab_size, config.layout_hidden_dim)
        self.pos_encoder = nn.Embedding(config.max_layout_seq_len, config.layout_hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.layout_hidden_dim,
            nhead=config.layout_num_heads,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config.layout_num_layers)
        self.fc_out = nn.Linear(config.layout_hidden_dim, config.layout_vocab_size)

        self.config = config

    def forward(self, qwen_latent, tgt_seq, tgt_mask=None):
        """
        qwen_latent: (B, 4096)
        tgt_seq: (B, Seq_Len)
        """
        B, seq_len = tgt_seq.size()

        # Create Condition Prefix (B, 1, Hidden_Dim)
        condition = self.condition_proj(qwen_latent).unsqueeze(1)

        # Embed Target Sequence
        positions = torch.arange(0, seq_len, device=tgt_seq.device).unsqueeze(0).expand(B, seq_len)
        tgt_emb = self.embedding(tgt_seq) + self.pos_encoder(positions)

        # Transformer Decoder expects: tgt, memory
        # Here, memory is just our condition prefix.
        output = self.transformer_decoder(tgt_emb, condition, tgt_mask=tgt_mask)
        logits = self.fc_out(output)
        return logits

    def generate(self, qwen_latent, max_length=100):
        """Autoregressive generation for inference."""
        B = qwen_latent.size(0)
        device = qwen_latent.device

        # Start with SOS token
        current_seq = torch.full((B, 1), self.config.sos_token_id, dtype=torch.long, device=device)

        for _ in range(max_length):
            # Causal mask
            seq_len = current_seq.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

            logits = self(qwen_latent, current_seq, tgt_mask=tgt_mask)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            current_seq = torch.cat([current_seq, next_token], dim=1)

            # Simple stopping criteria
            if (next_token == self.config.eos_token_id).all():
                break

        return current_seq


# =============================================================================
# Phase 3: Structural Decoder (JSON Tree)
# =============================================================================
class StructuralDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.condition_proj = nn.Sequential(
            nn.Linear(config.qwen_embed_dim, config.json_hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.json_hidden_dim * 2, config.json_hidden_dim)
        )

        self.embedding = nn.Embedding(config.json_vocab_size, config.json_hidden_dim)
        self.pos_encoder = nn.Embedding(config.max_json_seq_len, config.json_hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.json_hidden_dim,
            nhead=config.json_num_heads,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config.json_num_layers)
        self.fc_out = nn.Linear(config.json_hidden_dim, config.json_vocab_size)
        self.config = config

    def forward(self, qwen_latent, tgt_seq, tgt_mask=None):
        B, seq_len = tgt_seq.size()
        condition = self.condition_proj(qwen_latent).unsqueeze(1)

        positions = torch.arange(0, seq_len, device=tgt_seq.device).unsqueeze(0).expand(B, seq_len)
        tgt_emb = self.embedding(tgt_seq) + self.pos_encoder(positions)

        output = self.transformer_decoder(tgt_emb, condition, tgt_mask=tgt_mask)
        logits = self.fc_out(output)
        return logits

    def generate(self, qwen_latent, max_length=200):
        B = qwen_latent.size(0)
        device = qwen_latent.device
        current_seq = torch.full((B, 1), self.config.sos_token_id, dtype=torch.long, device=device)

        for _ in range(max_length):
            seq_len = current_seq.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            logits = self(qwen_latent, current_seq, tgt_mask=tgt_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            current_seq = torch.cat([current_seq, next_token], dim=1)
            if (next_token == self.config.eos_token_id).all():
                break
        return current_seq


# =============================================================================
# Phase 4: Token Translation & Decoding Functions
# =============================================================================
def decode_layout_tokens(token_list, config):
    """Translates raw token IDs into human-readable Bounding Boxes and UI Classes."""
    decoded_boxes = []
    tokens = token_list[1:] if token_list and token_list[0] == config.sos_token_id else token_list

    for i in range(0, len(tokens), 5):
        chunk = tokens[i:i + 5]
        if config.eos_token_id in chunk:
            break
        if len(chunk) == 5:
            # Reversing the +3 offset applied during tokenization
            x1, y1, x2, y2 = [val - 3 for val in chunk[:4]]
            cls_id = chunk[4]
            box = {
                "class_id": f"UI_Element_{cls_id}",
                "bbox": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            }
            decoded_boxes.append(box)
    return decoded_boxes


def decode_json_tokens(token_list, config):
    """Translates raw structural token IDs back into a valid JSON tree string."""
    tokens = token_list[1:] if token_list and token_list[0] == config.sos_token_id else token_list

    elements = []
    for t in tokens:
        if t == config.eos_token_id:
            break
        if t not in [config.pad_token_id]:
            node_type = "container" if t % 2 == 0 else "component"
            elements.append({"node_id": t, "type": node_type})

    mock_tree = {
        "type": "root",
        "children": elements
    }
    return json.dumps(mock_tree, indent=2)


# =============================================================================
# Main Pipeline execution logic
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load actual data from CSV
    try:
        all_embeddings = load_embeddings_from_csv("../rico-data/embeddings/embeddings_qwen.csv")
    except Exception as e:
        print(f"Critical error: {e}")
        return

    # Split dataset: Use 80% for training
    train_size = int(len(all_embeddings) * 0.8)
    train_embeddings = all_embeddings.head(train_size)
    print(f"Using {len(train_embeddings)} samples for training (80%).")

    # Initialize Models
    layout_decoder = LayoutDecoder(cfg).to(device)
    structural_decoder = StructuralDecoder(cfg).to(device)

    # Optimizers
    opt_layout = torch.optim.Adam(layout_decoder.parameters(), lr=0.001)
    opt_struct = torch.optim.Adam(structural_decoder.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)

    print("\n--- Starting Training Phase on CSV Data ---")

    avg_layout_loss, avg_struct_loss = -1, -1

    num_epochs = 50
    for epoch in range(num_epochs):
        layout_loss_total = 0
        struct_loss_total = 0
        valid_samples = 0

        for i, row in tqdm(train_embeddings.iterrows(),
                           total=len(train_embeddings),
                           desc=f"Epoch {epoch + 1}/{num_epochs} | Layout Loss: {avg_layout_loss:.4f} | Struct Loss: {avg_struct_loss:.4f}"):
            # 1. Load targets from actual RICO file
            tgt_layout, tgt_json = parse_rico_json(row["json-path"], cfg)

            if tgt_layout is None or tgt_json is None:
                # Skip if file was missing or invalid JSON
                continue

            tgt_layout, tgt_json = tgt_layout.to(device), tgt_json.to(device)

            # Convert pandas list to tensor if it isn't one already
            if not torch.is_tensor(row["embedding"]):
                latent_vec = torch.tensor(row["embedding"], dtype=torch.float32).unsqueeze(0).to(device)
            else:
                latent_vec = row["embedding"].unsqueeze(0).to(device)

            valid_samples += 1

            # 2. Forward Pass: Layout
            opt_layout.zero_grad()
            layout_mask = nn.Transformer.generate_square_subsequent_mask(tgt_layout.size(1)).to(device)
            layout_logits = layout_decoder(latent_vec, tgt_layout, tgt_mask=layout_mask)
            loss_layout = criterion(layout_logits[:, :-1, :].reshape(-1, cfg.layout_vocab_size),
                                    tgt_layout[:, 1:].reshape(-1))
            loss_layout.backward()
            opt_layout.step()
            layout_loss_total += loss_layout.item()

            # 3. Forward Pass: Structure
            opt_struct.zero_grad()
            json_mask = nn.Transformer.generate_square_subsequent_mask(tgt_json.size(1)).to(device)
            json_logits = structural_decoder(latent_vec, tgt_json, tgt_mask=json_mask)
            loss_struct = criterion(json_logits[:, :-1, :].reshape(-1, cfg.json_vocab_size),
                                    tgt_json[:, 1:].reshape(-1))
            loss_struct.backward()
            opt_struct.step()
            struct_loss_total += loss_struct.item()

        avg_layout_loss = layout_loss_total / max(1, valid_samples)
        avg_struct_loss = struct_loss_total / max(1, valid_samples)

    print("\n--- Saving Trained Models ---")
    torch.save(layout_decoder.state_dict(), "layout_decoder.pth")
    torch.save(structural_decoder.state_dict(), "structural_decoder.pth")
    print("Successfully saved 'layout_decoder.pth' and 'structural_decoder.pth'")

    print("\n--- Starting Phase 4: Inference Pipeline (Using Last Loaded Embedding) ---")

    layout_decoder.eval()
    structural_decoder.eval()

    with torch.no_grad():
        # Use the very last embedding from the CSV for the inference test (part of the 20% holdout)
        last_row = all_embeddings.iloc[-1]

        if not torch.is_tensor(last_row["embedding"]):
            vec_input = torch.tensor(last_row["embedding"], dtype=torch.float32).unsqueeze(0).to(device)
        else:
            vec_input = last_row["embedding"].unsqueeze(0).to(device)

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

        print("\nUI Generation from CSV Embedding completed!")


if __name__ == "__main__":
    main()
