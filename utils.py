import os
import json
import orjson
import pandas as pd
import torch
import torch.nn as nn


CUDA_VISIBLE_DEVICE: str = str(3)


# =============================================================================
# Configuration & Hyperparameters
# =============================================================================
class Config:
    qwen_embed_dim = 4096

    layout_vocab_size = 1100
    layout_hidden_dim = 512
    layout_num_layers = 4
    layout_num_heads = 8
    max_layout_seq_len = 500

    json_vocab_size = 2000
    json_hidden_dim = 512
    json_num_layers = 4
    json_num_heads = 8
    max_json_seq_len = 1000

    pad_token_id = 0
    sos_token_id = 1
    eos_token_id = 2


# =============================================================================
# Decoders
# =============================================================================
class LayoutDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.condition_proj = nn.Linear(config.qwen_embed_dim, config.layout_hidden_dim)
        self.embedding = nn.Embedding(config.layout_vocab_size, config.layout_hidden_dim)
        self.pos_encoder = nn.Embedding(config.max_layout_seq_len, config.layout_hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.layout_hidden_dim, nhead=config.layout_num_heads,
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config.layout_num_layers)
        self.fc_out = nn.Linear(config.layout_hidden_dim, config.layout_vocab_size)
        self.config = config

    def forward(self, qwen_latent, tgt_seq, tgt_mask=None):
        B, seq_len = tgt_seq.size()
        condition = self.condition_proj(qwen_latent).unsqueeze(1)
        positions = torch.arange(0, seq_len, device=tgt_seq.device).unsqueeze(0).expand(B, seq_len)
        tgt_emb = self.embedding(tgt_seq) + self.pos_encoder(positions)
        output = self.transformer_decoder(tgt_emb, condition, tgt_mask=tgt_mask)
        return self.fc_out(output)

    def generate(self, qwen_latent, max_length=100):
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
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.json_hidden_dim, nhead=config.json_num_heads,
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config.json_num_layers)
        self.fc_out = nn.Linear(config.json_hidden_dim, config.json_vocab_size)
        self.config = config

    def forward(self, qwen_latent, tgt_seq, tgt_mask=None):
        B, seq_len = tgt_seq.size()
        condition = self.condition_proj(qwen_latent).unsqueeze(1)
        positions = torch.arange(0, seq_len, device=tgt_seq.device).unsqueeze(0).expand(B, seq_len)
        tgt_emb = self.embedding(tgt_seq) + self.pos_encoder(positions)
        output = self.transformer_decoder(tgt_emb, condition, tgt_mask=tgt_mask)
        return self.fc_out(output)

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
# Shared Functions & Parsers
# =============================================================================
def parse_rico_json(filepath, config):
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

        if 'bounds' in node:
            b = node['bounds']
            for val in b:
                bin_val = max(0, min(999, int(val))) + 3
                if len(layout_tokens) < config.max_layout_seq_len - 2:
                    layout_tokens.append(bin_val)

            ui_class = node.get('class', 'Unknown')
            class_id = (hash(ui_class) % 96) + 1003
            if len(layout_tokens) < config.max_layout_seq_len - 2:
                layout_tokens.append(class_id)

        node_type = node.get('class', 'Unknown')
        token_hash = (hash(node_type) % (config.json_vocab_size - 3)) + 3
        if len(json_tokens) < config.max_json_seq_len - 2:
            json_tokens.append(token_hash)

        if 'children' in node:
            for child in node['children']:
                traverse(child)

    # Locate the actual UI root in the RICO format
    if 'activity' in data and 'root' in data['activity']:
        ui_root = data['activity']['root']
    else:
        ui_root = data

    traverse(ui_root)

    layout_tokens.append(config.eos_token_id)
    json_tokens.append(config.eos_token_id)

    return layout_tokens, json_tokens


def decode_layout_tokens(token_list, config):
    decoded_boxes = []
    tokens = token_list[1:] if token_list and token_list[0] == config.sos_token_id else token_list
    for i in range(0, len(tokens), 5):
        chunk = tokens[i:i + 5]
        if config.eos_token_id in chunk:
            break
        if len(chunk) == 5:
            x1, y1, x2, y2 = [val - 3 for val in chunk[:4]]
            cls_id = chunk[4]
            box = {"class_id": f"UI_Element_{cls_id}", "bbox": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]}
            decoded_boxes.append(box)
    return decoded_boxes


def decode_json_tokens(token_list, config):
    tokens = token_list[1:] if token_list and token_list[0] == config.sos_token_id else token_list
    elements = []
    for t in tokens:
        if t == config.eos_token_id:
            break
        if t not in [config.pad_token_id]:
            node_type = "container" if t % 2 == 0 else "component"
            elements.append({"node_id": t, "type": node_type})
    mock_tree = {"type": "root", "children": elements}
    return json.dumps(mock_tree, indent=2)


def load_embeddings_from_csv(filepath="../rico-data/embeddings/embeddings_qwen.csv"):
    """Loads all pre-computed Qwen embeddings from the CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The required file '{filepath}' was not found. Execution aborted.")

    df = pd.read_csv(filepath, names=["json-path", "embedding"], header=None)

    # Convert string representation of list back to an actual Python list and fix paths
    df["embedding"] = [orjson.loads(x) for x in df["embedding"]]
    df["json-path"] = df["json-path"].apply(lambda x: x.replace("/lustre/scratch/sgjolieb-screen2vec", ".."))
    df["jpg-path"] = df["json-path"].apply(lambda x: x.replace(".json", ".jpg"))

    print(f"Successfully loaded {len(df)} embeddings.")
    return df


def minify_json(json_data):
    """
    Removes all unnecessary whitespaces from a JSON object.
    This creates the most compact valid JSON representation.
    """
    return json.dumps(json_data, separators=(',', ':'))


def flatten_rico_to_text(node):
    """
    Recursively extracts meaningful UI elements from a RICO JSON node
    and flattens them into a clean, token-efficient text representation
    ideal for LLM embeddings.
    """
    if not node:
        return ""

    elements = []

    # Extract useful attributes commonly found in RICO dataset
    # We split the class to remove the verbose 'android.widget.' prefix
    ui_class = node.get('class', '').split('.')[-1]
    text = node.get('text', '')
    content_desc = node.get('content-desc', '')
    bounds = node.get('bounds', [])

    # Fix for content_desc being a list like [null] or ["string"] in RICO dataset
    if isinstance(content_desc, list):
        content_desc = " ".join([str(item) for item in content_desc if item is not None])
    elif content_desc is None:
        content_desc = ""

    # Clean strings (added isinstance checks to prevent future type errors)
    text = text.strip() if isinstance(text, str) and text else ""
    content_desc = content_desc.strip() if isinstance(content_desc, str) and content_desc else ""

    # Build a semantic string for the current node
    parts = []
    if ui_class:
        parts.append(f"[{ui_class}]")
    if text:
        parts.append(f"Text: '{text}'")
    elif content_desc:  # Fallback to content-desc if no text is present
        parts.append(f"Desc: '{content_desc}'")

    # Include bounding boxes if they are available
    if bounds and isinstance(bounds, list) and len(bounds) == 4:
        parts.append(f"Bounds: {bounds}")

    # Only keep nodes that have visible semantic value or positional bounds to save tokens
    # You can modify this condition if you want to keep structural empty layouts
    if parts and (text or content_desc or bounds):
        elements.append(" ".join(parts))

    # Recursively process children
    if 'children' in node and node['children']:
        for child in node['children']:
            child_text = flatten_rico_to_text(child)
            if child_text:
                elements.append(child_text)

    # Join with newlines so each element is clearly separated for the embedding model
    return "\n".join(elements)


def process_rico_data(json_data, flatten=True):
    """
    Processes RICO JSON data for embeddings based on the flatten switch.
    If flatten is False, it returns the minified JSON string.
    If flatten is True, it returns the flattened semantic text.
    """
    if not flatten:
        return minify_json(json_data)

    root_node = json_data.get('activity', {}).get('root', {})
    return flatten_rico_to_text(root_node)


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Mocking a small chunk of RICO UI data for demonstration
    sample_rico_data = json.load(open("data/rico-examples/0.json", "r"))

    print("--- Testing with flatten=False (Only Minify) ---")
    result_minified = process_rico_data(sample_rico_data, flatten=False)
    print(result_minified)
    print("\n")

    print("--- Testing with flatten=True (Flatten & Filter) ---")
    result_flattened = process_rico_data(sample_rico_data, flatten=True)
    print(result_flattened)