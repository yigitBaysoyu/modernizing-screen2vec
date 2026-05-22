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
def process_json_file(json_path,
                      binning=True,
                      reduce=True,
                      minify=True):
    """
    Processes RICO JSON data for embeddings based on the flatten switch.
    If flatten is False, it returns the minified JSON string.
    If flatten is True, it returns the flattened semantic text.
    """
    assert os.path.exists(json_path), f"Could not find path: {json_path}"

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # all of this operates on a dict
    if binning:
        json_data = _bin_json_coordinates(json_data)
    if reduce:
        json_data = _reduce_json_nodes(json_data)
    if minify:
        return _minify_json(json_data)
    else:
        return json.dumps(json_data, indent=2)


def _bin_json_coordinates(json_data):
    """
    Replaces absolute pixel coordinates with binned percentages (1-100)
    based on the maximum dimensions found in the JSON data.
    """
    root_node = json_data.get('activity', {}).get('root', {}) if isinstance(json_data, dict) else json_data

    max_x = 1
    max_y = 1

    def _find_max_bounds(node):
        nonlocal max_x, max_y
        if not isinstance(node, dict):
            return
        if 'bounds' in node:
            b = node['bounds']
            if isinstance(b, list) and len(b) == 4:
                try:
                    x2 = int(b[2])
                    y2 = int(b[3])
                    if x2 > max_x: max_x = x2
                    if y2 > max_y: max_y = y2
                except (ValueError, TypeError):
                    pass

        if 'children' in node and isinstance(node['children'], list):
            for child in node['children']:
                _find_max_bounds(child)

    _find_max_bounds(root_node)

    screen_width = max_x
    screen_height = max_y

    def _apply_binning(node):
        if not isinstance(node, dict):
            return
        if 'bounds' in node:
            b = node['bounds']
            if isinstance(b, list) and len(b) == 4:
                reduced_bounds = []
                for idx, val in enumerate(b):
                    try:
                        val_num = int(val)
                    except (ValueError, TypeError):
                        val_num = 0

                    denom = screen_width if idx % 2 == 0 else screen_height
                    if denom <= 0:
                        denom = 1

                    pct = round((val_num / denom) * 100)
                    pct_clamped = max(1, min(100, pct))
                    reduced_bounds.append(pct_clamped)
                node['bounds'] = reduced_bounds

        if 'children' in node and isinstance(node['children'], list):
            for child in node['children']:
                _apply_binning(child)

    if isinstance(json_data, dict) and 'activity' in json_data and 'root' in json_data['activity']:
        _apply_binning(json_data['activity']['root'])
    else:
        _apply_binning(json_data)

    return json_data


def _reduce_json_nodes(json_data):
    """
    Strips away unimportant data, leaving only structural and semantic
    keys necessary for UI reconstruction.
    """

    def _apply_reduce(node):
        if not isinstance(node, dict):
            return node

        reduced_node = {}
        keys_to_keep = ['class', 'text', 'content-desc', 'clickable', 'checked', 'bounds']
        for k in keys_to_keep:
            if k in node:
                reduced_node[k] = node[k]

        if 'children' in node and isinstance(node['children'], list):
            children = []
            for child in node['children']:
                child_reduced = _apply_reduce(child)
                if child_reduced is not None:
                    children.append(child_reduced)
            if children:
                reduced_node['children'] = children

        # Return None if the node ended up completely empty to allow proper filtering
        return reduced_node if reduced_node else None

    if isinstance(json_data, dict) and 'activity' in json_data and 'root' in json_data['activity']:
        json_data['activity']['root'] = _apply_reduce(json_data['activity']['root'])
    else:
        json_data = _apply_reduce(json_data)

    return json.dumps(json_data, indent=2)


def _minify_json(json_data):
    """
    Removes all unnecessary whitespaces from a JSON object.
    This creates the most compact valid JSON representation.
    """
    return json.dumps(json_data, separators=(',', ':'))


# ==========================================
# Decoding Stuff
# ==========================================

def _decode_layout_tokens(token_list, config):
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


def _decode_json_tokens(token_list, config):
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


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Mocking a small chunk of RICO UI data for demonstration
    sample_rico_data = json.load(open("data/rico-examples/0.json", "r"))

    print("--- Testing with flatten=False (Only Minify) ---")
    result_minified = process_rico_json(sample_rico_data, flatten=False)
    print(result_minified)
    print("\n")

    print("--- Testing with flatten=True (Flatten & Filter) ---")
    result_flattened = process_rico_json(sample_rico_data, flatten=True)
    print(result_flattened)