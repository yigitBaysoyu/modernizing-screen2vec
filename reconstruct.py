import os
import json
import orjson
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import argparse
import difflib


# =============================================================================
# Configuration (Must exactly match training script)
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


cfg = Config()


# =============================================================================
# Core Classes & Functions (Reused from training)
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
        if not isinstance(node, dict): return
        if 'bounds' in node:
            for val in node['bounds']:
                layout_tokens.append(max(0, min(999, int(val))) + 3)
            ui_class = node.get('class', 'Unknown')
            layout_tokens.append((hash(ui_class) % 96) + 1003)

        node_type = node.get('class', 'Unknown')
        json_tokens.append((hash(node_type) % (config.json_vocab_size - 3)) + 3)

        if 'children' in node:
            for child in node['children']:
                traverse(child)

    traverse(data)
    layout_tokens.append(config.eos_token_id)
    json_tokens.append(config.eos_token_id)
    return layout_tokens, json_tokens


def decode_layout_tokens(token_list, config):
    decoded_boxes = []
    tokens = token_list[1:] if token_list and token_list[0] == config.sos_token_id else token_list
    for i in range(0, len(tokens), 5):
        chunk = tokens[i:i + 5]
        if config.eos_token_id in chunk: break
        if len(chunk) == 5:
            x1, y1, x2, y2 = [val - 3 for val in chunk[:4]]
            cls_id = chunk[4]
            decoded_boxes.append(
                {"class_id": f"UI_Element_{cls_id}", "bbox": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]})
    return decoded_boxes


def decode_json_tokens(token_list, config):
    tokens = token_list[1:] if token_list and token_list[0] == config.sos_token_id else token_list
    elements = []
    for t in tokens:
        if t == config.eos_token_id: break
        if t not in [config.pad_token_id]:
            node_type = "container" if t % 2 == 0 else "component"
            elements.append({"node_id": t, "type": node_type})
    return json.dumps({"type": "root", "children": elements}, indent=2)


# =============================================================================
# Evaluation Metrics
# =============================================================================
def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea) if float(boxAArea + boxBArea - interArea) > 0 else 0.0
    return iou


def evaluate_layout(pred_layout, gt_layout):
    if not pred_layout or not gt_layout:
        return 0.0

    total_iou = 0.0
    matches = 0

    for p_box in pred_layout:
        best_iou = 0.0
        for g_box in gt_layout:
            if p_box["class_id"] == g_box["class_id"]:
                iou = calculate_iou(p_box["bbox"], g_box["bbox"])
                if iou > best_iou:
                    best_iou = iou
        if best_iou > 0:
            total_iou += best_iou
            matches += 1

    return total_iou / len(pred_layout) if len(pred_layout) > 0 else 0.0


def evaluate_structure(pred_str, gt_str):
    return difflib.SequenceMatcher(None, pred_str, gt_str).ratio()


# =============================================================================
# Main Execution
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Reconstruct and Evaluate RICO UI from Embedding.")
    parser.add_argument("--layout_model", type=str, required=True, help="Path to trained layout decoder (.pth)")
    parser.add_argument("--struct_model", type=str, required=True, help="Path to trained structural decoder (.pth)")
    parser.add_argument("--index", type=int, required=True, help="Index of the embedding in the CSV file")
    parser.add_argument("--csv_path", type=str, default="../rico-data/embeddings/embeddings_qwen.csv",
                        help="Path to embeddings CSV")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing on {device}...")

    print(f"[*] Loading CSV from: {args.csv_path}")
    df = pd.read_csv(args.csv_path, names=["json-path", "embedding"], header=None)

    if args.index >= len(df) or args.index < 0:
        print(f"[-] Error: Index {args.index} is out of bounds for dataset of size {len(df)}.")
        return

    row = df.iloc[args.index]
    json_path = row["json-path"].replace("/lustre/scratch/sgjolieb-screen2vec", "..")
    raw_embedding = orjson.loads(row["embedding"])

    print("[*] Loading Model Weights...")
    layout_decoder = LayoutDecoder(cfg).to(device)
    structural_decoder = StructuralDecoder(cfg).to(device)

    layout_decoder.load_state_dict(torch.load(args.layout_model, map_location=device))
    structural_decoder.load_state_dict(torch.load(args.struct_model, map_location=device))

    layout_decoder.eval()
    structural_decoder.eval()

    print(f"[*] Processing item at index {args.index} (Path: {json_path})")

    # Load original raw JSON for debugging
    raw_json_str = "Error loading raw JSON"
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                raw_json_data = json.load(f)
                raw_json_str = json.dumps(raw_json_data, indent=2)
            except json.JSONDecodeError:
                pass

    latent_vec = torch.tensor(raw_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    latent_vec = F.normalize(latent_vec, p=2, dim=-1)

    with torch.no_grad():
        gen_layout_tokens = layout_decoder.generate(latent_vec, max_length=cfg.max_layout_seq_len)[0].tolist()
        gen_struct_tokens = structural_decoder.generate(latent_vec, max_length=cfg.max_json_seq_len)[0].tolist()

    pred_layout_decoded = decode_layout_tokens(gen_layout_tokens, cfg)
    pred_struct_decoded = decode_json_tokens(gen_struct_tokens, cfg)

    gt_layout_tokens, gt_struct_tokens = parse_rico_json(json_path, cfg)
    if not gt_layout_tokens:
        print("[-] Error: Could not read Ground Truth JSON file to compare metrics.")
        gt_layout_decoded = []
        gt_struct_decoded = ""
    else:
        gt_layout_decoded = decode_layout_tokens(gt_layout_tokens, cfg)
        gt_struct_decoded = decode_json_tokens(gt_struct_tokens, cfg)

    struct_sim = evaluate_structure(pred_struct_decoded, gt_struct_decoded)
    layout_iou = evaluate_layout(pred_layout_decoded, gt_layout_decoded)

    min_l_len = min(len(gen_layout_tokens), len(gt_layout_tokens))
    layout_acc = sum(1 for i in range(min_l_len) if gen_layout_tokens[i] == gt_layout_tokens[i]) / max(1,
                                                                                                       len(gt_layout_tokens))

    min_s_len = min(len(gen_struct_tokens), len(gt_struct_tokens))
    struct_acc = sum(1 for i in range(min_s_len) if gen_struct_tokens[i] == gt_struct_tokens[i]) / max(1,
                                                                                                       len(gt_struct_tokens))

    print("\n" + "=" * 50)
    print("                EVALUATION REPORT")
    print("=" * 50)

    print("\n[ GROUND TRUTH JSON (Raw, truncated to 1500 chars) ]")
    print(raw_json_str[:1500] + ("...\n(truncated)" if len(raw_json_str) > 1500 else ""))

    print("\n[ GROUND TRUTH STRUCTURE (Parsed) ]")
    print(gt_struct_decoded[:500] + ("...\n(truncated)" if len(gt_struct_decoded) > 500 else ""))

    print("\n[ PREDICTED STRUCTURE ]")
    print(pred_struct_decoded[:500] + ("...\n(truncated)" if len(pred_struct_decoded) > 500 else ""))

    print("\n[ GROUND TRUTH LAYOUT (First 3 elements) ]")
    print(json.dumps(gt_layout_decoded[:3], indent=2))

    print("\n[ PREDICTED LAYOUT (First 3 elements) ]")
    print(json.dumps(pred_layout_decoded[:3], indent=2))

    print("\n" + "-" * 50)
    print("METRICS:")
    print("-" * 50)
    print(f"Structural String Similarity : {struct_sim:.4f} (1.0 is perfect match)")
    print(f"Structural Token Exact Match : {struct_acc:.4f}")
    print(f"Layout Average BBox IoU      : {layout_iou:.4f} (1.0 is perfect overlap)")
    print(f"Layout Token Exact Match     : {layout_acc:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()