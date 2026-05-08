import os
import json
import orjson
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import argparse
import difflib

from utils import (Config, LayoutDecoder, StructuralDecoder, parse_rico_json,
                   decode_layout_tokens, decode_json_tokens, CUDA_VISIBLE_DEVICE)

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICE

# =============================================================================
# Configuration Initialization
# =============================================================================
cfg = Config()

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

    # Merge structural and layout predictions into a mock raw JSON for side-by-side comparison
    pred_merged_str = ""
    try:
        struct_data = json.loads(pred_struct_decoded)
        for i, child in enumerate(struct_data.get("children", [])):
            if i < len(pred_layout_decoded):
                child["class"] = pred_layout_decoded[i].get("class_id", "Unknown")
                child["bounds"] = pred_layout_decoded[i].get("bbox", [])
        pred_merged_str = json.dumps(struct_data, indent=2)
    except Exception as e:
        pred_merged_str = json.dumps({"error": f"Merge failed: {e}"})

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

    print("\n" + "=" * 115)
    print(" " * 45 + "EVALUATION REPORT")
    print("=" * 115)

    # Side-by-Side Formatting
    col_width = 55
    left_str = raw_json_str[:1500] + ("...\n(truncated)" if len(raw_json_str) > 1500 else "")
    right_str = pred_merged_str[:1500] + ("...\n(truncated)" if len(pred_merged_str) > 1500 else "")

    left_lines = left_str.split("\n")
    right_lines = right_str.split("\n")
    max_lines = max(len(left_lines), len(right_lines))

    print(f"\n{'-' * 115}")
    print(f"{'GROUND TRUTH (Raw JSON)':<{col_width}} | {'PREDICTION (Merged JSON)':<{col_width}}")
    print(f"{'-' * 115}")
    for i in range(max_lines):
        l = left_lines[i][:col_width] if i < len(left_lines) else ""
        r = right_lines[i][:col_width] if i < len(right_lines) else ""
        print(f"{l:<{col_width}} | {r:<{col_width}}")
    print(f"{'-' * 115}\n")

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