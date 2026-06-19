# Reconstruction & Evaluation Script.
#
# Loads the trained LLaVA-style adapter (EmbeddingProjector + LoRA) and
# generates a cleaned JSON from a Qwen3-VL embedding.  Compares the
# prediction against the ground-truth cleaned JSON produced by
# process_json_file() and reports quality metrics.
#
# Requires:
#   - Trained projector weights (.pth)
#   - Trained LoRA adapter directory
#   - Path to embeddings CSV
#   - Index of the embedding to reconstruct

import os
import json
import importlib
import torch
import torch.nn.functional as F
import pandas as pd
import orjson
import argparse
import difflib
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from utils import process_json_file, CUDA_VISIBLE_DEVICE, load_embeddings

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICE

# Import shared components from the training script (which has a leading
# digit in its filename, so we use importlib).
_adapter = importlib.import_module("2_train_adapter")
EmbeddingProjector = _adapter.EmbeddingProjector
AdapterConfig = _adapter.AdapterConfig
generate_from_prefix = _adapter.generate_from_prefix


# =============================================================================
# Evaluation Helpers
# =============================================================================
def flatten_components(screen, prefix=""):
    """Recursively flatten the component tree into a list of (path, node) tuples."""
    components = []
    for i, comp in enumerate(screen.get("components", [])):
        path = f"{prefix}/{i}"
        components.append((path, comp))
        if "children" in comp:
            components.extend(
                flatten_components({"components": comp["children"]}, prefix=path)
            )
    return components


def calculate_iou(box_a, box_b):
    """Intersection-over-Union for two [x1, y1, x2, y2] boxes."""
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def evaluate_json_quality(pred_str, gt_str):
    """
    Comprehensive evaluation of a generated JSON against ground truth.

    Returns a dict of metrics:
        string_similarity  – SequenceMatcher ratio (0-1)
        json_valid          – whether the prediction parses as JSON
        app_match           – app field matches
        activity_match      – activity field matches
        component_count_gt  – # components in ground truth (recursive)
        component_count_pred– # components in prediction (recursive)
        class_accuracy      – fraction of paired components with matching class
        avg_iou             – mean bounding-box IoU across paired components
    """
    metrics = {}

    # 1. Raw string similarity
    metrics["string_similarity"] = difflib.SequenceMatcher(
        None, pred_str, gt_str
    ).ratio()

    # 2. JSON validity
    pred_data, gt_data = None, None
    try:
        pred_data = json.loads(pred_str)
        metrics["json_valid"] = True
    except json.JSONDecodeError:
        metrics["json_valid"] = False

    try:
        gt_data = json.loads(gt_str)
    except json.JSONDecodeError:
        pass

    if not pred_data or not gt_data:
        metrics["app_match"] = False
        metrics["activity_match"] = False
        metrics["component_count_gt"] = 0
        metrics["component_count_pred"] = 0
        metrics["class_accuracy"] = 0.0
        metrics["avg_iou"] = 0.0
        return metrics

    # 3. App / Activity match
    metrics["app_match"] = pred_data.get("app") == gt_data.get("app")
    metrics["activity_match"] = pred_data.get("activity") == gt_data.get("activity")

    # 4. Component-level analysis (flatten both trees)
    pred_comps = flatten_components(pred_data)
    gt_comps = flatten_components(gt_data)

    metrics["component_count_pred"] = len(pred_comps)
    metrics["component_count_gt"] = len(gt_comps)

    # 5. Paired comparison (by DFS tree order)
    min_len = min(len(pred_comps), len(gt_comps))
    if min_len == 0:
        metrics["class_accuracy"] = 0.0
        metrics["avg_iou"] = 0.0
        return metrics

    total_iou = 0.0
    class_matches = 0

    for i in range(min_len):
        _, p = pred_comps[i]
        _, g = gt_comps[i]

        if p.get("class") == g.get("class"):
            class_matches += 1

        pb = p.get("bounds", [0, 0, 0, 0])
        gb = g.get("bounds", [0, 0, 0, 0])
        if len(pb) == 4 and len(gb) == 4:
            total_iou += calculate_iou(pb, gb)

    metrics["class_accuracy"] = class_matches / min_len
    metrics["avg_iou"] = total_iou / min_len

    return metrics


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct and evaluate a UI JSON from a Qwen3-VL "
            "embedding using the trained LLaVA-style adapter."
        )
    )
    parser.add_argument(
        "--projector_model", type=str, required=True,
        help="Path to trained EmbeddingProjector weights (.pth)",
    )
    parser.add_argument(
        "--lora_model", type=str, required=True,
        help="Path to trained LoRA adapter directory",
    )
    parser.add_argument(
        "--index", type=int, required=True,
        help="Index of the embedding to reconstruct",
    )
    parser.add_argument(
        "--embeddings_dir", type=str, default="embeddings",
        help="Path to embeddings directory",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to raw Rico JSON/JPG files (optional)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="Maximum tokens to generate",
    )

    args = parser.parse_args()
    cfg = AdapterConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing on {device}...")

    # -------------------------------------------------------------------------
    # 1. Load embeddings
    # -------------------------------------------------------------------------
    embeddings_dir = args.embeddings_dir
    if not os.path.exists(embeddings_dir):
        print(f"[-] Error: Embeddings directory not found at {embeddings_dir}")
        return

    print(f"[*] Scanning and loading embeddings from: {embeddings_dir}")
    try:
        df = load_embeddings(embeddings_dir=embeddings_dir, data_dir=args.data_dir)
    except Exception as e:
        print(f"[-] Error loading embeddings: {e}")
        return

    if df.empty:
        print("[-] Error: No matching embeddings loaded.")
        return

    if args.index >= len(df) or args.index < 0:
        print(
            f"[-] Error: Index {args.index} is out of bounds "
            f"for dataset of size {len(df)}."
        )
        return

    row = df.iloc[args.index]
    json_path = row["json-path"]
    raw_embedding = row["embedding"]

    # -------------------------------------------------------------------------
    # 2. Load models
    # -------------------------------------------------------------------------
    print(f"[*] Loading base language model: {cfg.lm_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.lm_model_id,
        torch_dtype=torch.float32,
    )

    print(f"[*] Loading LoRA adapter from: {args.lora_model}")
    lm_model = PeftModel.from_pretrained(base_model, args.lora_model)
    lm_model.to(device)
    lm_model.eval()

    lm_hidden_dim = lm_model.config.hidden_size
    print(f"[*] Loading projector from: {args.projector_model}")
    projector = EmbeddingProjector(
        cfg.qwen_embed_dim, lm_hidden_dim, cfg.num_prefix_tokens
    ).to(device)
    projector.load_state_dict(
        torch.load(args.projector_model, map_location=device)
    )
    projector.eval()

    # -------------------------------------------------------------------------
    # 3. Generate from embedding
    # -------------------------------------------------------------------------
    print(f"[*] Processing item at index {args.index} (Path: {json_path})")

    latent_vec = (
        torch.tensor(raw_embedding, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )
    latent_vec = F.normalize(latent_vec, p=2, dim=-1)

    with torch.no_grad():
        prefix_embeds = projector(latent_vec)
        generated_ids = generate_from_prefix(
            lm_model, tokenizer, prefix_embeds,
            max_new_tokens=args.max_tokens,
        )
        pred_json_str = tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

    # -------------------------------------------------------------------------
    # 4. Load ground truth (cleaned JSON from the same pipeline used for training)
    # -------------------------------------------------------------------------
    gt_json_str = ""
    if os.path.exists(json_path):
        try:
            gt_json_str = process_json_file(json_path, minify=True)
        except Exception as e:
            print(f"[-] Warning: could not clean GT JSON: {e}")
    else:
        print(f"[-] Warning: GT file not found at {json_path}")

    # -------------------------------------------------------------------------
    # 5. Evaluate
    # -------------------------------------------------------------------------
    metrics = evaluate_json_quality(pred_json_str, gt_json_str)

    # -------------------------------------------------------------------------
    # 6. Display results
    # -------------------------------------------------------------------------
    col_width = 55

    # Pretty-print both JSONs for the side-by-side view
    try:
        gt_pretty = json.dumps(json.loads(gt_json_str), indent=2)
    except (json.JSONDecodeError, TypeError):
        gt_pretty = gt_json_str
    try:
        pred_pretty = json.dumps(json.loads(pred_json_str), indent=2)
    except (json.JSONDecodeError, TypeError):
        pred_pretty = pred_json_str

    gt_display = gt_pretty[:2000] + (
        "\n...(truncated)" if len(gt_pretty) > 2000 else ""
    )
    pred_display = pred_pretty[:2000] + (
        "\n...(truncated)" if len(pred_pretty) > 2000 else ""
    )

    print("\n" + "=" * 115)
    print(" " * 45 + "EVALUATION REPORT")
    print("=" * 115)

    gt_lines = gt_display.split("\n")
    pred_lines = pred_display.split("\n")
    max_lines = max(len(gt_lines), len(pred_lines))

    print(f"\n{'-' * 115}")
    header_l = "GROUND TRUTH (Cleaned JSON)"
    header_r = "PREDICTION (Generated JSON)"
    print(f"{header_l:<{col_width}} | {header_r:<{col_width}}")
    print(f"{'-' * 115}")
    for i in range(max_lines):
        left = gt_lines[i][:col_width] if i < len(gt_lines) else ""
        right = pred_lines[i][:col_width] if i < len(pred_lines) else ""
        print(f"{left:<{col_width}} | {right:<{col_width}}")
    print(f"{'-' * 115}\n")

    print("\n" + "-" * 50)
    print("METRICS:")
    print("-" * 50)
    valid_icon = "Yes" if metrics["json_valid"] else "No"
    print(f"JSON Valid               : {valid_icon}")
    print(
        f"String Similarity        : {metrics['string_similarity']:.4f} "
        f"(1.0 = perfect)"
    )
    app_icon = "Yes" if metrics["app_match"] else "No"
    act_icon = "Yes" if metrics["activity_match"] else "No"
    print(f"App Match                : {app_icon}")
    print(f"Activity Match           : {act_icon}")
    print(f"Component Count (GT)     : {metrics['component_count_gt']}")
    print(f"Component Count (Pred)   : {metrics['component_count_pred']}")
    print(f"Class Accuracy           : {metrics['class_accuracy']:.4f}")
    print(
        f"Average Bounds IoU       : {metrics['avg_iou']:.4f} "
        f"(1.0 = perfect)"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()