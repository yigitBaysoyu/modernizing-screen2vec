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
import sys
import json
import importlib
import torch
import torch.nn.functional as F
import pandas as pd
import orjson
import argparse
import difflib
import numpy as np
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# See calculate_levenshtein_distance() below; kept optional so the script still
# runs in an environment where rapidfuzz is missing.
try:
    from rapidfuzz.distance import Levenshtein as _rapidfuzz_levenshtein
except ImportError:
    _rapidfuzz_levenshtein = None

# Optional: only needed for --json_repair.
try:
    import json_repair
except ImportError:
    json_repair = None

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


def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the exact Levenshtein (edit) distance between two strings.

    The pure-Python fallback below is O(len(s1) * len(s2)) interpreted
    operations, which costs seconds to minutes per item on these JSONs.
    rapidfuzz computes the same quantity -- unit cost for insertion, deletion
    and substitution -- in C, so the reported metric is unchanged.
    """
    if _rapidfuzz_levenshtein is not None:
        return _rapidfuzz_levenshtein.distance(s1, s2)

    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1] + [0] * len(s2)
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    return previous_row[-1]


def repair_json_str(pred_str):
    """
    Best-effort recovery of a prediction that does not parse, returned as a
    JSON string so the existing evaluator can score it unchanged.

    Returns None when nothing usable can be salvaged. The target schema is
    always an object, so a list or scalar counts as unsalvageable -- see the
    same guard in 3_json_reconstruction_par.py, which carries the measured
    recovery rates.
    """
    if json_repair is None or not pred_str:
        return None
    try:
        repaired = json_repair.loads(pred_str)
    except Exception:
        return None
    if not isinstance(repaired, dict) or not repaired:
        return None
    return json.dumps(repaired, separators=(",", ":"))


def evaluate_json_quality(pred_str, gt_str):
    """
    Comprehensive evaluation of a generated JSON against ground truth.

    Returns a dict of metrics:
        string_similarity       – SequenceMatcher ratio (0-1) [Gestalt Pattern Matching]
        levenshtein_distance    – Edit distance on minified JSONs
        levenshtein_similarity  – Normalized Levenshtein similarity (0-1)
        json_valid              – whether the prediction parses as JSON
        app_match               – app field matches
        activity_match          – activity field matches
        component_count_gt      – # components in ground truth (recursive)
        component_count_pred    – # components in prediction (recursive)
        class_accuracy          – fraction of paired components with matching class
        avg_iou                 – mean bounding-box IoU across paired components
    """
    metrics = {}

    # 1. Sequence similarity (Gestalt / difflib SequenceMatcher)
    metrics["string_similarity"] = difflib.SequenceMatcher(
        None, pred_str, gt_str
    ).ratio()

    # 2. JSON validity and minified JSON extraction
    pred_data, gt_data = None, None
    try:
        pred_data = json.loads(pred_str)
        metrics["json_valid"] = True
    except json.JSONDecodeError:
        metrics["json_valid"] = False

    try:
        gt_data = json.loads(gt_str)
    except (json.JSONDecodeError, TypeError):
        pass

    # Build minified JSON strings for GT and Prediction (or fallback to raw strings if unparseable)
    pred_min_str = (
        json.dumps(pred_data, separators=(",", ":"))
        if pred_data is not None
        else pred_str
    )
    gt_min_str = (
        json.dumps(gt_data, separators=(",", ":"))
        if gt_data is not None
        else gt_str
    )

    # 3. Levenshtein Distance & Similarity on minified JSONs
    lev_dist = calculate_levenshtein_distance(pred_min_str, gt_min_str)
    max_len = max(len(pred_min_str), len(gt_min_str))
    lev_sim = 1.0 - (lev_dist / max_len) if max_len > 0 else 1.0

    metrics["levenshtein_distance"] = lev_dist
    metrics["levenshtein_similarity"] = lev_sim

    if not pred_data or not gt_data:
        metrics["app_match"] = False
        metrics["activity_match"] = False
        metrics["component_count_gt"] = 0
        metrics["component_count_pred"] = 0
        metrics["class_accuracy"] = 0.0
        metrics["avg_iou"] = 0.0
        return metrics

    # 4. App / Activity match
    metrics["app_match"] = pred_data.get("app") == gt_data.get("app")
    metrics["activity_match"] = pred_data.get("activity") == gt_data.get("activity")

    # 5. Component-level analysis (flatten both trees)
    pred_comps = flatten_components(pred_data)
    gt_comps = flatten_components(gt_data)

    metrics["component_count_pred"] = len(pred_comps)
    metrics["component_count_gt"] = len(gt_comps)

    # 6. Paired comparison (by DFS tree order)
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
        "--embeddings_dir", type=str, default="embeddings_4096_last/",
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
    parser.add_argument(
        "--full-output", action="store_true",
        help="Print only the reconstructed JSON string, suppressing metrics and logs",
    )
    parser.add_argument(
        "--json_repair", action="store_true",
        help=(
            "Additionally score the prediction through json_repair when it "
            "does not parse, and print the result under '(repaired)' labels. "
            "The strict metrics above are left exactly as they were."
        ),
    )
    parser.add_argument(
        "--lm_model_id", type=str, default=None,
        help=(
            "Override the base LM. By default it is read from the LoRA "
            "checkpoint's adapter_config.json, which is what it was trained on."
        ),
    )

    args = parser.parse_args()
    
    def log_info(msg):
        if not args.full_output:
            print(msg)
    cfg = AdapterConfig()
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"[*] Initializing on {device}...")

    # -------------------------------------------------------------------------
    # 1. Load embeddings
    # -------------------------------------------------------------------------
    embeddings_dir = args.embeddings_dir
    if not os.path.exists(embeddings_dir):
        print(f"[-] Error: Embeddings directory not found at {embeddings_dir}", file=sys.stderr)
        return

    log_info(f"[*] Scanning embeddings directory: {embeddings_dir}")
    try:
        df = load_embeddings(embeddings_dir=embeddings_dir, data_dir=args.data_dir, load_data=False)
    except Exception as e:
        print(f"[-] Error loading embeddings metadata: {e}", file=sys.stderr)
        return

    if df.empty:
        print("[-] Error: No matching embeddings loaded.", file=sys.stderr)
        return

    if args.index >= len(df) or args.index < 0:
        print(
            f"[-] Error: Index {args.index} is out of bounds "
            f"for dataset of size {len(df)}.",
            file=sys.stderr
        )
        return

    row = df.iloc[args.index]
    json_path = row["json-path"]
    
    # Load only the single requested embedding from disk
    embedding_path = row["embedding_path"]
    log_info(f"[*] Loading single embedding at index {args.index} from: {embedding_path}")
    try:
        with np.load(embedding_path) as f_data:
            raw_embedding = f_data["embedding"].astype(np.float32)
    except Exception as e:
        print(f"[-] Error loading single embedding file: {e}", file=sys.stderr)
        return

    # -------------------------------------------------------------------------
    # 2. Load models
    # -------------------------------------------------------------------------
    # A checkpoint is only loadable by the exact architecture it was trained
    # with, and AdapterConfig moves between experiments. So take every
    # architectural decision from the checkpoint itself rather than from cfg.
    projector_sd = torch.load(args.projector_model, map_location="cpu")

    lm_model_id = args.lm_model_id
    if lm_model_id is None:
        adapter_cfg_path = os.path.join(args.lora_model, "adapter_config.json")
        try:
            with open(adapter_cfg_path) as f:
                lm_model_id = json.load(f)["base_model_name_or_path"]
        except Exception as e:
            print(
                f"[-] Error: could not read base model from {adapter_cfg_path}: {e}\n"
                f"    Pass --lm_model_id explicitly.",
                file=sys.stderr,
            )
            return

    # The projector was saved in the precision it was trained in; matching it
    # keeps generation numerically identical to validation.
    model_dtype = projector_sd["projector.0.weight"].dtype
    ckpt_hidden_dim = projector_sd["projector.2.weight"].shape[0]
    use_prefix_pos = "prefix_pos.weight" in projector_sd
    num_prefix_tokens = (
        projector_sd["prefix_pos.weight"].shape[0]
        if use_prefix_pos
        else cfg.num_prefix_tokens
    )

    log_info(f"[*] Loading base language model: {lm_model_id} ({model_dtype})")
    tokenizer = AutoTokenizer.from_pretrained(lm_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        lm_model_id,
        torch_dtype=model_dtype,
    )

    if base_model.config.hidden_size != ckpt_hidden_dim:
        print(
            f"[-] Error: projector was trained for hidden_size {ckpt_hidden_dim}, "
            f"but {lm_model_id} has {base_model.config.hidden_size}. The "
            f"projector and the LoRA adapter come from different runs.",
            file=sys.stderr,
        )
        return

    log_info(f"[*] Loading LoRA adapter from: {args.lora_model}")
    lm_model = PeftModel.from_pretrained(base_model, args.lora_model)
    lm_model.to(device)
    lm_model.eval()

    lm_hidden_dim = lm_model.config.hidden_size
    log_info(
        f"[*] Loading projector from: {args.projector_model} "
        f"(prefix_pos={'yes' if use_prefix_pos else 'no'})"
    )
    projector = EmbeddingProjector(
        cfg.qwen_embed_dim, lm_hidden_dim, num_prefix_tokens,
        use_prefix_pos=use_prefix_pos,
    ).to(device)
    projector.load_state_dict(projector_sd)
    projector = projector.to(device=device, dtype=model_dtype)
    projector.eval()

    # -------------------------------------------------------------------------
    # 3. Generate from embedding
    # -------------------------------------------------------------------------
    log_info(f"[*] Processing item at index {args.index} (Path: {json_path})")

    latent_vec = (
        torch.tensor(raw_embedding, dtype=model_dtype)
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

    if args.full_output:
        try:
            pred_pretty = json.dumps(json.loads(pred_json_str), indent=2)
            print(pred_pretty)
        except (json.JSONDecodeError, TypeError):
            print(pred_json_str)
        return

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

    # Scored a second time on the repaired string rather than folded into the
    # numbers above: the strict metrics are what every report on disk holds.
    repaired_metrics = None
    if args.json_repair and not metrics["json_valid"]:
        repaired_str = repair_json_str(pred_json_str)
        if repaired_str is not None:
            repaired_metrics = evaluate_json_quality(repaired_str, gt_json_str)

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

    valid_icon = "Yes" if metrics["json_valid"] else "No"
    app_icon = "Yes" if metrics["app_match"] else "No"
    act_icon = "Yes" if metrics["activity_match"] else "No"

    print("\n" + "-" * 50)
    print("METRICS:")
    print("-" * 50)
    print(f"JSON Valid                         : {valid_icon}")
    print(f"Sequence Similarity (Gestalt)*     : {metrics['string_similarity']:.4f} ")
    print(f"Levenshtein Distance (minified)    : {metrics['levenshtein_distance']} ")
    print(f"Levenshtein Similarity (minified)* : {metrics['levenshtein_similarity']:.4f} ")
    print(f"App Match                          : {app_icon}")
    print(f"Activity Match                     : {act_icon}")
    print(f"Component Count (GT)               : {metrics['component_count_gt']}")
    print(f"Component Count (Pred)             : {metrics['component_count_pred']}")
    print(f"Class Accuracy*                    : {metrics['class_accuracy']:.4f}")
    print(f"Average Bounds IoU*                : {metrics['avg_iou']:.4f}")
    if args.json_repair:
        print("-" * 50)
        if repaired_metrics is None:
            recovered = "Yes (already valid)" if metrics["json_valid"] else "No"
            print(f"JSON Valid (repaired)              : {recovered}")
        else:
            print(f"JSON Valid (repaired)              : Yes")
            print(f"Levenshtein Similarity (repaired)* : {repaired_metrics['levenshtein_similarity']:.4f} ")
            print(f"Component Count (Pred, repaired)   : {repaired_metrics['component_count_pred']}")
            print(f"Class Accuracy (repaired)*         : {repaired_metrics['class_accuracy']:.4f}")
            print(f"Average Bounds IoU (repaired)*     : {repaired_metrics['avg_iou']:.4f}")
    print("="* 50)
    print("Note: * indicates that 1.0 is a perfect match.")
    print(f"Execution Time                     : {time.time() - start_time:.2f} seconds")
    print("=" * 50)


if __name__ == "__main__":
    main()