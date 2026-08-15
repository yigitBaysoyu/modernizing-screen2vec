# Parallelized Reconstruction & Evaluation Script for HPC.
#
# Multi-GPU version of 3_json_reconstruction.py.  Spawns N worker processes
# (one per GPU) via torch.multiprocessing.  Each worker loads the full model
# stack (Qwen3.5-4B base + LoRA + projector) once, then processes its
# contiguous slice of the RICO dataset.
#
# For each RICO element, produces two output files:
#   - <index>_reconstructed.json  (full pretty-printed JSON reconstruction)
#   - <index>_report.txt          (evaluation metrics report)
#
# All workers write to the same shared output directory.
#
# Usage:
#     # 8 GPU workers (default for H200 node)
#     python 3_json_reconstruction_par.py --num_workers 8
#
#     # Single GPU (standalone / debug)
#     python 3_json_reconstruction_par.py --num_workers 1
#
# Requires:
#   - Trained projector weights (.pth)
#   - Trained LoRA adapter directory
#   - Embeddings directory (embeddings_4096_last/)
#   - Raw Rico JSON files for ground-truth comparison

import os
import sys
import json
import importlib
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import argparse
import difflib
import time
import traceback
from pathlib import Path

# rapidfuzz carries a C implementation of the edit distance.  It is a hard
# requirement in requirements.txt, but the pure-Python fallback in
# calculate_levenshtein_distance() keeps the script runnable without it.
try:
    from rapidfuzz.distance import Levenshtein as _rapidfuzz_levenshtein
except ImportError:
    _rapidfuzz_levenshtein = None

# Import shared components from the training script (which has a leading
# digit in its filename, so we use importlib).
_adapter = importlib.import_module("2_train_adapter")
EmbeddingProjector = _adapter.EmbeddingProjector
AdapterConfig = _adapter.AdapterConfig
generate_from_prefix = _adapter.generate_from_prefix

# Token budget used when --max_tokens <= 0 ("unlimited").  Generation still
# stops at EOS; this is only the ceiling for a degenerate, never-terminating
# generation, and is far beyond any JSON the model was trained to emit.
UNLIMITED_TOKEN_BUDGET = 1_000_000


# =============================================================================
# Evaluation Helpers (copied from 3_json_reconstruction.py to keep this
# script self-contained and avoid import issues with multiprocessing spawn)
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

    The pure-Python fallback is O(len(s1) * len(s2)) *interpreted* operations,
    and on these inputs that is not a rounding error.  Measured over 15 sampled
    reconstructions: median 4.6s, mean 13.6s per item, peaking at 114s for a
    25k-character runaway generation scored against a 10k-character ground
    truth.  This runs inline in the reconstruction worker loop, so every one of
    those seconds is a second the GPU spends idle.

    rapidfuzz computes the same quantity -- unit cost for insertion, deletion
    and substitution -- with a bit-parallel algorithm in C, in single-digit
    milliseconds, so the metric is unchanged.
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
# Progress Display
# =============================================================================
def format_duration(seconds):
    """Format a duration in seconds as 'HH:MM:SS'."""
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_progress(done, total, successes, failures, elapsed, bar_width=30):
    """
    Render a single-line progress indicator:

        [#########---------------------]  312/2000 ( 15.6%) | 310 ok, 2 err |
        2.41s/item | elapsed 00:12:31 | ETA 01:07:45
    """
    frac = done / total if total else 1.0
    filled = int(round(bar_width * frac))
    bar = "#" * filled + "-" * (bar_width - filled)

    avg = elapsed / done if done else 0.0
    eta = (total - done) * avg

    return (
        f"[{bar}] {done}/{total} ({frac * 100:5.1f}%) | "
        f"{successes} ok, {failures} err | "
        f"{avg:.2f}s/item | elapsed {format_duration(elapsed)} | "
        f"ETA {format_duration(eta)}"
    )


# =============================================================================
# Report Formatting
# =============================================================================
def format_report(rico_id, json_path, embedding_path, pred_json_str, gt_json_str, metrics, elapsed):
    """Format an evaluation report as a text string (mirrors 3_json_reconstruction.py output)."""
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

    lines = []
    lines.append("=" * 115)
    lines.append(" " * 45 + "EVALUATION REPORT")
    lines.append("=" * 115)
    lines.append(f"Item Index      : {rico_id}")
    lines.append(f"JSON Path       : {json_path}")
    lines.append(f"Embedding Path  : {embedding_path}")

    gt_lines = gt_display.split("\n")
    pred_lines = pred_display.split("\n")
    max_lines = max(len(gt_lines), len(pred_lines))

    lines.append(f"\n{'-' * 115}")
    header_l = "GROUND TRUTH (Cleaned JSON)"
    header_r = "PREDICTION (Generated JSON)"
    lines.append(f"{header_l:<{col_width}} | {header_r:<{col_width}}")
    lines.append(f"{'-' * 115}")
    for i in range(max_lines):
        left = gt_lines[i][:col_width] if i < len(gt_lines) else ""
        right = pred_lines[i][:col_width] if i < len(pred_lines) else ""
        lines.append(f"{left:<{col_width}} | {right:<{col_width}}")
    lines.append(f"{'-' * 115}\n")

    valid_icon = "Yes" if metrics["json_valid"] else "No"
    app_icon = "Yes" if metrics["app_match"] else "No"
    act_icon = "Yes" if metrics["activity_match"] else "No"

    lines.append("-" * 50)
    lines.append("METRICS:")
    lines.append("-" * 50)
    lines.append(f"JSON Valid                         : {valid_icon}")
    lines.append(f"Sequence Similarity (Gestalt)*     : {metrics['string_similarity']:.4f} ")
    lines.append(f"Levenshtein Distance (minified)    : {metrics['levenshtein_distance']} ")
    lines.append(f"Levenshtein Similarity (minified)* : {metrics['levenshtein_similarity']:.4f} ")
    lines.append(f"App Match                          : {app_icon}")
    lines.append(f"Activity Match                     : {act_icon}")
    lines.append(f"Component Count (GT)               : {metrics['component_count_gt']}")
    lines.append(f"Component Count (Pred)             : {metrics['component_count_pred']}")
    lines.append(f"Class Accuracy*                    : {metrics['class_accuracy']:.4f}")
    lines.append(f"Average Bounds IoU*                : {metrics['avg_iou']:.4f}")
    lines.append("=" * 50)
    lines.append("Note: * indicates that 1.0 is a perfect match.")
    lines.append(f"Execution Time                     : {elapsed:.2f} seconds")
    lines.append("=" * 50)

    return "\n".join(lines)


# =============================================================================
# Dataset Index Builder
# =============================================================================
def build_dataset_index(embeddings_dir, data_dir):
    """
    Scan embeddings_dir for .npz files and match them to ground-truth JSONs.
    Returns a list of dicts: [{rico_id, embedding_path, json_path}, ...]
    sorted by rico_id.
    """
    embeddings_path = Path(embeddings_dir)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

    npz_files = sorted(embeddings_path.glob("*.npz"), key=lambda p: int(p.stem))

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in '{embeddings_dir}'.")

    # Resolve data directories for ground-truth JSON lookup
    if data_dir is None:
        search_dirs = ["../rico-data/data/layouts", "data/rico-examples", "rico/examples"]
    elif isinstance(data_dir, (list, tuple)):
        search_dirs = list(data_dir)
    else:
        search_dirs = [data_dir]

    dataset = []
    for npz_path in npz_files:
        rico_id = int(npz_path.stem)
        json_name = f"{rico_id}.json"

        # Find matching ground-truth JSON
        found_json_path = None
        for d in search_dirs:
            candidate = os.path.join(d, json_name)
            if os.path.exists(candidate):
                found_json_path = candidate
                break

        dataset.append({
            "rico_id": rico_id,
            "embedding_path": str(npz_path),
            "json_path": found_json_path,  # May be None if GT not found
        })

    return dataset


# =============================================================================
# Single-Element Processing
# =============================================================================
def process_single_element(
    item, projector, lm_model, tokenizer, cfg, device, model_dtype,
    output_dir, max_tokens,
):
    """
    Process a single RICO element: generate reconstruction + evaluate.
    Writes <rico_id>_reconstructed.json and <rico_id>_report.txt to output_dir.

    Returns (rico_id, success: bool, elapsed: float).
    """
    from utils import process_json_file

    rico_id = item["rico_id"]
    embedding_path = item["embedding_path"]
    json_path = item["json_path"]

    reconstructed_path = os.path.join(output_dir, f"{rico_id}_reconstructed.json")
    report_path = os.path.join(output_dir, f"{rico_id}_report.txt")

    # Skip if both outputs already exist (restart resilience)
    if os.path.exists(reconstructed_path) and os.path.exists(report_path):
        return rico_id, True, 0.0

    start_time = time.time()

    try:
        # 1. Load the single embedding
        with np.load(embedding_path) as f_data:
            raw_embedding = f_data["embedding"].astype(np.float32)

        # 2. Generate reconstruction
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
                max_new_tokens=max_tokens if max_tokens > 0 else UNLIMITED_TOKEN_BUDGET,
            )
            pred_json_str = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )

        # 3. Write full reconstruction JSON
        try:
            pred_pretty = json.dumps(json.loads(pred_json_str), indent=2)
        except (json.JSONDecodeError, TypeError):
            pred_pretty = pred_json_str

        # Atomic write: write to .tmp then rename
        tmp_reconstructed = reconstructed_path + ".tmp"
        with open(tmp_reconstructed, "w", encoding="utf-8") as f:
            f.write(pred_pretty)
        os.replace(tmp_reconstructed, reconstructed_path)

        # 4. Load ground truth and evaluate
        gt_json_str = ""
        if json_path and os.path.exists(json_path):
            try:
                gt_json_str = process_json_file(json_path, minify=True)
            except Exception:
                pass

        metrics = evaluate_json_quality(pred_json_str, gt_json_str)
        elapsed = time.time() - start_time

        # 5. Write report
        report_text = format_report(
            rico_id, json_path or "(not found)", embedding_path,
            pred_json_str, gt_json_str, metrics, elapsed,
        )
        tmp_report = report_path + ".tmp"
        with open(tmp_report, "w", encoding="utf-8") as f:
            f.write(report_text)
        os.replace(tmp_report, report_path)

        return rico_id, True, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        # Write error report so we know what failed
        try:
            error_report = (
                f"ERROR processing RICO element {rico_id}\n"
                f"Embedding: {embedding_path}\n"
                f"JSON: {json_path}\n"
                f"Error: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}\n"
            )
            tmp_report = report_path + ".tmp"
            with open(tmp_report, "w", encoding="utf-8") as f:
                f.write(error_report)
            os.replace(tmp_report, report_path)
        except Exception:
            pass
        return rico_id, False, elapsed


# =============================================================================
# GPU Worker Function (runs in a spawned subprocess)
# =============================================================================
def gpu_worker(worker_id, num_workers, dataset, args):
    """
    Worker function for torch.multiprocessing.  Each worker:
      1. Pins itself to a specific GPU
      2. Loads the full model stack
      3. Processes its contiguous slice of the dataset

    Workers are mapped round-robin onto the available GPUs, so `num_workers`
    may exceed the GPU count.  Decoding at batch size 1 leaves the device
    mostly idle -- it is bound by kernel launches and by the CPU-side scoring
    between generations, not by arithmetic -- so several workers sharing a GPU
    interleave into each other's gaps instead of competing.  A 1.5B model in
    float32 is ~6 GB of weights plus a few hundred MB of KV cache, which leaves
    an H200 with room for far more than the two or three this is meant for.
    """
    # Pin to a specific GPU (round-robin when oversubscribing).
    gpu_id = worker_id % max(1, args.num_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")  # Always cuda:0 since we set CUDA_VISIBLE_DEVICES

    # With several workers per GPU there are more processes than there is CPU
    # work to spread, and torch's default intra-op pool (one thread per visible
    # core) would have every one of them fan out across the whole node.  Nothing
    # here is a parallel CPU tensor op; the host side is a launch loop.
    torch.set_num_threads(1)

    # Compute this worker's slice
    total = len(dataset)
    chunk_size = (total + num_workers - 1) // num_workers  # ceil division
    start_idx = worker_id * chunk_size
    end_idx = min(start_idx + chunk_size, total)
    worker_dataset = dataset[start_idx:end_idx]

    print(f"[Worker {worker_id}] GPU {gpu_id} | Elements {start_idx}–{end_idx - 1} "
          f"({len(worker_dataset)} of {total})")

    if not worker_dataset:
        print(f"[Worker {worker_id}] No elements to process. Exiting.")
        return

    # Load models
    cfg = AdapterConfig()

    # A checkpoint is only loadable by the exact architecture it was trained
    # with, and AdapterConfig moves between experiments. So take every
    # architectural decision from the checkpoint itself rather than from cfg.
    projector_sd = torch.load(args.projector_model, map_location="cpu")

    model_dtype = projector_sd["projector.0.weight"].dtype
    ckpt_hidden_dim = projector_sd["projector.2.weight"].shape[0]
    use_prefix_pos = "prefix_pos.weight" in projector_sd
    num_prefix_tokens = (
        projector_sd["prefix_pos.weight"].shape[0]
        if use_prefix_pos
        else cfg.num_prefix_tokens
    )

    lm_model_id = args.lm_model_id
    if lm_model_id is None:
        adapter_cfg_path = os.path.join(args.lora_model, "adapter_config.json")
        with open(adapter_cfg_path) as f:
            lm_model_id = json.load(f)["base_model_name_or_path"]

    print(f"[Worker {worker_id}] Loading base model: {lm_model_id} ({model_dtype})")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(lm_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        lm_model_id,
        dtype=model_dtype,
    )

    if base_model.config.hidden_size != ckpt_hidden_dim:
        raise RuntimeError(
            f"projector was trained for hidden_size {ckpt_hidden_dim}, but "
            f"{lm_model_id} has {base_model.config.hidden_size}. The projector "
            f"and the LoRA adapter come from different runs."
        )

    print(f"[Worker {worker_id}] Loading LoRA adapter: {args.lora_model}")
    lm_model = PeftModel.from_pretrained(base_model, args.lora_model)
    lm_model.to(device)

    # Fold the LoRA deltas into the base weights.  An unmerged PeftModel keeps
    # every adapted projection wrapped in a Python module that runs two extra
    # small matmuls per call -- roughly 200 of them per decoded token here.
    # That is negligible arithmetic and a large amount of host-side work in a
    # loop that is already launch-bound.  Merging is exact up to float rounding
    # (W + BA * scaling), so reconstructions may differ in the last bits.
    lm_model = lm_model.merge_and_unload()
    lm_model.eval()

    lm_hidden_dim = lm_model.config.hidden_size
    print(
        f"[Worker {worker_id}] Loading projector: {args.projector_model} "
        f"(prefix_pos={'yes' if use_prefix_pos else 'no'}, "
        f"num_prefix_tokens={num_prefix_tokens})"
    )
    projector = EmbeddingProjector(
        cfg.qwen_embed_dim, lm_hidden_dim, num_prefix_tokens,
        use_prefix_pos=use_prefix_pos,
    ).to(device)
    projector.load_state_dict(projector_sd)
    projector = projector.to(device=device, dtype=model_dtype)
    projector.eval()

    print(f"[Worker {worker_id}] Model loaded. Starting inference on {len(worker_dataset)} elements...")

    # Create output directory (safe for concurrent creation)
    os.makedirs(args.output_dir, exist_ok=True)

    # Process elements
    n_items = len(worker_dataset)
    successes = 0
    failures = 0
    worker_start = time.time()

    # On a terminal the bar is redrawn in place; in a Slurm log file (no TTY)
    # every update is a separate line, printed only every `progress_every`
    # elements so the log stays readable.
    interactive = sys.stdout.isatty()
    progress_every = max(1, getattr(args, "progress_every", 25))

    for i, item in enumerate(worker_dataset):
        rico_id, success, elapsed = process_single_element(
            item, projector, lm_model, tokenizer, cfg, device, model_dtype,
            args.output_dir, args.max_tokens,
        )

        if success:
            successes += 1
        else:
            failures += 1

        done = i + 1
        if interactive or done % progress_every == 0 or done == n_items or done == 1:
            line = format_progress(
                done, n_items, successes, failures, time.time() - worker_start,
            )
            if interactive:
                print(f"\r[Worker {worker_id}] {line}", end="", flush=True)
            else:
                print(f"[Worker {worker_id}] {line}", flush=True)

    if interactive:
        print(flush=True)  # terminate the in-place progress line

    worker_elapsed = time.time() - worker_start
    print(
        f"[Worker {worker_id}] DONE. "
        f"{successes} succeeded, {failures} failed. "
        f"Total time: {format_duration(worker_elapsed)} "
        f"({worker_elapsed / 3600:.2f}h)",
        flush=True,
    )


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parallelized JSON reconstruction from Qwen3-VL embeddings. "
            "Spawns N workers (one per GPU) to process the full RICO dataset."
        )
    )
    parser.add_argument(
        "--projector_model", type=str, default="adapter_model_out/best_projector.pth",
        help="Path to trained EmbeddingProjector weights (.pth)",
    )
    parser.add_argument(
        "--lora_model", type=str, default="adapter_model_out/best_lora_adapter",
        help="Path to trained LoRA adapter directory",
    )
    parser.add_argument(
        "--embeddings_dir", type=str, default="embeddings_4096_last",
        help="Path to embeddings directory (default: embeddings_4096_last)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="../rico-data/data/layouts",
        help="Path to raw Rico JSON files for ground-truth comparison",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=8192,
        help=(
            "Maximum tokens to generate per element. Generation stops at EOS "
            "anyway; this only caps a degenerate, never-terminating generation. "
            "Use 0 for an effectively unlimited budget."
        ),
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help=(
            "Number of worker processes. May exceed --num_gpus; workers are "
            "assigned to GPUs round-robin. Batch-size-1 decoding does not "
            "saturate a GPU, so 2-3 workers per GPU is usually faster than 1."
        ),
    )
    parser.add_argument(
        "--num_gpus", type=int, default=0,
        help=(
            "Number of GPUs to spread the workers over. 0 (default) "
            "auto-detects via torch.cuda.device_count()."
        ),
    )
    parser.add_argument(
        "--progress_every", type=int, default=25,
        help=(
            "Print a progress line every N elements when stdout is not a "
            "terminal (e.g. a Slurm log). On a terminal the progress bar is "
            "redrawn after every element."
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default="reconstructions_qwen3.5-4B",
        help="Output directory for reconstructed JSONs and reports",
    )
    parser.add_argument(
        "--lm_model_id", type=str, default=None,
        help=(
            "Override the base LM. By default it is read from the LoRA "
            "checkpoint's adapter_config.json, which is what it was trained on."
        ),
    )
    args = parser.parse_args()

    # Resolve the GPU count once, in the parent, so every worker agrees on the
    # round-robin mapping.  The workers cannot do this themselves: they set
    # CUDA_VISIBLE_DEVICES before touching CUDA and would each see one device.
    if args.num_gpus <= 0:
        args.num_gpus = max(1, torch.cuda.device_count())

    print("=" * 70)
    print("  Parallelized JSON Reconstruction (Multi-GPU)")
    print("=" * 70)
    print(f"  Projector model : {args.projector_model}")
    print(f"  LoRA model      : {args.lora_model}")
    print(f"  Embeddings dir  : {args.embeddings_dir}")
    print(f"  Data dir        : {args.data_dir}")
    print(f"  Max tokens      : {args.max_tokens if args.max_tokens > 0 else 'unlimited'}")
    print(f"  Num workers     : {args.num_workers}")
    print(f"  Num GPUs        : {args.num_gpus} "
          f"({args.num_workers / args.num_gpus:.1f} workers/GPU)")
    print(f"  Progress every  : {max(1, args.progress_every)} elements")
    print(f"  Output dir      : {args.output_dir}")
    print("=" * 70)

    # 1. Build dataset index
    print("\n[Main] Building dataset index...")
    dataset = build_dataset_index(args.embeddings_dir, args.data_dir)
    print(f"[Main] Found {len(dataset)} RICO elements")

    gt_found = sum(1 for item in dataset if item["json_path"] is not None)
    print(f"[Main] Ground-truth JSONs found: {gt_found}/{len(dataset)}")

    # 2. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Main] Output directory: {os.path.abspath(args.output_dir)}")

    # 3. Check for already-completed elements
    existing_reconstructed = set()
    existing_reports = set()
    for f in os.listdir(args.output_dir):
        if f.endswith("_reconstructed.json"):
            try:
                existing_reconstructed.add(int(f.replace("_reconstructed.json", "")))
            except ValueError:
                pass
        elif f.endswith("_report.txt"):
            try:
                existing_reports.add(int(f.replace("_report.txt", "")))
            except ValueError:
                pass
    already_done = existing_reconstructed & existing_reports
    print(f"[Main] Already completed: {len(already_done)}/{len(dataset)} elements")

    # 4. Spawn workers
    total_start = time.time()

    if args.num_workers == 1:
        # Single worker mode — run directly (easier debugging)
        print("\n[Main] Running in single-worker mode (no multiprocessing)")
        gpu_worker(0, 1, dataset, args)
    else:
        print(f"\n[Main] Spawning {args.num_workers} GPU workers via multiprocessing...")
        # Use fork-server or spawn for CUDA compatibility
        mp.set_start_method("spawn", force=True)

        processes = []
        for worker_id in range(args.num_workers):
            p = mp.Process(
                target=gpu_worker,
                args=(worker_id, args.num_workers, dataset, args),
            )
            p.start()
            processes.append(p)
            print(f"[Main] Started worker {worker_id} (PID: {p.pid})")

        # Wait for all workers to finish
        for worker_id, p in enumerate(processes):
            p.join()
            if p.exitcode != 0:
                print(f"[Main] WARNING: Worker {worker_id} exited with code {p.exitcode}")
            else:
                print(f"[Main] Worker {worker_id} completed successfully")

    total_elapsed = time.time() - total_start

    # 5. Summary
    final_reconstructed = len(list(Path(args.output_dir).glob("*_reconstructed.json")))
    final_reports = len(list(Path(args.output_dir).glob("*_report.txt")))

    print(f"\n{'=' * 70}")
    print(f"  RECONSTRUCTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Output directory    : {os.path.abspath(args.output_dir)}")
    print(f"  Reconstructed JSONs : {final_reconstructed}")
    print(f"  Report files        : {final_reports}")
    print(f"  Total elements      : {len(dataset)}")
    print(f"  Total time          : {total_elapsed / 3600:.2f} hours")
    print(f"  Avg time per element: {total_elapsed / max(len(dataset), 1):.2f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
