# Parallelized Reconstruction & Evaluation Script for HPC.
#
# Batched, shardable version of 3_json_reconstruction.py.  Each invocation is
# one shard: it loads the model stack (base LM + LoRA + projector) once, decodes
# its stride of the RICO dataset `--batch_size` elements at a time, and hands
# the finished generations to a pool of CPU processes that score them and write
# the reports.
#
# For each RICO element, produces two output files:
#   - <index>_reconstructed.json  (full pretty-printed JSON reconstruction)
#   - <index>_report.txt          (evaluation metrics report)
#
# Work is split on two levels, both of them pure functions of the command line,
# so shards never have to talk to each other:
#   - across Slurm array tasks via --shard_index / --num_shards
#   - within a task via --num_workers (round-robin over the visible GPUs)
# Every shard writes into the same --output_dir, which is stamped with the run's
# configuration (see claim_output_dir) so two different runs cannot silently
# interleave their reconstructions there.
#
# Usage:
#     # One Slurm array task out of eight, one GPU, batches of 64
#     python 3_json_reconstruction_par.py --shard_index 3 --num_shards 8 \
#         --batch_size 64 --output_dir /lustre/.../reconstructions_...
#
#     # Standalone / debug: everything in one process
#     python 3_json_reconstruction_par.py --batch_size 8
#
#     # Aggregate the reports of a finished run
#     python 3_json_reconstruction_par.py --summarize --output_dir ...
#
# Requires:
#   - Trained projector weights (.pth)
#   - Trained LoRA adapter directory
#   - Embeddings directory (embeddings_4096_last/)
#   - Raw Rico JSON files for ground-truth comparison

import os
import re
import sys
import json
import importlib
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import argparse
import difflib
import inspect
import multiprocessing
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
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

# Token budget used when --max_tokens <= 0 ("unlimited").  Generation still
# stops at EOS; this is only the ceiling for a degenerate, never-terminating
# generation, and is far beyond any JSON the model was trained to emit.
UNLIMITED_TOKEN_BUDGET = 1_000_000

# Written into the shared output directory by whichever shard gets there first.
# See claim_output_dir() for what it is for.
RUN_MANIFEST_NAME = "_run_manifest.json"


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
# Sharding, Ordering and Run Bookkeeping
# =============================================================================
def order_dataset_for_batching(dataset):
    """
    Globally order the dataset by ground-truth file size, ascending.

    This is the coarse pass, and its job is load balancing rather than batching:
    slice_for_worker() strides this list, so every shard has to derive the same
    order from the same inputs without talking to the others, and one stat() per
    file is all that can be afforded across the full 66k-element dataset.

    Measured on the real Rico layouts, striding this order gives shards that
    differ by at most one element and by 0.4% (8 shards) to 1.5% (32 shards) in
    total ground-truth bytes.  Items without a ground truth carry no length
    signal at all and go last, in rico_id order.

    It is a poor *batching* order on its own -- raw layout bytes correlate with
    the cleaned target only at Spearman 0.81, which leaves a batch of 128
    spanning far more length than it needs to.  refine_order_by_target_length()
    fixes that within a shard, where it is affordable.
    """
    def sort_key(item):
        try:
            return (0, os.path.getsize(item["json_path"]))
        except (OSError, TypeError):
            return (1, 0)

    return sorted(dataset, key=sort_key)


def target_length(item):
    """Length of the cleaned ground-truth JSON, or None if there is not one."""
    from utils import process_json_file

    if not item["json_path"]:
        return None
    try:
        return len(process_json_file(item["json_path"], minify=True))
    except Exception:
        return None


def refine_order_by_target_length(items, io_workers=8):
    """
    Re-sort one shard's items by the length of the cleaned target, ascending.

    A batch decodes until its *longest* member emits EOS, so the spread of
    target lengths inside a batch is what decides how much of the work is real
    and how much is padding EOS.  The global file-size order is only a proxy for
    that, and not a strong one.  Measured over shard 0 of the Rico dataset at
    batch 128, counting decode steps:

        shuffled                    4.8x the ideal work
        by raw file size            3.1x
        by cleaned target length    1.1x

    i.e. exact ordering is worth 2.3x fewer forward passes over the proxy -- the
    single largest factor in this script after the batching itself.

    It is affordable here and nowhere else: the cost is one process_json_file()
    per item, and a shard is a fraction of the dataset.  At ~2-15 ms per file
    that is seconds to a couple of minutes once, against hours of decoding.
    Items without a usable ground truth keep their relative order and go last;
    they still have to be reconstructed, there is just nothing to sort them by.
    """
    with ThreadPoolExecutor(max_workers=max(1, io_workers)) as pool:
        lengths = list(pool.map(target_length, items))

    known = [(n, i) for i, n in enumerate(lengths) if n is not None]
    unknown = [i for i, n in enumerate(lengths) if n is None]
    known.sort()

    return ([items[i] for _, i in known] + [items[i] for i in unknown],
            len(unknown))


def slice_for_worker(dataset, global_worker_id, global_num_workers):
    """
    Take every `global_num_workers`-th element, starting at `global_worker_id`.

    Strided rather than contiguous, because order_dataset_for_batching() has
    already sorted the dataset by expected output length: a contiguous slice
    would hand one worker every short screen and another every long one, and the
    array would finish at the pace of its unluckiest task.  Striding gives every
    worker the same length distribution while leaving each worker's own list
    sorted, which is what the batching wants.
    """
    return dataset[global_worker_id::global_num_workers]


def claim_output_dir(output_dir, fingerprint):
    """
    Stamp the shared output directory with this run's configuration, or verify
    that the stamp already there matches it.

    Every array task writes into one directory, and that directory is meant to
    be reused across submissions so an interrupted run can be resumed.  That
    makes it the one place where two different configurations could silently
    interleave their reconstructions -- and afterwards nothing would show it,
    because the files are named by Rico ID, not by what produced them.  So the
    first task to arrive writes the fingerprint and everybody else has to match.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, RUN_MANIFEST_NAME)

    record = dict(fingerprint)
    record["created_at"] = datetime.now().isoformat(timespec="seconds")

    try:
        # "x" is O_CREAT|O_EXCL: exactly one of the tasks starting at the same
        # moment creates the file, the rest fall through to the check below.
        with open(manifest_path, "x", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        return record
    except FileExistsError:
        pass

    with open(manifest_path, encoding="utf-8") as f:
        existing = json.load(f)

    mismatched = {
        key: (existing.get(key), value)
        for key, value in fingerprint.items()
        if existing.get(key) != value
    }
    if mismatched:
        details = ", ".join(
            f"{key}: on disk {have!r} vs. requested {want!r}"
            for key, (have, want) in mismatched.items()
        )
        raise RuntimeError(
            f"'{output_dir}' already holds reconstructions from a different "
            f"configuration, and mixing them would be undetectable afterwards. "
            f"{details}. Use a different --output_dir."
        )
    return existing


def pending_items(items, output_dir):
    """
    Drop the items whose two output files both already exist.

    Checked once up front rather than inside the per-item path, so a resumed run
    pays one pair of stat() calls per item instead of interleaving Lustre
    metadata lookups with the decode loop.
    """
    pending = []
    for item in items:
        rico_id = item["rico_id"]
        reconstructed = os.path.join(output_dir, f"{rico_id}_reconstructed.json")
        report = os.path.join(output_dir, f"{rico_id}_report.txt")
        if os.path.exists(reconstructed) and os.path.exists(report):
            continue
        pending.append(item)
    return pending


# =============================================================================
# Scoring and Report Writing (runs off the GPU worker's critical path)
# =============================================================================
def write_error_report(rico_id, embedding_path, json_path, output_dir, error, elapsed):
    """Record a failed item so a rerun can tell "not done yet" from "broke"."""
    report_path = os.path.join(output_dir, f"{rico_id}_report.txt")
    try:
        text = (
            f"ERROR processing RICO element {rico_id}\n"
            f"Embedding: {embedding_path}\n"
            f"JSON: {json_path}\n"
            f"Error: {error}\n"
            f"Traceback:\n{traceback.format_exc()}\n"
        )
        tmp_report = report_path + ".tmp"
        with open(tmp_report, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_report, report_path)
    except Exception:
        pass
    return rico_id, False, elapsed


def score_and_write_result(payload):
    """
    Evaluate one generated JSON against its ground truth and write both output
    files.  Pure CPU work: no CUDA, no model, nothing unpicklable.

    Kept at module level and handed a flat tuple so it can go straight into a
    ProcessPoolExecutor.  Scoring one item means a difflib SequenceMatcher and a
    Levenshtein distance over multi-kilobyte strings; at batch size 64 that adds
    up to a substantial stretch of single-threaded Python per batch, and every
    second of it is a second the GPU would otherwise spend idle.

    Returns (rico_id, success, elapsed).
    """
    from utils import process_json_file

    (rico_id, json_path, embedding_path, pred_json_str,
     output_dir, elapsed) = payload

    reconstructed_path = os.path.join(output_dir, f"{rico_id}_reconstructed.json")
    report_path = os.path.join(output_dir, f"{rico_id}_report.txt")

    try:
        # 1. Write full reconstruction JSON (atomic: write .tmp, then rename)
        try:
            pred_pretty = json.dumps(json.loads(pred_json_str), indent=2)
        except (json.JSONDecodeError, TypeError):
            pred_pretty = pred_json_str

        tmp_reconstructed = reconstructed_path + ".tmp"
        with open(tmp_reconstructed, "w", encoding="utf-8") as f:
            f.write(pred_pretty)
        os.replace(tmp_reconstructed, reconstructed_path)

        # 2. Load ground truth and evaluate
        gt_json_str = ""
        if json_path and os.path.exists(json_path):
            try:
                gt_json_str = process_json_file(json_path, minify=True)
            except Exception:
                pass

        metrics = evaluate_json_quality(pred_json_str, gt_json_str)

        # 3. Write report
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
        return write_error_report(
            rico_id, embedding_path, json_path, output_dir, e, elapsed
        )


# =============================================================================
# Batched Generation
# =============================================================================
def last_position_logits_kwarg(model):
    """
    Return the kwarg that limits a forward pass to the last position's logits,
    or {} if this transformers version does not have one.

    Without it the prefix pass materialises logits for *every* prefix token: at
    batch 64, 128 prefix tokens and Qwen2.5's 152k vocabulary that is a 5 GB
    float32 tensor, allocated and discarded once per batch, of which a single
    row is ever read.  The parameter was called num_logits_to_keep before
    transformers renamed it to logits_to_keep, so both are accepted.
    """
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return {}

    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return {name: 1}
    return {}


def generate_batch_from_prefix(
    model, tokenizer, prefix_embeds, max_new_tokens,
    eos_check_every=16, prefix_forward_kwargs=None,
):
    """
    Greedy-decode a whole batch of prefixes in one loop.

    generate_from_prefix() in 2_train_adapter.py takes a batch dimension but
    only stops once every sequence emits EOS at the *same* step, and then
    truncates all of them at that one position -- semantics that are right at
    batch 1 and wrong for a real batch.  This version tracks completion per
    sequence: a finished row is pinned to EOS for the rest of the run, so its KV
    cache stays well-formed and its tail is trivially trimmed afterwards, and
    decoding stops once every row is finished.

    Every prefix in a batch has the same number of tokens (batches are grouped
    by embedding shape), so no padding or attention mask is needed on the prefix
    side, and at batch 1 this returns exactly what generate_from_prefix() does.

    Args:
        prefix_embeds: [B, num_prefix, hidden_dim] projected prefix embeddings.

    Returns:
        [B, T] token IDs; each row still has to be truncated at its own EOS,
        which decode_generated_batch() does.
    """
    eos_token_id = tokenizer.eos_token_id
    prefix_forward_kwargs = prefix_forward_kwargs or {}

    # First forward pass: consume the prefix, populate the KV cache
    outputs = model(
        inputs_embeds=prefix_embeds, use_cache=True, **prefix_forward_kwargs
    )
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    generated = [next_token]
    finished = next_token.squeeze(1) == eos_token_id
    eos_column = torch.full_like(next_token, eos_token_id)
    steps_since_check = 1

    for _ in range(max_new_tokens - 1):
        if steps_since_check >= eos_check_every:
            # Reading `finished` forces a device synchronisation, which stops
            # the host from queueing the next step's kernels.  Doing it every
            # `eos_check_every` steps instead of every step costs at most that
            # many extra decode steps after the last sequence finishes -- see
            # the note in generate_from_prefix().
            if bool(finished.all()):
                break
            steps_since_check = 0

        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # Hold finished rows at EOS rather than letting them ramble on past it.
        next_token = torch.where(finished.unsqueeze(1), eos_column, next_token)

        generated.append(next_token)
        finished = finished | (next_token.squeeze(1) == eos_token_id)
        steps_since_check += 1

    return torch.cat(generated, dim=1)


def decode_generated_batch(tokenizer, generated_ids, eos_token_id):
    """Truncate every row at its own first EOS and decode it to a string."""
    rows = generated_ids.to("cpu")

    texts = []
    for row in rows:
        eos_positions = (row == eos_token_id).nonzero()
        end = eos_positions[0].item() + 1 if eos_positions.numel() else row.numel()
        texts.append(tokenizer.decode(row[:end], skip_special_tokens=True))
    return texts


def load_embedding(item):
    """Read one .npz embedding. Returns (item, array, error) -- never raises."""
    try:
        with np.load(item["embedding_path"]) as f_data:
            return item, f_data["embedding"].astype(np.float32), None
    except Exception as e:
        return item, None, e


def group_by_embedding_shape(loaded):
    """
    Split one chunk of loaded embeddings into shape-homogeneous groups.

    A directory's embeddings are uniform in practice, but a (4096,) and an
    (N, 4096) embedding cannot share a forward pass, so a shape change simply
    closes the current group.  Load failures are returned separately.

    Returns (groups, failures), where groups is [(items, arrays), ...] and
    failures is [(item, error), ...].
    """
    groups, failures = [], []
    items, arrays = [], []

    for item, array, error in loaded:
        if error is not None:
            failures.append((item, error))
            continue
        if arrays and array.shape != arrays[0].shape:
            groups.append((items, arrays))
            items, arrays = [], []
        items.append(item)
        arrays.append(array)

    if arrays:
        groups.append((items, arrays))
    return groups, failures


def generate_for_group(
    arrays, projector, lm_model, tokenizer, device, model_dtype,
    max_tokens, prefix_forward_kwargs,
):
    """Project a group of embeddings and decode all of them in one batch."""
    latent = torch.from_numpy(np.stack(arrays)).to(device=device, dtype=model_dtype)
    latent = F.normalize(latent, p=2, dim=-1)

    with torch.no_grad():
        prefix_embeds = projector(latent)
        generated_ids = generate_batch_from_prefix(
            lm_model, tokenizer, prefix_embeds,
            max_new_tokens=max_tokens if max_tokens > 0 else UNLIMITED_TOKEN_BUDGET,
            prefix_forward_kwargs=prefix_forward_kwargs,
        )

    return decode_generated_batch(tokenizer, generated_ids, tokenizer.eos_token_id)


# =============================================================================
# GPU Worker Function (runs in a spawned subprocess)
# =============================================================================
def gpu_worker(worker_id, num_workers, dataset, args):
    """
    Worker function for torch.multiprocessing.  Each worker:
      1. Pins itself to a specific GPU
      2. Loads the full model stack
      3. Decodes its stride of the dataset in batches
      4. Hands the finished generations to a pool of CPU processes for scoring

    Sharding is two-level.  The Slurm array supplies --shard_index of
    --num_shards, and within a task the spawned workers supply worker_id of
    num_workers, so the worker's position in the global ordering is
    `shard_index * num_workers + worker_id` out of `num_shards * num_workers`.
    Nothing coordinates between tasks at runtime: the split is a pure function
    of those four numbers, and every task derives the same dataset ordering from
    the same embeddings directory.

    Workers within a task are mapped round-robin onto the visible GPUs, so
    num_workers may exceed the GPU count.  With batched decoding that is rarely
    worth doing any more -- a batch of 64 keeps the device busy on its own,
    which batch-size-1 decoding never did.
    """
    # Pin to a specific GPU (round-robin when oversubscribing).  Read from
    # --device rather than probing torch.cuda: probing initialises the driver,
    # and after that CUDA_VISIBLE_DEVICES no longer has any effect on which
    # device this process gets.
    gpu_id = worker_id % max(1, args.num_gpus)
    if args.device == "cpu":
        device, device_label = torch.device("cpu"), "cpu"
    elif num_workers == 1:
        # Single worker: this is running in the parent, which already called
        # torch.cuda.device_count() and so has the driver initialised.  Setting
        # CUDA_VISIBLE_DEVICES now would be ignored, so address the device.
        device, device_label = torch.device(f"cuda:{gpu_id}"), f"GPU {gpu_id}"
    else:
        # A spawned worker has not touched CUDA yet, so masking works and keeps
        # the process from mapping memory on devices it will never use.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device, device_label = torch.device("cuda:0"), f"GPU {gpu_id}"

    # Nothing in the decode loop is a parallel CPU tensor op -- the host side is
    # a kernel launch loop -- and torch's default intra-op pool would give every
    # worker a thread per visible core.  The scoring pool below is where the CPU
    # parallelism actually belongs.
    torch.set_num_threads(1)

    # Compute this worker's stride of the (length-ordered) dataset.
    global_num_workers = max(1, args.num_shards) * num_workers
    global_worker_id = args.shard_index * num_workers + worker_id
    worker_dataset = slice_for_worker(dataset, global_worker_id, global_num_workers)

    print(f"[Worker {worker_id}] {device_label} | global worker "
          f"{global_worker_id + 1}/{global_num_workers} | "
          f"{len(worker_dataset)} of {len(dataset)} elements")

    if not worker_dataset:
        print(f"[Worker {worker_id}] No elements to process. Exiting.")
        return

    # Skip what a previous run already finished, before loading any model.
    todo = pending_items(worker_dataset, args.output_dir)
    if len(todo) != len(worker_dataset):
        print(f"[Worker {worker_id}] {len(worker_dataset) - len(todo)} elements "
              f"already on disk, {len(todo)} left to do")
    if not todo:
        print(f"[Worker {worker_id}] Nothing left to do. Exiting.")
        return

    # Tighten the batching order. Worth doing before the model is loaded: it is
    # pure file I/O, and a wrong order costs multiples of the decode time.
    if args.order_by == "exact":
        order_start = time.time()
        todo, without_gt = refine_order_by_target_length(todo, args.io_workers)
        print(f"[Worker {worker_id}] Ordered {len(todo)} elements by target "
              f"length in {time.time() - order_start:.0f}s "
              f"({without_gt} without ground truth, appended)", flush=True)

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
    # small matmuls per call -- roughly 200 of them per decode step here.
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

    prefix_forward_kwargs = last_position_logits_kwarg(lm_model)

    batch_size = max(1, args.batch_size)
    print(f"[Worker {worker_id}] Model loaded. Decoding {len(todo)} elements "
          f"at batch size {batch_size}...")

    os.makedirs(args.output_dir, exist_ok=True)

    # Scoring runs in its own processes so the GPU can start the next batch
    # while the previous one is still being diffed against ground truth.  Spawn
    # rather than fork: this process holds a live CUDA context, and forking one
    # is a documented way to get a deadlock or a corrupted child.
    eval_pool = None
    if args.eval_workers > 0:
        eval_pool = ProcessPoolExecutor(
            max_workers=args.eval_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        print(f"[Worker {worker_id}] Scoring in {args.eval_workers} CPU processes")

    n_items = len(todo)
    generated_count = 0
    failures = 0
    pending_futures = []
    worker_start = time.time()

    # On a terminal the bar is redrawn in place; in a Slurm log file (no TTY)
    # every update is a separate line, printed only every `progress_every`
    # elements so the log stays readable.
    interactive = sys.stdout.isatty()
    progress_every = max(1, getattr(args, "progress_every", 25))
    last_reported = 0

    def collect(futures, block):
        """Fold whatever scoring has finished into the failure count."""
        nonlocal failures
        still_running = []
        for future in futures:
            if not block and not future.done():
                still_running.append(future)
                continue
            try:
                _, success, _ = future.result()
            except Exception as e:
                print(f"[Worker {worker_id}] Scoring task failed: {e}", flush=True)
                success = False
            if not success:
                failures += 1
        return still_running

    # The .npz files are read through a small thread pool: on Lustre it is the
    # open() latency that costs, not the few kilobytes of payload, and numpy
    # releases the GIL while reading.
    with ThreadPoolExecutor(max_workers=max(1, args.io_workers)) as io_pool:
        for start_idx in range(0, n_items, batch_size):
            chunk = todo[start_idx:start_idx + batch_size]
            loaded = list(io_pool.map(load_embedding, chunk))
            groups, load_failures = group_by_embedding_shape(loaded)

            for item, error in load_failures:
                write_error_report(
                    item["rico_id"], item["embedding_path"], item["json_path"],
                    args.output_dir, error, 0.0,
                )
                failures += 1
                generated_count += 1

            for items, arrays in groups:
                batch_start = time.time()
                try:
                    texts = generate_for_group(
                        arrays, projector, lm_model, tokenizer, device,
                        model_dtype, args.max_tokens, prefix_forward_kwargs,
                    )
                except Exception as error:
                    # One bad batch must not take the whole shard down; the
                    # items stay unreported and a rerun will pick them up.
                    print(f"[Worker {worker_id}] Batch of {len(items)} failed "
                          f"({type(error).__name__}: {error}). If this is an "
                          f"OOM, lower --batch_size.", flush=True)
                    traceback.print_exc()
                    # Hand the half-built KV cache back, or an OOM on one batch
                    # leaves the allocator fragmented enough to fail the next.
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    failures += len(items)
                    generated_count += len(items)
                    continue

                # Reported per item as the batch's wall time divided by its
                # size, which is what an item actually costs now that they are
                # decoded together.
                per_item = (time.time() - batch_start) / max(1, len(items))

                payloads = [
                    (item["rico_id"], item["json_path"], item["embedding_path"],
                     text, args.output_dir, per_item)
                    for item, text in zip(items, texts)
                ]

                if eval_pool is None:
                    for payload in payloads:
                        _, success, _ = score_and_write_result(payload)
                        if not success:
                            failures += 1
                else:
                    pending_futures.extend(
                        eval_pool.submit(score_and_write_result, p) for p in payloads
                    )
                    pending_futures = collect(pending_futures, block=False)

                generated_count += len(items)

            if (interactive
                    or generated_count - last_reported >= progress_every
                    or generated_count >= n_items):
                last_reported = generated_count
                line = format_progress(
                    generated_count, n_items, generated_count - failures,
                    failures, time.time() - worker_start,
                )
                if interactive:
                    print(f"\r[Worker {worker_id}] {line}", end="", flush=True)
                else:
                    print(f"[Worker {worker_id}] {line}", flush=True)

    if interactive:
        print(flush=True)  # terminate the in-place progress line

    if eval_pool is not None:
        print(f"[Worker {worker_id}] Waiting for {len(pending_futures)} "
              f"outstanding scoring tasks...", flush=True)
        collect(pending_futures, block=True)
        eval_pool.shutdown(wait=True)

    worker_elapsed = time.time() - worker_start
    print(
        f"[Worker {worker_id}] DONE. "
        f"{generated_count - failures} succeeded, {failures} failed. "
        f"Total time: {format_duration(worker_elapsed)} "
        f"({worker_elapsed / 3600:.2f}h)",
        flush=True,
    )


# =============================================================================
# Aggregation over a finished (or partial) output directory
# =============================================================================
_METRIC_LINE_RE = re.compile(r"^([A-Za-z][^:]*?)\s*:\s*(.+?)\s*$")

# Report label -> (key, caster).  Mirrors the table in _results_reconstruction.ipynb;
# this is only meant to give the aggregation job something to put in the mail,
# the notebook stays the place where the analysis happens.
_SUMMARY_FIELDS = {
    "JSON Valid": ("json_valid", lambda v: v.strip().lower() == "yes"),
    "Sequence Similarity (Gestalt)*": ("sequence_similarity", float),
    "Levenshtein Similarity (minified)*": ("levenshtein_similarity", float),
    "App Match": ("app_match", lambda v: v.strip().lower() == "yes"),
    "Activity Match": ("activity_match", lambda v: v.strip().lower() == "yes"),
    "Class Accuracy*": ("class_accuracy", float),
    "Average Bounds IoU*": ("average_bounds_iou", float),
}


def parse_report_metrics(path):
    """Pull the METRICS block out of one report file. Returns {} on an error report."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return {}

    metrics_block = text.partition("METRICS:")[2]
    if not metrics_block:
        return {}

    out = {}
    for line in metrics_block.splitlines():
        match = _METRIC_LINE_RE.match(line)
        if match is None:
            continue
        field = _SUMMARY_FIELDS.get(match.group(1))
        if field is None:
            continue
        key, cast = field
        try:
            out[key] = cast(match.group(2))
        except ValueError:
            pass
    return out


def summarize_output_dir(output_dir):
    """
    Print aggregate metrics over every report in `output_dir`.

    Runs as its own small Slurm job once the array has finished, so the
    completion mail carries the headline numbers instead of just an exit code.
    """
    report_paths = sorted(Path(output_dir).glob("*_report.txt"))
    print(f"[Summary] {len(report_paths)} report files in {output_dir}")

    if not report_paths:
        return

    with ThreadPoolExecutor() as pool:
        parsed = [m for m in pool.map(parse_report_metrics, report_paths) if m]

    broken = len(report_paths) - len(parsed)
    print(f"[Summary] {len(parsed)} scored, {broken} error reports")

    manifest_path = os.path.join(output_dir, RUN_MANIFEST_NAME)
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            print("[Summary] Run manifest: " + json.dumps(json.load(f)))

    print(f"\n{'=' * 70}")
    print(f"  RECONSTRUCTION SUMMARY")
    print(f"{'=' * 70}")
    for key, _ in _SUMMARY_FIELDS.values():
        values = [m[key] for m in parsed if key in m]
        if not values:
            continue
        mean = sum(float(v) for v in values) / len(values)
        if isinstance(values[0], bool):
            print(f"  {key:<28}: {mean * 100:6.2f}%  ({sum(values)}/{len(values)})")
        else:
            print(f"  {key:<28}: {mean:8.4f}  (n={len(values)})")
    print(f"{'=' * 70}")


# =============================================================================
# Main
# =============================================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Parallelized JSON reconstruction from Qwen3-VL embeddings. "
            "Decodes the RICO dataset in batches, sharded across a Slurm array."
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
        "--batch_size", type=int, default=256,
        help=(
            "How many elements to decode in one forward pass. A batch runs "
            "until its longest member emits EOS, so --order_by decides how "
            "much of that work is real. Modelled over the Rico target "
            "lengths, 256 is best or near-best whatever the fixed per-step "
            "cost turns out to be; 512 needs 68 GB of KV to justify itself."
        ),
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help=(
            "Worker processes per Slurm task. May exceed --num_gpus; workers "
            "are assigned to GPUs round-robin. With batched decoding one worker "
            "per GPU is normally enough to saturate the device."
        ),
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=("cuda", "cpu"),
        help=(
            "Where to run. 'cpu' exists to let the whole worker path be "
            "exercised on a machine without a GPU; it is far too slow for the "
            "real dataset."
        ),
    )
    parser.add_argument(
        "--num_gpus", type=int, default=0,
        help=(
            "Number of GPUs to spread this task's workers over. 0 (default) "
            "auto-detects via torch.cuda.device_count()."
        ),
    )
    parser.add_argument(
        "--shard_index", type=int, default=0,
        help="This task's index in the Slurm array (0-based).",
    )
    parser.add_argument(
        "--num_shards", type=int, default=1,
        help=(
            "Total number of Slurm array tasks. Together with --shard_index and "
            "--num_workers this determines the dataset split; no runtime "
            "coordination between tasks is involved."
        ),
    )
    parser.add_argument(
        "--eval_workers", type=int, default=6,
        help=(
            "CPU processes that score generations and write reports, off the "
            "GPU's critical path. 0 scores inline in the worker process."
        ),
    )
    parser.add_argument(
        "--io_workers", type=int, default=8,
        help=(
            "Threads used to read a batch's .npz embeddings from Lustre, and "
            "to measure target lengths for --order_by exact."
        ),
    )
    parser.add_argument(
        "--order_by", type=str, default="exact", choices=("exact", "size", "none"),
        help=(
            "How to order a shard's elements before batching. 'exact' measures "
            "each cleaned ground truth once (seconds per shard) and is worth "
            "~2.3x fewer forward passes over 'size', which keeps the cheap "
            "global file-size order. 'none' is the same as 'size' here, since "
            "the global order is what the shards are strided out of."
        ),
    )
    parser.add_argument(
        "--progress_every", type=int, default=64,
        help=(
            "Print a progress line every N elements when stdout is not a "
            "terminal (e.g. a Slurm log). On a terminal the progress bar is "
            "redrawn after every batch."
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default="reconstructions_qwen3.5-4B",
        help=(
            "Output directory for reconstructed JSONs and reports. Shared by "
            "every array task; make it an absolute path on shared storage."
        ),
    )
    parser.add_argument(
        "--lm_model_id", type=str, default=None,
        help=(
            "Override the base LM. By default it is read from the LoRA "
            "checkpoint's adapter_config.json, which is what it was trained on."
        ),
    )
    parser.add_argument(
        "--job_json", type=str, default=None,
        help=(
            "Path to a job.json written by the HPC submit script. Any key in it "
            "that matches an argument name overrides that argument."
        ),
    )
    parser.add_argument(
        "--summarize", action="store_true",
        help=(
            "Do not reconstruct anything: parse the reports already in "
            "--output_dir and print aggregate metrics. Used by the aggregation "
            "job that runs after the array finishes."
        ),
    )
    return parser


def apply_job_json(args, parser):
    """
    Overlay job.json onto the parsed arguments.

    The submit script writes one job.json next to the code and every array task
    reads the same file; only --shard_index differs between tasks, and that
    comes from SLURM_ARRAY_TASK_ID on the command line.  Keys that are not
    arguments are ignored so the file can also carry documentation for a human.
    """
    with open(args.job_json, "r") as f:
        job = json.load(f)

    known = {action.dest for action in parser._actions}
    applied = {}
    for key, value in job.items():
        if key in known and value is not None:
            setattr(args, key, value)
            applied[key] = value

    print(f"[Main] Loaded job config from '{args.job_json}': "
          f"{json.dumps(applied, indent=2)}")
    return args


def run_fingerprint(args):
    """
    The configuration that decides what ends up in the output directory.

    Deliberately not the whole argument namespace: --shard_index, --batch_size
    or --eval_workers change how the work is spread, not what it produces, and
    including them would make a resumed run with a different shard count look
    like a conflict.
    """
    return {
        "projector_model": args.projector_model,
        "lora_model": args.lora_model,
        "embeddings_dir": args.embeddings_dir,
        "data_dir": args.data_dir,
        "max_tokens": args.max_tokens,
        "lm_model_id": args.lm_model_id,
    }


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.job_json:
        args = apply_job_json(args, parser)

    if args.summarize:
        summarize_output_dir(args.output_dir)
        return

    # Resolve the GPU count once, in the parent, so every worker agrees on the
    # round-robin mapping.  The workers cannot do this themselves: they set
    # CUDA_VISIBLE_DEVICES before touching CUDA and would each see one device.
    if args.num_gpus <= 0:
        args.num_gpus = max(1, torch.cuda.device_count() if args.device == "cuda" else 1)

    args.num_shards = max(1, args.num_shards)
    args.num_workers = max(1, args.num_workers)
    global_num_workers = args.num_shards * args.num_workers

    print("=" * 70)
    print("  Parallelized JSON Reconstruction (batched, array-sharded)")
    print("=" * 70)
    print(f"  Projector model : {args.projector_model}")
    print(f"  LoRA model      : {args.lora_model}")
    print(f"  Embeddings dir  : {args.embeddings_dir}")
    print(f"  Data dir        : {args.data_dir}")
    print(f"  Max tokens      : {args.max_tokens if args.max_tokens > 0 else 'unlimited'}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Shard           : {args.shard_index + 1}/{args.num_shards}")
    print(f"  Workers/shard   : {args.num_workers} "
          f"({global_num_workers} workers globally)")
    print(f"  Num GPUs        : {args.num_gpus} "
          f"({args.num_workers / args.num_gpus:.1f} workers/GPU)")
    print(f"  Batch order     : {args.order_by}")
    print(f"  Eval workers    : {args.eval_workers or 'inline'}")
    print(f"  Output dir      : {args.output_dir}")
    print("=" * 70)

    # 1. Claim the shared output directory (or refuse to mix configurations)
    claim_output_dir(args.output_dir, run_fingerprint(args))
    print(f"[Main] Output directory: {os.path.abspath(args.output_dir)}")

    # 2. Build the dataset index and put it in the order every task agrees on
    print("\n[Main] Building dataset index...")
    dataset = build_dataset_index(args.embeddings_dir, args.data_dir)
    gt_found = sum(1 for item in dataset if item["json_path"] is not None)
    print(f"[Main] Found {len(dataset)} RICO elements "
          f"({gt_found} with ground truth)")

    dataset = order_dataset_for_batching(dataset)

    # 3. Spawn workers
    total_start = time.time()

    if args.num_workers == 1:
        # Single worker per task -- run directly, no multiprocessing layer.
        print("\n[Main] Running in single-worker mode (no multiprocessing)")
        gpu_worker(0, 1, dataset, args)
    else:
        print(f"\n[Main] Spawning {args.num_workers} GPU workers via multiprocessing...")
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

        for worker_id, p in enumerate(processes):
            p.join()
            if p.exitcode != 0:
                print(f"[Main] WARNING: Worker {worker_id} exited with code {p.exitcode}")
            else:
                print(f"[Main] Worker {worker_id} completed successfully")

    total_elapsed = time.time() - total_start

    # 4. Summary.  The counts cover the whole shared directory, so with a Slurm
    # array they include what the other tasks have written so far.
    final_reconstructed = len(list(Path(args.output_dir).glob("*_reconstructed.json")))
    final_reports = len(list(Path(args.output_dir).glob("*_report.txt")))

    print(f"\n{'=' * 70}")
    print(f"  SHARD {args.shard_index + 1}/{args.num_shards} COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Output directory    : {os.path.abspath(args.output_dir)}")
    print(f"  Reconstructed JSONs : {final_reconstructed} (all shards)")
    print(f"  Report files        : {final_reports} (all shards)")
    print(f"  Total elements      : {len(dataset)}")
    print(f"  Shard time          : {total_elapsed / 3600:.2f} hours")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
