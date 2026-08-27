"""
repair_rescore.py

Worker side of the "Repaired JSON" pass in _results_reconstruction.ipynb.

The reports written by 3_json_reconstruction_par.py hold the *strict* score: a
prediction that does not parse counts as zero components, zero class accuracy,
zero IoU.  This module re-reads the raw predictions from disk, runs them
through json_repair (and, optionally, through the loop collapser), and scores
them again with the very same functions the reports were produced with --
reconstruction_metrics.score_against_ground_truth().

It lives in a module rather than in a notebook cell for one reason: on Windows
a ProcessPoolExecutor spawns fresh interpreters, and those can only unpickle a
worker function that is importable by name.  Notebook-local functions are not.
"""

import json
import os

from reconstruction_metrics import (
    UNORDERED_METRIC_KEYS,
    collapse_component_loops,
    flatten_components,
    repair_json,
    score_against_ground_truth,
    unordered_metrics,
)

# Set once per worker process by init_worker(); see rescore_item().
COLLAPSE_LOOPS = False


def init_worker(collapse_loops):
    """ProcessPoolExecutor initializer: pin the flags for this worker."""
    global COLLAPSE_LOOPS
    COLLAPSE_LOOPS = collapse_loops


def reconstruction_path(dim, idx):
    """Path of the raw prediction belonging to one (dim, idx) row of `df`."""
    return os.path.join(f"reconstructions_{dim}", f"{idx}_reconstructed.json")


def build_tasks(df, only_invalid=True):
    """
    Group the rows of the results DataFrame into one task per RICO index.

    Every dimension reconstructs the *same* screens, so the ground truth of an
    index is shared by up to one row per dimension.  Cleaning a raw Rico layout
    is not free, so the tasks are cut along the index rather than along the
    row: one task loads its ground truth once and scores every dimension that
    needs it.

    With `only_invalid` (the default) a row is only included when its strict
    parse failed -- for a prediction that already parsed, repairing is a no-op
    and the strict numbers in `df` are the repaired numbers.

    Returns a list of (idx, json_path, [(dim, reconstruction_path), ...]).
    """
    frame = df.reset_index()

    if only_invalid:
        frame = frame[frame["json_valid"] != True]  # noqa: E712 -- None counts as invalid

    tasks = []
    for idx, group in frame.groupby("idx", sort=False):
        json_paths = [p for p in group["json_path"].tolist() if isinstance(p, str)]
        if not json_paths:
            continue
        entries = [
            (row.dim, reconstruction_path(row.dim, row.idx))
            for row in group.itertuples()
        ]
        tasks.append((idx, json_paths[0], entries))

    return tasks


def _load_ground_truth(json_path):
    """Cleaned, minified ground-truth JSON string -- or "" if unavailable."""
    # Imported lazily: utils pulls in numpy/pandas, and a worker that only ever
    # hits missing files should not pay for that at startup.
    from utils import process_json_file

    if not json_path or not os.path.exists(json_path):
        return ""
    try:
        return process_json_file(json_path, minify=True)
    except Exception:
        return ""


def rescore_item(task):
    """
    Repair and re-score every dimension of one RICO index.

    Returns a list of per-row dicts (dim, idx, and the repaired metrics under
    the same names the strict columns use, suffixed "_repaired").
    """
    idx, json_path, entries = task

    gt_str = _load_ground_truth(json_path)
    try:
        gt_data = json.loads(gt_str)
    except (json.JSONDecodeError, TypeError):
        gt_data = None

    rows = []
    for dim, path in entries:
        row = {"dim": dim, "idx": idx}

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                pred_str = fh.read()
        except OSError:
            row["rescore_error"] = "prediction file missing"
            rows.append(row)
            continue

        # The strict pass already told us this did not parse, so go straight to
        # the repair; for a RESCORE_ALL run the prediction may well be valid,
        # and json.loads is then the cheaper and more faithful answer.
        try:
            pred_data = json.loads(pred_str)
            repair_used = False
        except (json.JSONDecodeError, TypeError):
            pred_data = repair_json(pred_str)
            repair_used = pred_data is not None

        # The strict parse, before any repair or loop collapsing.  Every other
        # strict column is read out of the reports on disk, but the order-free
        # metrics were added after those were written, so there is nothing to
        # parse -- their strict value has to be produced here as well.
        strict_data = pred_data if not repair_used else None

        removed = 0
        if COLLAPSE_LOOPS and isinstance(pred_data, dict):
            before = len(flatten_components(pred_data))
            pred_data = collapse_component_loops(pred_data)
            removed = before - len(flatten_components(pred_data))

        metrics = score_against_ground_truth(pred_data, gt_data, pred_str, gt_str)

        row["json_valid_repaired"] = pred_data is not None
        row["json_repair_used"] = repair_used
        row["components_collapsed"] = removed
        row["levenshtein_distance_repaired"] = metrics["levenshtein_distance"]
        row["levenshtein_similarity_repaired"] = metrics["levenshtein_similarity"]
        row["app_match_repaired"] = metrics["app_match"]
        row["activity_match_repaired"] = metrics["activity_match"]
        row["component_count_gt_repaired"] = metrics["component_count_gt"]
        row["component_count_pred_repaired"] = metrics["component_count_pred"]
        row["class_accuracy_repaired"] = metrics["class_accuracy"]
        # Named after the report label ("Average Bounds IoU"), not after the
        # metrics key, so the repaired column sits next to its strict twin.
        row["average_bounds_iou_repaired"] = metrics["avg_iou"]

        # Order-free metrics, both halves.  The repaired values already come
        # out of score_against_ground_truth() above -- it computes them for
        # whatever parse it was handed -- so only the strict half needs a
        # second pass, over the unrepaired, uncollapsed parse.
        strict_unordered = unordered_metrics(strict_data, gt_data)
        for key in UNORDERED_METRIC_KEYS:
            row[key] = strict_unordered[key]
            row[f"{key}_repaired"] = metrics[key]

        rows.append(row)

    return rows
