"""
reconstruction_metrics.py

Scoring helpers for the JSON reconstruction task, shared by the generation
scripts and the result notebooks.

These functions used to live inline in 3_json_reconstruction_par.py (itself a
copy of the ones in 3_json_reconstruction.py).  They are pure -- no torch, no
transformers, no CUDA -- so keeping them in a module of their own means a
result notebook, or any `ProcessPoolExecutor` worker started from one, can
import the exact scoring code the reports on disk were produced with, without
loading a model stack it has no use for.
"""

import difflib
import json

# Optional: only needed for the repair scoring pass.  Kept soft so an
# environment without it still runs the strict evaluation unchanged.
try:
    import json_repair
except ImportError:
    json_repair = None

# rapidfuzz carries a C implementation of the edit distance.  It is a hard
# requirement in requirements.txt, but the pure-Python fallback in
# calculate_levenshtein_distance() keeps the module usable without it.
try:
    from rapidfuzz.distance import Levenshtein as _rapidfuzz_levenshtein
except ImportError:
    _rapidfuzz_levenshtein = None


def flatten_components(screen, prefix=""):
    """Recursively flatten the component tree into a list of (path, node) tuples."""
    components = []
    for i, comp in enumerate(screen.get("components", [])):
        path = f"{prefix}/{i}"
        components.append((path, comp))
        # A repaired prediction can put anything in this list -- a bare string,
        # a nested list -- so the node type is checked before it is walked.
        if isinstance(comp, dict) and isinstance(comp.get("children"), list):
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


def _numeric_box(bounds):
    """
    A "bounds" field as 4 floats, or None if it is not a clean box.

    `len(bounds) == 4` alone is not a strong enough guard: json_repair can
    salvage "bounds" as a 4-character string (which also has length 4, but
    indexes into characters, not coordinates) or as a list mixing numbers
    with strings.  calculate_iou() then compares an int against a str and
    raises.  Coercing every entry through float() here, and giving up on
    the whole box if any entry does not convert, keeps that invalid input
    out of calculate_iou() instead of catching it after the fact.
    """
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    try:
        return [float(v) for v in bounds]
    except (TypeError, ValueError):
        return None


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


def repair_json(pred_str):
    """
    Best-effort recovery of a prediction that does not parse.

    Returns the parsed object, or None if nothing usable can be salvaged.

    The decoder fails in two distinct ways, and only one of them is worth
    repairing.  Roughly 43% of the invalid outputs are a well-formed prefix that
    simply stops -- an unterminated string, unbalanced brackets, a trailing
    comma -- and those carry most of a correct answer.  The rest have collapsed
    into degenerate repetition (the same short unit emitted until the token
    budget runs out); json_repair will still return *something* for those, which
    is why the caller keeps the strict `json_valid` flag separate and never
    lets a repaired parse overwrite it.

    Measured on 4x4096_avg: json_repair recovers 88% of invalid predictions,
    against 43% for closing brackets by hand.  The recovered ones score about
    0.15 class accuracy where naturally valid ones score 0.35, so this lifts the
    floor rather than the ceiling -- see --json-guided for preventing the
    breakage instead of patching it.
    """
    if json_repair is None or not pred_str:
        return None
    try:
        repaired = json_repair.loads(pred_str)
    except Exception:
        return None
    # json_repair answers "" / {} / [] rather than failing when there is nothing
    # to salvage; those parse but carry no components, so treat them as failure.
    if repaired in ("", {}, [], None):
        return None
    # The target schema is always an object.  json_repair happily returns a
    # list or a bare scalar when the prediction collapsed into something else,
    # and that cannot be scored against a screen -- treat it as unsalvageable.
    if not isinstance(repaired, dict):
        return None
    return repaired


def score_against_ground_truth(pred_data, gt_data, pred_str, gt_str):
    """
    All metrics that depend on the *parsed* prediction, for one already
    decided parse.  `pred_data` is None when the prediction did not parse.

    Split out of evaluate_json_quality() so the same scoring can run twice
    over one prediction -- once on the strict parse, once on the repaired
    one -- without the two copies drifting apart.
    """
    metrics = {}

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

    # Levenshtein Distance & Similarity on minified JSONs
    lev_dist = calculate_levenshtein_distance(pred_min_str, gt_min_str)
    max_len = max(len(pred_min_str), len(gt_min_str))
    lev_sim = 1.0 - (lev_dist / max_len) if max_len > 0 else 1.0

    metrics["levenshtein_distance"] = lev_dist
    metrics["levenshtein_similarity"] = lev_sim

    # isinstance rather than a plain truth test: a prediction that parses as a
    # list or a scalar reaches .get() below and raises.  That was reachable in
    # the strict path too, just rare; the repaired path hits it regularly.
    if not isinstance(pred_data, dict) or not isinstance(gt_data, dict):
        metrics["app_match"] = False
        metrics["activity_match"] = False
        metrics["component_count_gt"] = 0
        metrics["component_count_pred"] = 0
        metrics["class_accuracy"] = 0.0
        metrics["avg_iou"] = 0.0
        return metrics

    # App / Activity match
    metrics["app_match"] = pred_data.get("app") == gt_data.get("app")
    metrics["activity_match"] = pred_data.get("activity") == gt_data.get("activity")

    # Component-level analysis (flatten both trees)
    pred_comps = flatten_components(pred_data)
    gt_comps = flatten_components(gt_data)

    metrics["component_count_pred"] = len(pred_comps)
    metrics["component_count_gt"] = len(gt_comps)

    # Paired comparison (by DFS tree order)
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

        # Same reason as in flatten_components(): a salvaged parse can hold
        # entries that are not components at all.  They stay in the denominator
        # -- they are positions the prediction got wrong -- but score nothing.
        if not isinstance(p, dict) or not isinstance(g, dict):
            continue

        if p.get("class") == g.get("class"):
            class_matches += 1

        pb = _numeric_box(p.get("bounds", [0, 0, 0, 0]))
        gb = _numeric_box(g.get("bounds", [0, 0, 0, 0]))
        if pb is not None and gb is not None:
            total_iou += calculate_iou(pb, gb)

    metrics["class_accuracy"] = class_matches / min_len
    metrics["avg_iou"] = total_iou / min_len

    return metrics


def evaluate_json_quality(pred_str, gt_str, repair=False):
    """
    Comprehensive evaluation of a generated JSON against ground truth.

    Returns a dict of metrics:
        string_similarity       - SequenceMatcher ratio (0-1) [Gestalt Pattern Matching]
        levenshtein_distance    - Edit distance on minified JSONs
        levenshtein_similarity  - Normalized Levenshtein similarity (0-1)
        json_valid              - whether the prediction parses as JSON
        app_match               - app field matches
        activity_match          - activity field matches
        component_count_gt      - # components in ground truth (recursive)
        component_count_pred    - # components in prediction (recursive)
        class_accuracy          - fraction of paired components with matching class
        avg_iou                 - mean bounding-box IoU across paired components

    With `repair`, every parse-dependent metric above is additionally
    reported under a "_repaired" suffix, computed on the repair_json()
    reconstruction of an otherwise unparsable prediction, plus:
        json_valid_repaired     - whether anything could be salvaged
        json_repair_used        - whether the repair actually had to step in

    The strict keys keep their exact meaning either way.  That matters: the
    reconstruction folders on disk already hold hundreds of thousands of
    reports scored the strict way, and a repaired number silently replacing
    a strict one would make every one of those incomparable.
    """
    metrics = {}

    # 1. Sequence similarity (Gestalt / difflib SequenceMatcher)
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
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. Strict scoring -- unchanged, and the only thing older reports hold
    metrics.update(
        score_against_ground_truth(pred_data, gt_data, pred_str, gt_str)
    )

    if not repair:
        return metrics

    # 4. Repaired scoring, reported alongside rather than instead
    if metrics["json_valid"]:
        repaired_data = pred_data
        metrics["json_repair_used"] = False
    else:
        repaired_data = repair_json(pred_str)
        metrics["json_repair_used"] = repaired_data is not None

    metrics["json_valid_repaired"] = repaired_data is not None
    for key, value in score_against_ground_truth(
        repaired_data, gt_data, pred_str, gt_str
    ).items():
        metrics[f"{key}_repaired"] = value

    return metrics



def _canonical(component):
    """Stable string form of a component subtree, used as a dedupe key."""
    return json.dumps(component, sort_keys=True, separators=(",", ":"))


def collapse_component_loops(screen):
    """
    Remove looped repetition from a decoded screen.

    A degenerate generation does not stop after the screen is described: it
    keeps emitting the same component -- or the same short cycle of two or
    three components -- until the token budget runs out.  Those extra copies
    are not predictions of anything; they only inflate the component count and
    drag the paired class/IoU comparison out of alignment with the ground
    truth, because every duplicate shifts the remaining pairs by one.

    The rule is exact-duplicate removal among *siblings*, keyed on the whole
    canonicalised subtree and applied recursively.  It therefore collapses both
    a straight run (A A A A) and a cycle (A B A B A B), while two genuinely
    different list rows -- which differ at least in their bounds -- are kept,
    since their canonical forms differ.

    Returns a new screen dict; the input is not modified.
    """
    if not isinstance(screen, dict):
        return screen

    def dedupe(components):
        if not isinstance(components, list):
            return components

        seen = set()
        out = []
        for comp in components:
            if not isinstance(comp, dict):
                # Non-dict entries cannot be scored anyway; keep them once.
                key = _canonical(comp)
                if key not in seen:
                    seen.add(key)
                    out.append(comp)
                continue

            key = _canonical(comp)
            if key in seen:
                continue
            seen.add(key)

            if isinstance(comp.get("children"), list):
                comp = dict(comp)
                comp["children"] = dedupe(comp["children"])
            out.append(comp)
        return out

    collapsed = dict(screen)
    if isinstance(collapsed.get("components"), list):
        collapsed["components"] = dedupe(collapsed["components"])
    return collapsed
