"""
trace_index.py

Joins RICO interaction-trace screens to the RICO layouts corpus.

Why this exists
---------------
The two RICO downloads number their screens independently:

  * ``data/layouts/<id>.json``  - the "UI Screenshots and View Hierarchies"
    corpus. One flat directory, ids 0..66260, assigned globally across the
    whole crawl. Our Qwen embeddings are generated from these files and are
    therefore keyed by this *global* id (1_qwen_generate_embeddings.py).

  * ``data/traces/filtered_traces/<app>/<trace>/gestures.json`` - the
    "Interaction Traces" download. Its keys are UI indices *within one
    recording session* and restart in every trace.

Reading a trace key as a global layouts id silently pairs each trace with
unrelated screens: verified 0 of 731 sampled trace screens matched
``layouts/<same-id>.json`` by activity name, and only 5,178 distinct ids are
reachable that way out of 66,261 embeddings.

The two corpora do contain the same screens, only renumbered, so they can be
joined on content. This module builds that join once and persists it.

Join key
--------
Primary: SHA-1 of the canonical JSON of the *entire* document::

    json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

Measured over 3,999 sampled trace screens, this yields 99.7% unique matches,
0.0% ambiguous, and it is the only scope with zero collisions on the layouts
side (66,261 distinct hashes for 66,261 files). Hashing raw file *bytes* does
not work at all (0% match): the two corpora serialize the same JSON with
different key order and whitespace, which is why canonicalization is required.

Fallback: SHA-1 of the canonical ``activity.root`` subtree, which drops
``request_id``. ``request_id`` survives the renumbering for 99.9% of screens,
so this only rescues the small remainder. It collapses 212 screens into
duplicate groups, hence fallback only.

Anything still unmatched is dropped and logged rather than guessed.

Assignment is injective by construction: a layouts id is consumed at most
once, so two trace screens can never alias onto the same embedding.
"""

import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# Bump when the hashing scheme changes so stale indexes can be detected.
INDEX_SCHEMA_VERSION = 1

DEFAULT_INDEX_PATH = "trace_layout_index.csv"
DEFAULT_LAYOUTS_DIR = "../rico-data/data/layouts"
DEFAULT_TRACES_DIR = "../rico-data/data/traces/filtered_traces"

CSV_FIELDS = ["app", "trace", "screen_key", "layout_id", "match_method"]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def doc_hashes(path):
    """
    Return (full_hash, root_hash) for one view-hierarchy JSON file,
    or None if it cannot be read or carries no view hierarchy.
    """
    try:
        with open(path, "rb") as f:
            doc = json.loads(f.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    activity = doc.get("activity") or {}
    root = activity.get("root") if isinstance(activity, dict) else None
    if root is None:
        return None
    return _sha1(_canon(doc)), _sha1(_canon(root))


# ---------------------------------------------------------------------------
# Collecting both sides
# ---------------------------------------------------------------------------
def collect_layouts(layouts_dir, workers=16, progress=None):
    """Map each layouts screen id to its (full_hash, root_hash)."""
    files = sorted(Path(layouts_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No layout JSON files found in '{layouts_dir}'")
    if progress:
        progress(f"Hashing {len(files)} layout files from '{layouts_dir}' ...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(doc_hashes, files))

    by_id = {}
    unreadable = []
    for path, res in zip(files, results):
        try:
            sid = int(path.stem)
        except ValueError:
            continue
        if res is None:
            unreadable.append(path.name)
        else:
            by_id[sid] = res
    return by_id, unreadable


def collect_trace_screens(traces_dir, workers=16, progress=None):
    """
    Every gesture screen of every trace, as
    (app, trace, screen_key, (full_hash, root_hash) or None).
    """
    root = Path(traces_dir)
    if not root.exists():
        raise FileNotFoundError(f"Traces directory not found: {traces_dir}")

    screens = []
    for app_dir in sorted(root.iterdir()):
        if not app_dir.is_dir():
            continue
        for trace_dir in sorted(app_dir.iterdir()):
            if not trace_dir.is_dir() or not trace_dir.name.startswith("trace_"):
                continue
            gestures_path = trace_dir / "gestures.json"
            if not gestures_path.exists():
                continue
            try:
                with open(gestures_path, "r", encoding="utf-8") as f:
                    gestures = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            for key in gestures.keys():
                try:
                    screen_key = int(key)
                except ValueError:
                    continue
                screens.append((
                    app_dir.name,
                    trace_dir.name,
                    screen_key,
                    trace_dir / "view_hierarchies" / f"{key}.json",
                ))

    if progress:
        progress(f"Hashing {len(screens)} trace gesture screens from '{traces_dir}' ...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        hashes = list(ex.map(doc_hashes, [s[3] for s in screens]))
    return [(a, t, k, h) for (a, t, k, _), h in zip(screens, hashes)]


# ---------------------------------------------------------------------------
# Building the index
# ---------------------------------------------------------------------------
def build_index(layouts_dir=DEFAULT_LAYOUTS_DIR, traces_dir=DEFAULT_TRACES_DIR,
                workers=16, progress=None):
    """
    Build the trace-screen -> layouts-id assignment.

    Returns (rows, stats, dropped) where rows are dicts with CSV_FIELDS and
    dropped is a list of (app, trace, screen_key, reason).
    """
    layouts, layout_unreadable = collect_layouts(layouts_dir, workers, progress)
    trace_screens = collect_trace_screens(traces_dir, workers, progress)

    full_pool = defaultdict(list)
    root_pool = defaultdict(list)
    for sid, (h_full, h_root) in layouts.items():
        full_pool[h_full].append(sid)
        root_pool[h_root].append(sid)
    for pool in (full_pool, root_pool):
        for hashval in pool:
            pool[hashval].sort()

    rows = []
    dropped = []
    used = set()
    # Deterministic order so a rebuild produces an identical index.
    ordered = sorted(
        [s for s in trace_screens if s[3] is not None],
        key=lambda s: (s[0], s[1], s[2]),
    )
    for app, trace, key, hashes in trace_screens:
        if hashes is None:
            dropped.append((app, trace, key, "unhashable_trace_file"))

    def claim(pool, hashval):
        """Take the lowest still-unused layouts id carrying this hash."""
        for sid in pool.get(hashval, ()):
            if sid not in used:
                return sid
        return None

    # Pass 1: full-document hash. Pass 2: activity.root subtree.
    pending = []
    for app, trace, key, hashes in ordered:
        h_full, h_root = hashes
        sid = claim(full_pool, h_full)
        if sid is None:
            pending.append((app, trace, key, h_root))
            continue
        used.add(sid)
        rows.append({"app": app, "trace": trace, "screen_key": key,
                     "layout_id": sid, "match_method": "full_doc_hash"})

    for app, trace, key, h_root in pending:
        sid = claim(root_pool, h_root)
        if sid is None:
            dropped.append((app, trace, key, "no_layout_match"))
            continue
        used.add(sid)
        rows.append({"app": app, "trace": trace, "screen_key": key,
                     "layout_id": sid, "match_method": "root_hash"})

    rows.sort(key=lambda r: (r["app"], r["trace"], r["screen_key"]))

    # Injectivity is the whole point of `used`; assert it rather than trust it.
    layout_ids = [r["layout_id"] for r in rows]
    assert len(layout_ids) == len(set(layout_ids)), \
        "index is not injective - two trace screens share a layouts id"

    n_trace = len(trace_screens)
    stats = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "layout_files": len(layouts),
        "layout_files_unreadable": len(layout_unreadable),
        "layout_distinct_full_hashes": len(full_pool),
        "layout_distinct_root_hashes": len(root_pool),
        "trace_screens": n_trace,
        "mapped": len(rows),
        "mapped_full_doc_hash": sum(1 for r in rows if r["match_method"] == "full_doc_hash"),
        "mapped_root_hash": sum(1 for r in rows if r["match_method"] == "root_hash"),
        "dropped": len(dropped),
        "dropped_unhashable": sum(1 for d in dropped if d[3] == "unhashable_trace_file"),
        "dropped_no_match": sum(1 for d in dropped if d[3] == "no_layout_match"),
        "coverage": (len(rows) / n_trace) if n_trace else 0.0,
        "layout_ids_consumed": len(used),
    }
    return rows, stats, dropped


def write_index(rows, stats, dropped, out_path=DEFAULT_INDEX_PATH,
                layouts_dir=DEFAULT_LAYOUTS_DIR, traces_dir=DEFAULT_TRACES_DIR):
    """Write the index CSV, a meta sidecar, and a dropped-screen audit CSV."""
    out = Path(out_path)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    meta = dict(stats)
    meta.update({
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layouts_dir": str(layouts_dir),
        "traces_dir": str(traces_dir),
        "hash_algorithm": "sha1",
        "canonicalization": 'json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)',
        "primary_key": "entire document",
        "fallback_key": "activity.root subtree",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    })
    meta_path = out.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    dropped_path = out.with_name(out.stem + "_dropped.csv")
    with open(dropped_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["app", "trace", "screen_key", "reason"])
        writer.writerows(dropped)

    return out, meta_path, dropped_path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_index(path=DEFAULT_INDEX_PATH):
    """
    Load the index as {(app, trace, screen_key): layout_id}.

    Raises FileNotFoundError with build instructions if it is missing. The
    scripts deliberately refuse to run without it: reading a trace key as a
    layouts id is silently wrong, and a fallback would reintroduce that bug.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Trace->layout index not found at '{path}'.\n"
            f"RICO trace screen keys and layouts screen ids are different "
            f"namespaces and cannot be used interchangeably.\n"
            f"Build the index first:\n"
            f"    python 0_build_trace_index.py"
        )
    index = {}
    with open(p, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            index[(row["app"], row["trace"], int(row["screen_key"]))] = int(row["layout_id"])
    if not index:
        raise ValueError(f"Trace->layout index at '{path}' is empty.")
    return index


def layout_id_to_app(index):
    """Invert the index: layouts id -> app package that the trace came from."""
    return {layout_id: app for (app, _trace, _key), layout_id in index.items()}
