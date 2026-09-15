"""
screen_observability.py

Per-screen "target observability" scan of the RICO layouts corpus, and the
whitelist / blacklist decision derived from it.

The reconstruction task decodes an embedding back into the cleaned view
hierarchy (utils.clean_screen).  For some screens that hierarchy cannot
describe what the screenshot shows: the pixels are drawn by a component that
renders its own content outside the Android view system -- a WebView, a
video or OpenGL surface, a game engine, an ad container, a map.  The cleaned
JSON then holds one opaque box where the screenshot holds a whole page.  Such
screens are a poor reconstruction target and a poor training signal, and this
module measures how much of every screen is hidden that way.

Definitions
-----------
opaque class
    A class that draws outside the view system.  Matched by family
    (OPAQUE_FAMILIES) on the full class path, case-insensitively.

opaque subtree
    A cleaned component of an opaque class whose *entire* subtree carries no
    `text` and no `content-desc`.  The "no text anywhere below" condition is
    what separates a WebView whose page content the accessibility layer did
    expose (children with text -> the JSON does describe the page) from an ad
    container wrapped in obfuscated framework views (children, but nothing
    readable -> still opaque).  A childless opaque node is the common case.

opaque coverage
    Union area of all opaque subtrees on the screen, as a fraction of the
    device frame.  Computed on the cleaner's 1-100 grid, so it is exact for
    the binned boxes the decoder actually has to reproduce.  Union rather than
    sum: two overlapping opaque boxes do not count twice.

decision
    blacklist  if the cleaned screen has no components at all ("empty"), or
               opaque coverage >= threshold ("opaque");
    whitelist  otherwise.
    The threshold is a parameter (default 0.5); the continuous coverage is
    kept in the output so the cut can be revisited without rescanning.

Everything here is pure CPU: orjson, numpy, pandas and utils.clean_node.  The
worker function is module-level so a ProcessPoolExecutor can pickle it on
Windows as well as on POSIX.

Usage
-----
    python screen_observability.py                       # full corpus -> screen_observability.csv
    python screen_observability.py --threshold 0.5 --workers 8
    python screen_observability.py --layouts_dir data/rico-examples --out /tmp/x.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

import utils


# =============================================================================
# Opaque class families
# =============================================================================
# Each family: substrings matched against the lower-cased *full* class path,
# optional package prefixes, and suffixes matched as CamelCase token tuples
# against the short class name (see camel_tokens).  Order matters only for
# attribution (the first matching family wins); the coverage decision uses
# the union over all families.
#
# The lists were checked against a census of every class name in the 66,261
# RICO layouts (see _results_screen_observability.ipynb); the frequent members
# of each family are named in the comments.
OPAQUE_FAMILIES: dict[str, dict[str, tuple]] = {
    # Ad containers.  Internally a WebView, typically a banner strip; area
    # weighting keeps a small banner from blacklisting a screen.  Frequent:
    # gms.ads.AdView (8.9k screens) and its obfuscated internals, MoPubView,
    # facebook.ads.AdView, PublisherAdView, amazon.device.ads.AdLayout,
    # gms.ads.internal.overlay.* (full-screen interstitials).
    "ad": {
        "substrings": (
            "gms.ads", "com.google.ads", ".ads.", ".advert", "advertising",
            "com.mopub", "mraid", "inmobi", "millennialmedia", "chartboost",
            "applovin", "amazon.device.ads", "smaato", "startapp", "vungle",
            "adcolony", "tapjoy", "ironsource", "appnext", "airpush",
            "leadbolt", "appodeal", "admarvel", "vervewireless", "freewheel",
        ),
        "suffixes": (("ad", "view"), ("ad", "layout"), ("admob", "view"), ("ad", "container")),
    },
    # Map tiles are rendered by a native surface; overlaid controls are real
    # views and stay visible to the hierarchy.  Frequent: the obfuscated
    # com.google.maps.api.android.lib6.* and maps.* classes, gms.maps.MapView,
    # HERE / Nokia map canvases.
    "map": {
        "substrings": (
            "gms.maps", "google.maps", "maps.api.android", "android.maps",
            "apps.gmm.map", "here.mapcanvas", "nokia.maps", "mapbox",
            "osmdroid", "osmand",
        ),
        "prefixes": ("maps.",),
        "suffixes": (("map", "view"), ("map", "canvas", "view")),
    },
    # Game / cross-platform engines drawing into a single surface.  Frequent:
    # UnityPlayer, AIRWindowSurfaceView, Cocos2dxGLSurfaceView, libGDX,
    # Corona, AndEngine.
    "engine": {
        "substrings": (
            "unity3d.player", "cocos2dx", "io.flutter", "badlogic.gdx", "libsdl",
            "adobe.air", "adobe.flashruntime", "qtproject", "ansca.corona",
            "andengine", "godot", "epicgames", "yoyogames", "unreal",
            "org.love2d", "org.renpy", "marmalade", "gameclosure", "com.defold",
        ),
        "suffixes": (),
    },
    # Web containers, including the Cordova / Crosswalk / Tencent X5 variants.
    # Frequent: android.webkit.WebView (5.6k screens), BrowserWebView,
    # facebook BasicWebView, cordova SystemWebView, XWalkViewBridge.
    # Chrome's own browser chrome (org.chromium.chrome.browser.*) is native
    # and deliberately NOT matched; its render surface is caught below.
    "webview": {
        "substrings": ("webview", "xwalk", "cordova.engine", "crosswalk", "tencent.smtt"),
        "suffixes": (("web", "view"),),
    },
    # Raw surfaces: OpenGL, video, camera preview, browser compositors.
    # Frequent: SurfaceView, TextureView, VideoView, GLRootView,
    # YouTubePlayerView, GLSurfaceView.
    "surface": {
        "substrings": ("youtube.player", "org.mozilla.gecko.gfx"),
        "suffixes": (
            ("surface", "view"), ("texture", "view"), ("video", "view"),
            ("gl", "root", "view"), ("gl", "view"),
            ("content", "view", "render", "view"), ("compositor", "view"),
        ),
    },
}
FAMILY_ORDER = tuple(OPAQUE_FAMILIES)

# The cleaner bins every coordinate to 1..100 of the device frame, so the
# frame itself is [1, 1, 100, 100]: 99 x 99 unit cells.
GRID = 100
FRAME_CELLS = float((GRID - 1) * (GRID - 1))

DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_CLASS_SCREENS = 5
DEFAULT_LAYOUTS_DIR = Path("../rico-data/data/layouts")
DEFAULT_OUT = Path("screen_observability.csv")


# =============================================================================
# Class helpers
# =============================================================================
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def camel_tokens(short_name: str) -> tuple[str, ...]:
    """
    Lower-cased CamelCase tokens of a short class name:
        "GLSurfaceView"           -> ("gl", "surface", "view")
        "MraidBridge$MraidWebView" -> ("mraid", "web", "view")   (innermost class)
    Suffixes are matched on these tokens rather than on raw characters, so
    "AdView" matches NowPlayingAdView but not HeadView / ReadView / LoadView,
    and "MapView" matches OTCMapView but not BitmapView.
    """
    inner = short_name.rsplit("$", 1)[-1]
    return tuple(t.lower() for t in _CAMEL_RE.findall(inner))


def family_of(class_name: str | None) -> str | None:
    """Opaque family of a class path, or None if it is an ordinary view."""
    if not class_name:
        return None
    full = class_name.lower()
    tokens = camel_tokens(class_name.rsplit(".", 1)[-1])
    for family in FAMILY_ORDER:
        spec = OPAQUE_FAMILIES[family]
        if any(s in full for s in spec["substrings"]):
            return family
        if any(full.startswith(p) for p in spec.get("prefixes", ())):
            return family
        for suffix in spec["suffixes"]:
            if len(tokens) >= len(suffix) and tokens[-len(suffix):] == suffix:
                return family
    return None


def _short(class_name: str | None) -> str:
    return (class_name or "").rsplit(".", 1)[-1]


def _is_image_class(class_name: str | None) -> bool:
    return "image" in _short(class_name).lower()


def _has_label(node: dict) -> bool:
    return bool(node.get("text")) or bool(node.get("content-desc"))


def subtree_has_text(node: dict) -> bool:
    """True if this component or any descendant carries text / content-desc."""
    stack = [node]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        if _has_label(current):
            return True
        stack.extend(current.get("children") or [])
    return False


def box_cells(bounds) -> tuple[slice, slice] | None:
    """Grid slices covered by a cleaned [x1, y1, x2, y2] box, or None."""
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in bounds)
    except (TypeError, ValueError):
        return None
    x1, x2 = max(1, min(GRID, x1)), max(1, min(GRID, x2))
    y1, y2 = max(1, min(GRID, y1)), max(1, min(GRID, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return slice(y1 - 1, y2 - 1), slice(x1 - 1, x2 - 1)


def box_fraction(bounds) -> float:
    cells = box_cells(bounds)
    if cells is None:
        return 0.0
    rows, cols = cells
    return (rows.stop - rows.start) * (cols.stop - cols.start) / FRAME_CELLS


# =============================================================================
# Per-screen analysis
# =============================================================================
def analyse_components(components: list) -> dict:
    """
    Walk a cleaned component forest once and return every per-screen measure
    that depends on it.  Pure function of the cleaned tree, so it can be unit
    tested on synthetic screens.
    """
    opaque_grid = np.zeros((GRID, GRID), dtype=bool)
    family_grids = {family: np.zeros((GRID, GRID), dtype=bool) for family in FAMILY_ORDER}
    leaf_grid = np.zeros((GRID, GRID), dtype=bool)

    n_components = 0
    n_leaves = 0
    n_text = 0
    max_depth = 0
    n_opaque_subtrees = 0
    n_opaque_exempt = 0          # opaque-class nodes kept because their subtree has text
    largest_opaque = 0.0
    largest_opaque_class = None
    largest_image_leaf = 0.0
    largest_component = 0.0
    class_counter: Counter = Counter()

    stack = [(node, 1) for node in reversed(components)]
    while stack:
        node, depth = stack.pop()
        if not isinstance(node, dict):
            continue
        n_components += 1
        max_depth = max(max_depth, depth)
        cls = node.get("class")
        class_counter[cls or "<no-class>"] += 1
        if _has_label(node):
            n_text += 1
        children = node.get("children") or []
        bounds = node.get("bounds")
        fraction = box_fraction(bounds)
        largest_component = max(largest_component, fraction)

        family = family_of(cls)
        if family is not None:
            if subtree_has_text(node):
                n_opaque_exempt += 1
            else:
                # Whole subtree is one opaque region: paint it, do not descend.
                n_opaque_subtrees += 1
                cells = box_cells(bounds)
                if cells is not None:
                    opaque_grid[cells] = True
                    family_grids[family][cells] = True
                if fraction > largest_opaque:
                    largest_opaque = fraction
                    largest_opaque_class = cls
                # Descendants still count as components (they are in the
                # target JSON), but they are not walked for further opaque
                # regions -- they lie inside this one anyway.
                inner = list(children)
                while inner:
                    child = inner.pop()
                    if isinstance(child, dict):
                        n_components += 1
                        class_counter[child.get("class") or "<no-class>"] += 1
                        if not child.get("children"):
                            n_leaves += 1
                        inner.extend(child.get("children") or [])
                continue

        if not children:
            n_leaves += 1
            cells = box_cells(bounds)
            if cells is not None:
                leaf_grid[cells] = True
            if (
                _is_image_class(cls)
                and not _has_label(node)
                and not node.get("resource-id")
            ):
                largest_image_leaf = max(largest_image_leaf, fraction)
        else:
            stack.extend((child, depth + 1) for child in reversed(children))

    row = {
        "clean_component_count": n_components,
        "clean_leaf_count": n_leaves,
        "text_component_count": n_text,
        "max_depth": max_depth,
        "opaque_coverage": float(opaque_grid.sum() / FRAME_CELLS),
        "opaque_subtree_count": n_opaque_subtrees,
        "opaque_exempt_count": n_opaque_exempt,
        "largest_opaque_fraction": largest_opaque,
        "largest_opaque_class": largest_opaque_class,
        "leaf_coverage": float(leaf_grid.sum() / FRAME_CELLS),
        "largest_component_fraction": largest_component,
        "largest_image_leaf_fraction": largest_image_leaf,
    }
    for family in FAMILY_ORDER:
        row[f"cov_{family}"] = float(family_grids[family].sum() / FRAME_CELLS)
    row["_class_counter"] = class_counter
    return row


def analyse_raw(root) -> dict:
    """Counts over the raw (uncleaned) hierarchy, for the raw-vs-clean note."""
    n_nodes = 0
    n_opaque = 0
    n_opaque_visible = 0
    stack = [root] if isinstance(root, dict) else []
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        n_nodes += 1
        if family_of(node.get("class")) is not None:
            n_opaque += 1
            if utils.is_visible(node) and utils.area_ok(node.get("bounds")):
                n_opaque_visible += 1
        stack.extend(node.get("children") or [])
    return {
        "raw_node_count": n_nodes,
        "raw_opaque_nodes": n_opaque,
        "raw_opaque_visible_nodes": n_opaque_visible,
    }


def scan_screen(json_path: str | Path) -> dict:
    """One raw RICO layout -> one flat row (plus the per-screen class Counter)."""
    json_path = Path(json_path)
    screen_id = int(json_path.stem)
    with open(json_path, "rb") as fh:
        raw = orjson.loads(fh.read())

    app, activity = utils.parse_activity_name(raw)
    root = (raw.get("activity") or {}).get("root")
    components = utils.clean_node(root, *utils.device_frame(root)) if root else []

    row = {"screen_id": screen_id, "app": app, "activity": activity}
    row.update(analyse_raw(root))
    row.update(analyse_components(components))
    return row


# =============================================================================
# Corpus scan
# =============================================================================
def layout_paths(layouts_dir: str | Path) -> list[Path]:
    paths = [p for p in Path(layouts_dir).glob("*.json") if p.stem.isdigit()]
    return sorted(paths, key=lambda p: int(p.stem))


def scan_dataset(paths, workers: int | None = None, chunksize: int = 64, progress=None) -> pd.DataFrame:
    """
    Scan every path with a process pool.  Returns one row per screen; the
    per-screen class Counter is kept in the private `_class_counter` column
    until add_rare_class_share() has consumed it.
    """
    paths = [str(p) for p in paths]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        iterator = pool.map(scan_screen, paths, chunksize=chunksize)
        if progress is not None:
            iterator = progress(iterator, total=len(paths))
        rows.extend(iterator)
    return pd.DataFrame(rows).set_index("screen_id").sort_index()


def add_rare_class_share(frame: pd.DataFrame, min_screens: int = DEFAULT_MIN_CLASS_SCREENS) -> pd.DataFrame:
    """
    Share of a screen's cleaned components whose class name occurs in fewer
    than `min_screens` screens of the scanned corpus.  A decoder has to spell
    every class name; rare custom classes are a difficulty of their own, kept
    separate from opaqueness so the two are not confused in the analysis.
    """
    counters = frame["_class_counter"]
    screens_per_class: Counter = Counter()
    for counter in counters:
        screens_per_class.update(counter.keys())
    rare = {cls for cls, n in screens_per_class.items() if n < min_screens}

    shares = []
    distinct = []
    for counter in counters:
        total = sum(counter.values())
        rare_count = sum(n for cls, n in counter.items() if cls in rare)
        shares.append(rare_count / total if total else 0.0)
        distinct.append(len(counter))

    out = frame.drop(columns=["_class_counter"]).copy()
    out["clean_distinct_classes"] = distinct
    out["rare_class_share"] = shares
    out.attrs["distinct_classes_in_corpus"] = len(screens_per_class)
    out.attrs["rare_classes_in_corpus"] = len(rare)
    out.attrs["min_class_screens"] = min_screens
    return out


def assign_lists(frame: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Add the `list` and `reason` columns for one threshold."""
    out = frame.copy()
    empty = out["clean_component_count"].eq(0)
    opaque = out["opaque_coverage"].ge(threshold) & ~empty
    out["reason"] = np.where(empty, "empty", np.where(opaque, "opaque", ""))
    out["list"] = np.where(empty | opaque, "blacklist", "whitelist")
    out.attrs["threshold"] = threshold
    return out


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_metadata(frame: pd.DataFrame, layouts_dir, threshold: float, elapsed: float | None = None) -> dict:
    """Provenance record written next to the CSV (mirrors trace_layout_index.meta.json)."""
    here = Path(__file__).resolve().parent
    counts = frame["list"].value_counts()
    reasons = frame["reason"].value_counts()
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layouts_dir": str(layouts_dir),
        "screens": int(len(frame)),
        "threshold": threshold,
        "whitelist": int(counts.get("whitelist", 0)),
        "blacklist": int(counts.get("blacklist", 0)),
        "blacklist_empty": int(reasons.get("empty", 0)),
        "blacklist_opaque": int(reasons.get("opaque", 0)),
        "min_class_screens": frame.attrs.get("min_class_screens"),
        "distinct_classes_in_corpus": frame.attrs.get("distinct_classes_in_corpus"),
        "rare_classes_in_corpus": frame.attrs.get("rare_classes_in_corpus"),
        "opaque_families": {k: dict(v) for k, v in OPAQUE_FAMILIES.items()},
        "utils_sha256": _sha256(here / "utils.py"),
        "module_sha256": _sha256(here / "screen_observability.py"),
        "elapsed_seconds": None if elapsed is None else round(elapsed, 1),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def run_scan(layouts_dir=DEFAULT_LAYOUTS_DIR, threshold=DEFAULT_THRESHOLD,
             min_class_screens=DEFAULT_MIN_CLASS_SCREENS, workers=None,
             progress=None) -> tuple[pd.DataFrame, dict]:
    """Full pipeline: scan -> rare-class share -> lists.  Returns (frame, meta)."""
    started = time.time()
    paths = layout_paths(layouts_dir)
    if not paths:
        raise FileNotFoundError(f"No <id>.json layouts under {layouts_dir}")
    frame = scan_dataset(paths, workers=workers, progress=progress)
    frame = add_rare_class_share(frame, min_screens=min_class_screens)
    frame = assign_lists(frame, threshold=threshold)
    meta = scan_metadata(frame, layouts_dir, threshold, elapsed=time.time() - started)
    return frame, meta


def write_outputs(frame: pd.DataFrame, meta: dict, out_csv: str | Path) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, float_format="%.6g")
    out_csv.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_csv


def load_scan(csv_path: str | Path = DEFAULT_OUT) -> pd.DataFrame:
    """Read a scan back with the same dtypes the scan produced."""
    frame = pd.read_csv(csv_path, index_col="screen_id", keep_default_na=False,
                        dtype={"app": str, "activity": str, "largest_opaque_class": str,
                               "list": str, "reason": str})
    frame["largest_opaque_class"] = frame["largest_opaque_class"].replace("", np.nan)
    return frame


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--layouts_dir", default=str(DEFAULT_LAYOUTS_DIR),
                        help=f"Raw RICO layouts <id>.json (default: {DEFAULT_LAYOUTS_DIR})")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Output CSV; a .meta.json is written next to it (default: {DEFAULT_OUT})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Opaque coverage at or above which a screen is blacklisted (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--min_class_screens", type=int, default=DEFAULT_MIN_CLASS_SCREENS,
                        help="A class seen in fewer screens than this counts as rare (default: 5)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes (default: all cores)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from tqdm import tqdm

    frame, meta = run_scan(
        layouts_dir=args.layouts_dir, threshold=args.threshold,
        min_class_screens=args.min_class_screens, workers=args.workers,
        progress=lambda it, total: tqdm(it, total=total, desc="Scanning screens"),
    )
    out = write_outputs(frame, meta, args.out)
    print(f"screens        : {meta['screens']:,}")
    print(f"whitelist      : {meta['whitelist']:,}")
    print(f"blacklist      : {meta['blacklist']:,}  "
          f"(empty {meta['blacklist_empty']:,}, opaque {meta['blacklist_opaque']:,}) "
          f"at threshold {meta['threshold']}")
    print(f"written        : {out} (+ {out.with_suffix('.meta.json').name})")
    print(f"elapsed        : {meta['elapsed_seconds']} s")


if __name__ == "__main__":
    sys.exit(main())
