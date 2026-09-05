"""Shared, read-only helpers for the paper embedding-space notebooks."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
RICO_ROOT = PROJECT_ROOT.parent / "rico-data"
LAYOUTS_DIR = RICO_ROOT / "data" / "layouts"
TRACES_DIR = RICO_ROOT / "data" / "traces" / "filtered_traces"
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings_4096_avg"
DEFAULT_TRACE_INDEX = PROJECT_ROOT / "trace_layout_index.csv"
DEFAULT_ENRICO_TOPICS = PROJECT_ROOT / "enrico_design_topics.csv"

ACTIVITY_RE = re.compile(r'"activity_name"\s*:\s*"([^"]+)"')


def numeric_id(value: object) -> int:
    """Return a global RICO layout id as an integer."""
    return int(Path(str(value).replace("\\", "/")).stem)


def _embedding_path(embeddings_dir: Path, screen_id: int) -> Path:
    return embeddings_dir / f"{screen_id}.npz"


def available_embedding_ids(embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR) -> set[int]:
    return {numeric_id(path.stem) for path in embeddings_dir.glob("*.npz")}


def load_embedding_file(path: Path) -> np.ndarray:
    """Load one genuine N=1 embedding; never derive it from wider pooled tokens."""
    with np.load(path, allow_pickle=False) as archive:
        key = "embedding" if "embedding" in archive.files else archive.files[0]
        vector = np.asarray(archive[key], dtype=np.float32).squeeze()
    if vector.ndim != 1:
        raise ValueError(f"Expected one vector in {path}, found shape {vector.shape}")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"Invalid embedding norm in {path}: {norm}")
    return vector / norm


def load_embeddings(
    screen_ids: Iterable[int] | None = None,
    embeddings_dir: Path = DEFAULT_EMBEDDINGS_DIR,
    workers: int = 12,
    description: str = "Loading embeddings",
) -> tuple[list[int], np.ndarray]:
    """Load normalized embeddings in deterministic global-id order."""
    if screen_ids is None:
        paths = sorted(embeddings_dir.glob("*.npz"), key=lambda p: numeric_id(p.stem))
    else:
        ids = sorted({numeric_id(screen_id) for screen_id in screen_ids})
        paths = [_embedding_path(embeddings_dir, screen_id) for screen_id in ids]
        paths = [path for path in paths if path.exists()]
    if not paths:
        raise ValueError(f"No embeddings found in {embeddings_dir}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        vectors = list(tqdm(executor.map(load_embedding_file, paths), total=len(paths), desc=description))
    dimensions = {vector.size for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent embedding dimensions: {sorted(dimensions)}")
    return [numeric_id(path.stem) for path in paths], np.stack(vectors).astype(np.float32)


def fit_umap(
    embeddings: np.ndarray,
    random_state: int = 42,
    n_neighbors: int = 30,
    min_dist: float = 0.15,
    precomponents: int = 50,
) -> tuple[np.ndarray, PCA]:
    """PCA-denoise normalized vectors, then compute a deterministic 2-D UMAP."""
    n_components = min(precomponents, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    reduced = pca.fit_transform(embeddings)
    coordinates = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(embeddings) - 1),
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
        n_jobs=1,
    ).fit_transform(reduced)
    return coordinates, pca


def deterministic_sample(values: Iterable[int], size: int, random_state: int = 42) -> list[int]:
    values = np.asarray(sorted({numeric_id(value) for value in values}), dtype=np.int64)
    if len(values) <= size:
        return values.tolist()
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(values, size=size, replace=False)).tolist()


def package_from_layout(layout_path: Path) -> str | None:
    with layout_path.open("r", encoding="utf-8", errors="replace") as handle:
        prefix = handle.read(16_384)
    match = ACTIVITY_RE.search(prefix)
    return match.group(1).split("/", 1)[0] if match else None


def read_enrico_topics(path: Path = DEFAULT_ENRICO_TOPICS) -> pd.DataFrame:
    topics = pd.read_csv(path, dtype={"screen_id": "int64", "topic": "string"})
    required = {"screen_id", "topic"}
    if not required.issubset(topics.columns):
        raise ValueError(f"Expected columns {sorted(required)}, found {topics.columns.tolist()}")
    topics = topics.dropna(subset=["screen_id", "topic"])[["screen_id", "topic"]].copy()
    topics["topic"] = topics["topic"].str.strip()
    if topics["screen_id"].duplicated().any():
        raise ValueError("ENRICO topic file contains duplicate screen IDs")
    return topics


def hierarchy_metrics(layout_path: Path) -> dict[str, float | int | str]:
    """Extract interpretable raw view-hierarchy complexity measures."""
    with layout_path.open("r", encoding="utf-8", errors="replace") as handle:
        document = json.load(handle)
    root = document.get("activity", {}).get("root")
    if not isinstance(root, dict):
        raise ValueError(f"No activity.root in {layout_path}")

    node_count = 0
    max_depth = 0
    leaf_count = 0
    clickable_count = 0
    text_node_count = 0
    text_char_count = 0
    image_node_count = 0
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        node_count += 1
        max_depth = max(max_depth, depth)
        clickable_count += int(bool(node.get("clickable")))
        text = node.get("text")
        if isinstance(text, str) and text.strip():
            text_node_count += 1
            text_char_count += len(text.strip())
        class_name = str(node.get("class", "")).lower()
        image_node_count += int("image" in class_name)
        children = [child for child in node.get("children", []) if isinstance(child, dict)]
        if children:
            stack.extend((child, depth + 1) for child in children)
        else:
            leaf_count += 1

    package = str(document.get("activity_name", "")).split("/", 1)[0] or None
    return {
        "screen_id": numeric_id(layout_path.stem),
        "package": package,
        "node_count": node_count,
        "max_depth": max_depth,
        "leaf_count": leaf_count,
        "clickable_count": clickable_count,
        "text_node_count": text_node_count,
        "text_char_count": text_char_count,
        "image_node_count": image_node_count,
        "clickable_fraction": clickable_count / node_count,
        "text_node_fraction": text_node_count / node_count,
    }


def load_hierarchy_metrics(
    screen_ids: Iterable[int],
    layouts_dir: Path = LAYOUTS_DIR,
    workers: int = 16,
) -> pd.DataFrame:
    paths = [layouts_dir / f"{numeric_id(screen_id)}.json" for screen_id in screen_ids]
    paths = [path for path in paths if path.exists()]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(tqdm(executor.map(hierarchy_metrics, paths), total=len(paths), desc="Reading hierarchies"))
    return pd.DataFrame(rows)


def load_trace_sequences(
    available_ids: set[int],
    trace_index_path: Path = DEFAULT_TRACE_INDEX,
    traces_dir: Path = TRACES_DIR,
    min_length: int = 3,
) -> list[dict[str, object]]:
    """Resolve per-trace keys through the mandatory global trace-layout index."""
    from trace_index import load_index

    index = load_index(trace_index_path)
    sequences: list[dict[str, object]] = []
    for app_dir in tqdm(sorted(traces_dir.iterdir()), desc="Reading traces"):
        if not app_dir.is_dir():
            continue
        for trace_dir in sorted(app_dir.iterdir()):
            gestures_path = trace_dir / "gestures.json"
            if not trace_dir.is_dir() or not gestures_path.exists():
                continue
            try:
                with gestures_path.open("r", encoding="utf-8") as handle:
                    gestures = json.load(handle)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            layout_ids = []
            for key in gestures.keys():
                try:
                    screen_key = int(key)
                except ValueError:
                    continue
                layout_id = index.get((app_dir.name, trace_dir.name, screen_key))
                if layout_id is not None and layout_id in available_ids:
                    layout_ids.append(layout_id)
            if len(layout_ids) >= min_length:
                sequences.append({"app": app_dir.name, "trace": trace_dir.name, "layout_ids": layout_ids})
    return sequences


def output_dir(name: str) -> Path:
    path = PROJECT_ROOT / "graphics" / name
    path.mkdir(parents=True, exist_ok=True)
    return path
