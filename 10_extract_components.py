"""
10_extract_components.py

Step 1 of the GUI *component* prediction pipeline (Screen2Vec paper, Table 2).

Walks the RICO screens, cleans each one with utils.clean_screen() -- the same
representation the screen-level pipeline embeds in 1_qwen_generate_embeddings.py
-- and flattens it into an ordered list of GUI components. Every component keeps
its text, its Screen2Vec class label (paper Table 1), its binned bounds, and its
parent index, which is everything scripts 12/13 need to rebuild the EUCLIDEAN
and HIERARCHICAL context of any component.

The union of all component texts is the prediction vocabulary, the counterpart
of screen2vec-master/UI_embedding/extract_vocab.py. Top-k%% accuracy in Table 2
is measured against this vocabulary, so its size is reported in meta.json.

Outputs (into --out_dir, default component_data/):
    components.jsonl.gz  one screen per line:
                         {"id", "app", "c": [[text_idx, cls, x1,y1,x2,y2, parent], ...]}
    vocab.json           list of unique texts; index 0 is "" (never a target)
    meta.json            counts and settings of this extraction run

Usage:
    python 10_extract_components.py
    python 10_extract_components.py --layouts_dir ../rico-data/data/layouts \\
        --out_dir component_data --n_jobs -1
"""

import argparse
import json
import os
import time
from glob import glob

from joblib import Parallel, delayed
from tqdm import tqdm

from component_utils import flatten_components, write_screens
from utils import clean_screen


# ============================================================================
# 1. Per-screen extraction
# ============================================================================
def extract_one(json_path):
    """Clean and flatten one RICO screen.

    Returns (screen_id, app, [(text, cls, x1, y1, x2, y2, parent), ...]) or None
    when the file cannot be parsed or holds no components.
    """
    try:
        screen = clean_screen(json_path)
    except Exception:
        return None

    comps = flatten_components(screen.get("components") or [])
    if not comps:
        return None

    try:
        screen_id = int(os.path.splitext(os.path.basename(json_path))[0])
    except ValueError:
        return None

    rows = []
    for c in comps:
        b = c["bounds"]
        rows.append((c["text"], c["cls"], b[0], b[1], b[2], b[3], c["parent"]))
    return screen_id, screen.get("app") or "", rows


def extract_batch(paths):
    """Extract a chunk of screens in one worker (keeps task overhead low)."""
    return [r for r in (extract_one(p) for p in paths) if r is not None]


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Extract GUI components and the text vocabulary from RICO screens"
    )
    parser.add_argument(
        "--layouts_dir", type=str, default="../rico-data/data/layouts",
        help="Directory of raw RICO view hierarchies (*.json)",
    )
    parser.add_argument(
        "--out_dir", type=str, default="component_data",
        help="Where components.jsonl.gz / vocab.json / meta.json are written",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1,
        help="Parallel workers (default: -1, all cores)",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=256,
        help="Screens per worker task",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N screens (for smoke tests)",
    )
    args = parser.parse_args()

    start = time.time()

    # ---------------------------------------------------------------
    # 1. Collect input screens
    # ---------------------------------------------------------------
    print(f"\n[Step 1/3] Scanning '{args.layouts_dir}'...")
    json_files = sorted(glob(os.path.join(args.layouts_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in '{args.layouts_dir}'")
    if args.limit:
        json_files = json_files[:args.limit]
    print(f"  {len(json_files)} screens to extract")

    # ---------------------------------------------------------------
    # 2. Clean + flatten in parallel
    # ---------------------------------------------------------------
    print(f"\n[Step 2/3] Extracting components (n_jobs={args.n_jobs})...")
    batches = list(chunked(json_files, args.chunk_size))
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(extract_batch)(batch)
        for batch in tqdm(batches, desc="Extracting screens")
    )

    # ---------------------------------------------------------------
    # 3. Build the vocabulary and write the dataset
    # ---------------------------------------------------------------
    print("\n[Step 3/3] Building vocabulary and writing dataset...")
    # Index 0 is the empty string: components without text are context only.
    vocab = [""]
    text_to_index = {"": 0}
    screens = []
    total_components = 0
    total_text_components = 0

    for batch in results:
        for screen_id, app, rows in batch:
            encoded = []
            for text, cls, x1, y1, x2, y2, parent in rows:
                idx = text_to_index.get(text)
                if idx is None:
                    idx = len(vocab)
                    text_to_index[text] = idx
                    vocab.append(text)
                encoded.append([idx, cls, x1, y1, x2, y2, parent])
                if idx != 0:
                    total_text_components += 1
            total_components += len(encoded)
            screens.append({"id": screen_id, "app": app, "c": encoded})

    screens.sort(key=lambda s: s["id"])

    os.makedirs(args.out_dir, exist_ok=True)
    components_path = os.path.join(args.out_dir, "components.jsonl.gz")
    vocab_path = os.path.join(args.out_dir, "vocab.json")
    meta_path = os.path.join(args.out_dir, "meta.json")

    write_screens(components_path, screens)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    meta = {
        "layouts_dir": args.layouts_dir,
        "screens": len(screens),
        "components": total_components,
        "text_components": total_text_components,
        "vocab_size": len(vocab),
        "mean_components_per_screen": total_components / max(1, len(screens)),
        "mean_text_components_per_screen": total_text_components / max(1, len(screens)),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - start
    print(f"\n  Screens extracted        : {meta['screens']}")
    print(f"  Components               : {meta['components']}")
    print(f"  ...with non-empty text   : {meta['text_components']}")
    print(f"  Vocabulary size          : {meta['vocab_size']}")
    print(f"  Components per screen    : {meta['mean_components_per_screen']:.2f} "
          f"({meta['mean_text_components_per_screen']:.2f} with text)")
    print(f"\nWrote {components_path}")
    print(f"      {vocab_path}")
    print(f"      {meta_path}")
    print(f"Total extraction time: {elapsed:.1f}s")
    print("\nNext step — embed the vocabulary with:")
    print(f"  python 11_embed_component_vocab.py --data_dir {args.out_dir}")


if __name__ == "__main__":
    main()
