#!/usr/bin/env python3
"""
Script to apply PCA to pre-computed Qwen3-VL embeddings to reduce dimensions to 64.
"""

import os
import sys
import glob
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA


def load_single_embedding(path):
    """Loads a single embedding from a compressed .npz file."""
    with np.load(path) as data:
        return data["embedding"].astype(np.float32)


def save_single_embedding(args_tuple):
    """Saves a single transformed embedding as a compressed .npz file."""
    path, embedding = args_tuple
    np.savez_compressed(path, embedding=embedding.astype(np.float16))


def main():
    parser = argparse.ArgumentParser(description="Reduce embedding dimensions to 64 using PCA.")
    parser.add_argument("--src", type=str, default="embeddings", help="Source directory of embeddings.")
    parser.add_argument("--dst", type=str, default="embeddings_64", help="Destination directory.")
    parser.add_argument("--n-components", type=int, default=64, help="Number of components for PCA.")
    args = parser.parse_args()

    if not os.path.exists(args.src):
        print(f"Error: Source directory '{args.src}' does not exist.")
        sys.exit(1)

    os.makedirs(args.dst, exist_ok=True)

    # Sort files numerically
    def numerical_sort_key(path):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(name)
        except ValueError:
            return name

    npz_paths = sorted(glob.glob(os.path.join(args.src, "*.npz")), key=numerical_sort_key)
    N = len(npz_paths)

    if N == 0:
        print(f"Error: No .npz files found in '{args.src}'.")
        sys.exit(1)

    print(f"Loading {N} embeddings in parallel...")
    with ThreadPoolExecutor() as executor:
        embeddings = list(tqdm(executor.map(load_single_embedding, npz_paths), total=N, desc="Loading"))

    X = np.stack(embeddings)  # shape: (N, 4096)

    # Run the PCA
    n_comp = args.n_components
    print(f"Applying PCA (fitting {n_comp} components)...")
    pca = PCA(n_components=n_comp, random_state=42)
    X_reduced = pca.fit_transform(X)

    print(f"Saving reduced embeddings to '{args.dst}'...")
    save_tasks = []
    for path, emb_reduced in zip(npz_paths, X_reduced):
        dest_path = os.path.join(args.dst, os.path.basename(path))
        save_tasks.append((dest_path, emb_reduced))

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(save_single_embedding, save_tasks), total=len(save_tasks), desc="Saving"))

    print(f"Successfully processed and saved {N} embeddings.")


if __name__ == "__main__":
    main()
