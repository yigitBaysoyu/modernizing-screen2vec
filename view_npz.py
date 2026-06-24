#!/usr/bin/env python3
"""
Script to load a pre-computed Qwen3-VL embedding .npz file by index
and display its shape and raw data.
"""

import os
import sys
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Load an embedding .npz file by index and display its shape and raw data."
    )
    parser.add_argument(
        "index",
        type=str,
        help="The index of the embedding (e.g., 1 for embeddings/1.npz)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="embeddings",
        help="Directory where the .npz files are stored (default: 'embeddings')",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the entire array without truncating the output",
    )
    args = parser.parse_args()

    # Construct the file path
    # If the user passed something like "1.npz", strip the extension
    index_clean = args.index
    if index_clean.endswith(".npz"):
        index_clean = index_clean[:-4]

    file_name = f"{index_clean}.npz"
    file_path = os.path.join(args.dir, file_name)

    print(f"Resolving index '{args.index}' to file: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        print(
            f"Please ensure you are running this in the correct folder, "
            f"or specify the embeddings directory using --dir option."
        )
        sys.exit(1)

    try:
        # Load the npz file
        with np.load(file_path) as data:
            keys = list(data.keys())
            print(f"Successfully loaded '{file_name}'")
            print(f"Keys found in npz: {keys}")

            if not keys:
                print("Warning: The loaded .npz file is empty.")
                sys.exit(0)

            # Check for the default 'embedding' key used in the codebase
            target_key = "embedding" if "embedding" in keys else keys[0]
            if target_key != "embedding":
                print(f"Note: Key 'embedding' not found. Using key '{target_key}' instead.")

            array_data = data[target_key]

            print("\n" + "=" * 50)
            print(f"Array Details for key '{target_key}':")
            print(f"  Shape: {array_data.shape}")
            print(f"  Dtype: {array_data.dtype}")
            print("=" * 50)

            # Option to print the full array or the truncated default NumPy print
            if args.full:
                np.set_printoptions(threshold=sys.maxsize)

            print("\nRaw Data:")
            print(array_data)

    except Exception as e:
        print(f"Error loading or parsing .npz file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
