import os
import json
import argparse
import pandas as pd

from utils import Config, parse_rico_json, decode_layout_tokens, decode_json_tokens

cfg = Config()


# =============================================================================
# Main Execution
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Test RICO Data Parsing Logic.")
    parser.add_argument("--index", type=int, required=True, help="Index of the UI in the CSV file")
    parser.add_argument("--csv_path", type=str, default="../rico-data/embeddings/embeddings_qwen.csv",
                        help="Path to embeddings CSV")

    args = parser.parse_args()

    print(f"[*] Loading CSV from: {args.csv_path}")
    if not os.path.exists(args.csv_path):
        print(f"[-] Error: Cannot find CSV at {args.csv_path}")
        return

    df = pd.read_csv(args.csv_path, names=["json-path", "embedding"], header=None)

    if args.index >= len(df) or args.index < 0:
        print(f"[-] Error: Index {args.index} is out of bounds (Dataset size: {len(df)}).")
        return

    row = df.iloc[args.index]
    json_path = row["json-path"].replace("/lustre/scratch/sgjolieb-screen2vec", "..")

    print(f"[*] Extracting data for index {args.index} -> {json_path}\n")

    layout_tokens, struct_tokens = parse_rico_json(json_path, cfg)

    if layout_tokens is None or struct_tokens is None:
        print("[-] Error parsing JSON: tokens are None")
        return

    # Load raw data for reporting
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"[-] Error loading raw JSON: {e}")
        return

    # Decode tokens back to readable format
    decoded_layout = decode_layout_tokens(layout_tokens, cfg)
    decoded_struct = decode_json_tokens(struct_tokens, cfg)

    raw_json_str = json.dumps(raw_data, indent=2)

    # ---------------------------------------------------------
    # PRINTING RESULTS
    # ---------------------------------------------------------
    print("=" * 100)
    print(" 1. ORIGINAL RAW JSON (Truncated to 1000 chars)")
    print("=" * 100)
    print(raw_json_str[:1000] + ("\n... [TRUNCATED]" if len(raw_json_str) > 1000 else ""))

    print("\n" + "=" * 100)
    print(" 2. STRUCTURAL MODEL INPUT")
    print("=" * 100)
    print(f"Token Array (Length {len(struct_tokens)}):")
    print(f"{str(struct_tokens[:30])[:-1]} ... ]")
    print("\nDecoded Structure (What the model 'sees'):")
    print(decoded_struct[:800] + ("\n... [TRUNCATED]" if len(decoded_struct) > 800 else ""))

    print("\n" + "=" * 100)
    print(" 3. LAYOUT MODEL INPUT")
    print("=" * 100)
    print(f"Token Array (Length {len(layout_tokens)}):")
    print(f"{str(layout_tokens[:30])[:-1]} ... ]")
    print("\nDecoded Layout Bounding Boxes (First 5 elements):")
    print(json.dumps(decoded_layout[:5], indent=2))

    if len(layout_tokens) <= 2:
        print("\n[!] WARNING: The layout token array is empty [SOS, EOS]. This UI has no valid bounds.")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()