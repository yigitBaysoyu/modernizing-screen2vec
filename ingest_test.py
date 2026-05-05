import os
import json
import argparse
import pandas as pd


# =============================================================================
# Configuration (Matches training script)
# =============================================================================
class Config:
    layout_vocab_size = 1100
    json_vocab_size = 2000
    max_layout_seq_len = 500
    max_json_seq_len = 1000
    pad_token_id = 0
    sos_token_id = 1
    eos_token_id = 2


cfg = Config()


# =============================================================================
# Core Traversal & Parsing (The EXACT logic the model uses)
# =============================================================================
def parse_rico_json(filepath, config):
    if not os.path.exists(filepath):
        return None, None, "File not found"

    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return None, None, "Invalid JSON"

    layout_tokens = [config.sos_token_id]
    json_tokens = [config.sos_token_id]

    def traverse(node):
        if not isinstance(node, dict): return

        # --- 1. Layout Extraction ---
        if 'bounds' in node:
            # Clamp coordinates and apply offset
            for val in node['bounds']:
                layout_tokens.append(max(0, min(999, int(val))) + 3)

            # Map UI Class to vocabulary
            ui_class = node.get('class', 'Unknown')
            layout_tokens.append((hash(ui_class) % 96) + 1003)

        # --- 2. Structure Extraction ---
        node_type = node.get('class', 'Unknown')
        json_tokens.append((hash(node_type) % (config.json_vocab_size - 3)) + 3)

        # --- 3. Recursive Call ---
        if 'children' in node:
            for child in node['children']:
                traverse(child)

    # Locate the actual UI root in the RICO format (The Fix)
    if 'activity' in data and 'root' in data['activity']:
        ui_root = data['activity']['root']
    else:
        ui_root = data

    traverse(ui_root)
    layout_tokens.append(config.eos_token_id)
    json_tokens.append(config.eos_token_id)

    return layout_tokens, json_tokens, data


# =============================================================================
# Decoding logic (To make tokens human-readable)
# =============================================================================
def decode_layout_tokens(token_list, config):
    decoded_boxes = []
    tokens = token_list[1:] if token_list and token_list[0] == config.sos_token_id else token_list
    for i in range(0, len(tokens), 5):
        chunk = tokens[i:i + 5]
        if config.eos_token_id in chunk: break
        if len(chunk) == 5:
            x1, y1, x2, y2 = [val - 3 for val in chunk[:4]]
            cls_id = chunk[4]
            decoded_boxes.append(
                {"class_id": f"UI_Element_{cls_id}", "bbox": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]})
    return decoded_boxes


def decode_json_tokens(token_list, config):
    tokens = token_list[1:] if token_list and token_list[0] == config.sos_token_id else token_list
    elements = []
    for t in tokens:
        if t == config.eos_token_id: break
        if t not in [config.pad_token_id]:
            node_type = "container" if t % 2 == 0 else "component"
            elements.append({"node_id": t, "type": node_type})
    return json.dumps({"type": "root", "children": elements}, indent=2)


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

    layout_tokens, struct_tokens, raw_data = parse_rico_json(json_path, cfg)

    if raw_data is None:
        print(f"[-] Error parsing JSON: {raw_data}")
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