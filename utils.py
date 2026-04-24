import os
import json
from glob import glob
import json

import orjson
import pandas as pd


def load_embeddings_from_csv(filepath="../rico-data/embeddings/embeddings_qwen.csv"):
    """Loads all pre-computed Qwen embeddings from the CSV file."""
    df = pd.read_csv(filepath, names=["json-path", "embedding"], header=None)

    # Convert string representation of list back to an actual Python list and fix paths
    df["embedding"] = [orjson.loads(x) for x in df["embedding"]]
    df["json-path"] = df["json-path"].apply(lambda x: x.replace("/lustre/scratch/sgjolieb-screen2vec", ".."))
    df["jpg-path"] = df["json-path"].apply(lambda x: x.replace(".json", ".jpg"))

    assert len(df) == 65795

    return df


def minify_json(json_data):
    """
    Removes all unnecessary whitespaces from a JSON object.
    This creates the most compact valid JSON representation.
    """
    return json.dumps(json_data, separators=(',', ':'))


def flatten_rico_to_text(node):
    """
    Recursively extracts meaningful UI elements from a RICO JSON node
    and flattens them into a clean, token-efficient text representation
    ideal for LLM embeddings.
    """
    if not node:
        return ""

    elements = []

    # Extract useful attributes commonly found in RICO dataset
    # We split the class to remove the verbose 'android.widget.' prefix
    ui_class = node.get('class', '').split('.')[-1]
    text = node.get('text', '')
    content_desc = node.get('content-desc', '')
    bounds = node.get('bounds', [])

    # Fix for content_desc being a list like [null] or ["string"] in RICO dataset
    if isinstance(content_desc, list):
        content_desc = " ".join([str(item) for item in content_desc if item is not None])
    elif content_desc is None:
        content_desc = ""

    # Clean strings (added isinstance checks to prevent future type errors)
    text = text.strip() if isinstance(text, str) and text else ""
    content_desc = content_desc.strip() if isinstance(content_desc, str) and content_desc else ""

    # Build a semantic string for the current node
    parts = []
    if ui_class:
        parts.append(f"[{ui_class}]")
    if text:
        parts.append(f"Text: '{text}'")
    elif content_desc:  # Fallback to content-desc if no text is present
        parts.append(f"Desc: '{content_desc}'")

    # Include bounding boxes if they are available
    if bounds and isinstance(bounds, list) and len(bounds) == 4:
        parts.append(f"Bounds: {bounds}")

    # Only keep nodes that have visible semantic value or positional bounds to save tokens
    # You can modify this condition if you want to keep structural empty layouts
    if parts and (text or content_desc or bounds):
        elements.append(" ".join(parts))

    # Recursively process children
    if 'children' in node and node['children']:
        for child in node['children']:
            child_text = flatten_rico_to_text(child)
            if child_text:
                elements.append(child_text)

    # Join with newlines so each element is clearly separated for the embedding model
    return "\n".join(elements)


def process_rico_data(json_data, flatten=True):
    """
    Processes RICO JSON data for embeddings based on the flatten switch.
    If flatten is False, it returns the minified JSON string.
    If flatten is True, it returns the flattened semantic text.
    """
    if not flatten:
        return minify_json(json_data)

    root_node = json_data.get('activity', {}).get('root', {})
    return flatten_rico_to_text(root_node)


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Mocking a small chunk of RICO UI data for demonstration
    sample_rico_data = json.load(open("data/rico-examples/0.json", "r"))

    print("--- Testing with flatten=False (Only Minify) ---")
    result_minified = process_rico_data(sample_rico_data, flatten=False)
    print(result_minified)
    print("\n")

    print("--- Testing with flatten=True (Flatten & Filter) ---")
    result_flattened = process_rico_data(sample_rico_data, flatten=True)
    print(result_flattened)