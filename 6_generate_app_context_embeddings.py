import argparse
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer

# Standard helper to split activity name from utils
def parse_activity_name(activity_name):
    if not activity_name:
        return "", ""
    if "/" in activity_name:
        pkg, activity = activity_name.split("/", 1)
    else:
        pkg, activity = "", activity_name
    return pkg, activity.split(".")[-1]

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    parser = argparse.ArgumentParser(
        description="Concatenate App Store metadata embeddings to projected screen embeddings."
    )
    parser.add_argument(
        "--embeddings_dir", type=str, default="embeddings_seq_128",
        help="Input embeddings directory containing screen_id.npz files",
    )
    parser.add_argument(
        "--output_dir", type=str, default="embeddings_seq_128_with_app",
        help="Output directory for concatenated embeddings",
    )
    parser.add_argument(
        "--csv_path", type=str, default="../rico-data/data/play_store_metadata/app_details.csv",
        help="Path to app_details.csv",
    )
    parser.add_argument(
        "--traces_dir", type=str, default="../rico-data/data/traces/filtered_traces",
        help="Path to RICO filtered_traces directory",
    )
    parser.add_argument(
        "--layouts_dir", type=str, default="../rico-data/data/layouts",
        help="Path to RICO layouts directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load app details CSV
    print(f"Loading app details from {args.csv_path}...")
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"App details CSV not found at {args.csv_path}")
    
    df = pd.read_csv(args.csv_path)
    # Map App Package Name to text: "Play Store Name - Category"
    app_text_map = {}
    for idx, row in df.iterrows():
        pkg = row.get("App Package Name")
        name = row.get("Play Store Name")
        category = row.get("Category")
        if pd.isna(name) or not str(name).strip():
            name = ""
        if pd.isna(category) or not str(category).strip():
            category = ""
        
        # Build text description
        if name and category:
            text = f"{name} - {category}"
        elif name:
            text = name
        elif category:
            text = category
        else:
            text = ""
        
        if pkg:
            app_text_map[str(pkg).strip()] = text

    print(f"Loaded {len(app_text_map)} app metadata entries.")

    # 2. Get all screen IDs from embeddings directory
    npz_files = sorted(Path(args.embeddings_dir).glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {args.embeddings_dir}")
    print(f"Found {len(npz_files)} projected screen embeddings.")

    # 3. Map screen IDs to app packages
    print("Mapping screen IDs to app package names from traces...")
    screen_to_app = {}
    traces_root = Path(args.traces_dir)
    if traces_root.exists():
        app_dirs = sorted(traces_root.iterdir())
        for app_dir in tqdm(app_dirs, desc="Parsing traces for app packages"):
            if not app_dir.is_dir():
                continue
            app_pkg = app_dir.name
            for trace_dir in sorted(app_dir.iterdir()):
                if not trace_dir.is_dir() or not trace_dir.name.startswith("trace_"):
                    continue
                gestures_path = trace_dir / "gestures.json"
                if not gestures_path.exists():
                    continue
                try:
                    with open(gestures_path, "r", encoding="utf-8") as f:
                        gestures = json.load(f)
                    for key in gestures.keys():
                        try:
                            sid = int(key)
                            screen_to_app[sid] = app_pkg
                        except ValueError:
                            continue
                except Exception:
                    continue

    print(f"Mapped {len(screen_to_app)} screens via traces.")

    # Resolve fallbacks for missing screens via layout JSON files in parallel
    from concurrent.futures import ThreadPoolExecutor

    def parse_single_layout(sid, layouts_dir):
        json_path = os.path.join(layouts_dir, f"{sid}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                pkg, _ = parse_activity_name(raw.get("activity_name"))
                if pkg:
                    return sid, pkg
            except Exception:
                pass
        return sid, None

    missing_sids = [int(p.stem) for p in npz_files if int(p.stem) not in screen_to_app]
    if missing_sids:
        print(f"Resolving app package name for {len(missing_sids)} screens via layout JSON files in parallel...")
        
        def process_sid(sid):
            return parse_single_layout(sid, args.layouts_dir)
            
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(process_sid, missing_sids),
                total=len(missing_sids),
                desc="Parsing layout JSONs"
            ))
            
        for sid, pkg in results:
            if pkg:
                screen_to_app[sid] = pkg

    # 4. Generate app text strings
    print("Preparing text input for embedding model...")
    app_texts = {} # package_name -> descriptive text
    unique_packages = set(screen_to_app.values())
    for pkg in unique_packages:
        if pkg in app_text_map and app_text_map[pkg]:
            app_texts[pkg] = app_text_map[pkg]
        else:
            # Fallback: clean package name components
            parts = pkg.split(".")
            # Filter out standard keywords like "com", "android", "apps"
            clean_parts = [p for p in parts if p not in ("com", "org", "net", "android", "app", "apps", "google")]
            if not clean_parts:
                clean_parts = parts
            app_texts[pkg] = " ".join(clean_parts)

    # 5. Embed unique texts using sentence-transformers
    print("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    model.eval()

    print("Embedding app packages...")
    unique_pkgs_list = list(app_texts.keys())
    unique_texts_list = [app_texts[pkg] for pkg in unique_pkgs_list]

    pkg_to_emb = {}
    batch_size = 128
    
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_texts_list), batch_size), desc="Embedding batches"):
            batch_texts = unique_texts_list[i : i + batch_size]
            batch_pkgs = unique_pkgs_list[i : i + batch_size]
            
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
            model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            sentence_embeddings = sentence_embeddings.cpu().numpy()
            for pkg, emb in zip(batch_pkgs, sentence_embeddings):
                pkg_to_emb[pkg] = emb

    # 6. Concatenate embeddings and save to disk
    print(f"Concatenating and saving embeddings to {args.output_dir}...")
    
    # We also prepare a default 0-embedding of size 384 for absolute fallbacks
    default_app_emb = np.zeros(384, dtype=np.float32)

    for npz_path in tqdm(npz_files, desc="Saving concatenated embeddings"):
        screen_id = int(npz_path.stem)
        
        # Load screen embedding
        with np.load(npz_path) as f_data:
            screen_emb = f_data["embedding"].astype(np.float32)
        
        # L2-normalize the sequence-projected screen embedding first
        screen_emb_norm = np.linalg.norm(screen_emb)
        if screen_emb_norm > 1e-12:
            screen_emb = screen_emb / screen_emb_norm
        
        # Get matching app embedding
        app_pkg = screen_to_app.get(screen_id)
        if app_pkg and app_pkg in pkg_to_emb:
            app_emb = pkg_to_emb[app_pkg]
        else:
            app_emb = default_app_emb
            
        # Concatenate: (128,) + (384,) -> (512,)
        concat_emb = np.concatenate([screen_emb, app_emb], axis=0).astype(np.float32)
        
        # Save output
        out_path = os.path.join(args.output_dir, f"{screen_id}.npz")
        np.savez_compressed(out_path, embedding=concat_emb)

    print("Success! Concatenated embeddings created and saved.")

if __name__ == "__main__":
    main()
