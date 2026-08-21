import argparse
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer

import trace_index

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
    token_embeddings = model_output[0]
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
        "--csv_path", type=str, default="../rico-data/data/play_store_metadata/app_details_enriched.csv",
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
    parser.add_argument(
        "--trace_index", type=str, default=trace_index.DEFAULT_INDEX_PATH,
        help="Trace->layout index CSV from 0_build_trace_index.py. Required: "
             "trace gestures.json keys are per-trace indices, not the global "
             "layouts ids the embeddings are keyed by.",
    )
    parser.add_argument(
        "--description_col", type=str, default="gp_description",
        help="Column name in the CSV that contains full app descriptions. "
             "If not found, falls back to 'Play Store Name - Category'. "
             "Screen2Vec uses full Play Store descriptions for their 768-dim BERT embedding.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load app details CSV — try to find a full description column
    print(f"Loading app details from {args.csv_path}...")
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"App details CSV not found at {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    print(f"  CSV columns: {list(df.columns)}")

    # Check if a Description column exists
    desc_col = args.description_col if args.description_col in df.columns else None
    if desc_col:
        n_with_desc = df[desc_col].notna().sum()
        print(f"  Found description column '{desc_col}' with {n_with_desc}/{len(df)} non-null entries.")
        print(f"  Using FULL descriptions (matching Screen2Vec's approach).")
        # Use 768-dim model to match Screen2Vec's Sentence-BERT
        sbert_model_name = "sentence-transformers/all-mpnet-base-v2"  # 768-dim
        embed_dim_app = 768
    else:
        print(f"  No '{args.description_col}' column found. Falling back to 'Name - Category'.")
        print(f"  NOTE: Screen2Vec uses full Play Store descriptions scraped live via")
        print(f"  google-play-scraper. Their descriptions are much richer than Name+Category.")
        print(f"  This is a known limitation if you don't have the full descriptions.")
        sbert_model_name = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim
        embed_dim_app = 384

    # Build app_text_map
    app_text_map = {}
    for idx, row in df.iterrows():
        pkg = row.get("App Package Name", "").strip()
        name = row.get("Play Store Name", "").strip()
        category = row.get("Category", "").strip()
        scraped_desc = row.get("gp_description").strip() if not pd.isna(row.get("gp_description")) else ""
        app_text_map[pkg] = f"{pkg} - {name} - {category} - {scraped_desc}"


    print(f"Loaded {len(app_text_map)} app metadata entries.")

    # 2. Get all screen IDs from embeddings directory
    npz_files = sorted(Path(args.embeddings_dir).glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {args.embeddings_dir}")
    print(f"Found {len(npz_files)} projected screen embeddings.")

    # 3. Map screen IDs to app packages
    #
    # This MUST go through the trace->layout index. A trace's gestures.json
    # keys are per-trace indices, not the global layouts ids the embeddings
    # are keyed by, so assigning screen_to_app[int(gesture_key)] attributes
    # each app's screens to unrelated ids (and, because those ids collide
    # across traces, whichever app is parsed last wins). See trace_index.py.
    #
    # The index is also what makes the attribution *correct* rather than just
    # consistent: Screen2Vec describes a screen by the Play Store entry of the
    # app whose trace it came from, and the traced app differs from the
    # activity's own task package for ~8% of screens (ad activities,
    # cross-app screens). Only the index knows which trace a screen came from.
    print("Mapping screen IDs to app package names via the trace->layout index...")
    index = trace_index.load_index(args.trace_index)
    screen_to_app = trace_index.layout_id_to_app(index)
    print(f"Mapped {len(screen_to_app)} screens via the trace->layout index "
          f"('{args.trace_index}').")

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
    print(f"Loading SBERT model '{sbert_model_name}' (embed_dim={embed_dim_app})...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(sbert_model_name)
    model = AutoModel.from_pretrained(sbert_model_name).to(device)
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
    
    # We also prepare a default 0-embedding matching the embedding size
    default_app_emb = np.zeros(embed_dim_app, dtype=np.float32)



    # Determine screen embedding dim from first file
    screen_emb_dim = None

    for npz_path in tqdm(npz_files, desc="Saving concatenated embeddings"):
        screen_id = int(npz_path.stem)
        
        # Load screen embedding
        with np.load(npz_path) as f_data:
            screen_emb = f_data["embedding"].astype(np.float32)
        
        if screen_emb_dim is None:
            screen_emb_dim = screen_emb.shape[0]
            print(f"Concatenating screen ({screen_emb_dim}-dim) + app ({embed_dim_app}-dim) embeddings...")
            print(f"Output dimension: {screen_emb_dim + embed_dim_app}-dim")

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
            
        # Concatenate screen emb + app emb
        concat_emb = np.concatenate([screen_emb, app_emb], axis=0).astype(np.float32)
        
        # Save output (uncompressed np.savez is significantly faster than np.savez_compressed)
        out_path = os.path.join(args.output_dir, f"{screen_id}.npz")
        np.savez(out_path, embedding=concat_emb)

    print("Success! Concatenated embeddings created and saved.")

if __name__ == "__main__":
    main()
