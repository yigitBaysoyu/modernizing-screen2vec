import argparse
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import math
from transformers import AutoModel, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed

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

def load_seq_embeddings(embeddings_dir):
    npz_files = sorted(Path(embeddings_dir).glob("*.npz"))
    print(f"Loading {len(npz_files)} sequence embeddings...")
    id_to_idx = {}
    embeddings = []
    for idx, npz_path in enumerate(npz_files):
        screen_id = int(npz_path.stem)
        with np.load(npz_path) as f:
            emb = f["embedding"].astype(np.float32)
        id_to_idx[screen_id] = idx
        embeddings.append(emb)
    return id_to_idx, np.stack(embeddings, axis=0)

def parse_traces(traces_dir, available_ids):
    traces_root = Path(traces_dir)
    traces = []
    app_dirs = sorted(traces_root.iterdir())
    for app_dir in app_dirs:
        if not app_dir.is_dir():
            continue
        for trace_dir in sorted(app_dir.iterdir()):
            if not trace_dir.is_dir() or not trace_dir.name.startswith("trace_"):
                continue
            gestures_path = trace_dir / "gestures.json"
            if not gestures_path.exists():
                continue
            try:
                with open(gestures_path, "r", encoding="utf-8") as f:
                    gestures = json.load(f)
            except Exception:
                continue
            screen_ids = []
            for key in gestures.keys():
                try:
                    sid = int(key)
                except ValueError:
                    continue
                if sid in available_ids:
                    screen_ids.append(sid)
            if len(screen_ids) < 3:
                continue
            traces.append(screen_ids)
    return traces

def eval_single_trace(
    trace,
    id_to_idx,
    embedding_matrix_normed,
    k_001pct,
    k_01pct,
    k_1pct,
    k_5pct,
):
    trace_indices = [id_to_idx[sid] for sid in trace]
    trace_normed = embedding_matrix_normed[trace_indices]
    T = len(trace)

    trace_normed_sum = trace_normed.sum(axis=0, keepdims=True)
    pred = (trace_normed_sum - trace_normed) / (T - 1)
    pred_norm = np.linalg.norm(pred, axis=1, keepdims=True)
    pred_normed = pred / np.maximum(pred_norm, 1e-12)

    similarities = cosine_similarity(pred_normed, embedding_matrix_normed)
    target_sims = similarities[np.arange(T), trace_indices]
    ranks = np.sum(similarities >= target_sims[:, None], axis=1)

    hits_top1 = int(np.sum(ranks <= 1))
    hits_top001pct = int(np.sum(ranks <= k_001pct))
    hits_top01pct = int(np.sum(ranks <= k_01pct))
    hits_top1pct = int(np.sum(ranks <= k_1pct))
    hits_top5pct = int(np.sum(ranks <= k_5pct))

    return T, hits_top1, hits_top001pct, hits_top01pct, hits_top1pct, hits_top5pct

def evaluate(traces, id_to_idx, embedding_matrix, n_jobs=-1):
    N = embedding_matrix.shape[0]
    k_001pct = max(1, math.ceil(0.0001 * N))
    k_01pct = max(1, math.ceil(0.001 * N))
    k_1pct = max(1, math.ceil(0.01 * N))
    k_5pct = max(1, math.ceil(0.05 * N))

    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embedding_matrix_normed = embedding_matrix / norms

    results = Parallel(n_jobs=n_jobs)(
        delayed(eval_single_trace)(
            trace, id_to_idx, embedding_matrix_normed, k_001pct, k_01pct, k_1pct, k_5pct
        )
        for trace in traces
    )

    total_predictions = 0
    hits_top1 = 0
    hits_top001pct = 0
    hits_top01pct = 0
    hits_top1pct = 0
    hits_top5pct = 0

    for res in results:
        T, h1, h001, h01, h1pct, h5pct = res
        total_predictions += T
        hits_top1 += h1
        hits_top001pct += h001
        hits_top01pct += h01
        hits_top1pct += h1pct
        hits_top5pct += h5pct

    return {
        "top1": hits_top1 / total_predictions,
        "top001pct": hits_top001pct / total_predictions,
        "top01pct": hits_top01pct / total_predictions,
        "top1pct": hits_top1pct / total_predictions,
        "top5pct": hits_top5pct / total_predictions,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", type=str, default="embeddings_seq_128")
    parser.add_argument("--traces_dir", type=str, default="../rico-data/data/traces/filtered_traces")
    parser.add_argument("--layouts_dir", type=str, default="../rico-data/data/layouts")
    parser.add_argument("--csv_path", type=str, default="../rico-data/data/play_store_metadata/app_details.csv")
    args = parser.parse_args()

    # 1. Load sequence embeddings
    id_to_idx, screen_embs = load_seq_embeddings(args.embeddings_dir)
    available_ids = set(id_to_idx.keys())
    idx_to_id = {v: k for k, v in id_to_idx.items()}

    # 2. Parse traces and sample 10% for fast evaluation
    all_traces = parse_traces(args.traces_dir, available_ids)
    np.random.seed(42)
    sample_indices = np.random.choice(len(all_traces), size=max(1, int(len(all_traces) * 0.1)), replace=False)
    eval_traces = [all_traces[i] for i in sample_indices]
    print(f"Sampled {len(eval_traces)} traces for sweep evaluation.")

    # 3. Map screen IDs to app packages
    print("Mapping screen IDs to app package names...")
    screen_to_app = {}
    traces_root = Path(args.traces_dir)
    if traces_root.exists():
        app_dirs = sorted(traces_root.iterdir())
        for app_dir in app_dirs:
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
                        screen_to_app[int(key)] = app_pkg
                except Exception:
                    continue

    missing_sids = [sid for sid in available_ids if sid not in screen_to_app]
    if missing_sids:
        print(f"Parsing {len(missing_sids)} layout files in parallel...")
        def parse_single_layout(sid):
            json_path = os.path.join(args.layouts_dir, f"{sid}.json")
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

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(parse_single_layout, missing_sids))
        for sid, pkg in results:
            if pkg:
                screen_to_app[sid] = pkg

    # Fill default pkg for any remaining
    for sid in available_ids:
        if sid not in screen_to_app:
            screen_to_app[sid] = "unknown.app"

    # 4. Generate app semantic text descriptions
    df = pd.read_csv(args.csv_path)
    app_text_map = {}
    for idx, row in df.iterrows():
        pkg = row.get("App Package Name")
        name = row.get("Play Store Name")
        category = row.get("Category")
        if pd.isna(name) or not str(name).strip():
            name = ""
        if pd.isna(category) or not str(category).strip():
            category = ""
        text = f"{name} - {category}" if name and category else (name or category or "")
        if pkg:
            app_text_map[str(pkg).strip()] = text

    app_texts = {}
    unique_packages = sorted(list(set(screen_to_app.values())))
    for pkg in unique_packages:
        if pkg in app_text_map and app_text_map[pkg]:
            app_texts[pkg] = app_text_map[pkg]
        else:
            parts = pkg.split(".")
            clean_parts = [p for p in parts if p not in ("com", "org", "net", "android", "app", "apps", "google")]
            app_texts[pkg] = " ".join(clean_parts) if clean_parts else pkg

    # 5. Generate semantic embeddings
    print("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    model.eval()

    unique_texts = [app_texts[pkg] for pkg in unique_packages]
    semantic_embs = {}
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(unique_texts), batch_size):
            batch_texts = unique_texts[i : i + batch_size]
            batch_pkgs = unique_packages[i : i + batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = model(**encoded)
            embeddings = mean_pooling(outputs, encoded['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()
            for pkg, emb in zip(batch_pkgs, embeddings):
                semantic_embs[pkg] = emb

    # 6. Generate unique app fingerprint embeddings (orthogonal/random projection)
    print("Generating unique app fingerprint embeddings...")
    rng = np.random.default_rng(42)
    unique_embs = {}
    for pkg in unique_packages:
        vec = rng.normal(size=384).astype(np.float32)
        unique_embs[pkg] = vec / np.linalg.norm(vec)

    # 7. Sweep weights
    # w_screen: weight for sequence projected screen embedding (dim 128)
    # w_semantic: weight for semantic category embedding (dim 384)
    # w_unique: weight for unique package embedding (dim 384)
    sweeps = [
        # format: (w_screen, w_semantic, w_unique)
        (1.0, 0.0, 0.0), # baseline screen only
        (1.0, 1.0, 0.0), # screen + semantic (what we ran)
        (1.0, 0.0, 1.0), # screen + unique fingerprint (strong filter)
        (1.0, 1.0, 1.0), # screen + semantic + fingerprint
        (1.0, 1.0, 3.0), # screen + semantic + high-weight fingerprint
        (1.0, 0.5, 5.0), # screen + low semantic + very high-weight fingerprint
        (1.0, 0.2, 10.0), # screen + very low semantic + ultra high-weight fingerprint
    ]

    print("\nStarting weight sweeps on 10% trace sample:")
    print("-" * 80)
    for w_s, w_sem, w_uniq in sweeps:
        # Construct concatenated embedding matrix
        matrix = []
        for idx in range(screen_embs.shape[0]):
            screen_id = idx_to_id[idx]
            app = screen_to_app[screen_id]
            
            s_emb = screen_embs[idx]
            s_emb = s_emb / np.maximum(np.linalg.norm(s_emb), 1e-12)
            
            sem_emb = semantic_embs[app]
            uniq_emb = unique_embs[app]
            
            parts = [s_emb * w_s]
            if w_sem > 0:
                parts.append(sem_emb * w_sem)
            if w_uniq > 0:
                parts.append(uniq_emb * w_uniq)
                
            concat = np.concatenate(parts, axis=0)
            matrix.append(concat)
            
        embedding_matrix = np.stack(matrix, axis=0)
        
        # Evaluate
        metrics = evaluate(eval_traces, id_to_idx, embedding_matrix, n_jobs=-1)
        print(f"Weights (Screen={w_s:.1f}, Semantic={w_sem:.1f}, Unique={w_uniq:.1f}):")
        print(f"  Top-1: {metrics['top1']:.4f} | Top-0.01%: {metrics['top001pct']:.4f} | Top-0.1%: {metrics['top01pct']:.4f} | Top-1%: {metrics['top1pct']:.4f} | Top-5%: {metrics['top5pct']:.4f}")
        print("-" * 80)

if __name__ == "__main__":
    main()
