"""
11_embed_component_vocab.py

Step 2 of the GUI component prediction pipeline: embeds every unique GUI
component text extracted by 10_extract_components.py.

This is the counterpart of Screen2Vec's vocab_emb.npy (extract_vocab.py encodes
the vocabulary with Sentence-BERT 'bert-base-nli-mean-tokens'). We use the same
embedder family as the rest of this repo -- Qwen3-VL-Embedding-8B, text-only,
last-token pooled exactly like 1_qwen_generate_embeddings.py pools a screen --
so component and screen embeddings live in the same space.

The prediction task in 12/13 ranks a predicted vector against this matrix, so
the encoder choice is what mainly moves the Table 2 numbers.

Outputs (into --data_dir):
    vocab_emb.npy   (V, D) float16, L2-normalised, row i = embedding of vocab[i]

Usage:
    python 11_embed_component_vocab.py --data_dir component_data
    # smaller/CPU-friendly encoder for a smoke test:
    python 11_embed_component_vocab.py --data_dir component_data \\
        --model sentence-transformers/all-MiniLM-L6-v2 --batch_size 512
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Same local checkout 1_qwen_generate_embeddings.py loads.
DEFAULT_MODEL = "../rico-data/models/Qwen3-VL-Embedding-8B"


# ============================================================================
# Pooling — mirrors pool_hidden_states() in 1_qwen_generate_embeddings.py
# ============================================================================
def pool(last_hidden_states, attention_mask, pooling):
    """Pool a batch of token sequences into one vector per sequence.

    "last" takes the final non-padding token (a causal model has attended to the
    whole string there); "mean" averages the non-padding tokens, which is what
    bidirectional sentence encoders such as Sentence-BERT/MiniLM expect.
    """
    if pooling == "last":
        flipped = attention_mask.flip(dims=[1])
        last_one_positions = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
        pooled = last_hidden_states[row, col]
    else:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
        pooled = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return F.normalize(pooled.float(), p=2, dim=-1)


def default_pooling_for(model_path):
    """Causal Qwen embedders summarise at the last token; BERT-likes average."""
    return "last" if "qwen" in os.path.basename(model_path).lower() else "mean"


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Embed the GUI component text vocabulary"
    )
    parser.add_argument(
        "--data_dir", type=str, default="component_data",
        help="Directory written by 10_extract_components.py",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Local path or HF id of the text encoder (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pooling", choices=("last", "mean", "auto"), default="auto",
        help="Token pooling; 'auto' picks 'last' for Qwen, 'mean' otherwise",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=64,
                        help="Token cap per component text (GUI labels are short)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output .npy (default: <data_dir>/vocab_emb.npy)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only embed the first N vocabulary entries (smoke tests)")
    args = parser.parse_args()

    pooling = default_pooling_for(args.model) if args.pooling == "auto" else args.pooling
    out_path = args.out or os.path.join(args.data_dir, "vocab_emb.npy")

    start = time.time()

    # ---------------------------------------------------------------
    # 1. Vocabulary
    # ---------------------------------------------------------------
    vocab_path = os.path.join(args.data_dir, "vocab.json")
    print(f"\n[Step 1/3] Loading vocabulary from '{vocab_path}'...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    if args.limit:
        vocab = vocab[:args.limit]
    print(f"  {len(vocab)} unique component texts")

    # ---------------------------------------------------------------
    # 2. Encoder
    # ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Step 2/3] Loading encoder '{args.model}' on {device} (pooling={pooling})...")
    if os.path.sep in args.model or args.model.startswith(".."):
        if not os.path.exists(args.model):
            print(f"\nLocal model not found at {args.model}")
            print("Download it with:")
            print("  hf download Qwen/Qwen3-VL-Embedding-8B "
                  "--local-dir ../rico-data/models/Qwen3-VL-Embedding-8B")
            raise SystemExit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    # Multimodal checkpoints wrap the text tower; use it directly for text-only.
    if hasattr(model, "language_model"):
        model = model.language_model
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        model = model.model.language_model
    model = model.to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------------
    # 3. Encode
    # ---------------------------------------------------------------
    print(f"\n[Step 3/3] Encoding in batches of {args.batch_size}...")
    embeddings = []
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(vocab), args.batch_size), desc="Embedding vocab"):
            batch = vocab[start_idx:start_idx + args.batch_size]
            # The empty string still needs a row; give it a single pad token so
            # the tokenizer returns a non-degenerate sequence.
            batch = [t if t else tokenizer.pad_token for t in batch]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt",
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1] if outputs.hidden_states is not None \
                else outputs.last_hidden_state
            pooled = pool(hidden, inputs["attention_mask"], pooling)
            embeddings.append(pooled.cpu().numpy().astype(np.float16))

    matrix = np.concatenate(embeddings, axis=0)
    # Index 0 is the "no text" slot: keep it at the origin so it can never win a
    # ranking and so padded context tokens contribute nothing.
    matrix[0] = 0.0

    os.makedirs(args.data_dir, exist_ok=True)
    np.save(out_path, matrix)

    elapsed = time.time() - start
    print(f"\n  Embedding matrix: {matrix.shape} ({matrix.dtype})")
    print(f"Wrote {out_path}")
    print(f"Total embedding time: {elapsed:.1f}s")
    print("\nNext step — train the predictor with:")
    print(f"  python 12_train_component_predictor.py --data_dir {args.data_dir}")


if __name__ == "__main__":
    main()
