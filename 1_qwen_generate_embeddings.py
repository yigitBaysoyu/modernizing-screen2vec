# Generates the embeddings using Model QWEN VLM Embedder.
#
# The sequence of hidden states is compressed into POOL_TOKENS pooled tokens:
#   POOL_TOKENS = 1   -> (4096,)      single summary vector  [default]
#   POOL_TOKENS = N   -> (N, 4096)    N-bucket sequence trace
#
# Higher N retains more of the positional structure of the input (and is
# therefore reconstructable in more detail downstream), at N-times the
# storage cost.  Use --pool-tokens / --pooling to select.
#
# --pool-tokens takes several N at once (e.g. --pool-tokens 2 4 8 16 32 64).
# All of them are pooled out of the *same* forward pass and written to their
# own directory, so a whole sweep costs one inference run instead of one per N.
# Note that the shapes are not derivable from each other after the fact:
# adaptive_avg_pool1d puts its bucket borders at floor(i*L/N)/ceil((i+1)*L/N),
# which only line up between two N when the valid length happens to divide
# evenly, and each bucket is L2-normalized individually on top of that.  Hence
# they have to be taken from the live hidden states, which is what happens here.
#
# Both settings are encoded in the output directory name
# (embeddings_4096_last, embeddings_32x4096_avg, ...), so a folder is never
# ambiguous about how it was produced.

import os
import json
import sys
import gc
import socket
import time
import argparse
import numpy as np

from utils import _minify_json, process_json_file

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor
from qwen_vl_utils import process_vision_info
from glob import glob
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# =============================================================================
# Pooling configuration
# =============================================================================
# Hidden size of the embedder's last layer.  Used to derive the output
# directory name; asserted against the real embedding shape at runtime.
EMBED_DIM = 4096

# Number of pooled tokens the token sequence is compressed into.  A tuple:
# every entry is produced from the same forward pass and stored separately.
POOL_TOKENS = (1,)
SUPPORTED_POOL_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128)

# "last" – hidden state of the final non-padding token (only valid for
#          POOL_TOKENS == 1).  This is what embeddings_4096_last/ holds.
# "avg"  – adaptive average pooling of the sequence into POOL_TOKENS buckets.
#          The only mode defined for every N, so use it for the whole sweep
#          (including N=1) if you want the runs to be comparable.
POOLING = "last"


def embeddings_dir_for(num_pool_tokens, pooling, embed_dim=EMBED_DIM):
    """
    Derives the output directory name.

    The pooling strategy is part of the name, because "last" and "avg" are
    different operations rather than special cases of one another -- mixing
    them in one sweep silently confounds pooling with bucket count.

        (1, "last")   -> embeddings_4096_last
        (1, "avg")    -> embeddings_4096_avg
        (32, "avg")   -> embeddings_32x4096_avg
    """
    shape = f"{embed_dim}" if num_pool_tokens == 1 else f"{num_pool_tokens}x{embed_dim}"
    return f"embeddings_{shape}_{pooling}"


def pool_hidden_states_multi(last_hidden_states, attention_mask, num_pool_tokens, pooling):
    """
    Compresses [1, seq_len, D] hidden states into several embedding shapes.

    All requested shapes come out of the same hidden states, i.e. out of a
    single forward pass -- the pooling itself is negligible next to the model
    call, so a sweep over N costs essentially the same as one N.

    Args:
        last_hidden_states: [1, seq_len, D] final-layer hidden states.
        attention_mask:     [1, seq_len] mask marking the non-padding tokens.
        num_pool_tokens:    Iterable of pooled-token counts to produce.
        pooling:            "last" or "avg" (see POOLING above).

    Returns:
        {N: tensor}, where the tensor is [D] for N == 1 and [N, D] otherwise.
    """
    num_pool_tokens = list(num_pool_tokens)

    if pooling == "last":
        # Attention-mask-based last-token pooling: in a causal model this
        # position has attended to the whole input, so it acts as a summary.
        # Defined for a single token only, hence never more than one entry.
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
        pooled = last_hidden_states[row, col]  # [1, D]
        return {1: F.normalize(pooled, p=2, dim=-1).squeeze(0)}  # [D]

    # "avg": drop padding, then adaptive-average-pool the sequence dimension
    # into exactly N contiguous buckets.  Each bucket keeps a coarse-grained
    # trace of its slice of the input, so component order and position survive
    # the compression.
    #
    # The padding is dropped and the tensor permuted to [batch, channels,
    # seq_len] once for all N; only the pooling op itself runs per N.
    valid_len = int(attention_mask[0].sum().item())
    hidden = last_hidden_states[:, :valid_len, :].float().permute(0, 2, 1)  # [1, D, L]

    pooled_by_n = {}
    for n in num_pool_tokens:
        pooled = F.adaptive_avg_pool1d(hidden, n).permute(0, 2, 1)  # [1, N, D]

        # L2 normalize each pooled token.  2_train_adapter.py normalizes along
        # the last dim anyway, so doing it here keeps every N consistent on disk.
        pooled = F.normalize(pooled, p=2, dim=-1).squeeze(0)  # [N, D]

        pooled_by_n[n] = pooled.squeeze(0) if n == 1 else pooled

    return pooled_by_n


def pool_hidden_states(last_hidden_states, attention_mask, num_pool_tokens, pooling):
    """Single-shape convenience wrapper around pool_hidden_states_multi()."""
    pooled = pool_hidden_states_multi(
        last_hidden_states, attention_mask, [num_pool_tokens], pooling
    )
    return pooled[1 if pooling == "last" else num_pool_tokens]


def embedding_path_for(file_path, emb_dir):
    """Target .npz path of a source JSON inside one embedding directory."""
    base_name = os.path.basename(file_path).replace('.json', '.npz')
    return os.path.join(emb_dir, base_name)


def save_embedding_npz(file_path, embedding, emb_dir):
    """Saves embedding to a compressed .npz file named after the source JSON."""
    os.makedirs(emb_dir, exist_ok=True)
    emb_path = embedding_path_for(file_path, emb_dir)

    # Save as compressed numpy array (fp16 to save space)
    np.savez_compressed(emb_path, embedding=embedding.astype(np.float16))


def main(json_path, image_path, model, processor, num_pool_tokens=POOL_TOKENS, pooling=POOLING):
    """
    Embeds one screen and returns {N: ndarray} for every requested N, or None
    if the screen's JSON or screenshot is missing.
    """
    # 1. Read JSON Data
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return None

    # 2. Check for Image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    # 3. Construct the Multimodal Input
    # According to Qwen-VL-Utils standard formatting:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image_path}",
                    # "max_pixels": 1000000, # Uncomment if you still get OOMs due to extremely tall screenshots
                },
                {
                    "type": "text",
                    # "text": f"Represent this UI component and its context: {json_content}",
                    "text": f"{process_json_file(json_path)}",
                },
            ],
        }
    ]

    # Preparation for generation
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # For embedding models, we typically do a simple forward pass without generating tokens
    # and extract the specific layer's output representation.
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Extract the hidden states from the last layer for all tokens
        last_hidden_states = outputs.hidden_states[-1] # Shape: [1, seq_len, 4096]

        # Compress the sequence into every configured number of pooled tokens.
        # One forward pass feeds all of them.
        pooled_by_n = pool_hidden_states_multi(
            last_hidden_states, inputs["attention_mask"], num_pool_tokens, pooling
        )

        # Shape per entry: [4096] for N=1, else [N, 4096]
        embeddings = {
            n: pooled.cpu().float().numpy() for n, pooled in pooled_by_n.items()
        }

    # Explicitly free VRAM to prevent memory leaks in the loop
    del inputs
    del outputs
    del last_hidden_states
    torch.cuda.empty_cache()
    gc.collect()

    return embeddings


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate Qwen3-VL embeddings for Rico screens, pooled into one or "
            "several configurable numbers of tokens. Several --pool-tokens "
            "share a single forward pass per screen."
        )
    )
    parser.add_argument(
        "job_file", nargs="?", default=None,
        help="Optional job.json listing the screens to embed. "
             "Without it, all layouts under ../rico-data/data/layouts are used.",
    )
    parser.add_argument(
        "--pool-tokens", type=int, nargs="+", default=list(POOL_TOKENS),
        choices=SUPPORTED_POOL_TOKENS, metavar="N",
        help=f"Number(s) of pooled tokens per screen (default: "
             f"{' '.join(str(n) for n in POOL_TOKENS)}, i.e. shape "
             f"({EMBED_DIM},)). Several values are pooled from the same "
             f"forward pass and each land in their own directory, e.g. "
             f"--pool-tokens 2 4 8 16 32 64. "
             f"Supported: {', '.join(str(n) for n in SUPPORTED_POOL_TOKENS)}.",
    )
    parser.add_argument(
        "--pooling", choices=("last", "avg"), default=POOLING,
        help=f"Pooling strategy (default: {POOLING}). 'last' is only defined "
             "for --pool-tokens 1; use 'avg' for a comparable sweep over N.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory. Only allowed with a single --pool-tokens value; "
             "otherwise the names are derived from --pool-tokens and --pooling "
             "(embeddings_4096_last, embeddings_32x4096_avg, ...).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a screen when its .npz already exists in every output "
             "directory. Lets a timed-out job be resubmitted without redoing "
             "the screens it already finished.",
    )

    args = parser.parse_args()

    # Deduplicate and sort, so the printed plan and the directory order match
    # regardless of how the values were passed on the command line.
    args.pool_tokens = sorted(set(args.pool_tokens))

    if args.pooling == "last" and args.pool_tokens != [1]:
        parser.error(
            "--pooling last produces a single vector and cannot be combined "
            f"with --pool-tokens {' '.join(str(n) for n in args.pool_tokens)}; "
            "use --pooling avg."
        )
    if args.out_dir and len(args.pool_tokens) > 1:
        parser.error(
            "--out-dir names a single directory and cannot be combined with "
            f"{len(args.pool_tokens)} --pool-tokens values; drop it and let the "
            "directory names be derived per N."
        )
    return args


if __name__ == "__main__":
    start_time = time.time()
    successful_embeddings_count = 0
    skipped_existing_count = 0

    args = parse_args()
    emb_dirs = {
        n: (args.out_dir or embeddings_dir_for(n, args.pooling))
        for n in args.pool_tokens
    }
    expected_shapes = {
        n: ((EMBED_DIM,) if n == 1 else (n, EMBED_DIM)) for n in args.pool_tokens
    }
    print(f"Pooling: {args.pooling} | one forward pass per screen | outputs:")
    for n in args.pool_tokens:
        print(f"  pool tokens: {n:>3} | shape: {expected_shapes[n]} | {emb_dirs[n]}/")

    if args.job_file:
        with open(args.job_file, 'r', encoding='utf-8') as f:
            job_data = json.load(f)
        # Parse job list; handle both list of dicts and list of strings
        tasks = []
        for item in job_data:
            if isinstance(item, dict):
                json_path = item["json"]
                image_path = item.get("image") or json_path.replace(".json", ".jpg")
            else:
                json_path = item
                image_path = json_path.replace(".json", ".jpg")
            tasks.append((json_path, image_path))
    else:
        json_files = glob("../rico-data/data/layouts/*json", recursive=True)
        json_files = sorted(json_files)
        tasks = [(g, g.replace(".json", ".jpg")) for g in json_files]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loaded files (showing first 10):", [t[0] for t in tasks[:10]])
    
    # Verify we have tasks to run
    assert len(tasks) >= 1

    print("Loading model and processor... (This may take a while and requires ~16GB of VRAM/RAM for 8B)")

    model_id = "../rico-data/models/Qwen3-VL-Embedding-8B"
    # model_id = "./models/Qwen3-VL-Embedding-8B"

    if not os.path.exists(model_id):
        print(f"\nLocal model not found at {model_id}")
        print("Please download it using the huggingface-cli:")
        print("hf download Qwen/Qwen3-VL-Embedding-8B --local-dir ./models/Qwen3-VL-Embedding-8B")
        sys.exit(1)

    try:
        # Load the model directly to the assigned device
        # Using bfloat16 to avoid Out-Of-Memory errors on 16GB GPUs
        model = AutoModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"\nFailed to load model from local path: {e}")
        print("Tip: Ensure the model data is not corrupted.")
        sys.exit(1)

    # Only update tqdm on 1% step
    step_size = len(tasks) // 100 or 1
    pbar = tqdm(tasks, miniters=step_size)
    for i, (json_path, image_path) in enumerate(pbar):
        if i % step_size == 0:
            pbar.set_description(f"Processing: {json_path}")

        if args.skip_existing and all(
            os.path.exists(embedding_path_for(json_path, d)) for d in emb_dirs.values()
        ):
            skipped_existing_count += 1
            continue

        try:
            embeddings = main(
                json_path, image_path, model, processor,
                num_pool_tokens=args.pool_tokens, pooling=args.pooling,
            )
        except torch.OutOfMemoryError as oom:
            print("Skipping", json_path, "due to torch.OutOfMemoryError:", str(oom))
            continue

        if embeddings is not None:
            for n, embedding in embeddings.items():
                assert embedding.shape == expected_shapes[n], (
                    f"Expected {expected_shapes[n]}, got {embedding.shape} for "
                    f"{json_path} at N={n}. Does the embedder's hidden size "
                    "still match EMBED_DIM?"
                )
                save_embedding_npz(json_path, embedding, emb_dirs[n])
            successful_embeddings_count += 1

    end_time = time.time()
    elapsed_seconds = int(end_time - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n--- Execution Summary ---")
    print(f"Total time elapsed: {hours} hours, {minutes} minutes, {seconds} seconds")
    print(f"Screens embedded: {successful_embeddings_count}")
    if args.skip_existing:
        print(f"Screens skipped (already complete): {skipped_existing_count}")
    print(
        f"Embeddings written: "
        f"{successful_embeddings_count * len(args.pool_tokens)} "
        f"across {len(args.pool_tokens)} directories"
    )
    for n in args.pool_tokens:
        written = len(glob(os.path.join(emb_dirs[n], "*.npz")))
        print(f"  {emb_dirs[n]}/: {written} files")