# Generates the embeddings using Model QWEN VLM Embedder.

import os
import json
import sys
import gc
import socket
import time
import numpy as np

from utils import _minify_json, process_json_file

import torch
from transformers import AutoModel, AutoProcessor
from qwen_vl_utils import process_vision_info
from glob import glob
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def save_embedding_npz(file_path, embedding):
    """Saves embedding to a compressed .npz file named after the source JSON."""
    emb_dir = "embeddings"
    os.makedirs(emb_dir, exist_ok=True)
    base_name = os.path.basename(file_path).replace('.json', '.npz')
    emb_path = os.path.join(emb_dir, base_name)
    
    # Save as compressed numpy array (fp16 to save space)
    np.savez_compressed(emb_path, embedding=embedding.astype(np.float16))


def main(json_path, image_path, model, processor):
    # 1. Read JSON Data
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        json_content = f.read()

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

        # Use adaptive average pooling to pool the sequence dimension down to exactly 1 token (shape: [1, 1, 4096])
        # We must permute to [batch, channels, seq_len] for 1d pooling, then permute back.
        pooled = torch.nn.functional.adaptive_avg_pool1d(last_hidden_states.permute(0, 2, 1), 1).permute(0, 2, 1)

        embedding = pooled.squeeze(0).squeeze(0).cpu().float().numpy() # Shape: [4096]

    # Explicitly free VRAM to prevent memory leaks in the loop
    del inputs
    del outputs
    del last_hidden_states
    torch.cuda.empty_cache()
    gc.collect()

    return embedding


if __name__ == "__main__":
    start_time = time.time()
    successful_embeddings_count = 0

    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            job_data = json.load(f)
        to_glob = [item["json"] for item in job_data]
    else:
        to_glob = glob("rico/examples/*.json")
        to_glob = glob("../rico-data/data/combined/*json", recursive=True)
        to_glob = sorted(to_glob)

    from accelerate import Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_local_main_process:
        print("Loaded files (showing first 10):", to_glob[:10])
    assert len(to_glob) > 1

    # Shard the dataset across processes
    to_glob = to_glob[accelerator.process_index::accelerator.num_processes]

    if accelerator.is_local_main_process:
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
        print("Tip: Ensure the model data is not corrupted and accelerate is installed.")
        sys.exit(1)

    # Only update tqdm on 1% step
    step_size = len(to_glob) // 100 or 1
    pbar = tqdm(to_glob, miniters=step_size, disable=not accelerator.is_local_main_process)
    for i, g in enumerate(pbar):
        if i % step_size == 0:
            pbar.set_description(f"Processing: {g}")

        json_path = g
        image_path = g.replace(".json", ".jpg")

        try:
            embedding = main(json_path, image_path, model, processor)
        except torch.OutOfMemoryError as oom:
            print("Skipping", json_path, "due to torch.OutOfMemoryError:", str(oom))
            continue

        if embedding is not None:
            save_embedding_npz(g, embedding)
            successful_embeddings_count += 1

    # Wait for all processes to finish before printing summary
    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        end_time = time.time()
        elapsed_seconds = int(end_time - start_time)
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\n--- Execution Summary ---")
        print(f"Total time elapsed: {hours} hours, {minutes} minutes, {seconds} seconds")
        # Note: count is per process, so this is just for the main process shard
        print(f"Embeddings performed on main process: {successful_embeddings_count}")