# Generates the embeddings using Model QWEN VLM Embedder.

import os
import json
import sys
import gc
import csv
import socket
import time

from utils import minify_json

try:
    import fcntl
except ImportError:
    fcntl = None

import torch
from transformers import AutoModel, AutoProcessor
from qwen_vl_utils import process_vision_info
from glob import glob
from tqdm import tqdm


if socket.gethostname().lower() == "hci-nc":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def append_to_csv_safely(csv_path, file_path, embedding):
    """
    Safely appends a file path and its corresponding embedding to a CSV file.
    Uses file locking to ensure thread/process safety.
    """
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        if fcntl:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            writer.writerow([file_path, json.dumps(embedding.tolist())])
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk immediately
        finally:
            if fcntl:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


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
                    #"max_pixels": 1000000, # Uncomment if you still get OOMs due to extremely tall screenshots
                },
                {
                    "type": "text",
                    #"text": f"Represent this UI component and its context: {json_content}",
                    "text": f"{minify_json(json_content)}",
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
        last_hidden_states = outputs.hidden_states[-1]
        
        # Simple mean pooling over the sequence dimension to generate a single vector
        # Note: The official Qwen3-VL-Embedding might have specific EOS token pooling instructions,
        # but mean pooling is a solid fallback for multimodal representation.
        embedding = last_hidden_states.mean(dim=1).squeeze().cpu().float().numpy()

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
    
    print("Loaded files (showing first 10):", to_glob[:10])
    assert len(to_glob) > 1

    print("Loading model and processor... (This may take a while and requires ~16GB of VRAM/RAM for 8B)")
    
    model_id = "../rico-data/models/Qwen3-VL-Embedding-8B"
    #model_id = "./models/Qwen3-VL-Embedding-8B"

    if not os.path.exists(model_id):
        print(f"\nLocal model not found at {model_id}")
        print("Please download it using the huggingface-cli:")
        print("hf download Qwen/Qwen3-VL-Embedding-8B --local-dir ./models/Qwen3-VL-Embedding-8B")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("\nWARNING: CUDA not detected. Running on CPU. This will be very slow.")
        print("Please ensure you have a CUDA-enabled PyTorch version installed.")

    try:
        # Load the model with automatic device mapping (requires accelerate)
        # Using bfloat16 to avoid Out-Of-Memory errors on 16GB GPUs
        model = AutoModel.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"\nFailed to load model from local path: {e}")
        print("Tip: Ensure the model data is not corrupted and accelerate is installed.")
        sys.exit(1)

    csv_output_file = "embeddings.csv"

    # Only update tqdm on 1% step
    step_size = len(to_glob) // 100 or 1
    pbar = tqdm(to_glob, miniters=step_size)
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
            append_to_csv_safely(csv_output_file, g, embedding)
            successful_embeddings_count += 1

    end_time = time.time()
    elapsed_seconds = int(end_time - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n--- Execution Summary ---")
    print(f"Total time elapsed: {hours} hours, {minutes} minutes, {seconds} seconds")
    print(f"Total embeddings performed: {successful_embeddings_count}")