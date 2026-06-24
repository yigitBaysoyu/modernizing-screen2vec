# screen2vec-revisited

The goal of this repository is to create new user interface screens following the RICO dataset. 
It is a research work that should exceed the results of the screen2vec-paper.
The screen2vec-paper is provided in HTML format in folder `screen2vec-paper`.
In general, we seek to embed user interfaces using a VLM model (by embedding the JPEG+JSON of the screen)
and then to perform a llava-style reconstruction.

Our goal is to create better screens than the original authors could with their approach.



## Code Execution Flow (= pipeline)

1. Load RICO-Dataset
2. Clean RICO-Dataset
    - Remove unnecessary keys from JSON.
    - Apply binning of bounding boxes (1-100%).
    - Implemented in: `utils.py::process_json_file()`.
3. Create Embeddings (`1_qwen_generate_embeddings.py`).
4. Train the Adapter!
    - To translate the encoders (qwen-embed-8b) embedding space to the decoder embedding space.
    - File: `2_train_adapter.py`.
5. Predict a reconstructed UI from validation-set embeddings and save the reconstructed JSON (`3_json_reconstruction.py`).
6. Compare the reconstructed JSON to original but cleaned JSON.
7. Calculate some metric over all elements.

## Important Information

- All embeddings and adapter training needs to be redone, when the JSON processing code is touched!

## Process Sketch

![](assets/process-draft.png)

## Running JSON Reconstruction

To run the JSON reconstruction and evaluation script, use the following command:

```bash
python 3_json_reconstruction.py \
  --projector_model adapter_model_out/projector_<timestamp>.pth \
  --lora_model adapter_model_out/lora_adapter_<timestamp> \
  --index <index> \
  [--embeddings_dir embeddings] \
  [--data_dir <path_to_rico_combined>] \
  [--max_tokens 1024]
```

### Parameters:
- `--projector_model`: Path to the trained projector weights (`.pth`) file.
- `--lora_model`: Path to the trained LoRA adapter directory.
- `--index`: Index of the embedding to reconstruct.
- `--embeddings_dir` *(optional)*: Path to the embeddings directory (defaults to `embeddings`).
- `--data_dir` *(optional)*: Path to raw Rico JSON/JPG directory (optional, searches standard paths if omitted).
- `--max_tokens` *(optional)*: Maximum tokens to generate (defaults to `1024`).

### Example:
```bash
python 3_json_reconstruction.py --projector_model adapter_model_out/best_projector.pth --lora_model adapter_model_out/best_lora_adapter/ --index 3
```

(use `--index 2` because the early indexes are progressive web apps only having a webview.)

Indexes are always the respective element in the dataset, e.g., index 1 equals toembeddings/ 1.npz. 