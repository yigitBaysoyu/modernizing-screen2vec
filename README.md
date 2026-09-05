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

## Generating Several Embedding Shapes at Once

`1_qwen_generate_embeddings.py` takes several `--pool-tokens` values. They are all
pooled out of the *same* forward pass and written to their own directory, so a
sweep over N costs one inference run rather than one per N:

```bash
python 1_qwen_generate_embeddings.py job.json --pool-tokens 2 4 8 16 32 64 --pooling avg
# -> embeddings_2x4096_avg/ ... embeddings_64x4096_avg/
```

The shapes are *not* derivable from one another afterwards: `adaptive_avg_pool1d`
puts its bucket borders at `floor(i*L/N)` / `ceil((i+1)*L/N)`, which only line up
between two N when the valid sequence length divides evenly, and each bucket is
L2-normalized individually on top. So they have to be taken from the live hidden
states, which is what a multi-value `--pool-tokens` does.

Add `--skip-existing` to resume a run that hit its wall time; it skips screens
whose `.npz` already exists in *every* target directory.

On the cluster, set `POOL_TOKENS` in `hpc/_hpc1_start_embedding.py` to the list
of N you want and submit once.

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
  [--data_dir <path_to_rico_layouts>] \
  [--max_tokens 1024]
```

Example:

```bash
 python 3_json_reconstruction.py 
          --projector_model adapter_model_out/best_projector.pth \
          --lora_model adapter_model_out/best_lora_adapter/ \
          --index 3  \  # 0 .. 66260
          --embeddings_dir embeddings_4096_last/   \ 
          [--full-output]
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

### Recovering the five Qwen3-4B-Base runs on HPC

After placing the five adapters, embeddings, and partial reconstruction folders
directly in `/lustre/scratch/sgjolieb-screen2vec`, run the dedicated recovery
launcher on the HPC login node:

```bash
cd /lustre/scratch/sgjolieb-screen2vec/screen2vec-revisited
python hpc/_hpc6_resume_qwen3_4b_reconstructions.py --dry-run
python hpc/_hpc6_resume_qwen3_4b_reconstructions.py
```

It submits five variants times eight shards as a 40-task Slurm array capped at
eight concurrent H200s. Each task uses batch size 64, recursively splits a batch
on CUDA OOM, and skips IDs that already have a successful reconstruction/report
pair. A dependent CPU job audits all five folders for 66,261 complete pairs.

To recover only the separate Qwen3-1.7B-Base `4096_last` run (for example while
the 4B recovery is still active), select it explicitly:

```bash
python hpc/_hpc6_resume_qwen3_4b_reconstructions.py \
  --only qwen3-1.7b-base --dry-run
python hpc/_hpc6_resume_qwen3_4b_reconstructions.py \
  --only qwen3-1.7b-base
```

Important Notes:
- The original screen2vec paper is stored in the file "screen2vec.txt". Please read it.
- The original source code is stored in folder "screen2vec-master".

## Replicating & Beating Screen2Vec Table 3 (GUI Screen Prediction)

To reproduce the results that significantly outperform Screen2Vec's Table 3 on the GUI Screen Prediction task, run the following pipeline sequence:

### Pipeline v1

1. **Train Sequence Projector:**
   Trains a projector network (4096 -> 1024 -> 768) on the raw Qwen3-VL embeddings using InfoNCE loss.
   ```bash
   python 5_train_sequence_projector.py --proj_dim 768
   ```

2. **Generate App Context Embeddings:**
   Combines the projected 768-dim screen embeddings with the App Store descriptions (via Sentence-BERT) to replicate Screen2Vec's app-context concatenation trick.
   ```bash
   python 6_generate_app_context_embeddings.py --embeddings_dir embeddings_seq_768 --output_dir embeddings_seq_768_with_app
   ```

3. **Train LSTM Trace Predictor:**
   Trains a unidirectional LSTM to predict the next screen given the previous context screens in an interaction trace.
   ```bash
   python 8_train_lstm_predictor.py --embeddings_dir embeddings_seq_768
   ```

4. **Evaluate and Compare:**
   Evaluates the trained LSTM model using the concatenated app-context embeddings, ensuring an apples-to-apples comparison with the original Screen2Vec evaluation protocol.
   ```bash
   python 4_evaluate_screen_prediction.py --embeddings_dir embeddings_seq_768_with_app --lstm_model lstm_predictor_768d.pth
   ```

### Pipeline v2

```bash
# Step 1-3: Only if you haven't already (you have — embeddings_4096_last/ exists)
# python 1_qwen_generate_embeddings.py

# Step 4: NEW — replaces both 5_train_sequence_projector.py AND 8_train_lstm_predictor.py
python 9_train_joint.py --proj_dim 768

# Step 5: Same as before
python 6_generate_app_context_embeddings.py --embeddings_dir embeddings_seq_768 --output_dir embeddings_seq_768_with_app_desc

# Step 6: Same as before
python 4_evaluate_screen_prediction.py --embeddings_dir embeddings_seq_768_with_app --lstm_model lstm_predictor_768d.pth
```


TODOs
- [x] Benchmarken Table 3 stechen.
- Den User-Centric Context-Dependent Flow als Argument bringen, Icons bereitzustellen und Co un Nutzerinteraktionen einfacher zu machen.
- Eigene Benchmarks für Reconstruction Quality
- Evtl. Android in the Wild

Parameter:
- Embeddingspace verkleinern und eine abflachende Kurve zeigen
- Embedding-Modell verkleinern um weniger Strom zu verbrauchen und performanter zu wreden
- Rechenzeit und Speicherverbrauch gegen Screen2Vec Approach testen
- Anzahl Prefixtokens untersuchen

Title Ideas:
- Screen2Vec 2.0: Embedding Representations to Support Mobile Downstream Tasks for Mobile Interactions
-  Supporting Downstream Tasks in Mobile Multitask Interactions

Reviewer:
- Per Ola Kristensen

