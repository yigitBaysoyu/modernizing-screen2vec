# Dataset-level Reconstruction Evaluation.
#
# Extends 3_json_reconstruction.py from a single screen to the validation
# split of the dataset. Two phases:
#
#   generate  (GPU) – reconstruct JSONs for many validation screens and
#                     stream them to dataset_eval/predictions_<tag>.jsonl
#                     (resumable: already-generated screens are skipped).
#   evaluate  (CPU) – score stored predictions with evaluate_json_quality()
#                     and write per-screen metrics (CSV) plus an aggregate
#                     report (headline table + complexity breakdown).
#
# The validation split matches 2_train_adapter.py exactly: rows the trainer
# drops in build_target_column() (unusable JSONs, targets over
# AdapterConfig.max_target_len tokens) are filtered out first, then the first
# 80% of the remaining (numerically sorted) embeddings are training data, the
# rest validation.
#
# Usage:
#   python 3_evaluate_dataset_reconstruction.py generate \
#       --projector_model adapter_model_out/best_projector.pth \
#       --lora_model adapter_model_out/best_lora_adapter \
#       --tag adapter-v2 [--num_samples 1000 | --full] [--batch_size 32] \
#       [--gpu 0] [--seed 42] [--max_tokens 4096]
#
#   python 3_evaluate_dataset_reconstruction.py evaluate --tag adapter-v2

import os
import sys
import json
import time
import random
import argparse
import importlib

import numpy as np
import pandas as pd
from tqdm import tqdm

# Importing 3_json_reconstruction (leading digit -> importlib) reuses the
# exact single-screen metric functions and, transitively, the model
# components from 2_train_adapter. NOTE: this import sets CUDA_VISIBLE_DEVICES
# from utils.CUDA_VISIBLE_DEVICE; the --gpu flag re-overrides it in main()
# before CUDA is initialized.
_recon = importlib.import_module("3_json_reconstruction")
evaluate_json_quality = _recon.evaluate_json_quality
flatten_components = _recon.flatten_components
EmbeddingProjector = _recon.EmbeddingProjector
AdapterConfig = _recon.AdapterConfig
generate_from_prefix = _recon.generate_from_prefix

# The trainer's target-length filter, reused so the validation split here is
# computed over the same surviving rows the adapter was actually trained on.
_adapter = importlib.import_module("2_train_adapter")
fits_token_budget = _adapter.fits_token_budget

from utils import load_embeddings, process_json_file

OUTPUT_DIR = "dataset_eval"

# GT component-count buckets for the complexity breakdown table
COMPLEXITY_BUCKETS = [
    ("1", 1, 1),
    ("2-5", 2, 5),
    ("6-15", 6, 15),
    ("16+", 16, float("inf")),
]


# =============================================================================
# Split / selection helpers
# =============================================================================
def get_validation_positions(df, tokenizer, max_target_len):
    """Positions (iloc, in the raw frame) of the validation split.

    Matches 2_train_adapter.py exactly: build_target_column() drops rows
    (unusable JSONs and targets over max_target_len tokens) BEFORE the 80/20
    split, so the same filter has to run here.  Splitting the unfiltered
    frame would shift the boundary and leak trained-on screens into the
    "validation" sample.
    """
    kept_positions = []
    for pos, (_, row) in enumerate(tqdm(
            df.iterrows(), total=len(df),
            desc="Filtering targets (as in training)")):
        try:
            target_json = process_json_file(row["json-path"], minify=True)
        except Exception:
            target_json = None
        usable = bool(target_json) and len(target_json) >= 5
        if usable and fits_token_budget(target_json, tokenizer, max_target_len):
            kept_positions.append(pos)

    train_size = int(len(kept_positions) * 0.8)
    return kept_positions[train_size:]


def select_positions(val_positions, num_samples, full, seed):
    """Seeded random sample of the validation split (or all of it)."""
    if full or num_samples >= len(val_positions):
        return val_positions
    rng = random.Random(seed)
    return sorted(rng.sample(val_positions, num_samples))


# =============================================================================
# Phase 1: generate
# =============================================================================
def load_done_positions(predictions_path):
    """Dataset positions already present in the predictions file (for resume)."""
    done = set()
    if os.path.exists(predictions_path):
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["index"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def truncate_at_eos(token_ids, eos_token_id):
    """Cut a generated sequence at its first EOS token.

    generate_from_prefix() only stops early when ALL sequences in the batch
    hit EOS simultaneously, so sequences that finish early keep generating
    junk tokens afterwards.
    """
    eos_positions = (token_ids == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        return token_ids[: eos_positions[0].item()]
    return token_ids


def run_generate(args):
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    cfg = AdapterConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing on {device} "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")

    # -------------------------------------------------------------------------
    # 1. Load embedding metadata and select validation screens
    # -------------------------------------------------------------------------
    df = load_embeddings(embeddings_dir=args.embeddings_dir,
                         data_dir=args.data_dir, load_data=False)
    if df.empty:
        print("[-] Error: no embeddings found.", file=sys.stderr)
        return

    # The tokenizer is needed before the split can be computed: the trainer
    # drops over-long targets with it, and the split must match (see
    # get_validation_positions).  Resolving it needs the base model id, which
    # is read from the LoRA checkpoint unless overridden.
    lm_model_id = args.lm_model_id
    if lm_model_id is None:
        adapter_cfg_path = os.path.join(args.lora_model, "adapter_config.json")
        try:
            with open(adapter_cfg_path) as f:
                lm_model_id = json.load(f)["base_model_name_or_path"]
        except Exception as e:
            print(
                f"[-] Error: could not read base model from "
                f"{adapter_cfg_path}: {e}\n    Pass --lm_model_id explicitly.",
                file=sys.stderr,
            )
            return
    tokenizer = AutoTokenizer.from_pretrained(lm_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    val_positions = get_validation_positions(df, tokenizer, cfg.max_target_len)
    positions = select_positions(val_positions, args.num_samples, args.full,
                                 args.seed)
    print(f"[*] Validation split: {len(val_positions)} screens, "
          f"selected {len(positions)} "
          f"({'full split' if args.full else f'seed={args.seed}'})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    predictions_path = os.path.join(OUTPUT_DIR, f"predictions_{args.tag}.jsonl")
    done = load_done_positions(predictions_path)
    positions = [p for p in positions if p not in done]
    if done:
        print(f"[*] Resume: {len(done)} predictions already exist, "
              f"{len(positions)} remaining.")
    if not positions:
        print("[*] Nothing to do.")
        return

    # -------------------------------------------------------------------------
    # 2. Load models (checkpoint-driven, same procedure as
    #    3_json_reconstruction.py: every architectural decision is read from
    #    the checkpoint itself, because AdapterConfig moves between
    #    experiments and a checkpoint only loads into the exact architecture
    #    it was trained with)
    # -------------------------------------------------------------------------
    projector_sd = torch.load(args.projector_model, map_location="cpu")

    model_dtype = projector_sd["projector.0.weight"].dtype
    ckpt_hidden_dim = projector_sd["projector.2.weight"].shape[0]
    use_prefix_pos = "prefix_pos.weight" in projector_sd
    num_prefix_tokens = (
        projector_sd["prefix_pos.weight"].shape[0]
        if use_prefix_pos
        else cfg.num_prefix_tokens
    )

    print(f"[*] Loading base language model: {lm_model_id} ({model_dtype})")
    base_model = AutoModelForCausalLM.from_pretrained(
        lm_model_id,
        torch_dtype=model_dtype,
    )
    if base_model.config.hidden_size != ckpt_hidden_dim:
        print(
            f"[-] Error: projector was trained for hidden_size "
            f"{ckpt_hidden_dim}, but {lm_model_id} has "
            f"{base_model.config.hidden_size}. The projector and the LoRA "
            f"adapter come from different runs.",
            file=sys.stderr,
        )
        return

    print(f"[*] Loading LoRA adapter from: {args.lora_model}")
    lm_model = PeftModel.from_pretrained(base_model, args.lora_model)
    lm_model.to(device)
    lm_model.eval()

    lm_hidden_dim = lm_model.config.hidden_size
    print(f"[*] Loading projector from: {args.projector_model} "
          f"(prefix_pos={'yes' if use_prefix_pos else 'no'})")
    projector = EmbeddingProjector(
        cfg.qwen_embed_dim, lm_hidden_dim, num_prefix_tokens,
        use_prefix_pos=use_prefix_pos,
    ).to(device)
    projector.load_state_dict(projector_sd)
    projector = projector.to(device=device, dtype=model_dtype)
    projector.eval()

    # -------------------------------------------------------------------------
    # 3. Batched generation, streamed to disk
    # -------------------------------------------------------------------------
    batches = [
        positions[i: i + args.batch_size]
        for i in range(0, len(positions), args.batch_size)
    ]

    with open(predictions_path, "a", encoding="utf-8") as out_f:
        for batch in tqdm(batches, desc="Generating"):
            batch_start = time.time()

            embeddings = []
            for pos in batch:
                with np.load(df.iloc[pos]["embedding_path"]) as f_data:
                    embeddings.append(f_data["embedding"].astype(np.float32))

            latent = torch.tensor(
                np.stack(embeddings), dtype=model_dtype
            ).to(device)
            latent = F.normalize(latent, p=2, dim=-1)

            with torch.no_grad():
                prefix_embeds = projector(latent)
                generated_ids = generate_from_prefix(
                    lm_model, tokenizer, prefix_embeds,
                    max_new_tokens=args.max_tokens,
                )

            per_screen_time = (time.time() - batch_start) / len(batch)
            for b, pos in enumerate(batch):
                ids = truncate_at_eos(generated_ids[b], tokenizer.eos_token_id)
                pred_json_str = tokenizer.decode(ids, skip_special_tokens=True)
                record = {
                    "index": pos,
                    "embedding_path": df.iloc[pos]["embedding_path"],
                    "json_path": df.iloc[pos]["json-path"],
                    "predicted_json": pred_json_str,
                    "gen_time_s": round(per_screen_time, 3),
                }
                out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    print(f"[*] Done. Predictions written to {predictions_path}")
    print(f"[*] Next: python {os.path.basename(__file__)} evaluate --tag {args.tag}")


# =============================================================================
# Phase 2: evaluate
# =============================================================================
def gt_component_count(gt_str):
    """Component count of the ground truth, independent of prediction validity."""
    try:
        return len(flatten_components(json.loads(gt_str)))
    except (json.JSONDecodeError, TypeError):
        return 0


def relative_count_error(row):
    """|pred - gt| / gt. Uses the true GT count; invalid predictions count
    as a full miss (1.0) because their component counts are meaningless."""
    if not row["json_valid"]:
        return 1.0
    gt = row["gt_components"]
    if gt <= 0:
        return 0.0 if row["component_count_pred"] == 0 else 1.0
    return abs(row["component_count_pred"] - gt) / gt


def format_metrics_block(sub, label):
    """One aggregate row over a subset of the per-screen metrics DataFrame."""
    n = len(sub)
    if n == 0:
        return f"{label:<12} {'-':>8}  (no screens)"
    valid = sub[sub["json_valid"]]
    n_valid = len(valid)

    def pct(series):
        return f"{100.0 * series.mean():6.2f}%"

    def mean_pair(col):
        """valid-only mean / invalid-imputed-as-0 mean (A1: report both)."""
        valid_mean = valid[col].mean() if n_valid else 0.0
        imputed = sub[col].where(sub["json_valid"], 0.0).mean()
        return f"{valid_mean:.4f}/{imputed:.4f}"

    return (
        f"{label:<12} {n:>8} "
        f"{pct(sub['json_valid']):>9} "
        f"{pct(sub['app_match']):>9} "
        f"{pct(sub['activity_match']):>9} "
        f"{mean_pair('class_accuracy'):>15} "
        f"{mean_pair('avg_iou'):>15} "
        f"{mean_pair('string_similarity'):>15} "
        f"{sub['rel_count_error'].mean():>10.4f}"
    )


def run_evaluate(args):
    predictions_path = os.path.join(OUTPUT_DIR, f"predictions_{args.tag}.jsonl")
    if not os.path.exists(predictions_path):
        print(f"[-] Error: {predictions_path} not found. "
              f"Run the generate phase first.", file=sys.stderr)
        return

    records = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[*] Loaded {len(records)} predictions from {predictions_path}")

    rows = []
    for rec in records:
        try:
            gt_str = process_json_file(rec["json_path"], minify=True)
        except Exception as e:
            print(f"[-] Warning: could not clean GT for index {rec['index']}: {e}")
            continue

        metrics = evaluate_json_quality(rec["predicted_json"], gt_str)
        metrics["index"] = rec["index"]
        metrics["json_path"] = rec["json_path"]
        metrics["gen_time_s"] = rec.get("gen_time_s")
        metrics["gt_components"] = gt_component_count(gt_str)
        rows.append(metrics)

    if not rows:
        print("[-] Error: no screens could be evaluated.", file=sys.stderr)
        return

    df = pd.DataFrame(rows)
    df["rel_count_error"] = df.apply(relative_count_error, axis=1)

    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_{args.tag}.csv")
    df.to_csv(metrics_path, index=False)
    print(f"[*] Per-screen metrics written to {metrics_path}")

    # -------------------------------------------------------------------------
    # Aggregate report
    # -------------------------------------------------------------------------
    header = (
        f"{'':<12} {'screens':>8} {'valid':>9} {'app':>9} {'activity':>9} "
        f"{'class acc':>15} {'bounds IoU':>15} {'string sim':>15} {'count err':>10}"
    )
    lines = [
        "=" * len(header),
        f"DATASET RECONSTRUCTION EVALUATION  (tag: {args.tag})",
        f"Screens evaluated: {len(df)}   "
        f"Mean generation time: {df['gen_time_s'].mean():.2f}s/screen",
        "",
        "Continuous metrics are shown as 'valid-only/invalid-as-0'.",
        "count err = |pred - gt| / gt components (invalid prediction = 1.0).",
        "=" * len(header),
        header,
        "-" * len(header),
        format_metrics_block(df, "OVERALL"),
        "-" * len(header),
    ]
    for label, lo, hi in COMPLEXITY_BUCKETS:
        bucket = df[(df["gt_components"] >= lo) & (df["gt_components"] <= hi)]
        lines.append(format_metrics_block(bucket, f"gt comp {label}"))
    lines.append("=" * len(header))

    report = "\n".join(lines)
    report_path = os.path.join(OUTPUT_DIR, f"report_{args.tag}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"[*] Report written to {report_path}\n")
    print(report)


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Dataset-level evaluation of UI JSON reconstruction "
                    "on the validation split."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    gen = subparsers.add_parser("generate", help="Generate predictions (GPU)")
    gen.add_argument("--projector_model", type=str, required=True,
                     help="Path to trained EmbeddingProjector weights (.pth)")
    gen.add_argument("--lora_model", type=str, required=True,
                     help="Path to trained LoRA adapter directory")
    gen.add_argument("--tag", type=str, required=True,
                     help="Run tag used in all output file names")
    gen.add_argument("--num_samples", type=int, default=1000,
                     help="Seeded random sample size from the validation split")
    gen.add_argument("--full", action="store_true",
                     help="Evaluate the entire validation split")
    gen.add_argument("--seed", type=int, default=42,
                     help="Seed for the random sample")
    gen.add_argument("--batch_size", type=int, default=32,
                     help="Screens generated per batch")
    gen.add_argument("--gpu", type=str, default=None,
                     help="CUDA device id to use (overrides utils.CUDA_VISIBLE_DEVICE)")
    gen.add_argument("--max_tokens", type=int, default=4096,
                     help="Maximum tokens to generate per screen (matches "
                          "AdapterConfig.max_target_len: the adapter is "
                          "trained on targets up to this length)")
    gen.add_argument("--embeddings_dir", type=str,
                     default="embeddings_4096_last",
                     help="Path to embeddings directory (accepts both "
                          "(4096,) and (N, 4096) embeddings; must match the "
                          "config the adapter was trained on)")
    gen.add_argument("--data_dir", type=str, default=None,
                     help="Path to raw Rico JSON directory (optional)")
    gen.add_argument("--lm_model_id", type=str, default=None,
                     help="Override the base LM (default: read from the LoRA "
                          "checkpoint's adapter_config.json)")

    ev = subparsers.add_parser("evaluate", help="Score stored predictions (CPU)")
    ev.add_argument("--tag", type=str, required=True,
                    help="Run tag of the predictions to evaluate")

    args = parser.parse_args()

    if args.mode == "generate":
        # Override the pin inherited from importing 3_json_reconstruction.
        # Must happen before the first CUDA initialization.
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        run_generate(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
