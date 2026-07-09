# Embedding-dimension sweep for the Table 3 (GUI Screen Prediction) pipeline.
#
# For each projection dimension D, runs the full downstream chain on top of
# the frozen raw Qwen3-VL embeddings:
#
#   5_train_sequence_projector.py --proj_dim D --hidden_dim 4096
#       -> embeddings_seq_D/ + sequence_projector_Dd.pth
#   8_train_lstm_predictor.py --embeddings_dir embeddings_seq_D
#       -> lstm_predictor_Dd.pth
#   4_evaluate_screen_prediction.py --embeddings_dir embeddings_seq_D
#       --lstm_model lstm_predictor_Dd.pth
#       -> Table 3 metrics (parsed from the comparison table)
#
# Results accumulate in dimension_sweep/results.json; after every completed
# dimension the summary table (sweep_table.txt) and the curve figure
# (sweep_curve.png) are regenerated, so partial sweeps are always visual.
# Already-completed dimensions are skipped on re-run (use --force to redo).
#
# The projector hidden layer is fixed at 4096 for every point so that no
# dimension is rank-bottlenecked and all sweep points share an identical
# trunk — the only variable is the output width.
#
# Usage:
#   # Full sweep (run in a screen session; each dim takes ~10-30 min):
#   python 4_sweep_embedding_dimensions.py run --gpu 3
#
#   # Selected dimensions, plus the raw-4096 no-projector baseline:
#   python 4_sweep_embedding_dimensions.py run --dims 64,256,768 --include_raw
#
#   # Regenerate table + figure from saved results only:
#   python 4_sweep_embedding_dimensions.py plot

import os
import re
import sys
import json
import argparse
import importlib
import subprocess
from datetime import datetime

SWEEP_DIR = "dimension_sweep"
RESULTS_PATH = os.path.join(SWEEP_DIR, "results.json")
TABLE_PATH = os.path.join(SWEEP_DIR, "sweep_table.txt")
FIGURE_PATH = os.path.join(SWEEP_DIR, "sweep_curve.png")
LOG_DIR = os.path.join(SWEEP_DIR, "logs")

DEFAULT_DIMS = [32, 64, 128, 256, 512, 768, 1024]
RAW_KEY = "raw4096"  # untouched Qwen embeddings, no projector

METRIC_KEYS = ["top1", "top001pct", "top01pct", "top1pct", "top5pct", "nrmse"]
METRIC_LABELS = {
    "top1": "Top-1", "top001pct": "Top-0.01%", "top01pct": "Top-0.1%",
    "top1pct": "Top-1%", "top5pct": "Top-5%", "nrmse": "Norm.RMSE",
}


# =============================================================================
# Results store
# =============================================================================
def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"meta": {}, "runs": {}}


def save_results(results):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


# =============================================================================
# Pipeline stages (subprocesses; each stage streams to its own log file)
# =============================================================================
def run_stage(cmd, log_name, env):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, log_name)
    print(f"    $ {' '.join(cmd)}")
    print(f"      (log: {log_path})")
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Stage failed (exit {proc.returncode}) — see {log_path}"
        )
    return log_path


def parse_eval_metrics(log_path, label):
    """Extract the six metric floats from the 'Ours' row of the comparison
    table printed by 4_evaluate_screen_prediction.py."""
    row_re = re.compile(
        re.escape(label) + r"\s+"
        + r"\s+".join([r"([\d.]+)"] * 6)
    )
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = row_re.search(line)
            if m:
                return dict(zip(METRIC_KEYS, (float(g) for g in m.groups())))
    raise RuntimeError(f"Could not find metrics row '{label}' in {log_path}")


def run_dimension(dim, args, env):
    """Run projector -> LSTM -> evaluation for one dimension.
    dim is an int, or RAW_KEY for the no-projector baseline."""
    py = sys.executable
    is_raw = dim == RAW_KEY
    tag = RAW_KEY if is_raw else str(dim)

    if is_raw:
        emb_dir = args.embeddings_dir
        lstm_path = "lstm_predictor_raw4096.pth"
    else:
        emb_dir = f"embeddings_seq_{dim}"
        lstm_path = f"lstm_predictor_{dim}d.pth"

        # Stage 1: projector (skipped if its output already exists)
        if os.path.isdir(emb_dir) and os.listdir(emb_dir):
            print(f"    [skip] {emb_dir} already exists")
        else:
            print(f"  [1/3] Training projector ({dim}-dim)...")
            run_stage(
                [py, "5_train_sequence_projector.py",
                 "--proj_dim", str(dim),
                 "--hidden_dim", str(args.hidden_dim),
                 "--seed", str(args.seed)],
                f"dim{tag}_1_projector.log", env,
            )

    # Stage 2: LSTM predictor
    if os.path.exists(lstm_path) and not args.force:
        print(f"    [skip] {lstm_path} already exists")
    else:
        print(f"  [2/3] Training LSTM predictor on {emb_dir}...")
        run_stage(
            [py, "8_train_lstm_predictor.py",
             "--embeddings_dir", emb_dir,
             "--output", lstm_path,
             "--seed", str(args.seed)],
            f"dim{tag}_2_lstm.log", env,
        )

    # Stage 3: Table 3 evaluation
    print(f"  [3/3] Evaluating screen prediction...")
    label = f"OURS_{tag}"
    eval_cmd = [py, "4_evaluate_screen_prediction.py",
                "--embeddings_dir", emb_dir,
                "--lstm_model", lstm_path,
                "--label", label]
    if args.sample_fraction is not None:
        eval_cmd += ["--sample_fraction", str(args.sample_fraction)]
    log_path = run_stage(eval_cmd, f"dim{tag}_3_eval.log", env)

    metrics = parse_eval_metrics(log_path, label)
    metrics["completed_at"] = datetime.now().isoformat(timespec="seconds")
    return metrics


# =============================================================================
# Outputs: summary table + curve figure
# =============================================================================
def sorted_run_keys(runs):
    numeric = sorted(int(k) for k in runs if k != RAW_KEY)
    keys = [str(k) for k in numeric]
    if RAW_KEY in runs:
        keys.append(RAW_KEY)
    return keys


def paper_reference():
    """Screen2Vec's best Table 3 row, imported from the evaluation script."""
    try:
        mod = importlib.import_module("4_evaluate_screen_prediction")
        return mod.PAPER_TABLE3["Screen2Vec-EUCLIDEAN-spatial"]
    except Exception:
        return None


def write_table(results):
    runs = results["runs"]
    ref = paper_reference()
    header = f"{'dim':>10} " + " ".join(
        f"{METRIC_LABELS[k]:>10}" for k in METRIC_KEYS
    )
    lines = [
        f"Embedding-dimension sweep — GUI Screen Prediction (Table 3 protocol)",
        f"hidden_dim={results['meta'].get('hidden_dim')} "
        f"seed={results['meta'].get('seed')}",
        "=" * len(header),
        header,
        "-" * len(header),
    ]
    for key in sorted_run_keys(runs):
        m = runs[key]
        label = "raw 4096" if key == RAW_KEY else key
        lines.append(
            f"{label:>10} " + " ".join(f"{m[k]:>10.4f}" for k in METRIC_KEYS)
        )
    if ref:
        lines.append("-" * len(header))
        lines.append(
            f"{'S2V-spat.':>10} "
            + " ".join(f"{ref[k]:>10.4f}" for k in METRIC_KEYS)
            + "   (Screen2Vec-EUCLIDEAN-spatial, paper)"
        )
    lines.append("=" * len(header))
    table = "\n".join(lines)
    with open(TABLE_PATH, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    return table


def write_figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = results["runs"]
    dims = sorted(int(k) for k in runs if k != RAW_KEY)
    if not dims:
        return

    # Validated categorical palette (light mode), one hue per metric
    series = [
        ("top1", "#2a78d6"),      # blue
        ("top001pct", "#1baf7a"),  # aqua
        ("top01pct", "#eda100"),   # yellow
        ("top1pct", "#008300"),    # green
    ]
    ink, ink2, grid = "#1a1a19", "#575551", "#d9d8d2"
    ref = paper_reference()
    raw = runs.get(RAW_KEY)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.8), dpi=150,
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.patch.set_facecolor("white")

    for a in (ax, ax2):
        a.set_facecolor("white")
        a.grid(True, color=grid, linewidth=0.8, alpha=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(ink2)
        a.tick_params(colors=ink2, labelsize=9)
        a.set_xscale("log", base=2)
        a.set_xticks(dims)
        a.set_xticklabels([str(d) for d in dims], rotation=0)
        a.minorticks_off()
        a.set_xlabel("Projection dimension", color=ink, fontsize=10)

    # --- Left panel: Top-k accuracies (higher is better) ---
    for key, color in series:
        ys = [runs[str(d)][key] for d in dims]
        ax.plot(dims, ys, color=color, linewidth=2, marker="o",
                markersize=5, label=METRIC_LABELS[key])
        ax.annotate(METRIC_LABELS[key], (dims[-1], ys[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=ink, fontsize=8.5, va="center")
        if ref:
            ax.axhline(ref[key], color=color, linewidth=1.2,
                       linestyle=(0, (4, 3)), alpha=0.55)
        if raw:
            ax.plot([dims[-1] * 1.5], [raw[key]], marker="D",
                    markersize=5, color=color, linestyle="none")
    ax.set_ylabel("Accuracy", color=ink, fontsize=10)
    ax.set_ylim(0, 1.02)
    title = "GUI screen prediction accuracy vs. embedding dimension"
    extra = []
    if ref:
        extra.append("dashed = Screen2Vec-spatial (paper)")
    if raw:
        extra.append("◆ = raw 4096-dim, no projector")
    if extra:
        title += "\n(" + ";  ".join(extra) + ")"
    ax.set_title(title, color=ink, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=ink, loc="lower right")

    # --- Right panel: Normalized RMSE (lower is better) ---
    ys = [runs[str(d)]["nrmse"] for d in dims]
    ax2.plot(dims, ys, color="#2a78d6", linewidth=2, marker="o",
             markersize=5, label="Ours")
    if ref:
        ax2.axhline(ref["nrmse"], color="#2a78d6", linewidth=1.2,
                    linestyle=(0, (4, 3)), alpha=0.55)
    if raw:
        ax2.plot([dims[-1] * 1.5], [raw["nrmse"]], marker="D",
                 markersize=5, color="#2a78d6", linestyle="none")
    ax2.set_ylabel("Normalized RMSE  (lower is better)",
                   color=ink, fontsize=10)
    ax2.set_title("Prediction error", color=ink, fontsize=10.5, loc="left")

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(fig)


def refresh_outputs(results):
    table = write_table(results)
    write_figure(results)
    print(f"\n{table}\n")
    print(f"[*] Table:  {TABLE_PATH}")
    print(f"[*] Figure: {FIGURE_PATH}")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Sweep projection dimensions for the Table 3 pipeline "
                    "and visualize the accuracy/size trade-off."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    run_p = sub.add_parser("run", help="Run the sweep (resumable)")
    run_p.add_argument("--dims", type=str,
                       default=",".join(str(d) for d in DEFAULT_DIMS),
                       help="Comma-separated projection dimensions")
    run_p.add_argument("--include_raw", action="store_true",
                       help="Also evaluate the raw 4096-dim embeddings "
                            "without any projector (baseline)")
    run_p.add_argument("--hidden_dim", type=int, default=4096,
                       help="Projector hidden width, fixed across the sweep")
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--gpu", type=str, default=None,
                       help="CUDA device id for the training stages")
    run_p.add_argument("--force", action="store_true",
                       help="Re-run dimensions already in results.json")
    run_p.add_argument("--sample_fraction", type=float, default=None,
                       help="Evaluate on a fraction of traces (quick tests)")
    run_p.add_argument("--embeddings_dir", type=str, default="embeddings",
                       help="Directory with the raw Qwen embeddings")

    sub.add_parser("plot", help="Regenerate table + figure from results.json")

    args = parser.parse_args()
    results = load_results()

    if args.mode == "plot":
        if not results["runs"]:
            print("No results yet — run the sweep first.")
            return
        refresh_outputs(results)
        return

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    results["meta"] = {"hidden_dim": args.hidden_dim, "seed": args.seed}
    todo = [int(d) for d in args.dims.split(",") if d.strip()]
    keys = [str(d) for d in todo] + ([RAW_KEY] if args.include_raw else [])

    print(f"[*] Sweep plan: {keys}  "
          f"(hidden_dim={args.hidden_dim}, seed={args.seed}, "
          f"gpu={args.gpu or 'default'})")

    for i, key in enumerate(keys, 1):
        if key in results["runs"] and not args.force:
            print(f"\n[{i}/{len(keys)}] dim={key}: already done, skipping "
                  f"(--force to redo)")
            continue
        print(f"\n[{i}/{len(keys)}] dim={key}  "
              f"({datetime.now().strftime('%H:%M:%S')})")
        dim = RAW_KEY if key == RAW_KEY else int(key)
        try:
            metrics = run_dimension(dim, args, env)
        except RuntimeError as e:
            print(f"  [-] FAILED: {e}")
            print("      Fix the issue and re-run — completed dims are kept.")
            continue
        results["runs"][key] = metrics
        save_results(results)
        refresh_outputs(results)

    print("\n[*] Sweep finished.")


if __name__ == "__main__":
    main()
