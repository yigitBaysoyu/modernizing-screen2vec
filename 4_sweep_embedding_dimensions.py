"""
4_sweep_embedding_dimensions.py

Embedding-dimension sweep for the Table 3 (GUI Screen Prediction) pipeline.

Question answered: how small can the learned screen embedding be before
next-screen retrieval stops beating Screen2Vec's Table 3?  Screen2Vec uses
768 learned dimensions (+ 768 app-description dimensions).  The sweep runs
the repository's *current* Table 3 recipe once per projection dimension D
and collects the retrieval metrics into one table and one figure.

The recipe is the joint pipeline of _run_table3_pipeline2.ps1, verbatim
except for the dimension (and, optionally, hidden width / seed / protocol):

    1. 9_train_joint.py  --proj_dim D  [--hidden_dim H] [--seed S] [--strict | --nback K]
         -> embeddings_seq_D{proto}{tag}/        projected screen embeddings
            lstm_predictor_Dd{proto}{tag}.pth    LSTM checkpoint
            sequence_projector_Dd{proto}{tag}.pth
    2. 6_generate_app_context_embeddings.py
         -> embeddings_seq_D{proto}{tag}_with_app/   (D + 768 app dims)
    3. 4_evaluate_screen_prediction.py  [--strict | --nback K]
         -> Top-1, Top-k%, nRMSE, MRR, population sizes (parsed from stdout)

{proto} is the protocol suffix the scripts themselves use ("" permissive,
"_strict", "_nbackK"); {tag} is empty for the recipe defaults (hidden 1024,
seed 42) and "_h<H>" / "_s<S>" otherwise, so a hidden-width or seed repeat
can never overwrite the canonical checkpoints (9_train_joint.py
--artifact_tag).

The default sweep covers D = 16 ... 2048.  Because D=2048 needs a projector
hidden layer wider than the recipe's 1024, the default hidden width is 4096
for every point (one width across the whole curve, so only D varies); run
with --hidden_dim 1024 to reproduce the README recipe exactly for D <= 1024,
in which case the D=768 point writes the very artefacts the notebooks read.

Every stage resolves trace keys through trace_layout_index.csv (the scripts
refuse to run without it); numbers from before the 2026-08-21 index fix are
void and must not be mixed into a results file -- see FINDINGS.md.

Reading the results (FINDINGS.md sections 2-3): Top-1 and MRR are the
informative columns.  Top-0.1% and above are saturated by the app-description
half (an app-only oracle already reaches 0.9968 at Top-0.1%), and normalised
RMSE is not comparable to the paper's.  The figure therefore plots Top-1 and
MRR on their own panel; the other columns are kept for completeness.

Results accumulate in dimension_sweep/results{proto}{_hH}.json with one entry
per (dimension, seed); the summary table, CSV and figure are regenerated after
every completed run.  Completed (dimension, seed) pairs are skipped on re-run
(--force to redo).  Per-stage logs go to dimension_sweep/logs/ (gitignored).

Usage:
    # Default sweep (16 ... 2048, permissive protocol, seed 42, hidden 4096):
    python 4_sweep_embedding_dimensions.py run --gpu 0

    # README recipe width (hidden 1024; D=768 is then the canonical run):
    python 4_sweep_embedding_dimensions.py run --hidden_dim 1024 --dims 16,32,64,128,256,512,768,1024

    # Screen2Vec's own population, selected dimensions:
    python 4_sweep_embedding_dimensions.py run --protocol strict --dims 64,256,768

    # Seed repeats (mean +- std, error bars); non-42 intermediates are deleted
    # after their metrics are recorded unless --keep_artifacts:
    python 4_sweep_embedding_dimensions.py run --dims 32,128,768 --seeds 42,43,44

    # Show the commands without running anything:
    python 4_sweep_embedding_dimensions.py run --dims 768 --dry_run

    # Regenerate table / CSV / figure from the saved results:
    python 4_sweep_embedding_dimensions.py plot [--protocol strict] [--hidden_dim 4096]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import trace_index

SWEEP_DIR = Path("dimension_sweep")
DEFAULT_DIMS = (16, 32, 64, 128, 256, 512, 768, 1024, 2048)
# Width of the projector's hidden layer.  The README recipe uses 1024, which
# caps the sweep at D=1024 (a wider output would be rank-limited).  The
# default sweep therefore uses 4096 for EVERY point, so only D varies across
# the curve; artefacts and results of that width carry a _h4096 suffix, and
# --hidden_dim 1024 reproduces the README recipe exactly for D <= 1024.
RECIPE_HIDDEN_DIM = 1024
DEFAULT_HIDDEN_DIM = 4096
RECIPE_SEED = 42
RAW_EMBEDDINGS_DIR = "embeddings_4096_last"

METRIC_KEYS = ("top1", "top001pct", "top01pct", "top1pct", "top5pct", "nrmse", "mrr")
METRIC_LABELS = {
    "top1": "Top-1", "top001pct": "Top-0.01%", "top01pct": "Top-0.1%",
    "top1pct": "Top-1%", "top5pct": "Top-5%", "nrmse": "Norm.RMSE", "mrr": "MRR",
}
POPULATION_KEYS = ("n_traces", "n_screens_in_traces", "n_instances")

STAGES = {
    "train": "9_train_joint.py",
    "app": "6_generate_app_context_embeddings.py",
    "eval": "4_evaluate_screen_prediction.py",
}


# =============================================================================
# Naming: protocol suffix, artefact tag, result files
# =============================================================================
def protocol_suffix(protocol: str, nback: int | None) -> str:
    """The suffix 9_train_joint.py / 4_evaluate_screen_prediction.py use."""
    if protocol == "strict":
        return "_strict"
    if protocol == "nback":
        return f"_nback{nback}"
    return ""


def protocol_flags(protocol: str, nback: int | None) -> list[str]:
    """Flags shared by the trainer and the evaluator (both must get them)."""
    if protocol == "strict":
        return ["--strict"]
    if protocol == "nback":
        return ["--nback", str(nback)]
    return []


def artifact_tag(hidden_dim: int, seed: int) -> str:
    """Empty for the recipe defaults; otherwise _h<H>_s<S> parts as needed."""
    tag = ""
    if hidden_dim != RECIPE_HIDDEN_DIM:
        tag += f"_h{hidden_dim}"
    if seed != RECIPE_SEED:
        tag += f"_s{seed}"
    return tag


def artifact_paths(dim: int, proto: str, tag: str) -> dict[str, str]:
    stem = f"{dim}d{proto}{tag}"
    emb_dir = f"embeddings_seq_{dim}{proto}{tag}"
    return {
        "emb_dir": emb_dir,
        "app_dir": f"{emb_dir}_with_app",
        "lstm": f"lstm_predictor_{stem}.pth",
        "projector": f"sequence_projector_{stem}.pth",
        "plot": f"joint_training_{stem}.png",
    }


class SweepPaths:
    """Result files of one sweep configuration (protocol x hidden width)."""

    def __init__(self, protocol: str, nback: int | None, hidden_dim: int, sweep_dir: Path = SWEEP_DIR):
        suffix = protocol_suffix(protocol, nback)
        if hidden_dim != RECIPE_HIDDEN_DIM:
            suffix += f"_h{hidden_dim}"
        self.sweep_dir = Path(sweep_dir)
        self.results = self.sweep_dir / f"results{suffix}.json"
        self.table = self.sweep_dir / f"sweep_table{suffix}.txt"
        self.csv = self.sweep_dir / f"sweep_results{suffix}.csv"
        self.figure = self.sweep_dir / f"sweep_curve{suffix}.png"
        self.logs = self.sweep_dir / "logs"
        self.suffix = suffix


# =============================================================================
# Results store: runs[dim][seed] = metrics
# =============================================================================
def load_results(paths: SweepPaths) -> dict:
    if paths.results.exists():
        with open(paths.results, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"meta": {}, "runs": {}}


def save_results(paths: SweepPaths, results: dict) -> None:
    paths.sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.results, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def trace_index_meta(path: str) -> dict | None:
    meta_path = Path(path).with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {k: meta.get(k) for k in ("generated_utc", "mapped", "dropped", "coverage")}


# =============================================================================
# Stage execution
# =============================================================================
def run_stage(cmd: list[str], log_path: Path, env: dict) -> Path:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"    $ {' '.join(cmd)}")
    print(f"      (log: {log_path})")
    with open(log_path, "w", encoding="utf-8") as log_fh:
        proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Stage failed (exit {proc.returncode}); see {log_path}")
    return log_path


_MRR_RE = re.compile(r"MRR\s+([\d.]+)")
_POP_RE = re.compile(
    r"Eligible traces evaluated:\s*(\d+)\s*\((\d+) screens,\s*(\d+) prediction instances\)"
)


def parse_eval_metrics(text: str, label: str) -> dict:
    """
    Metrics of the labelled 'Ours' row in the comparison table printed by
    4_evaluate_screen_prediction.py, plus the MRR line and the population
    line printed after it.
    """
    row_re = re.compile(re.escape(label) + r"\s+" + r"\s+".join([r"([\d.]+)"] * 6))
    metrics = None
    for line in text.splitlines():
        match = row_re.search(line)
        if match:
            values = [float(g) for g in match.groups()]
            metrics = dict(zip(("top1", "top001pct", "top01pct", "top1pct", "top5pct", "nrmse"), values))
            break
    if metrics is None:
        raise RuntimeError(f"Could not find the metrics row '{label}' in the evaluator output")
    mrr = _MRR_RE.search(text)
    metrics["mrr"] = float(mrr.group(1)) if mrr else float("nan")
    pop = _POP_RE.search(text)
    if pop:
        metrics.update(zip(POPULATION_KEYS, (int(g) for g in pop.groups())))
    return metrics


def build_commands(dim: int, args, seed: int) -> list[tuple[str, list[str]]]:
    """The three stage commands for one (dimension, seed), in order."""
    py = sys.executable
    proto = protocol_suffix(args.protocol, args.nback)
    tag = artifact_tag(args.hidden_dim, seed)
    art = artifact_paths(dim, proto, tag)
    flags = protocol_flags(args.protocol, args.nback)

    train = [py, STAGES["train"],
             "--proj_dim", str(dim), "--hidden_dim", str(args.hidden_dim),
             "--seed", str(seed), "--output_dir", art["emb_dir"],
             "--raw_embeddings_dir", args.embeddings_dir,
             "--trace_index", args.trace_index, *flags]
    if tag:
        train += ["--artifact_tag", tag]
    app = [py, STAGES["app"],
           "--embeddings_dir", art["emb_dir"], "--output_dir", art["app_dir"],
           "--trace_index", args.trace_index]
    evaluate = [py, STAGES["eval"],
                "--embeddings_dir", art["app_dir"], "--lstm_model", art["lstm"],
                "--label", run_label(dim, seed), "--trace_index", args.trace_index, *flags]
    if args.sample_fraction is not None:
        evaluate += ["--sample_fraction", str(args.sample_fraction)]
    return [("train", train), ("app", app), ("eval", evaluate)]


def run_label(dim: int, seed: int) -> str:
    return f"SWEEP_D{dim}_S{seed}"


def run_dimension(dim: int, args, env: dict, paths: SweepPaths, seed: int) -> dict:
    """Recipe steps 1-3 for one (dimension, seed); returns the parsed metrics."""
    proto = protocol_suffix(args.protocol, args.nback)
    art = artifact_paths(dim, proto, artifact_tag(args.hidden_dim, seed))
    commands = dict(build_commands(dim, args, seed))
    stem = f"dim{dim}{proto}{artifact_tag(args.hidden_dim, seed)}"

    if os.path.isdir(art["emb_dir"]) and os.listdir(art["emb_dir"]) and os.path.exists(art["lstm"]) and not args.force:
        print(f"    [skip] {art['emb_dir']}/ and {art['lstm']} already exist")
    else:
        print(f"  [1/3] Joint projector + LSTM training (D={dim}, hidden {args.hidden_dim}, seed {seed})")
        run_stage(commands["train"], paths.logs / f"{stem}_1_train.log", env)
        if not os.path.exists(art["lstm"]):
            raise RuntimeError(f"Trainer finished but {art['lstm']} is missing")

    if os.path.isdir(art["app_dir"]) and os.listdir(art["app_dir"]) and not args.force:
        print(f"    [skip] {art['app_dir']}/ already exists")
    else:
        print("  [2/3] App-description concatenation")
        run_stage(commands["app"], paths.logs / f"{stem}_2_app.log", env)

    print("  [3/3] Screen prediction evaluation")
    log_path = run_stage(commands["eval"], paths.logs / f"{stem}_3_eval.log", env)
    metrics = parse_eval_metrics(log_path.read_text(encoding="utf-8"), run_label(dim, seed))
    metrics["seed"] = seed
    metrics["completed_at"] = datetime.now().isoformat(timespec="seconds")
    metrics["artifacts"] = art
    return metrics


def cleanup_artifacts(dim: int, args, seed: int) -> None:
    """Remove the bulky intermediates of a non-recipe seed once recorded."""
    if seed == RECIPE_SEED:
        return
    art = artifact_paths(dim, protocol_suffix(args.protocol, args.nback), artifact_tag(args.hidden_dim, seed))
    for key in ("emb_dir", "app_dir"):
        if os.path.isdir(art[key]):
            shutil.rmtree(art[key])
    for key in ("lstm", "projector", "plot"):
        if os.path.exists(art[key]):
            os.remove(art[key])
    print(f"    [cleanup] removed seed-{seed} intermediates for D={dim}")


# =============================================================================
# Aggregation and outputs
# =============================================================================
def paper_reference() -> dict | None:
    """Screen2Vec's best Table 3 row, read from the evaluator so it has one home."""
    try:
        import importlib
        return importlib.import_module("4_evaluate_screen_prediction").PAPER_TABLE3["Screen2Vec-EUCLIDEAN-spatial"]
    except Exception:
        return None


def seed_stats(results: dict) -> dict:
    """Per-dimension mean / std / n over seeds, for every metric."""
    stats = {}
    for dim_key, by_seed in results["runs"].items():
        entry = {"n": len(by_seed)}
        for key in METRIC_KEYS:
            values = [m[key] for m in by_seed.values() if key in m and m[key] == m[key]]
            entry[key] = {
                "mean": statistics.mean(values) if values else float("nan"),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        stats[dim_key] = entry
    return stats


def write_table(paths: SweepPaths, results: dict) -> str:
    stats = seed_stats(results)
    ref = paper_reference()
    meta = results.get("meta", {})
    multi = any(s["n"] > 1 for s in stats.values())
    col = 15 if multi else 10
    header = f"{'dim':>6} {'n':>3} " + " ".join(f"{METRIC_LABELS[k]:>{col}}" for k in METRIC_KEYS)
    lines = [
        "Embedding-dimension sweep - GUI Screen Prediction (joint recipe: 9_ -> 6_ -> 4_)",
        f"protocol={meta.get('protocol', '?')}  hidden_dim={meta.get('hidden_dim', '?')}  "
        f"raw={meta.get('raw_embeddings_dir', '?')}  seeds={meta.get('seeds', '?')}",
        "Final embedding per point = D screen dims + 768 app-description dims",
        "Top-1 and MRR are the informative columns; Top-0.1% and above are saturated by app identity,",
        "and Norm.RMSE is not comparable to the paper's (FINDINGS.md 2-3).",
        "=" * len(header), header, "-" * len(header),
    ]
    for dim_key in sorted(stats, key=int):
        s = stats[dim_key]
        if multi:
            cells = " ".join(f"{s[k]['mean']:.4f}+-{s[k]['std']:.4f}".rjust(col) for k in METRIC_KEYS)
        else:
            cells = " ".join(f"{s[k]['mean']:>{col}.4f}" for k in METRIC_KEYS)
        lines.append(f"{dim_key:>6} {s['n']:>3} {cells}")
    if ref:
        lines.append("-" * len(header))
        cells = " ".join((f"{ref[k]:.4f}" if k in ref else "n/a").rjust(col) for k in METRIC_KEYS)
        lines.append(f"{'S2V':>6} {'':>3} {cells}   (Screen2Vec-EUCLIDEAN-spatial, paper; 768+768 dims)")
    lines.append("=" * len(header))
    table = "\n".join(lines)
    paths.table.write_text(table + "\n", encoding="utf-8")
    return table


def write_csv(paths: SweepPaths, results: dict) -> None:
    """One row per (dimension, seed): flat, for the result notebooks."""
    import csv
    rows = []
    for dim_key, by_seed in results["runs"].items():
        for seed_key, m in by_seed.items():
            row = {"proj_dim": int(dim_key), "seed": int(seed_key)}
            row.update({k: m.get(k) for k in METRIC_KEYS})
            row.update({k: m.get(k) for k in POPULATION_KEYS})
            row["completed_at"] = m.get("completed_at")
            rows.append(row)
    rows.sort(key=lambda r: (r["proj_dim"], r["seed"]))
    with open(paths.csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["proj_dim", "seed"])
        writer.writeheader()
        writer.writerows(rows)


def write_figure(paths: SweepPaths, results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = seed_stats(results)
    dims = sorted(int(k) for k in stats)
    if not dims:
        return
    ref = paper_reference()
    meta = results.get("meta", {})
    n_values = [stats[str(d)]["n"] for d in dims]
    n_text = f"n={min(n_values)}" if min(n_values) == max(n_values) else f"n={min(n_values)}-{max(n_values)}"

    fig, (ax_main, ax_sat) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150, gridspec_kw={"width_ratios": [1.15, 1]})
    for ax in (ax_main, ax_sat):
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims])
        ax.minorticks_off()
        ax.set_xlabel("Screen-embedding dimension D (of D + 768)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def curve(ax, key, color, label):
        ys = [stats[str(d)][key]["mean"] for d in dims]
        es = [stats[str(d)][key]["std"] for d in dims]
        ax.errorbar(dims, ys, yerr=es if max(n_values) > 1 else None, color=color,
                    linewidth=2, marker="o", markersize=4, capsize=3, label=label)
        if ref and key in ref:
            ax.axhline(ref[key], color=color, linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.6)

    curve(ax_main, "top1", "#2a78d6", "Top-1")
    curve(ax_main, "mrr", "#d1495b", "MRR (no paper value)")
    ax_main.set_ylim(0, None)
    ax_main.set_ylabel("Retrieval quality")
    ax_main.set_title(f"Informative metrics (dashed = Screen2Vec-spatial, paper; {n_text} seeds)", loc="left", fontsize=10)
    ax_main.legend(frameon=False, fontsize=9, loc="lower right")

    curve(ax_sat, "top001pct", "#1baf7a", "Top-0.01%")
    curve(ax_sat, "top01pct", "#eda100", "Top-0.1%")
    curve(ax_sat, "top1pct", "#008300", "Top-1%")
    ax_sat.set_ylim(0, 1.02)
    ax_sat.set_ylabel("Accuracy")
    ax_sat.set_title("Top-k% columns (saturated by app identity from Top-0.1%)", loc="left", fontsize=10)
    ax_sat.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle(f"GUI screen prediction vs. embedding dimension - protocol {meta.get('protocol', '?')}, "
                 f"hidden {meta.get('hidden_dim', '?')}", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(paths.figure, bbox_inches="tight")
    plt.close(fig)


def refresh_outputs(paths: SweepPaths, results: dict) -> None:
    table = write_table(paths, results)
    write_csv(paths, results)
    write_figure(paths, results)
    print(f"\n{table}\n")
    print(f"[*] Table : {paths.table}\n[*] CSV   : {paths.csv}\n[*] Figure: {paths.figure}")


# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep the projection dimension through the joint Table 3 recipe and plot the curve."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--protocol", choices=("permissive", "strict", "nback"), default="permissive",
                        help="Evaluation protocol, passed to BOTH the trainer and the evaluator "
                             "(default: permissive, the README recipe). See 4_evaluate_screen_prediction.py.")
    common.add_argument("--nback", type=int, default=None, metavar="K",
                        help="Context length for --protocol nback")
    common.add_argument("--hidden_dim", type=int, default=DEFAULT_HIDDEN_DIM,
                        help=f"Projector hidden width, applied to every point (default {DEFAULT_HIDDEN_DIM} so "
                             f"that D=2048 fits; the README recipe uses {RECIPE_HIDDEN_DIM}). Must be >= every "
                             "swept dimension. Widths other than the recipe's write _h<H>-suffixed "
                             "artefacts and results.")

    run_p = sub.add_parser("run", parents=[common], help="Run the sweep (resumable)")
    run_p.add_argument("--dims", type=str, default=",".join(str(d) for d in DEFAULT_DIMS),
                       help="Comma-separated projection dimensions")
    run_p.add_argument("--seeds", type=str, default=str(RECIPE_SEED),
                       help="Comma-separated seeds; more than one gives mean +- std and error bars")
    run_p.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES for the stages")
    run_p.add_argument("--embeddings_dir", type=str, default=RAW_EMBEDDINGS_DIR,
                       help=f"Raw pooled Qwen embeddings (default: {RAW_EMBEDDINGS_DIR})")
    run_p.add_argument("--trace_index", type=str, default=trace_index.DEFAULT_INDEX_PATH,
                       help="Trace->layout index from 0_build_trace_index.py (required by every stage)")
    run_p.add_argument("--sample_fraction", type=float, default=None,
                       help="NON-RECIPE, plumbing tests only: evaluate on a fraction of traces")
    run_p.add_argument("--force", action="store_true", help="Re-run (dimension, seed) pairs already recorded")
    run_p.add_argument("--keep_artifacts", action="store_true",
                       help="Keep non-42-seed intermediates instead of deleting them after recording")
    run_p.add_argument("--dry_run", action="store_true", help="Print the stage commands and exit")

    sub.add_parser("plot", parents=[common], help="Regenerate table, CSV and figure from the results file")
    return parser


def validate(args, parser) -> None:
    if args.protocol == "nback" and (args.nback is None or args.nback < 1):
        parser.error("--protocol nback requires --nback K with K >= 1")
    if args.protocol != "nback" and args.nback is not None:
        parser.error("--nback only applies to --protocol nback")


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate(args, parser)
    paths = SweepPaths(args.protocol, args.nback, args.hidden_dim)

    if args.mode == "plot":
        results = load_results(paths)
        if not results["runs"]:
            print(f"No results in {paths.results}; run the sweep first.")
            return
        refresh_outputs(paths, results)
        return

    dims = [int(d) for d in args.dims.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if max(dims) > args.hidden_dim:
        parser.error(f"proj_dim {max(dims)} > hidden_dim {args.hidden_dim}: the projector output would be "
                     f"rank-limited by its hidden layer; use --hidden_dim >= {max(dims)}")
    jobs = [(dim, seed) for dim in dims for seed in seeds]

    if args.dry_run:
        print(f"[dry run] protocol={args.protocol} hidden_dim={args.hidden_dim} -> {paths.results}")
        for dim, seed in jobs:
            print(f"\n# D={dim} seed={seed}")
            for _, cmd in build_commands(dim, args, seed):
                print("  " + " ".join(cmd))
        return

    if not Path(args.trace_index).exists():
        raise SystemExit(f"{args.trace_index} missing; run 0_build_trace_index.py first (every stage needs it)")
    for script in STAGES.values():
        if not Path(script).exists():
            raise SystemExit(f"{script} not found; run from the repository root")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    results = load_results(paths)
    results["meta"] = {
        "recipe": "9_train_joint.py -> 6_generate_app_context_embeddings.py -> 4_evaluate_screen_prediction.py",
        "protocol": args.protocol if args.protocol != "nback" else f"nback{args.nback}",
        "hidden_dim": args.hidden_dim,
        "raw_embeddings_dir": args.embeddings_dir,
        "trace_index": args.trace_index,
        "trace_index_meta": trace_index_meta(args.trace_index),
        "seeds": sorted({*seeds, *(int(s) for by in results["runs"].values() for s in by)}),
        "git_revision": git_revision(),
        "sample_fraction": args.sample_fraction,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    print(f"[*] Sweep plan: dims {dims} x seeds {seeds}  (protocol={results['meta']['protocol']}, "
          f"hidden_dim={args.hidden_dim}, gpu={args.gpu or 'default'})")
    if args.sample_fraction is not None:
        print(f"[!] sample_fraction={args.sample_fraction}: NOT the recipe, plumbing tests only")

    for i, (dim, seed) in enumerate(jobs, 1):
        key, seed_key = str(dim), str(seed)
        if key in results["runs"] and seed_key in results["runs"][key] and not args.force:
            print(f"\n[{i}/{len(jobs)}] D={dim} seed={seed}: already recorded, skipping (--force to redo)")
            continue
        print(f"\n[{i}/{len(jobs)}] D={dim} seed={seed}  ({datetime.now().strftime('%H:%M:%S')})")
        try:
            metrics = run_dimension(dim, args, env, paths, seed)
        except RuntimeError as exc:
            print(f"  [-] FAILED: {exc}\n      Fix the issue and re-run; completed runs are kept.")
            continue
        if not args.keep_artifacts:
            cleanup_artifacts(dim, args, seed)
        results["runs"].setdefault(key, {})[seed_key] = metrics
        save_results(paths, results)
        refresh_outputs(paths, results)

    print("\n[*] Sweep finished.")


if __name__ == "__main__":
    main()
