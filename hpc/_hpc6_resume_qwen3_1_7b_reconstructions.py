"""Resume the incomplete Qwen3-1.7B-Base reconstruction run on HPC.

Run this on the login node from the checked-out project. It submits one Slurm
array with eight shards per selected run, capped at eight concurrent one-H200
tasks. Existing successful output pairs are skipped.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRATCH_ROOT = Path("/lustre/scratch/sgjolieb-screen2vec")
PROJECT_DIR = SCRATCH_ROOT / "screen2vec-revisited"
RECONSTRUCTION_SCRIPT = PROJECT_DIR / "3_json_reconstruction_par.py"
DATA_DIR = SCRATCH_ROOT / "rico-data" / "data" / "layouts"

EXPECTED_COUNT = 66_261
NUM_SHARDS = 8
MAX_CONCURRENT_GPUS = 8
BATCH_SIZE = 64
MAX_TOKENS = 4_608
CPUS_PER_TASK = 8
EVAL_WORKERS = 6
RECON_TIME = "2-00:00:00"
MAIL_TO = "jonathan.liebers@uni-due.de"

RUN_SPECS = (
    ("qwen3-1.7b-base", "4x4096_avg"),
    ("qwen3-1.7b-base", "16x4096_avg"),
    ("qwen3-1.7b-base", "64x4096_avg"),
    ("qwen3-1.7b-base", "4096_avg"),
    ("qwen3-1.7b-base", "4096_last"),
)


def run_paths(model_slug, variant):
    adapter_name = f"adapter_model_out_embeddings_{variant}_full_{model_slug}"
    reconstruction_name = (
        f"reconstructions_{variant}__via__"
        f"embeddings_{variant}_full_{model_slug}"
    )
    return {
        "model_slug": model_slug,
        "variant": variant,
        "label": f"{model_slug}/{variant}",
        "adapter_name": adapter_name,
        "adapter_dir": SCRATCH_ROOT / adapter_name,
        "embeddings_dir": SCRATCH_ROOT / f"embeddings_{variant}",
        "output_dir": SCRATCH_ROOT / reconstruction_name,
    }


RUNS = tuple(run_paths(model_slug, variant) for model_slug, variant in RUN_SPECS)


def old_fingerprint(run):
    """Fingerprint stored by the original script-5 run before relocation."""
    adapter_name = run["adapter_name"]
    return {
        "projector_model": f"{adapter_name}/best_projector.pth",
        "lora_model": f"{adapter_name}/best_lora_adapter",
        "embeddings_dir": f"embeddings_{run['variant']}",
        "data_dir": "../rico-data/data/layouts",
        "max_tokens": MAX_TOKENS,
        "lm_model_id": None,
    }


def canonical_fingerprint(run):
    """Fingerprint used now that all artifacts have canonical absolute paths."""
    return {
        "projector_model": str(run["adapter_dir"] / "best_projector.pth"),
        "lora_model": str(run["adapter_dir"] / "best_lora_adapter"),
        "embeddings_dir": str(run["embeddings_dir"]),
        "data_dir": str(DATA_DIR),
        "max_tokens": MAX_TOKENS,
        "lm_model_id": None,
    }


def fingerprint_matches(record, expected):
    return all(record.get(key) == value for key, value in expected.items())


def count_complete_pairs(output_dir):
    reconstructed = {
        path.name.removesuffix("_reconstructed.json")
        for path in output_dir.glob("*_reconstructed.json")
        if path.stat().st_size > 0
    }
    reports = set()
    errors = set()
    for path in output_dir.glob("*_report.txt"):
        if path.stat().st_size == 0:
            continue
        rico_id = path.name.removesuffix("_report.txt")
        reports.add(rico_id)
        with open(path, encoding="utf-8", errors="replace") as f:
            if f.read(32).startswith("ERROR processing"):
                errors.add(rico_id)
    return len((reconstructed & reports) - errors), len(errors)


def preflight(runs):
    problems = []
    manifests = []
    reference_embedding_ids = None
    reference_variant = None

    if not RECONSTRUCTION_SCRIPT.is_file():
        problems.append(f"reconstruction script missing: {RECONSTRUCTION_SCRIPT}")
    if not DATA_DIR.is_dir():
        problems.append(f"RICO layouts missing: {DATA_DIR}")

    print("Preflight:")
    for run in runs:
        variant = run["variant"]
        label = run["label"]
        projector = run["adapter_dir"] / "best_projector.pth"
        lora_config = run["adapter_dir"] / "best_lora_adapter" / "adapter_config.json"
        manifest_path = run["output_dir"] / "_run_manifest.json"

        for required in (run["adapter_dir"], run["embeddings_dir"], run["output_dir"]):
            if not required.is_dir():
                problems.append(f"directory missing: {required}")
        for required in (projector, lora_config, manifest_path):
            if not required.is_file():
                problems.append(f"file missing: {required}")

        embedding_ids = set()
        if run["embeddings_dir"].is_dir():
            for path in run["embeddings_dir"].glob("*.npz"):
                try:
                    embedding_ids.add(int(path.stem))
                except ValueError:
                    problems.append(f"non-numeric embedding filename: {path}")
            if len(embedding_ids) != EXPECTED_COUNT:
                problems.append(
                    f"{label}: expected {EXPECTED_COUNT} unique numeric embedding "
                    f"IDs, found {len(embedding_ids)}"
                )
            if reference_embedding_ids is None:
                reference_embedding_ids = embedding_ids
                reference_variant = label
            elif embedding_ids != reference_embedding_ids:
                missing = len(reference_embedding_ids - embedding_ids)
                extra = len(embedding_ids - reference_embedding_ids)
                problems.append(
                    f"{label}: embedding ID set differs from {reference_variant} "
                    f"({missing} missing, {extra} extra)"
                )

        record = None
        if manifest_path.is_file():
            try:
                record = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as error:
                problems.append(f"cannot read {manifest_path}: {error}")
            else:
                old = old_fingerprint(run)
                canonical = canonical_fingerprint(run)
                if not (
                    fingerprint_matches(record, old)
                    or fingerprint_matches(record, canonical)
                ):
                    problems.append(
                        f"{label}: manifest does not match either the original "
                        "or canonical recovery configuration"
                    )
                manifests.append((manifest_path, record, old, canonical))

        complete, errors = (
            count_complete_pairs(run["output_dir"])
            if run["output_dir"].is_dir() else (0, 0)
        )
        print(
            f"  {label:<36} embeddings={len(embedding_ids):>5}  "
            f"ids={min(embedding_ids, default=0)}..{max(embedding_ids, default=0)}  "
            f"complete={complete:>5}  pending={EXPECTED_COUNT - complete:>5}  "
            f"error_reports={errors}"
        )

    if problems:
        print("\nPreflight failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(1)

    return manifests


def migrate_manifests(manifests):
    """Atomically replace verified relative artifact paths with absolute ones."""
    for manifest_path, record, old, canonical in manifests:
        if fingerprint_matches(record, canonical):
            continue
        if not fingerprint_matches(record, old):
            raise RuntimeError(f"manifest changed after preflight: {manifest_path}")

        migrated = dict(record)
        migrated.update(canonical)
        migrated["relocated_from"] = {
            key: old[key]
            for key in ("projector_model", "lora_model", "embeddings_dir", "data_dir")
        }
        migrated["relocated_at"] = datetime.now().isoformat(timespec="seconds")

        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(migrated, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_path, manifest_path)
        print(f"Migrated manifest paths: {manifest_path}")


def build_job(run):
    job = canonical_fingerprint(run)
    job.update({
        "output_dir": str(run["output_dir"]),
        "batch_size": BATCH_SIZE,
        "num_workers": 1,
        "eval_workers": EVAL_WORKERS,
        "expected_count": EXPECTED_COUNT,
    })
    return job


def write_array_script(path, control_dir, job_paths):
    job_lines = "\n".join(f'  "{job_path}"' for job_path in job_paths)
    num_runs = len(job_paths)
    total_tasks = num_runs * NUM_SHARDS
    content = f"""#!/bin/bash
#SBATCH -J s2v_qwen3_1_7b_recovery
#SBATCH --output={control_dir}/output_%A_%a.txt
#SBATCH --error={control_dir}/output_%A_%a.txt
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={MAIL_TO}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --gres=gpu:1
#SBATCH --partition=GPU-H200
#SBATCH --time={RECON_TIME}
#SBATCH --array=0-{total_tasks - 1}%{MAX_CONCURRENT_GPUS}

set -euo pipefail
module load anaconda3/2023
export PATH=$HOME/.local/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

job_files=(
{job_lines}
)

task_id=${{SLURM_ARRAY_TASK_ID}}
run_index=$((task_id % {num_runs}))
shard_index=$((task_id / {num_runs}))
job_json=${{job_files[$run_index]}}

echo "Recovery task $task_id: run $run_index, shard $shard_index/{NUM_SHARDS}"
srun python "{RECONSTRUCTION_SCRIPT}" \\
    --job_json "$job_json" \\
    --shard_index "$shard_index" \\
    --num_shards {NUM_SHARDS} \\
    --oom_split \\
    --fail_on_error \\
    --require_complete_shard
"""
    path.write_text(content, encoding="utf-8")


def write_audit_script(path, control_dir, array_job_id, job_paths):
    job_lines = "\n".join(f'  "{job_path}"' for job_path in job_paths)
    content = f"""#!/bin/bash
#SBATCH -J s2v_qwen3_1_7b_audit
#SBATCH --output={control_dir}/output_audit_%j.txt
#SBATCH --error={control_dir}/output_audit_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user={MAIL_TO}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=STD-s-96h
#SBATCH --time=0-04:00:00
#SBATCH --dependency=afterany:{array_job_id}

set -u
module load anaconda3/2023
export PATH=$HOME/.local/bin:$PATH

job_files=(
{job_lines}
)

status=0
for job_json in "${{job_files[@]}}"; do
    srun python "{RECONSTRUCTION_SCRIPT}" --job_json "$job_json" --audit || status=1
done

if [[ $status -ne 0 ]]; then
    echo "One or more reconstruction directories are incomplete." >&2
    exit $status
fi

for job_json in "${{job_files[@]}}"; do
    srun python "{RECONSTRUCTION_SCRIPT}" --job_json "$job_json" --summarize
done
"""
    path.write_text(content, encoding="utf-8")


def submit(script_path):
    result = subprocess.run(
        ["sbatch", script_path.name],
        cwd=script_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = result.stdout.strip().split()[-1]
    print(f"Submitted {script_path.name}: {job_id}")
    return job_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run preflight and write scripts, but do not migrate manifests or submit.",
    )
    args = parser.parse_args()

    runs = RUNS
    manifests = preflight(runs)
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    selection_tag = "qwen3_1_7b_base"
    control_dir = SCRATCH_ROOT / f"recovery_{selection_tag}_{stamp}"
    control_dir.mkdir(parents=False, exist_ok=False)

    job_paths = []
    for index, run in enumerate(runs):
        job_path = control_dir / (
            f"job_{index}_{run['model_slug']}_{run['variant']}.json"
        )
        job_path.write_text(
            json.dumps(build_job(run), indent=2) + "\n", encoding="utf-8"
        )
        job_paths.append(job_path)

    array_script = control_dir / "recover.sh"
    write_array_script(array_script, control_dir, job_paths)

    print(f"\nRecovery control directory: {control_dir}")
    print(
        f"Load distribution: {len(runs)} run(s) x {NUM_SHARDS} shards = "
        f"{len(runs) * NUM_SHARDS} tasks, capped at {MAX_CONCURRENT_GPUS} H200s"
    )
    print(f"Batch size: {BATCH_SIZE}, with recursive OOM splitting")

    if args.dry_run:
        print("Dry run: manifests were not changed and no jobs were submitted.")
        print(f"Generated array script: {array_script}")
        return

    migrate_manifests(manifests)
    array_job_id = submit(array_script)

    audit_script = control_dir / "audit.sh"
    write_audit_script(audit_script, control_dir, array_job_id, job_paths)
    audit_job_id = submit(audit_script)

    print("\nRecovery submitted successfully:")
    print(f"  reconstruction array : {array_job_id}")
    print(f"  final audit          : {audit_job_id} (afterany:{array_job_id})")
    print(f"  logs and job files   : {control_dir}")


if __name__ == "__main__":
    main()
