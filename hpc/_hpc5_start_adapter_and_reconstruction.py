# Submits the whole adapter pipeline for ONE (base LM, embedding variant) pair
# as a chain of three dependent Slurm jobs:
#
#   1) train.sh        adapter training  (1 job, 8 GPUs, exclusive)
#   2) reconstruct.sh  reconstruction    (array of NUM_SHARDS tasks, 1 GPU each)
#                      --dependency=afterok:<train job>
#   3) aggregate.sh    summary + mail    (1 CPU job)
#                      --dependency=afterok:<reconstruction array>
#
# All three run inside ONE copied project folder, so the checkpoints the
# training writes are exactly the ones the reconstruction reads, and the
# reconstructions land next to them in that same folder.
#
# This replaces the two-step _hpc2 -> (download, edit, resubmit) -> _hpc3 dance.
# Both of those keep working unchanged and can still be used on their own.

import json
import platform
import re
import subprocess
import shutil
import sys
from datetime import datetime
from pathlib import Path


# ==============================================================================
# 1) SETTINGS -- the only two knobs for a normal run
# ==============================================================================

# (a) Base LM as a full Hugging Face id. Single source of truth: it is handed to
#     2_train_adapter.py via --lm_model_id (overriding AdapterConfig), and the
#     checkpoint directory name is derived from the very same string below. The
#     two can therefore no longer disagree -- which is what used to silently
#     produce "..._full_<slug-from-here>_<slug-from-AdapterConfig>" directories
#     that neither the mail step nor the reconstruction could find.
LM_MODEL_ID = "Qwen/Qwen3-1.7B-Base"

# (b) Which embeddings to train the adapter on and to reconstruct from.
#     Either "4x4096_avg" or the full "embeddings_4x4096_avg" -- both work.
#     Naming follows _hpc1_start_embedding.py:
#         (1, "last") -> embeddings_4096_last
#         (N, "avg")  -> embeddings_{N}x4096_avg
#     The directory name is the *only* place this has to be set: load_embeddings()
#     reads whatever shape is in the .npz files, and the projector maps an
#     (N, 4096) embedding to N prefix tokens for any N <= num_prefix_tokens (128).
EMBEDDINGS = "4096_last"

# ==============================================================================


# ------------------------------------------------------------------------------
# Everything below is derived. Do not hand-edit these -- that is the whole point.
# ------------------------------------------------------------------------------
EMBEDDINGS_DIR = (
    EMBEDDINGS if EMBEDDINGS.startswith("embeddings_") else f"embeddings_{EMBEDDINGS}"
)

# Must stay identical to 2_train_adapter.model_output_slug(), which is what
# main() appends to --output_dir when it is not already suffixed with it.
MODEL_SLUG = LM_MODEL_ID.split("/")[-1].lower()

ADAPTER_DIR = f"adapter_model_out_{EMBEDDINGS_DIR}_full_{MODEL_SLUG}"
PROJECTOR_MODEL = f"{ADAPTER_DIR}/best_projector.pth"
LORA_MODEL = f"{ADAPTER_DIR}/best_lora_adapter"

# Same tag _hpc3 builds, so the output folder keeps the name the analysis
# notebook expects: _results_reconstruction.ipynb globs "reconstructions*/*" and
# uses everything after "reconstructions_" as its `dim` key.
RUN_TAG = (f"{EMBEDDINGS_DIR.removeprefix('embeddings_')}"
           f"__via__{ADAPTER_DIR.removeprefix('adapter_model_out_')}")

# Relative on purpose: unlike _hpc3, which writes to a shared directory on
# scratch so resubmissions can resume it, everything from this submission lives
# in the one copied task folder. One task folder is one (model, embeddings) run,
# so nothing can get mixed up, and the whole result is a single directory to
# copy back into the repository root.
RECON_DIR = f"reconstructions_{RUN_TAG}"

DATA_DIR = "../rico-data/data/layouts"


# ==============================================================================
# 2) HPC SETTINGS -- rarely touched
# ==============================================================================
SCRATCH_ROOT = ("/lustre/scratch/sgjolieb-screen2vec"
                if not platform.system() == "Windows" else "C:\\Users\\Jonathan\\Dev")
PROJECT_NAME = "s2v"
DATA_DIR_NAME = "rico-data"
runtarget = "GPU-H200"  # GPU partition for both GPU phases
MAIL_FROM = "jonathan.liebers@uni-due.de"

NOW = datetime.now()
# Defined once so every generated script agrees on the task folder name
NOW_STR = NOW.strftime('%Y-%m-%d-%H-%M-%S')

# --- Phase 1: adapter training ------------------------------------------------
TRAIN_NUM_GPU = 8        # 2_train_adapter_amplitude.sh launches 8 processes
# Required for the wide embeddings: load_embeddings() casts to float32 and every
# rank holds the full DataFrame before sharding, so 8 ranks need ~850 GB of host
# RAM at (128, 4096). That fits into an H200 node only with nothing else on it.
TRAIN_EXCLUSIVE = True
TRAIN_TIME = "2-00:00:00"
# Throughput optimisations in 2_train_adapter.py. Arithmetically equivalent to
# the reference path (see FAST_PATH_NOTES there), just without the redundant
# all-reduces, host syncs and re-tokenization.
TRAIN_FAST = True
# Mail the trained adapter back as a split archive when training finishes.
# Runs before the reconstruction starts and does not gate it either way.
MAIL_ADAPTER = True

# --- Phase 2: reconstruction --------------------------------------------------
# One array task per shard, each asking for a single GPU. Slurm is free to
# scatter them and to start them as capacity appears, instead of holding one
# whole 8-GPU node until every shard is finished.
NUM_SHARDS = 8
RECON_NUM_GPU = 1         # GPUs per array task
WORKERS_PER_TASK = 1      # Worker processes per array task (one per GPU)
CPUS_PER_TASK = 8         # Enough for the scoring pool plus the decoder
EVAL_WORKERS = 6          # CPU processes that score generations and write reports
# How many elements go through one forward pass. Modelled over the real Rico
# target lengths, 256 costs 12.3k decode steps per shard against 18.2k at 128.
BATCH_SIZE = 256
# Must track AdapterConfig.max_target_len (4096) plus a small margin. Anything
# beyond that is a degenerate generation no budget can rescue.
MAX_TOKENS = 4608
RECON_TIME = "2-00:00:00"

# --- Safety net ---------------------------------------------------------------
# True: only print what would happen and write the generated scripts to
# DRY_RUN_DIR (default: ./_hpc5_dryrun). Nothing is copied, nothing is submitted.
DRY_RUN = False
DRY_RUN_DIR = None        # None -> "<project>/_hpc5_dryrun"

# AdapterConfig.num_prefix_tokens / .qwen_embed_dim, checked against the
# embeddings before anything is copied.
MAX_PREFIX_TOKENS = 128
EXPECTED_EMBED_DIM = 4096
# ==============================================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = f"{SCRATCH_ROOT}/{NOW_STR}_0"


# ==============================================================================
# 3) PREFLIGHT
# ==============================================================================
def parse_embedding_shape(embeddings_dir):
    """
    "embeddings_4x4096_avg" -> (4, 4096);  "embeddings_4096_last" -> (1, 4096).

    Returns (None, None) for anything that does not follow the scheme, so a
    non-standard directory is a warning rather than a hard stop.
    """
    match = re.match(r"^embeddings_(?:(\d+)x)?(\d+)_(\w+)$", embeddings_dir)
    if not match:
        return None, None
    tokens = int(match.group(1)) if match.group(1) else 1
    return tokens, int(match.group(2))


def preflight():
    """
    Fail here, in a second, rather than after two days of queueing.

    Everything checked is cheap and local: the embeddings must exist in the
    folder that is about to be copied, and their shape must be one the adapter
    can actually consume.
    """
    problems = []

    emb_path = PROJECT_ROOT / EMBEDDINGS_DIR
    if not emb_path.is_dir():
        problems.append(f"Embeddings directory '{emb_path}' does not exist.")
    else:
        npz_count = sum(1 for _ in emb_path.glob("*.npz"))
        if npz_count == 0:
            problems.append(f"Embeddings directory '{emb_path}' holds no .npz files.")
        else:
            print(f"  Embeddings      : {npz_count} .npz files in {emb_path}")

    tokens, dim = parse_embedding_shape(EMBEDDINGS_DIR)
    if tokens is None:
        print(f"  [warn] Cannot parse a shape out of '{EMBEDDINGS_DIR}'. "
              f"Skipping the shape checks.")
    else:
        print(f"  Embedding shape : {tokens} token(s) x {dim} dims (from the name)")
        if tokens > MAX_PREFIX_TOKENS:
            problems.append(
                f"'{EMBEDDINGS_DIR}' carries {tokens} tokens, but the projector "
                f"is built for at most AdapterConfig.num_prefix_tokens="
                f"{MAX_PREFIX_TOKENS}. Raise it in 2_train_adapter.py first."
            )
        if dim != EXPECTED_EMBED_DIM:
            problems.append(
                f"'{EMBEDDINGS_DIR}' has {dim}-dim features, but "
                f"AdapterConfig.qwen_embed_dim is {EXPECTED_EMBED_DIM}."
            )

    if "/" not in LM_MODEL_ID:
        print(f"  [warn] LM_MODEL_ID '{LM_MODEL_ID}' has no '<org>/' part. The "
              f"slug is derived as '{MODEL_SLUG}'.")

    if problems:
        print("\nPreflight failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        sys.exit(1)


# ==============================================================================
# 4) SLURM DIRECTIVES
# ==============================================================================
def gpu_directives():
    """
    Return (gres_line, partition_line, default_timelimit) for `runtarget`.
    """
    target = runtarget.lower()
    if target == "gpu-small":
        return "#SBATCH --gres=gpu:{n}", "#SBATCH --partition=GPU-small", "2-00:00:00"
    if target == "gpu-big":
        return "#SBATCH --gres=gpu:{n}", "#SBATCH --partition=GPU-big", "2-00:00:00"
    if target == "gpu-h200":
        return "#SBATCH --gres=gpu:{n}", "#SBATCH --partition=GPU-H200", "2-00:00:00"
    if target in ("cpu-shm", "shm"):
        return "", "#SBATCH --partition=SHM-s-96h", "4-00:00:00"
    if target in ("cpu", "cpu-std", "std", "std-s-96h"):
        return "", "#SBATCH --partition=STD-s-96h", "4-00:00:00"
    return "", "", "0-24:00:00"


def sbatch_header(jobname, *, num_gpu, timelimit, mail_type,
                  array_tasks=0, exclusive=False, cpus_per_task=0,
                  dependency=None, output_pattern=None):
    """
    Build one #SBATCH block.

    One function for all three jobs, because they differ only in the knobs above
    -- an array or not, a dependency or not, one GPU or eight.
    """
    gres_tpl, partition, _ = gpu_directives()
    gres = gres_tpl.format(n=num_gpu) if gres_tpl and num_gpu else ""

    if output_pattern is None:
        output_pattern = "output_%A_%a.txt" if array_tasks > 1 else "output_%j.txt"

    array_str = f"#SBATCH --array=0-{array_tasks - 1}" if array_tasks > 1 else ""
    exclusive_str = "#SBATCH --exclusive" if exclusive else ""
    cpus_str = f"#SBATCH --cpus-per-task={cpus_per_task}" if cpus_per_task else ""
    dependency_str = f"#SBATCH --dependency=afterok:{dependency}" if dependency else ""

    lines = [
        f"#SBATCH -J {jobname}",
        f"#SBATCH --output={output_pattern}",
        f"#SBATCH --error={output_pattern}",
        f"#SBATCH --mail-type={mail_type}",
        f"#SBATCH --mail-user={MAIL_FROM}",
        "#SBATCH --nodes=1",
        array_str,
        dependency_str,
        cpus_str,
        exclusive_str,
        gres,
        partition,
        f"#SBATCH --time={timelimit}",
    ]
    return "\n".join(line for line in lines if line)


PREAMBLE = f"""\
# Move to the single task folder shared by all three phases
cd "{TASK_DIR}"

# Load HPC modules
module load anaconda3/2023

# Ensure user-installed python binaries are in PATH
export PATH=$HOME/.local/bin:$PATH
"""


# ==============================================================================
# 5) JOB SCRIPTS
# ==============================================================================
def create_train_script(path):
    """
    Phase 1: train the adapter on 8 GPUs of one exclusive node.

    Not an array job: there is exactly one task, and a plain job id makes the
    --dependency of the reconstruction below unambiguous. Its exit code is what
    decides whether the rest of the chain runs at all, so the checkpoint gate at
    the bottom is the last thing in the script.
    """
    header = sbatch_header(
        "s2v_adapter_training",
        num_gpu=TRAIN_NUM_GPU,
        timelimit=TRAIN_TIME,
        mail_type="ALL",
        exclusive=TRAIN_EXCLUSIVE,
        output_pattern=None,
    )
    # stdout and stderr separately here: mail.py falls back to reading
    # hpc/stdout_<id>.txt + hpc/stderr_<id>.txt for the mail body.
    header = header.replace("#SBATCH --output=output_%j.txt\n"
                            "#SBATCH --error=output_%j.txt",
                            "#SBATCH --output=stdout_%j.txt\n"
                            "#SBATCH --error=stderr_%j.txt")

    fast_flag = " \\\n    --fast true" if TRAIN_FAST else ""

    mail_block = ""
    if MAIL_ADAPTER:
        mail_block = f"""
# Mail the adapter back as a split archive. In a subshell so its `cd hpc` does
# not move the working directory out from under the checkpoint gate below, and
# tolerant of failure: a bounced mail must not cancel the reconstruction.
export ADAPTER_OUTPUT_DIR="{ADAPTER_DIR}"
( cd hpc && srun python mail.py "${{SLURM_JOB_ID}}" "${{SLURM_JOB_NAME}}" ) || true
"""

    script_content = f"""#!/bin/bash
{header}

{PREAMBLE}
# Install requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Show environment
srun python hpc/env.py

# EXECUTE MAIN WORKLOAD -- adapter training.
# --lm_model_id makes this file the single source of truth for the base LM: the
# slug 2_train_adapter.py appends to --output_dir is derived from the same id,
# so --output_dir below is exactly the directory that gets written.
srun bash 2_train_adapter_amplitude.sh \\
    --embeddings_dir {EMBEDDINGS_DIR} \\
    --output_dir {ADAPTER_DIR} \\
    --lm_model_id {LM_MODEL_ID}{fast_flag}

# Collect this job's logs into the task folder for the mail below
cp "$SLURM_SUBMIT_DIR/stdout_${{SLURM_JOB_ID}}.txt" hpc/ 2>/dev/null || true
cp "$SLURM_SUBMIT_DIR/stderr_${{SLURM_JOB_ID}}.txt" hpc/ 2>/dev/null || true
{mail_block}
# CHECKPOINT GATE -- the exit code of this script is what --dependency=afterok
# on the reconstruction reads. Training can end "successfully" without usable
# checkpoints (empty dataset, see main()), and there is no point queueing eight
# GPUs behind that.
test -f "{PROJECTOR_MODEL}" \\
    || {{ echo "FATAL: {PROJECTOR_MODEL} was never written."; exit 1; }}
test -f "{LORA_MODEL}/adapter_config.json" \\
    || {{ echo "FATAL: {LORA_MODEL} was never written."; exit 1; }}
echo "Adapter checkpoints present. Reconstruction may proceed."
"""

    write_script(path, script_content)


def create_reconstruct_script(path, train_job_id):
    """
    Phase 2: one array task per shard, gated on the training job.

    Every task reads the same code, the same checkpoints and the same job.json
    out of the one task folder; only --shard_index differs, and that comes
    straight from $SLURM_ARRAY_TASK_ID.
    """
    script_name = "3_json_reconstruction_par.py"
    header = sbatch_header(
        "s2v_reconstruction_worker",
        num_gpu=RECON_NUM_GPU,
        timelimit=RECON_TIME,
        mail_type="FAIL",
        array_tasks=NUM_SHARDS,
        cpus_per_task=CPUS_PER_TASK,
        dependency=train_job_id,
    )

    script_content = f"""#!/bin/bash
{header}

{PREAMBLE}
# Install requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Show environment
srun python hpc/env.py

# EXECUTE MAIN WORKLOAD -- this task's shard of the reconstruction.
# Everything except the shard index comes from job.json, including the base LM,
# which 3_json_reconstruction_par.py reads back out of the LoRA checkpoint's
# adapter_config.json -- the model the adapter was actually trained against.
srun python {script_name} \\
    --job_json job.json \\
    --shard_index ${{SLURM_ARRAY_TASK_ID:-0}} \\
    --num_shards {NUM_SHARDS}
"""

    write_script(path, script_content)


def create_aggregate_script(path, array_job_id):
    """
    Phase 3: one small CPU job that runs once every shard is done, summarises
    the reconstruction directory and mails the result.
    """
    script_content = f"""#!/bin/bash
#SBATCH -J s2v_reconstruction_agg
#SBATCH --output=output_agg_%j.txt
#SBATCH --error=output_agg_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user={MAIL_FROM}
#SBATCH --nodes=1
#SBATCH --dependency=afterok:{array_job_id}
#SBATCH --partition=STD-s-96h
#SBATCH --time=0-02:00:00

{PREAMBLE}
# Install requirements (CPU wheels are enough for the aggregation)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# AGGREGATE: parse every report in the reconstruction directory
srun python 3_json_reconstruction_par.py \\
    --summarize \\
    --output_dir "{RECON_DIR}"

# Collect the array's logs plus this one so the mail carries all of them
cp "$SLURM_SUBMIT_DIR"/output_{array_job_id}_*.txt hpc/ 2>/dev/null || true
cp "$SLURM_SUBMIT_DIR/output_agg_${{SLURM_JOB_ID}}.txt" hpc/ 2>/dev/null || true

# Send final results email
cd hpc && srun python mail.py "agg_${{SLURM_JOB_ID}}" "${{SLURM_JOB_NAME}}"
"""

    write_script(path, script_content)


def write_script(path, content):
    with open(path, "w", newline="\n") as f:
        f.write(content)


# ==============================================================================
# 6) SUBMISSION
# ==============================================================================
def copy_folder(execution_id) -> Path:
    """
    Copies the entire project folder to a time-stamped scratch subfolder.
    """

    # to stop shutil from copying all
    def ignore_folders(dir, files):
        # ignore all dot folders
        return {f for f in files if f.startswith('.')}

    job_folder_name = f"{NOW_STR}_{execution_id}"
    target = Path(SCRATCH_ROOT) / job_folder_name
    print("Copying", PROJECT_ROOT, "->", target)
    shutil.copytree(PROJECT_ROOT, target, ignore=ignore_folders, symlinks=True)
    return target


def submit_sbatch(script_path: Path) -> str:
    """
    sbatch the script from its parent folder. Returns the Slurm job ID.
    """
    print(f"Submitting job via `sbatch` in {script_path}")
    result = subprocess.run(
        ["sbatch", script_path.name],
        cwd=script_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    # Parse: "Submitted batch job 12345678\n"
    job_id = result.stdout.strip().split()[-1]
    print(f"  -> Submitted. Job ID: {job_id}")
    return job_id


def build_job_config():
    """
    The reconstruction's job.json. Read by every array task and by the
    aggregation job through --job_json.

    lm_model_id is deliberately absent: 3_json_reconstruction_par.py then takes
    it from the LoRA checkpoint's adapter_config.json, which is the model the
    adapter was really trained against, not the one this file believes it was.
    """
    return {
        "projector_model": PROJECTOR_MODEL,
        "lora_model": LORA_MODEL,
        "embeddings_dir": EMBEDDINGS_DIR,
        "data_dir": DATA_DIR,
        "output_dir": RECON_DIR,
        "max_tokens": MAX_TOKENS,
        "batch_size": BATCH_SIZE,
        "num_workers": WORKERS_PER_TASK,
        "eval_workers": EVAL_WORKERS,
    }


def print_plan():
    print("=" * 72)
    print("  Adapter training + reconstruction (single chained submission)")
    print("=" * 72)
    print(f"  LM model id     : {LM_MODEL_ID}")
    print(f"  Model slug      : {MODEL_SLUG}")
    print(f"  Embeddings dir  : {EMBEDDINGS_DIR}")
    print(f"  Adapter dir     : {ADAPTER_DIR}")
    print(f"  Projector model : {PROJECTOR_MODEL}")
    print(f"  LoRA model      : {LORA_MODEL}")
    print(f"  Reconstructions : {RECON_DIR}")
    print(f"  Run tag         : {RUN_TAG}")
    print(f"  Data dir        : {DATA_DIR}")
    print(f"  Task directory  : {TASK_DIR}")
    print("-" * 72)
    print(f"  Train           : 1 job x {TRAIN_NUM_GPU} GPU "
          f"(exclusive={TRAIN_EXCLUSIVE}, fast={TRAIN_FAST}, "
          f"mail_adapter={MAIL_ADAPTER})")
    print(f"  Reconstruct     : {NUM_SHARDS} array tasks x {RECON_NUM_GPU} GPU "
          f"({WORKERS_PER_TASK} worker(s), {CPUS_PER_TASK} CPUs, "
          f"{EVAL_WORKERS} for scoring)")
    print(f"  Batch / tokens  : {BATCH_SIZE} / {MAX_TOKENS}")
    print(f"  GPU partition   : {runtarget}")
    print(f"  Dry run         : {DRY_RUN}")
    print("=" * 72)


# ==============================================================================
# 7) MAIN
# ==============================================================================
def main():
    print_plan()
    print("Preflight...")
    preflight()

    job_config = build_job_config()

    if DRY_RUN:
        target = Path(DRY_RUN_DIR) if DRY_RUN_DIR else PROJECT_ROOT / "_hpc5_dryrun"
        target.mkdir(parents=True, exist_ok=True)
        print(f"\n[DRY RUN] Nothing is copied and nothing is submitted.")
        print(f"[DRY RUN] Writing the generated files to {target}")

        with open(target / "job.json", "w") as f:
            json.dump(job_config, f, indent=2)

        create_train_script(target / "train.sh")
        create_reconstruct_script(target / "reconstruct.sh", "<TRAIN_JOB_ID>")
        create_aggregate_script(target / "aggregate.sh", "<ARRAY_JOB_ID>")

        for name in ("job.json", "train.sh", "reconstruct.sh", "aggregate.sh"):
            print(f"\n{'=' * 72}\n=== {name}\n{'=' * 72}")
            print((target / name).read_text())
        return

    # 1) One copy of the project for all three phases
    task_folder = copy_folder(0)

    # 2) The reconstruction's job.json. 2_train_adapter.py never reads job.json,
    #    so this single file serves the whole chain.
    with open(task_folder / "job.json", "w") as f:
        json.dump(job_config, f, indent=2)

    # 3) Training
    train_script = task_folder / "train.sh"
    create_train_script(train_script)
    train_job_id = submit_sbatch(train_script)

    # 4) Reconstruction array -- waits for the training job to exit 0
    reconstruct_script = task_folder / "reconstruct.sh"
    create_reconstruct_script(reconstruct_script, train_job_id)
    array_job_id = submit_sbatch(reconstruct_script)

    # 5) Aggregation -- waits for every task of the array
    aggregate_script = task_folder / "aggregate.sh"
    create_aggregate_script(aggregate_script, array_job_id)
    agg_job_id = submit_sbatch(aggregate_script)

    print(f"\n{'=' * 72}")
    print(f"  Training job:         {train_job_id}  ({TRAIN_NUM_GPU} GPUs)")
    print(f"  Reconstruction array: {array_job_id}  ({NUM_SHARDS} tasks, "
          f"afterok:{train_job_id})")
    print(f"  Aggregation job:      {agg_job_id}  (afterok:{array_job_id})")
    print(f"  Task directory:       {task_folder}")
    print(f"  Reconstructions in:   {task_folder / RECON_DIR}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
    if not DRY_RUN:
        print("HPC job chain submitted.")
