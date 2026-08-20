import json
import re
import platform
from glob import glob
import subprocess
import shutil
from datetime import datetime
from pathlib import Path


# ==============================================================================
# 1) SETTINGS
# ==============================================================================

# Model / reconstruction arguments
ADAPTER_FOLDER = "adapter_model_out_embeddings_4x4096_avg_full_qwen2.5-1.5b"

# How many elements go through one forward pass.  Modelled over the real Rico
# target lengths, 256 costs 12.3k decode steps per shard against 18.2k at 128,
# and is best or within a few percent of best whatever the fixed per-step cost
# is; 512 saves another 23% of steps but wants 68 GB of KV cache for it.
BATCH_SIZE = 256

# One array task per shard, each asking for a single GPU.  Slurm is free to
# scatter them across nodes and to start them as capacity appears, instead of
# holding one whole 8-GPU node until every shard is finished.
NUM_SHARDS = 8
NUM_GPU: int = 1          # GPUs per array task
WORKERS_PER_TASK = 1      # Worker processes per array task (one per GPU)
CPUS_PER_TASK = 8         # Enough for the scoring pool below plus the decoder
EVAL_WORKERS = 6          # CPU processes that score generations and write reports

# Filesystem settings
_parts = ADAPTER_FOLDER.split('_')
_idx = _parts.index('embeddings')
EMBEDDINGS_DIR = '_'.join(_parts[_idx:-2])
PROJECTOR_MODEL = f"{ADAPTER_FOLDER}/best_projector.pth"
LORA_MODEL = f"{ADAPTER_FOLDER}/best_lora_adapter"
DATA_DIR = "../rico-data/data/layouts"

# Must track AdapterConfig.max_target_len (4096) -- the longest target the
# adapter is ever trained to finish -- plus a small margin.  Anything beyond
# that is a degenerate generation no budget can rescue: across 65,927
# reconstructions from eight adapter variants, only 60 (0.09%) ever terminated
# between 2048 and 8192 tokens, while runaways ran to the ceiling and produced
# invalid JSON in every single case.
MAX_TOKENS = 4608


# ==============================================================================
# 2) HPC SETTINGS
# ==============================================================================
SCRATCH_ROOT = "/lustre/scratch/sgjolieb-screen2vec" if not platform.system() == "Windows" else "C:\\Users\\Jonathan\\Dev"
PROJECT_NAME = "s2v"
DATA_DIR_NAME = "rico-data"
runtarget = "GPU-H200"  # GPU partition to submit the array to

run_exclusive = False     # A single GPU per task -- never take a whole node
NOW = datetime.now()
# We define the timestamp once here to ensure the folder has a unique prefix
NOW_STR = NOW.strftime('%Y-%m-%d-%H-%M-%S')
MAIL_FROM = "jonathan.liebers@uni-due.de"

# ------------------------------------------------------------------------------
# Shared output directory, on scratch rather than inside a task folder.
#
# Every shard of every submission writes here, so the name has to identify the
# run on its own: the reconstruction of one embedding variant must never end up
# next to another's.  Both halves matter -- EMBEDDINGS_DIR is what is being
# reconstructed (and what its dimensionality is), ADAPTER_FOLDER is what does the
# reconstructing.  Deliberately no timestamp: a resubmission with identical
# settings then resumes the same directory instead of starting over, and the
# reconstruction script stamps a manifest into it that makes any mismatched
# resubmission fail loudly rather than interleave silently.
#
# The "reconstructions_" prefix is what _results_reconstruction.ipynb globs for,
# and everything after it becomes that notebook's `dim` key.
# ------------------------------------------------------------------------------
RUN_TAG = (f"{EMBEDDINGS_DIR.removeprefix('embeddings_')}"
           f"__via__{ADAPTER_FOLDER.removeprefix('adapter_model_out_')}")
SHARED_OUTPUT_DIR = f"{SCRATCH_ROOT}/reconstructions_{RUN_TAG}"
# ==============================================================================


# ==============================================================================
# 3) CREATE AND SUBMIT JOB
# ==============================================================================
def get_hpc_slurm_directives(jobname: str, num_tasks: int):
    """
    Return the #SBATCH block for the reconstruction array.
    """
    gres = ""
    partition = ""
    timelimit = "#SBATCH --time=0-24:00:00"

    if runtarget.lower() == "gpu-small".lower():
        gres = f"#SBATCH --gres=gpu:{NUM_GPU}"
        partition = "#SBATCH --partition=GPU-small"
        timelimit = "#SBATCH --time=2-00:00:00"
    elif runtarget.lower() == "gpu-big".lower():
        gres = f"#SBATCH --gres=gpu:{NUM_GPU}"
        partition = "#SBATCH --partition=GPU-big"
        timelimit = "#SBATCH --time=2-00:00:00"
    elif runtarget.lower() == "gpu-H200".lower():
        gres = f"#SBATCH --gres=gpu:{NUM_GPU}"
        partition = "#SBATCH --partition=GPU-H200"
        timelimit = "#SBATCH --time=2-00:00:00"
    elif runtarget.lower() in ("cpu-shm", "shm"):
        partition = "#SBATCH --partition=SHM-s-96h"
        timelimit = "#SBATCH --time=4-00:00:00"
    elif runtarget.lower() in ("cpu", "cpu-std", "std", "std-s-96h"):
        partition = "#SBATCH --partition=STD-s-96h"
        timelimit = "#SBATCH --time=4-00:00:00"

    nodes_str = "#SBATCH --nodes=1"
    exclusive_str = "#SBATCH --exclusive" if run_exclusive else ""
    cpus_str = f"#SBATCH --cpus-per-task={CPUS_PER_TASK}" if CPUS_PER_TASK else ""
    array_str = f"#SBATCH --array=0-{num_tasks - 1}" if num_tasks > 1 else ""
    output_str = ("#SBATCH --output=output_%A_%a.txt" if num_tasks > 1
                  else "#SBATCH --output=output_%j.txt")
    error_str = ("#SBATCH --error=output_%A_%a.txt" if num_tasks > 1
                 else "#SBATCH --error=output_%j.txt")

    sbatch_header = f"""\
#SBATCH -J s2v_reconstruction_worker
{output_str}            # output file (merged stdout and stderr)
{error_str}             # error file (redirected to output file)
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={MAIL_FROM}
{nodes_str}
{array_str}
{cpus_str}
{exclusive_str}
{gres}
{partition}
{timelimit}
"""
    return sbatch_header


def copy_folder(execution_id) -> Path:
    """
    Copies your entire project folder to a time-stamped scratch subfolder.
    """

    # to stop shutil from copying all
    def ignore_folders(dir, files):
        # ignore all dot folders
        return {f for f in files if f.startswith('.')}

    job_folder_name = f"{NOW_STR}_{execution_id}"  # job name
    current_project_folder = Path(__file__).resolve().parent.parent
    target = Path(SCRATCH_ROOT) / job_folder_name
    print("Copying", current_project_folder, "->", target)
    shutil.copytree(current_project_folder, target, ignore=ignore_folders, symlinks=True)
    return target


def create_worker_sbatch_script(path, num_tasks):
    """
    Write out start.sh -- the Slurm array that runs the reconstruction shards.

    Unlike the evaluation array in _hpc4_evaluate_screen_prediction_par.py there
    is only one task folder: every shard reads the same code, the same
    checkpoints and the same embeddings, and writes into the shared output
    directory.  The only thing that differs between tasks is --shard_index, and
    that comes straight from $SLURM_ARRAY_TASK_ID -- so there is nothing to gain
    from N copies of a multi-gigabyte project tree.
    """
    script_name = "3_json_reconstruction_par.py"
    sbatch_header = get_hpc_slurm_directives(script_name, num_tasks)

    script_content = f"""#!/bin/bash
{sbatch_header}

# All array tasks share one task folder (see create_worker_sbatch_script)
TASK_DIR="{SCRATCH_ROOT}/{NOW_STR}_0"

# Move to the task directory
cd $TASK_DIR

# Load HPC modules
module load anaconda3/2023

# Ensure user-installed python binaries are in PATH
export PATH=$HOME/.local/bin:$PATH

# Install requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Show environment
srun python hpc/env.py

# EXECUTE MAIN WORKLOAD -- this task's shard of the reconstruction.
# Everything except the shard index comes from job.json.
srun python {script_name} \\
    --job_json job.json \\
    --shard_index ${{SLURM_ARRAY_TASK_ID:-0}} \\
    --num_shards {num_tasks}
"""

    with open(path, "w") as f:
        f.write(script_content)


def create_aggregate_sbatch_script(path, array_job_id, task_dir):
    """
    Write out aggregate.sh -- one small job that runs after every array task has
    finished (--dependency=afterok), summarises the shared output directory and
    mails the result.  CPU only: it just parses report files.
    """
    script_content = f"""#!/bin/bash
#SBATCH -J s2v_reconstruction_agg
#SBATCH --output=output_agg_%j.txt
#SBATCH --error=output_agg_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user={MAIL_FROM}
#SBATCH --nodes=1
#SBATCH --partition=STD-s-96h
#SBATCH --time=0-02:00:00
#SBATCH --dependency=afterok:{array_job_id}

TASK_DIR="{task_dir}"
cd $TASK_DIR

# Load HPC modules
module load anaconda3/2023
export PATH=$HOME/.local/bin:$PATH

# Install requirements (CPU wheels are enough for the aggregation)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# AGGREGATE: parse every report in the shared output directory
srun python 3_json_reconstruction_par.py \\
    --summarize \\
    --output_dir "{SHARED_OUTPUT_DIR}"

# Collect the array's logs plus this one so the mail carries all of them
cp $SLURM_SUBMIT_DIR/output_{array_job_id}_*.txt hpc/ 2>/dev/null || true
cp $SLURM_SUBMIT_DIR/output_agg_${{SLURM_JOB_ID}}.txt hpc/ 2>/dev/null || true

# Send final results email
cd hpc && srun python mail.py "agg_${{SLURM_JOB_ID}}" "${{SLURM_JOB_NAME}}"
"""

    with open(path, "w") as f:
        f.write(script_content)


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


def prepare_task_folder(job: dict) -> Path:
    """
    1) Copy the entire codebase to a unique subfolder
    2) Write the shared job.json that every array task reads
    """
    dest_folder = copy_folder(0)

    json_path = dest_folder / "job.json"
    with open(json_path, "w") as f:
        json.dump(job, f, indent=2)

    return dest_folder


# ==============================================================================
# 4) MAIN
# ==============================================================================
def main():
    job_config = {
        "projector_model": PROJECTOR_MODEL,
        "lora_model": LORA_MODEL,
        "embeddings_dir": EMBEDDINGS_DIR,
        "data_dir": DATA_DIR,
        "output_dir": SHARED_OUTPUT_DIR,
        "max_tokens": MAX_TOKENS,
        "batch_size": BATCH_SIZE,
        "num_workers": WORKERS_PER_TASK,
        "eval_workers": EVAL_WORKERS,
    }

    print(f"Preparing task folder...")
    print(f"  Projector model : {PROJECTOR_MODEL}")
    print(f"  LoRA model      : {LORA_MODEL}")
    print(f"  Embeddings dir  : {EMBEDDINGS_DIR}")
    print(f"  Data dir        : {DATA_DIR}")
    print(f"  Output dir      : {SHARED_OUTPUT_DIR}")
    print(f"  Max tokens      : {MAX_TOKENS}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Array tasks     : {NUM_SHARDS} x {NUM_GPU} GPU "
          f"({WORKERS_PER_TASK} worker(s) each)")
    print(f"  CPUs per task   : {CPUS_PER_TASK} ({EVAL_WORKERS} for scoring)")
    print(f"  GPU partition   : {runtarget}")
    print(f"  Exclusive node  : {run_exclusive}")

    # 1) Create the single shared task folder with the job config
    task_folder = prepare_task_folder(job_config)

    # 2) Create and submit the worker array
    worker_script_path = task_folder / "start.sh"
    create_worker_sbatch_script(worker_script_path, NUM_SHARDS)
    array_job_id = submit_sbatch(worker_script_path)

    # 3) Create and submit the aggregation job (waits for the whole array)
    aggregate_script_path = task_folder / "aggregate.sh"
    create_aggregate_sbatch_script(
        aggregate_script_path, array_job_id=array_job_id, task_dir=str(task_folder),
    )
    agg_job_id = submit_sbatch(aggregate_script_path)

    print(f"\n{'=' * 60}")
    print(f"  Reconstruction array: {array_job_id}  ({NUM_SHARDS} tasks)")
    print(f"  Aggregation job:      {agg_job_id}  (runs after the array)")
    print(f"  Task directory:       {task_folder}")
    print(f"  Output directory:     {SHARED_OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
    print("HPC job submitted.")
