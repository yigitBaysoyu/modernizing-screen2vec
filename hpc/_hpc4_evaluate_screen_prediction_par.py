import json
import platform
from glob import glob
import subprocess
import shutil
from datetime import datetime
from pathlib import Path


# ==============================================================================
# 1) SETTINGS
# ==============================================================================

# How many parallel CPU nodes to use for trace evaluation
NUM_NODES = 10

# Evaluation script arguments
EMBEDDINGS_DIR = "embeddings_seq_768_with_app"
LSTM_MODEL = "lstm_predictor_768d.pth"       # Set to None for mean-of-context baseline
LABEL = None                                  # Custom label for the results table (auto-detected if None)

# ==============================================================================
# 2) HPC SETTINGS
# ==============================================================================
SCRATCH_ROOT = "/lustre/scratch/sgjolieb-screen2vec" if not platform.system() == "Windows" else "C:\\Users\\Jonathan\\Dev"
PROJECT_NAME = "s2v"
DATA_DIR_NAME = "rico-data"
runtarget = "cpu"  # "cpu" / "std" (STD-s-96h: 1-45 nodes), "cpu-shm" (SHM-s-96h: 1-6 nodes), "gpu-small", "gpu-big"
run_exclusive = False  # Set to False for standard shared nodes (STD), or True if requesting exclusive node access (e.g. SHM)
NUM_GPU: int = 0  # 1..4; how many GPUs to reserve per job
NOW = datetime.now()
# We define the timestamp once here to ensure all folders share the exact same prefix
NOW_STR = NOW.strftime('%Y-%m-%d-%H-%M-%S')
MAIL_FROM = "jonathan.liebers@uni-due.de"
# ==============================================================================


# ==============================================================================
# 3) HPC CONFIG: Define jobs (batches for the SLURM array)
# ==============================================================================

def create_batches():
    """Returns a list of batch configs, one per SLURM array task.
    Each batch tells its worker which slice of traces to evaluate."""
    return [
        {
            "batch_index": i,
            "num_batches": NUM_NODES,
            "results_dir": str(Path(SCRATCH_ROOT) / f"{NOW_STR}_eval_results"),
        }
        for i in range(NUM_NODES)
    ]


# ==============================================================================
# 4) CREATE AND SUBMIT JOB
# ==============================================================================
def get_hpc_slurm_directives(jobname: str, num_tasks: int):
    """
    Return #SBATCH script
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
    array_str = f"#SBATCH --array=0-{num_tasks - 1}" if num_tasks > 1 else ""
    output_str = "#SBATCH --output=output_%A_%a.txt" if num_tasks > 1 else "#SBATCH --output=output_%j.txt"
    error_str = "#SBATCH --error=output_%A_%a.txt" if num_tasks > 1 else "#SBATCH --error=output_%j.txt"

    sbatch_header = f"""\
#SBATCH -J s2v_screen_prediction
{output_str}            # output file (merged stdout and stderr)
{error_str}             # error file (redirected to output file)
#SBATCH --mail-type=FAIL
#SBATCH --mail-user={MAIL_FROM}
{nodes_str}
{array_str}
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


def create_worker_sbatch_script(path, total_array_elements):
    """
    Write out start.sh — the SLURM array job that runs the evaluation workers.
    Each array task evaluates a distinct slice of traces.
    """
    script_name = "4_evaluate_screen_prediction_par.py"
    sbatch_header = get_hpc_slurm_directives(script_name, total_array_elements)

    # Build the python command line
    lstm_arg = f"--lstm_model {LSTM_MODEL}" if LSTM_MODEL else ""

    script_content = f"""#!/bin/bash
{sbatch_header}

# Construct the path to the specific task folder based on current array ID
TASK_DIR="{SCRATCH_ROOT}/{NOW_STR}_${{SLURM_ARRAY_TASK_ID:-0}}"

# Move to the specific task directory
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

# EXECUTE MAIN WORKLOAD (worker mode — reads trace split from job.json)
srun python {script_name} --embeddings_dir {EMBEDDINGS_DIR} {lstm_arg} --job_json job.json
"""

    with open(path, "w") as f:
        f.write(script_content)


def create_aggregate_sbatch_script(path, array_job_id, results_dir, task_dir):
    """
    Write out aggregate.sh — a single SLURM job that runs AFTER all array
    workers complete (via --dependency=afterok). It reads all partial results,
    computes the final metrics, and sends the results via email.
    """
    # Determine label for the results table
    if LABEL:
        label_for_table = LABEL
    elif LSTM_MODEL:
        label_for_table = ">>> Qwen3-VL + LSTM (Ours) <<<"
    else:
        label_for_table = ">>> Qwen3-VL-Embedding (Ours) <<<"

    lstm_mode_arg = f"--lstm_model {LSTM_MODEL}" if LSTM_MODEL else ""

    # Use same partition as workers but with shorter time limit
    partition = ""
    if runtarget.lower() in ("cpu-shm", "shm"):
        partition = "#SBATCH --partition=SHM-s-96h"
    elif runtarget.lower() in ("cpu", "cpu-std", "std", "std-s-96h"):
        partition = "#SBATCH --partition=STD-s-96h"
    elif runtarget.lower() in ("gpu-small", "gpu-big", "gpu-h200"):
        partition = "#SBATCH --partition=STD-s-96h"  # aggregation is CPU-only

    script_content = f"""#!/bin/bash
#SBATCH -J s2v_eval_aggregate
#SBATCH --output=output_agg_%j.txt
#SBATCH --error=output_agg_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user={MAIL_FROM}
#SBATCH --nodes=1
{partition}
#SBATCH --time=0-02:00:00
#SBATCH --dependency=afterok:{array_job_id}

# Run from the first task folder (has the full codebase)
TASK_DIR="{task_dir}"
cd $TASK_DIR

# Load HPC modules
module load anaconda3/2023
export PATH=$HOME/.local/bin:$PATH

# Install requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# AGGREGATE all partial results into final metrics
srun python 4_evaluate_screen_prediction_par.py \\
    --aggregate \\
    --results_dir {results_dir} \\
    --label "{label_for_table}" \\
    {lstm_mode_arg}

# Copy aggregation log to hpc/ for the mailer
cp $SLURM_SUBMIT_DIR/output_agg_${{SLURM_JOB_ID}}.txt hpc/ 2>/dev/null || true

# Send final results email
cd hpc && srun python mail.py "${{SLURM_JOB_ID}}" "${{SLURM_JOB_NAME}}"
"""

    with open(path, "w") as f:
        f.write(script_content)


def submit_sbatch(script_path: Path) -> str:
    """
    sbatch the script from its parent folder.
    Returns the SLURM job ID (parsed from sbatch stdout).
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


def prepare_task_folder(array_id: int, job: dict):
    """
    1) Copy the entire codebase to a unique subfolder
    2) Generate individual job.json for this specific task
    """
    dest_folder = copy_folder(array_id)

    # write down job dict as `job.json`
    json_path = dest_folder / "job.json"
    with open(json_path, "w") as f:
        json.dump(job, f, indent=2)

    return dest_folder


# ==============================================================================
# 5) MAIN
# ==============================================================================
def main():
    batches = create_batches()
    total = len(batches)

    if total == 0:
        raise ValueError("NUM_NODES must be >= 1.")

    # Shared results directory on Lustre (all workers write here)
    results_dir = Path(SCRATCH_ROOT) / f"{NOW_STR}_eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Shared results directory: {results_dir}")

    print(f"Preparing {total} task folders...")

    first_folder = None

    # 1) Create all individual task folders (one per array element)
    for arrayid, job in enumerate(batches):
        try:
            folder_path = prepare_task_folder(arrayid, job)
            if arrayid == 0:
                first_folder = folder_path
        except FileNotFoundError as fe:
            print("Skipping task due to error...", str(fe))

    if not first_folder:
        raise RuntimeError("Failed to create the primary task folder.")

    # 2) Create the worker array job script inside the FIRST folder
    worker_script_path = first_folder / "start.sh"
    create_worker_sbatch_script(worker_script_path, total)

    # 3) Submit the worker array job and capture its ID
    array_job_id = submit_sbatch(worker_script_path)

    # 4) Create the aggregation job script (runs after all workers finish)
    aggregate_script_path = first_folder / "aggregate.sh"
    create_aggregate_sbatch_script(
        aggregate_script_path,
        array_job_id=array_job_id,
        results_dir=str(results_dir),
        task_dir=str(first_folder),
    )

    # 5) Submit the aggregation job (dependency: afterok on the array)
    agg_job_id = submit_sbatch(aggregate_script_path)

    print(f"\n{'=' * 60}")
    print(f"  Worker array job:   {array_job_id}  ({total} tasks)")
    print(f"  Aggregation job:    {agg_job_id}  (runs after workers)")
    print(f"  Results directory:  {results_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
    print("All HPC jobs submitted.")
