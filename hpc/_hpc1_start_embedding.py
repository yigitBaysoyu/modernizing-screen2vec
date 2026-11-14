import json
import platform
from glob import glob
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

# ==============================================================================
# 1) SETTINGS START
# ==============================================================================

# How many jobs out of the JOBS array each SLURM array task works on
JOBS_PER_BATCH = 8283

# Set to None to run all batches, otherwise set to number of batches to be run (all others are discarded)
MAX_BATCHES = 8
# Total jobs being run is JOBS_PER_BATCH x MAX_BATCHES if MAX_BATCHES is not None

input(f"Only {JOBS_PER_BATCH} x {MAX_BATCHES} = {JOBS_PER_BATCH*MAX_BATCHES} embeddings will be done! (press enter to continue)")

# Embedding shapes produced by 1_qwen_generate_embeddings.py.
# N = 1 -> (4096,), N -> (N, 4096). Results land in
# embeddings_4096_{POOLING}/ resp. embeddings_{N}x4096_{POOLING}/ inside
# each task folder.
#
# This is a list: every entry is pooled out of the *same* forward pass, so a
# whole sweep is one job and one inference run instead of one job per N.
# Supported: 1, 2, 4, 8, 16, 32, 64, 128
POOL_TOKENS = [2, 4, 8, 16, 32, 64]

# "last" (only valid for POOL_TOKENS = [1]) or "avg" (valid for every N).
# Use "avg" throughout when sweeping N, so the runs stay comparable.
POOLING = "avg"

# Skip screens whose .npz already exists in every output directory. Set to True
# when resubmitting a job that ran into its wall time, so it picks up where it
# stopped instead of re-embedding everything.
SKIP_EXISTING = False

# ==============================================================================
# 2) HPC SETTINGS
# ==============================================================================
SCRATCH_ROOT = "/lustre/scratch/sgjolieb-screen2vec" if not platform.system() == "Windows" else "C:\\Users\\Jonathan\\Dev"
PROJECT_NAME = "s2v"
DATA_DIR_NAME = "rico-data"
runtarget = "GPU-H200"  # or "gpu-small" / "cpu"
run_exclusive = False  # or True if you want the entire node
NUM_GPU: int = 1  # 1..4; how many GPUs to reserve per job
NOW = datetime.now()
# We define the timestamp once here to ensure all 7 folders share the exact same prefix
NOW_STR = NOW.strftime('%Y-%m-%d-%H-%M-%S')
MAIL_FROM = "user@example.org"  # address for Slurm job mails; replace with your own
# ==============================================================================


# ==============================================================================
# 3) HPC CONFIG: Define jobs (i.e., elements in slurm's task array)
# ==============================================================================

def create_batches():
    """Returns a list of batches. Each batch consists of one or more than one jobs and is executed by a single node
    following the "embarassingly parallel" principle. Therefore, one batch is one element within slurm's task array and
    contains a one or more than one atomic operations that is being executed by some node.
    If we, for example, return 7 elements, then we expect 7 nodes in slurm working on it, by having created a task array
    with 7 entries and each element contains one or many jobs."""

    # holds return values
    jobs = []

    for idx, g in enumerate(sorted(glob(f"{SCRATCH_ROOT}/{DATA_DIR_NAME}/data/layouts/*json"))):
        jobs.append({
            "task": "embed",
            "json": g,
            "image": g.replace('.json', '.jpg')
        })

    batches = [jobs[i:i + JOBS_PER_BATCH] for i in range(0, len(jobs), JOBS_PER_BATCH)]

    if MAX_BATCHES is not None:
        batches = batches[:MAX_BATCHES]

    return batches


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
    elif runtarget.lower() == "cpu".lower():
        partition = "#SBATCH --partition=SHM-s-96h"
        timelimit = "#SBATCH --time=4-00:00:00"

    exclusive_str = "#SBATCH --exclusive" if run_exclusive else ""
    array_str = f"#SBATCH --array=0-{num_tasks - 1}" if num_tasks > 0 else ""
    output_str = "#SBATCH --output=stdout_%A_%a.txt" if num_tasks > 0 else "#SBATCH --output=stdout_%j.txt"
    error_str = "#SBATCH --error=stderr_%A_%a.txt" if num_tasks > 0 else "#SBATCH --error=stderr_%j.txt"

    sbatch_header = f"""\
#SBATCH -J s2v_embedding
{output_str}            # stdout file
{error_str}             # stderr file
#SBATCH --mail-type=ALL
#SBATCH --mail-user={MAIL_FROM}
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


def create_master_sbatch_script(path, total_array_elements):
    """
    Write out the master start.sh that handles the array.
    """
    sbatch_header = get_hpc_slurm_directives("array", total_array_elements)

    # All N go into one invocation: 1_qwen_generate_embeddings.py pools them
    # from a single forward pass per screen.
    pool_tokens_arg = " ".join(str(n) for n in POOL_TOKENS)
    skip_existing_arg = " --skip-existing" if SKIP_EXISTING else ""

    script_content = f"""#!/bin/bash
{sbatch_header}

# Construct the path to the specific task folder based on current array ID
TASK_DIR="{SCRATCH_ROOT}/{NOW_STR}_${{SLURM_ARRAY_TASK_ID}}"

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

# EXECUTE MAIN WORKLOAD
srun python 1_qwen_generate_embeddings.py job.json --pool-tokens {pool_tokens_arg} --pooling {POOLING}{skip_existing_arg}

# Copy logs from the submission directory (the first folder) to this task's hpc folder
cp $SLURM_SUBMIT_DIR/stdout_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.txt hpc/ 2>/dev/null || true
cp $SLURM_SUBMIT_DIR/stderr_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.txt hpc/ 2>/dev/null || true

# Finish sending away results
cd hpc && srun python mail.py "${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}" "${{SLURM_JOB_NAME}}"
"""

    with open(path, "w") as f:
        f.write(script_content)


def submit_sbatch(script_path: Path):
    """
    sbatch the 'start.sh' script from its parent folder
    """
    print(f"Submitting job via `sbatch` in {script_path}")
    subprocess.run(["sbatch", script_path.name], cwd=script_path.parent, check=True)


def prepare_task_folder(array_id: int, job: dict):
    """
    1) Copy the entire codebase to a unique subfolder
    2) Generate individual job.json for this specific task
    """
    dest_folder = copy_folder(array_id)

    # write down job dict as `job.json`
    json_path = dest_folder / "job.json"
    with open(json_path, "w") as f:
        json.dump(job, f)

    return dest_folder


# ==============================================================================
# 4) MAIN
# ==============================================================================
def main():
    pool_tokens = sorted(set(POOL_TOKENS))
    unsupported = [n for n in pool_tokens if n not in (1, 2, 4, 8, 16, 32, 64, 128)]
    if unsupported:
        raise ValueError(f"Unsupported POOL_TOKENS: {unsupported}")
    if not pool_tokens:
        raise ValueError("POOL_TOKENS is empty; nothing to embed.")
    if POOLING == "last" and pool_tokens != [1]:
        raise ValueError(
            f"POOLING='last' produces a single vector; it cannot be combined "
            f"with POOL_TOKENS={pool_tokens}. Use POOLING='avg'."
        )

    print(f"Embeddings: pooling={POOLING}, one forward pass per screen, outputs:")
    for n in pool_tokens:
        shape = "(4096,)" if n == 1 else f"({n}, 4096)"
        stem = "4096" if n == 1 else f"{n}x4096"
        print(f"  N={n:<3} shape={shape:<12} -> embeddings_{stem}_{POOLING}/")

    # fp16 and barely compressible, so the raw size is a good estimate. Worth
    # printing: the full sweep is a lot larger than any single N, and scratch
    # quota is the thing that bites at the end of a 48h job.
    per_screen_mb = sum(pool_tokens) * 4096 * 2 / 1024 ** 2
    total_gb = per_screen_mb * JOBS_PER_BATCH * (MAX_BATCHES or 1) / 1024
    print(
        f"Estimated storage: ~{per_screen_mb:.2f} MB per screen over all shapes, "
        f"~{total_gb:.1f} GB in total."
    )

    batches = create_batches()
    total = len(batches)

    if total == 0:
        raise ValueError("No jobs (JOBS) found to submit.")

    print(f"Preparing {total} task folders...")

    first_folder = None

    # 1) Create all individual task folders
    for arrayid, job in enumerate(batches):
        try:
            folder_path = prepare_task_folder(arrayid, job)
            if arrayid == 0:
                first_folder = folder_path
        except FileNotFoundError as fe:
            print("Skipping task due to error...", str(fe))

    if not first_folder:
        raise RuntimeError("Failed to create the primary task folder.")

    # 2) Create the single master start script inside the FIRST copied folder
    master_script_path = first_folder / "start.sh"
    create_master_sbatch_script(master_script_path, total)

    # 3) Submit the single array job from the first folder
    submit_sbatch(master_script_path)

if __name__ == "__main__":
    main()
    print("All HPC jobs submitted as a single array.")