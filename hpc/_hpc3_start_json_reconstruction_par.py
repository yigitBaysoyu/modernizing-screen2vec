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

# Model / reconstruction arguments
ADAPTER_FOLDER = "adapter_model_out_embeddings_4x4096_avg_full_qwen2.5-1.5b"
EMBEDDINGS_DIR = "embeddings_4096_last"
PROJECTOR_MODEL = f"{ADAPTER_FOLDER}/best_projector.pth"
LORA_MODEL = f"{ADAPTER_FOLDER}/best_lora_adapter"
DATA_DIR = "../rico-data/data/layouts"
OUTPUT_DIR = f"reconstructions_for_{ADAPTER_FOLDER}"
# Must track AdapterConfig.max_target_len (4096) -- the longest target the
# adapter is ever trained to finish -- plus a small margin.  Anything beyond
# that is a degenerate generation no budget can rescue: across 65,927
# reconstructions from eight adapter variants, only 60 (0.09%) ever terminated
# between 2048 and 8192 tokens, while runaways ran to the ceiling and produced
# invalid JSON in every single case.
MAX_TOKENS = 4608
NUM_WORKERS = 4  # One per GPU

# ==============================================================================
# 2) HPC SETTINGS
# ==============================================================================
SCRATCH_ROOT = "/lustre/scratch/sgjolieb-screen2vec" if not platform.system() == "Windows" else "C:\\Users\\Jonathan\\Dev"
PROJECT_NAME = "s2v"
DATA_DIR_NAME = "rico-data"
runtarget = "GPU-H200"  # GPU-H200 partition with 8x H200 GPUs
run_exclusive = False  # Need exclusive access to all 8 GPUs
NUM_GPU: int = NUM_WORKERS  # 8 H200 GPUs on the node
NOW = datetime.now()
# We define the timestamp once here to ensure the folder has a unique prefix
NOW_STR = NOW.strftime('%Y-%m-%d-%H-%M-%S')
MAIL_FROM = "jonathan.liebers@uni-due.de"
# ==============================================================================


# ==============================================================================
# 3) CREATE AND SUBMIT JOB
# ==============================================================================
def get_hpc_slurm_directives(jobname: str):
    """
    Return #SBATCH script for a single GPU job (no array).
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

    sbatch_header = f"""\
#SBATCH -J {PROJECT_NAME}_{jobname}       # job name
#SBATCH --output=output_%j.txt            # output file (merged stdout and stderr)
#SBATCH --error=output_%j.txt             # error file (redirected to output file)
#SBATCH --mail-type=ALL
#SBATCH --mail-user={MAIL_FROM}
{nodes_str}
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


def create_master_sbatch_script(path):
    """
    Write out the master start.sh for the single multi-GPU reconstruction job.
    """
    script_name = "3_json_reconstruction_par.py"
    sbatch_header = get_hpc_slurm_directives(script_name)

    script_content = f"""#!/bin/bash
{sbatch_header}

# Construct the path to the task folder
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

# EXECUTE MAIN WORKLOAD — multi-GPU parallel JSON reconstruction
srun python {script_name} \\
    --projector_model {PROJECTOR_MODEL} \\
    --lora_model {LORA_MODEL} \\
    --embeddings_dir {EMBEDDINGS_DIR} \\
    --data_dir {DATA_DIR} \\
    --max_tokens {MAX_TOKENS} \\
    --num_workers {NUM_WORKERS} \\
    --output_dir {OUTPUT_DIR}

# Copy merged log from the submission directory to this task's hpc folder
cp $SLURM_SUBMIT_DIR/output_${{SLURM_JOB_ID}}.txt hpc/ 2>/dev/null || true

# Finish sending away results
cd hpc && srun python mail.py "${{SLURM_JOB_ID}}" "${{SLURM_JOB_NAME}}"
"""

    with open(path, "w") as f:
        f.write(script_content)


def submit_sbatch(script_path: Path):
    """
    sbatch the 'start.sh' script from its parent folder
    """
    print(f"Submitting job via `sbatch` in {script_path}")
    subprocess.run(["sbatch", script_path.name], cwd=script_path.parent, check=True)


def prepare_task_folder(job: dict = None):
    """
    1) Copy the entire codebase to a unique subfolder
    2) Generate individual job.json for this specific task (if provided)
    """
    dest_folder = copy_folder(0)

    # write down job dict as `job.json` if provided
    if job is not None:
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
        "output_dir": OUTPUT_DIR,
        "max_tokens": MAX_TOKENS,
        "num_workers": NUM_WORKERS,
    }

    print(f"Preparing task folder...")
    print(f"  Projector model : {PROJECTOR_MODEL}")
    print(f"  LoRA model      : {LORA_MODEL}")
    print(f"  Embeddings dir  : {EMBEDDINGS_DIR}")
    print(f"  Data dir        : {DATA_DIR}")
    print(f"  Output dir      : {OUTPUT_DIR}")
    print(f"  Max tokens      : {MAX_TOKENS}")
    print(f"  Num workers     : {NUM_WORKERS}")
    print(f"  GPU partition   : {runtarget}")
    print(f"  Num GPUs        : {NUM_GPU}")
    print(f"  Exclusive node  : {run_exclusive}")

    # 1) Create task folder with job config
    task_folder = prepare_task_folder(job_config)

    # 2) Create the master start script
    master_script_path = task_folder / "start.sh"
    create_master_sbatch_script(master_script_path)

    # 3) Submit the job
    submit_sbatch(master_script_path)

    print(f"\n{'=' * 60}")
    print(f"  Job submitted successfully!")
    print(f"  Task directory: {task_folder}")
    print(f"  Output will be in: {task_folder / OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
    print("HPC job submitted.")
