import json
import platform
from glob import glob
import subprocess
import shutil
from datetime import datetime
from pathlib import Path


# ==============================================================================
# 2) HPC SETTINGS
# ==============================================================================
SCRATCH_ROOT = "/lustre/scratch/sgjolieb-screen2vec" if not platform.system() == "Windows" else "C:\\Users\\Jonathan\\Dev"
PROJECT_NAME = "s2v"
DATA_DIR_NAME = "rico-data"
runtarget = "cpu"  # or "gpu-small" / "cpu"
run_exclusive = True  # or True if you want the entire node
NUM_GPU: int = 0  # 1..4; how many GPUs to reserve per job
NOW = datetime.now()
# We define the timestamp once here to ensure all 7 folders share the exact same prefix
NOW_STR = NOW.strftime('%Y-%m-%d-%H-%M-%S')
MAIL_FROM = "jonathan.liebers@uni-due.de"
# ==============================================================================



# ==============================================================================
# CREATE AND SUBMIT JOB
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
        partition = "#SBATCH --partition=STD-l-12h"
        timelimit = "#SBATCH --time=0-12:00:00"

    exclusive_str = "#SBATCH --exclusive" if run_exclusive else ""
    array_str = f"#SBATCH --array=0-{num_tasks - 1}" if num_tasks > 0 else ""
    output_str = "#SBATCH --output=stdout_%A_%a.txt" if num_tasks > 0 else "#SBATCH --output=stdout_%j.txt"
    error_str = "#SBATCH --error=stderr_%A_%a.txt" if num_tasks > 0 else "#SBATCH --error=stderr_%j.txt"

    sbatch_header = f"""\
#SBATCH -J {PROJECT_NAME}_{jobname}       # job name
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
srun python 4_evaluate_screen_prediction.py --embeddings_dir embeddings_seq_768_with_app --lstm_model lstm_predictor_768d.pth

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


def prepare_task_folder(array_id: int, job: dict = None):
    """
    1) Copy the entire codebase to a unique subfolder
    2) Generate individual job.json for this specific task (if provided)
    """
    dest_folder = copy_folder(array_id)

    # write down job dict as `job.json` if provided
    if job is not None:
        json_path = dest_folder / "job.json"
        with open(json_path, "w") as f:
            json.dump(job, f)

    return dest_folder


# ==============================================================================
# 4) MAIN
# ==============================================================================
def main():
    total = 1
    
    print(f"Preparing {total} task folders...")

    # 1) Create task folder
    first_folder = prepare_task_folder(0, None)

    # 2) Create the single master start script inside the FIRST copied folder
    master_script_path = first_folder / "start.sh"
    create_master_sbatch_script(master_script_path, total)

    # 3) Submit the single array job from the first folder
    submit_sbatch(master_script_path)

if __name__ == "__main__":
    main()
    print("All HPC jobs submitted as a single array.")