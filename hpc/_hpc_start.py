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
JOBS_PER_SLICE = 10000

# ==============================================================================
# 2) HPC SETTINGS
# ==============================================================================
SCRATCH_ROOT = "/lustre/scratch/sgjolieb-screen2vec" if not platform.system() == "Windows" else "C:\\Users\\Jonathan\\Downloads"
PROJECT_NAME = "screen2vec"
DATA_DIR_NAME = "rico-data"
runtarget = "gpu-big"  # or "gpu-small" / "cpu"
run_exclusive = False  # or True if you want the entire node
NUM_GPU: int = 1  # 1..4; how many GPUs to reserve
NOW = datetime.now()
MAIL_FROM = "jonathan.liebers@uni-due.de"
# ==============================================================================


# ==============================================================================
# 3) HPC CONFIG: Parameter space and parameters per job
# ==============================================================================

JOBS = []

for idx, g in enumerate(sorted(glob(f"{SCRATCH_ROOT}/{DATA_DIR_NAME}/data/combined/*json"))):
    JOBS.append({
        "task": "embed",
        "json": g,
        "image": g.replace('.json', '.jpg')
    })


# Added missing build_argument_list since it is called later
def build_argument_list(parameters: dict):
    return json.dumps(parameters)


# ==============================================================================
# 4) CREATE AND SUBMIT JOB
# ==============================================================================
def get_hpc_slurm_directives(jobname: str, num_tasks: int = 0):
    """
    Return #SBATCH script according to your HPC partition & runtarget preferences.
    """
    gres = ""
    partition = ""
    timelimit = "#SBATCH --time=0-24:00:00"

    if runtarget == "gpu-small":
        gres = f"#SBATCH --gres=gpu:{NUM_GPU}"
        partition = "#SBATCH --partition=GPU-small"
        timelimit = "#SBATCH --time=2-00:00:00"
    elif runtarget == "gpu-big":
        gres = f"#SBATCH --gres=gpu:{NUM_GPU}"
        partition = "#SBATCH --partition=GPU-big"
        timelimit = "#SBATCH --time=2-00:00:00"
    elif runtarget == "cpu":
        partition = "#SBATCH --partition=SHM-s-96h"
        timelimit = "#SBATCH --time=4-00:00:00"

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

    now_str = NOW.strftime('%Y-%m-%d-%H-%M-%S')
    job_folder_name = f"{now_str}_{execution_id}"  # job name
    current_project_folder = Path.cwd().parent  # assuming running _hpc_start.py from /hpc subfolder
    target = Path(SCRATCH_ROOT) / job_folder_name
    print("Copying", current_project_folder, "->", target)
    shutil.copytree(current_project_folder, target, ignore=ignore_folders)
    return target


def create_sbatch_script(path):
    """
    Write out the start.sh that will run via Slurm.
    """
    jobname = "array"
    sbatch_header = get_hpc_slurm_directives(jobname)

    script_content = f"""#!/bin/bash
{sbatch_header}

# Move to the job directory
cd $SLURM_SUBMIT_DIR

# Load HPC modules
module load anaconda3/2023

# Optionally install requirements if needed
# pip install -r requirements.txt

# Show environment
srun python --version

# Convert notebook to python only once on the first array task to prevent race conditions
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    srun python qwen_generate_embeddings.py
fi

# Wait for the notebook conversion to finish before starting workers
sleep 1

# Run the specific slice of jobs for this array task
bash chunks/chunk_${{SLURM_ARRAY_TASK_ID}}.sh

# Move outputs
cp stdout_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.txt .. 2>/dev/null || true
cp stderr_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.txt .. 2>/dev/null || true
"""

    with open(path, "w") as f:
        f.write(script_content)


def submit_sbatch(folder: Path):
    """
    sbatch the 'start.sh' script from inside <folder>/hpc
    """
    start_sh_path = folder / "start.sh"
    print(f"Submitting job via `sbatch` in {start_sh_path}")
    subprocess.run(["sbatch", "start.sh"], cwd=folder, check=True)


def start_array_job(arrayid: int, job: dict):
    """
    1) Copy the entire codebase to a unique folder in SCRATCH
    2) Generate individual bash scripts for each chunk
    3) Generate the HPC submission script
    4) Submit the job array
    """
    dest_folder = copy_folder(arrayid)

    # write down job dict as `job.json`
    json_path = dest_folder / "job.json"
    with open(json_path, "w") as f:
        json.dump(job, f)

    slurm_script_path = dest_folder / "start.sh"

    create_sbatch_script(slurm_script_path)
    submit_sbatch(dest_folder)


# ==============================================================================
# 4) MAIN
# ==============================================================================
def main():
    global JOBS

    JOBS = JOBS[:1000]  # FIXME

    total = len(JOBS)
    if total == 0:
        print("No jobs (JOBS) found to submit.")
        exit()

    # Slice the JOBS array based on the new JOBS_PER_SLICE parameter
    jobs = [JOBS[i:i + JOBS_PER_SLICE] for i in range(0, total, JOBS_PER_SLICE)]

    print(f"Submitting {len(jobs)} SLURM array tasks...")

    for arrayid, job in enumerate(jobs):
        try:
            start_array_job(arrayid, job)
        except FileNotFoundError as fe:
            print("Skipping exception ...", str(fe))

if __name__ == "__main__":
    main()
    print("All HPC jobs submitted.")
