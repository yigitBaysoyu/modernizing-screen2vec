"""Train Qwen3-4B-Base on 64x4096 embeddings, then reconstruct all RICO screens.

This is a fixed, safe preset for _hpc5_start_adapter_and_reconstruction.py:
training uses one exclusive 8-H200 job; reconstruction uses eight one-H200
array tasks. Reconstruction batches recursively split on CUDA OOM and completed
screen/report pairs are skipped automatically if a shard is rerun.

Run this generator from the project checkout on the HPC login node:

    python hpc/_hpc7_qwen3_4b_64x4096_all_in_one.py --dry-run
    python hpc/_hpc7_qwen3_4b_64x4096_all_in_one.py
"""

import argparse
import importlib.util
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_SCRIPT = Path(__file__).with_name("_hpc5_start_adapter_and_reconstruction.py")


def load_base_launcher():
    spec = importlib.util.spec_from_file_location("s2v_hpc5_launcher", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(module, *, dry_run):
    module.LM_MODEL_ID = "Qwen/Qwen3-4B-Base"
    module.EMBEDDINGS = "64x4096_avg"
    module.EMBEDDINGS_DIR = "embeddings_64x4096_avg"
    module.MODEL_SLUG = "qwen3-4b-base"
    module.ADAPTER_DIR = (
        "adapter_model_out_embeddings_64x4096_avg_full_qwen3-4b-base"
    )
    module.PROJECTOR_MODEL = f"{module.ADAPTER_DIR}/best_projector.pth"
    module.LORA_MODEL = f"{module.ADAPTER_DIR}/best_lora_adapter"
    module.RUN_TAG = "64x4096_avg__via__embeddings_64x4096_avg_full_qwen3-4b-base"
    module.RECON_DIR = f"reconstructions_{module.RUN_TAG}"

    module.runtarget = "GPU-H200"
    module.TRAIN_NUM_GPU = 8
    module.TRAIN_EXCLUSIVE = True
    module.NUM_SHARDS = 8
    module.RECON_NUM_GPU = 1
    # This was the recovery launcher's conservative H200 setting. OOM splitting
    # can reduce an individual batch further without losing completed outputs.
    module.BATCH_SIZE = 64
    module.DRY_RUN = dry_run
    module.DRY_RUN_DIR = PROJECT_ROOT / "_hpc7_dryrun"

    # A stability rerun must never inherit the previous checkpoint or outputs.
    # The generic launcher copies the checkout wholesale; replace that one
    # operation with a copy that omits this preset's two result directories.
    def copy_fresh_task(execution_id):
        target = Path(module.SCRATCH_ROOT) / f"{module.NOW_STR}_{execution_id}"
        excluded = {module.ADAPTER_DIR, module.RECON_DIR}

        def ignore_existing_results(directory, names):
            return {
                name for name in names
                if name.startswith(".") or name in excluded
            }

        print("Copying fresh replicate", module.PROJECT_ROOT, "->", target)
        print("Excluding prior result directories:", ", ".join(sorted(excluded)))
        shutil.copytree(
            module.PROJECT_ROOT,
            target,
            ignore=ignore_existing_results,
            symlinks=True,
        )
        return target

    module.copy_folder = copy_fresh_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preflight and render the generated files without copying or submitting.",
    )
    args = parser.parse_args()

    launcher = load_base_launcher()
    configure(launcher, dry_run=args.dry_run)
    launcher.main()
    if not args.dry_run:
        print("HPC Qwen3-4B-Base 64x4096 job chain submitted.")


if __name__ == "__main__":
    main()
