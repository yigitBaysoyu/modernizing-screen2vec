#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8           # Partition with 4× A100 GPUs
#SBATCH --qos=mcml                      # REQUIRED for mcml partitions
#SBATCH --gres=gpu:8                    # Request all 4 GPUs
#SBATCH --job-name=train_stylegan3
#SBATCH --output=train_stylegan3_%j.out
#SBATCH --error=train_stylegan3_%j.err
#SBATCH --time=2-00:00:00  # 2 days (D-HH:MM:SS)
#SBATCH --mem=128G

set -x  # Print all commands as they execute (for debugging)

# Define training parameters
TARGET_IMAGE_SIZE=1024
GAMMA=48.0

# Run StyleGAN3 training with host directories mounted to /workspace
srun --container-image="nvcr.io/nvidia/pytorch:24.02-py3" \
     --container-mounts=/dss/mcmlscratch/04/ra73net2:/workspace,\
/dss/dsshome1/04/ra73net2:/workspace-home \
     bash -c "cd /workspace/stylegan3-fashion && \
     python stylegan3/train.py \
       --outdir=/workspace/stylegan3-fashion/training-runs \
       --cfg=stylegan3-t \
       --data=/workspace/stylegan3-fashion/datasets/ffhq-${TARGET_IMAGE_SIZE}x${TARGET_IMAGE_SIZE}.zip \
       --gpus=8 \
       --batch=32 \
       --gamma=${GAMMA} \
       --mirror=1 \
       --kimg=10000 \
       --snap=50 \
       --metrics=none  \
       --zdim=15"
