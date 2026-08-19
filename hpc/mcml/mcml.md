# 1. General Description and Resources

Source: https://doku.lrz.de/1-general-description-and-resources-10746641.html

## Aim

This system is primarily for Big Data and AI users. It has a focus on GPU resource needs. Support for other use cases of these GPUs is currently limited.

## Compute Hardware

The table below shows the available compute hardware. It also shows the Slurm partitions. Jobs that target these resources must use these partitions. Partitions in grey are currently used mainly for interactive access via Interactive Web Servers. You can typically not target these partitions directly.

The default time limit for a job (allocation) is one hour. The maximum time limit is 2 days (`--time=2-00:00:00`).

| Architecture               | Slurm Partition       | Nodes | CPUs/Node         | Mem/Node    | GPUs/Node            | Mem/GPU    |
|----------------------------|-----------------------|-------|-------------------|-------------|----------------------|------------|
| HGX H100                   | lrz-hgx-h100-92x4     | 30    | 96                | 768 GB      | 4x NVIDIA H100       | 94 GB HBM2 |
| HGX A100                   | lrz-hgx-a100-80x4     | 5     | 96                | 1 TB        | 4x NVIDIA A100       | 80 GB HBM2 |
| DGX A100                   | lrz-dgx-a100-80x8     | 4     | 252               | 2 TB        | 8x NVIDIA A100       | 80 GB HBM2 |
| DGX A100                   | lrz-dgx-a100-40x8-mig | 1     | 252               | 1 TB        | 8x NVIDIA A100       | 40 GB HBM2 |
| DGX-1 V100                 | lrz-dgx-1-v100x8      | 1     | 76                | 512 GB      | 8x NVIDIA Tesla V100 | 16 GB HBM2 |
| DGX-1 P100                 | lrz-dgx-1-p100x8      | 1     | 76                | 512 GB      | 8x NVIDIA Tesla P100 | 16 GB HBM2 |
| HPE Intel Skylake + NVIDIA | lrz-hpe-p100x4        | 1     | 28                | 256 GB      | 4x NVIDIA Tesla P100 | 16 GB HBM2 |
| V100 GPU Nodes             | lrz-v100x2 (default)  | 4     | 19                | 368 GB      | 2x NVIDIA Tesla V100 | 16 GB HBM2 |
| CPU Nodes                  | lrz-cpu               | 12    | 18 / 28 / 38 / 94 | min. 360 GB | --                   | --         |

