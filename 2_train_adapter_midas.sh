accelerate launch --multi_gpu --gpu_ids 2,3 --num_processes 2 2_train_adapter.py "$@"
