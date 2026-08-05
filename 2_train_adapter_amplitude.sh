accelerate launch --multi_gpu --gpu_ids 0,1,2,3 --num_processes 4 2_train_adapter.py "$@"
