import os
import psutil
import socket
import getpass
import GPUtil
import subprocess
import tensorflow as tf

def main():
    # Print the environment variables
    print("Environment Variables:", os.environ)

    # Get and print the available system memory
    memory_info = psutil.virtual_memory()
    available_memory = memory_info.available / (1024 ** 2)  # Convert bytes to megabytes
    print(f"\nAvailable System Memory: {available_memory:.2f} MB")

    # Get and print the number of CPU cores
    cpu_cores = psutil.cpu_count(logical=True)
    print(f"Number of CPU Cores: {cpu_cores}")
    os.system("lscpu | head -n17")

    # Get and print the hostname
    hostname = socket.gethostname()
    print(f"Hostname: {hostname}")

    # Get and print the current user
    current_user = getpass.getuser()
    print(f"Current User: {current_user}")

    # Get and print GPU information
    gpus = GPUtil.getGPUs()
    print("GPU utilization", GPUtil.showUtilization())
    if gpus:
        print("\nAvailable GPUs:")
        for gpu in gpus:
            print(f"\tGPU ID: {gpu.id}")
            print(f"\tGPU Name: {gpu.name}")
            print(f"\tGPU Load: {gpu.load * 100:.2f}%")
            print(f"\tGPU Memory Free: {gpu.memoryFree:.2f} MB")
            print(f"\tGPU Memory Used: {gpu.memoryUsed:.2f} MB")
            print(f"\tGPU Memory Total: {gpu.memoryTotal:.2f} MB")
            print(f"\tGPU Temperature: {gpu.temperature:.2f} °C")
            print()
    else:
        print("\nNo GPUs found.")

    # Run nvidia-smi mig -l to get MIG instance details
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\nNVIDIA SMI:")
            print(result.stdout)
        else:
            print("\nFailed to retrieve nvidia-smi output.")
            print(result.stderr)
    except FileNotFoundError:
        print("\n'nvidia-smi' not found. Ensure that NVIDIA drivers are installed and 'nvidia-smi' is available in the PATH.")
    
    # Run tensorflow GPU info
    import tensorflow as tf

    # Get the list of GPUs
    device_type: str = "GPU"
    gpus = tf.config.experimental.list_physical_devices(device_type)
    
    if gpus:
        print(f"Number of TF GPUs available: {len(gpus)}", "for tensorflow with version", tf.__version__)
        print(tf.config.list_physical_devices(device_type))
        for i, gpu in enumerate(gpus):
            print(f"TF {device_type} {i}: {gpu}")
    
            # Get more details about each GPU
            details = tf.config.experimental.get_device_details(gpu)
            for key, value in details.items():
                print(f"\t{key}: {value}")
    else:
        print(f"No TF {device_type}s found.")

    # simple computation to check GPU usage
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print("TF matmul result:", c)


if __name__ == "__main__":
    print()
    print("== ENVIRONMENT INFORMATION START ==")
    main()
    print("== ENVIRONMENT INFORMATION END ==")
    print()

