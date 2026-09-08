import argparse
import json
import re
import statistics
import time

import paramiko
import requests


ssh_host = "10.95.154.240"
ssh_port = 8022
ssh_username = "u0_a285"
ssh_password = "0000"

embedding_url = "http://10.95.154.240:8080/embeddings"
health_url = embedding_url.replace("/embeddings", "/health")

battery_command = "timeout 20s termux-battery-status"
ssh_command_timeout_seconds = 60

ansi_escape = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

server_command = r"""
cd ~/qwen3-vl-embedding || exit 1
: > ./llama-server.log
: > ./llama-server-console.log
nohup ./llama.cpp/build/bin/llama-server \
  -m ./Qwen.Qwen3-VL-Embedding-2B.Q4_K_M.gguf \
  --mmproj ./mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf \
  --embedding \
  --pooling last \
  -c 4096 \
  -np 1 \
  --image-min-tokens 512 \
  --image-max-tokens 512 \
  --no-cache-prompt \
  --cache-ram 0 \
  --cache-reuse 0 \
  --host 0.0.0.0 \
  --port 8080 \
  --timeout 3600 \
  --log-prefix \
  --log-timestamps \
  --log-file ./llama-server.log \
  > ./llama-server-console.log 2>&1 < /dev/null &
echo $!
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure the idle power draw of the Pixel while llama-server is "
            "loaded and ready, but no embedding request is executed."
        )
    )
    parser.add_argument(
        "duration_seconds",
        nargs="?",
        type=float,
        default=180.0,
        help="Idle measurement duration in seconds. Default: 180.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent idle measurements. Default: 1.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=30.0,
        help=(
            "Cooldown between repeated measurements. "
            "The server is restarted for every run. Default: 30."
        ),
    )
    return parser.parse_args()


def clean_terminal_output(text):
    text = ansi_escape.sub("", text)
    text = text.replace("\r", "\n")

    return "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )


def connect_ssh():
    print("\n=== SSH connection ===")
    print("Connecting to Termux ...")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        ssh_host,
        port=ssh_port,
        username=ssh_username,
        password=ssh_password,
        timeout=30,
        banner_timeout=120,
        auth_timeout=60,
        allow_agent=False,
        look_for_keys=False,
    )

    transport = ssh.get_transport()
    transport.set_keepalive(15)

    print("SSH connection established.")
    print(f"Remote SSH version: {transport.remote_version}")
    print("SSH keepalive interval: 15 seconds")

    return ssh


def run_remote(ssh, command, description, check=True, show_output=True):
    print(f"\n[SSH] {description}")
    print(f"[SSH] Command: {command}")

    _, stdout, stderr = ssh.exec_command(
        command,
        timeout=ssh_command_timeout_seconds,
    )
    stdout_text = stdout.read().decode("utf-8").strip()
    stderr_text = stderr.read().decode("utf-8").strip()
    exit_status = stdout.channel.recv_exit_status()

    display_stdout = clean_terminal_output(stdout_text)
    display_stderr = clean_terminal_output(stderr_text)

    print(f"[SSH] Exit status: {exit_status}")

    if show_output and display_stdout:
        print("[SSH] Standard output:")
        print(display_stdout)

    if display_stderr:
        print("[SSH] Standard error:")
        print(display_stderr)

    if check and exit_status != 0:
        raise RuntimeError(
            f"Remote command failed: {description}\n"
            f"Exit status: {exit_status}\n"
            f"Standard output: {stdout_text}\n"
            f"Standard error: {stderr_text}"
        )

    return stdout_text, stderr_text, exit_status


def list_llama_servers(ssh, description):
    stdout_text, _, _ = run_remote(
        ssh,
        "pgrep -af '[l]lama-server' || true",
        description,
        check=False,
    )
    return clean_terminal_output(stdout_text)


def wait_for_health_endpoint_to_stop():
    print(f"[HEALTH] Waiting for {health_url} to stop responding ...")

    for attempt in range(1, 31):
        try:
            response = requests.get(health_url, timeout=1)
            print(
                f"[HEALTH] Shutdown check {attempt}/30: "
                f"still responding with HTTP {response.status_code}"
            )
        except requests.RequestException:
            print(
                f"[HEALTH] Shutdown confirmed after attempt "
                f"{attempt}/30."
            )
            return

        time.sleep(1)

    raise RuntimeError(
        f"The server process was stopped, but {health_url} is still "
        "responding. Another process is probably listening on port 8080."
    )


def stop_llama_servers(ssh, expected_pid, description):
    print(f"\n=== {description} ===")

    if expected_pid is not None:
        run_remote(
            ssh,
            f"if kill -0 {expected_pid} 2>/dev/null; then "
            f"echo 'Sending SIGTERM to expected PID {expected_pid}'; "
            f"kill {expected_pid}; "
            "else "
            f"echo 'Expected PID {expected_pid} is already gone'; "
            "fi",
            "Stop the exact llama-server PID",
            check=False,
        )

    remaining_processes = list_llama_servers(
        ssh,
        "List llama-server processes after PID-specific SIGTERM",
    )

    if remaining_processes:
        print(
            "[SERVER] Remaining process detected. "
            "Sending SIGTERM by full command-line match."
        )
        run_remote(
            ssh,
            "pids=$(pgrep -f '[l]lama-server' 2>/dev/null || true); "
            "if [ -n \"$pids\" ]; then kill $pids; fi",
            "Stop remaining llama-server processes",
            check=False,
        )

    for attempt in range(1, 11):
        remaining_processes = list_llama_servers(
            ssh,
            f"Verify shutdown, attempt {attempt}/10",
        )

        if not remaining_processes:
            print("[SERVER] No llama-server process remains.")
            break

        time.sleep(1)
    else:
        print(
            "[SERVER] SIGTERM did not stop all processes. "
            "Sending SIGKILL."
        )
        run_remote(
            ssh,
            "pids=$(pgrep -f '[l]lama-server' 2>/dev/null || true); "
            "if [ -n \"$pids\" ]; then kill -9 $pids; fi; sleep 1",
            "Force-stop remaining llama-server processes",
            check=False,
        )

        remaining_processes = list_llama_servers(
            ssh,
            "Verify shutdown after SIGKILL",
        )

        if remaining_processes:
            raise RuntimeError(
                "Could not stop all llama-server processes:\n"
                f"{remaining_processes}"
            )

    wait_for_health_endpoint_to_stop()


def verify_server_process(ssh, server_pid, description):
    _, _, status = run_remote(
        ssh,
        f"kill -0 {server_pid} 2>/dev/null && "
        f"printf 'PID {server_pid} command: ' && "
        f"tr '\\0' ' ' < /proc/{server_pid}/cmdline && echo",
        description,
        check=False,
    )

    if status != 0:
        run_remote(
            ssh,
            "tail -n 60 ~/qwen3-vl-embedding/llama-server-console.log",
            "Show console log from the failed server process",
            check=False,
        )
        raise RuntimeError(
            f"llama-server PID {server_pid} exited before becoming ready."
        )


def launch_llama_server(ssh):
    launch_stdout, _, _ = run_remote(
        ssh,
        server_command,
        "Launch a fresh llama-server process",
    )

    pid_lines = [
        line.strip()
        for line in launch_stdout.splitlines()
        if line.strip().isdigit()
    ]

    if not pid_lines:
        raise RuntimeError(
            "The launch command did not return a numeric server PID:\n"
            f"{launch_stdout}"
        )

    server_pid = int(pid_lines[-1])
    print(f"[SERVER] Remote process ID: {server_pid}")

    time.sleep(1)
    verify_server_process(
        ssh,
        server_pid,
        "Verify that the newly launched server PID is alive",
    )

    processes = list_llama_servers(
        ssh,
        "List processes after launching the new server",
    )
    process_lines = processes.splitlines()

    if len(process_lines) != 1 or not process_lines[0].startswith(
        f"{server_pid} "
    ):
        raise RuntimeError(
            "Expected exactly one llama-server process with PID "
            f"{server_pid}, received:\n{processes}"
        )

    return server_pid


def wait_for_llama_server(ssh, server_pid):
    health_started = time.perf_counter()
    last_health_message = None

    for health_attempt in range(1, 181):
        if health_attempt == 1 or health_attempt % 5 == 0:
            verify_server_process(
                ssh,
                server_pid,
                f"Verify server PID during health check "
                f"{health_attempt}/180",
            )

        try:
            health_response = requests.get(health_url, timeout=10)
            health_message = (
                f"HTTP {health_response.status_code}: "
                f"{health_response.text.strip()}"
            )

            if health_message != last_health_message or health_attempt % 5 == 0:
                print(
                    f"[HEALTH] Attempt {health_attempt}/180, "
                    f"{health_message}"
                )
                last_health_message = health_message

            if health_response.ok:
                verify_server_process(
                    ssh,
                    server_pid,
                    "Verify server PID after successful health check",
                )
                break
        except requests.RequestException as error:
            if health_attempt == 1 or health_attempt % 5 == 0:
                print(
                    f"[HEALTH] Attempt {health_attempt}/180 failed: "
                    f"{type(error).__name__}: {error}"
                )

        time.sleep(2)
    else:
        run_remote(
            ssh,
            "tail -n 60 ~/qwen3-vl-embedding/llama-server-console.log",
            "Show llama-server console log after startup failure",
            check=False,
        )
        raise RuntimeError(
            "llama-server did not become ready within 6 minutes"
        )

    health_elapsed = time.perf_counter() - health_started
    print(f"[SERVER] Ready after {health_elapsed:.3f} seconds")
    return health_elapsed


def read_battery(ssh, label):
    stdout_text, _, _ = run_remote(
        ssh,
        battery_command,
        f"Read battery status ({label})",
    )

    try:
        battery = json.loads(stdout_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "termux-battery-status did not return valid JSON:\n"
            f"{stdout_text}"
        ) from error

    print(f"[BATTERY] Parsed values ({label}):")
    print(json.dumps(battery, indent=2))

    if "energy" not in battery and "charge_counter" not in battery:
        raise RuntimeError(
            "The Pixel returned neither 'energy' nor 'charge_counter'. "
            "The battery fuel gauge therefore cannot provide an energy delta."
        )

    return battery


def calculate_energy(battery_before, battery_after, elapsed_seconds):
    energy_before = battery_before.get("energy")
    energy_after = battery_after.get("energy")
    charge_before = battery_before.get("charge_counter")
    charge_after = battery_after.get("charge_counter")
    voltage_before = battery_before.get("voltage")
    voltage_after = battery_after.get("voltage")
    current_before = battery_before.get("current")
    current_after = battery_after.get("current")

    used_microamp_hours = None
    mean_voltage_volts = None

    if charge_before is not None and charge_after is not None:
        used_microamp_hours = charge_before - charge_after

    if voltage_before is not None and voltage_after is not None:
        mean_voltage_volts = (
            (voltage_before + voltage_after) / 2
        ) / 1_000

    if energy_before is not None and energy_after is not None:
        measurement_method = "Android energy counter"
        energy_milliwatt_hours = (
            energy_before - energy_after
        ) / 1_000_000
    elif used_microamp_hours is not None and mean_voltage_volts is not None:
        measurement_method = "charge counter multiplied by mean voltage"
        energy_milliwatt_hours = (
            used_microamp_hours * mean_voltage_volts
        ) / 1_000
    else:
        raise RuntimeError(
            "The battery response does not contain enough values to "
            "calculate energy consumption."
        )

    energy_joules = energy_milliwatt_hours * 3.6
    average_power_watts = energy_joules / elapsed_seconds

    return {
        "measurement_method": measurement_method,
        "used_microamp_hours": used_microamp_hours,
        "mean_voltage_volts": mean_voltage_volts,
        "current_before": current_before,
        "current_after": current_after,
        "energy_milliwatt_hours": energy_milliwatt_hours,
        "energy_joules": energy_joules,
        "average_power_watts": average_power_watts,
    }


def run_idle_measurement(ssh, run_number, duration_seconds):
    server_pid = None

    try:
        print(f"\n=== Idle run {run_number} ===")

        stop_llama_servers(
            ssh,
            expected_pid=None,
            description="Pre-launch cleanup",
        )

        server_pid = launch_llama_server(ssh)
        wait_for_llama_server(ssh, server_pid)

        print("\n=== Idle measurement ===")
        print(
            "The llama-server is loaded and healthy. "
            "No embedding request will be sent."
        )
        print(
            f"Measurement duration: {duration_seconds:.3f} seconds"
        )

        battery_before = read_battery(
            ssh,
            f"before idle run {run_number}",
        )

        print(
            f"[IDLE] Start time: "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        start = time.perf_counter()
        time.sleep(duration_seconds)
        elapsed = time.perf_counter() - start

        print(
            f"[IDLE] End time: "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(
            f"[IDLE] Actual elapsed time: {elapsed:.3f} seconds"
        )

        battery_after = read_battery(
            ssh,
            f"after idle run {run_number}",
        )

        result = calculate_energy(
            battery_before,
            battery_after,
            elapsed,
        )

        print(f"[RESULT] Measurement method: {result['measurement_method']}")

        if result["used_microamp_hours"] is not None:
            print(
                f"[RESULT] Battery charge used: "
                f"{result['used_microamp_hours']:.0f} microamp-hours"
            )

        if result["mean_voltage_volts"] is not None:
            print(
                f"[RESULT] Mean battery voltage: "
                f"{result['mean_voltage_volts']:.3f} V"
            )

        print(
            f"[RESULT] Idle duration: "
            f"{elapsed:.3f} seconds"
        )
        print(
            f"[RESULT] Idle energy consumption: "
            f"{result['energy_milliwatt_hours']:.6f} mWh"
        )
        print(
            f"[RESULT] Idle energy consumption: "
            f"{result['energy_joules']:.6f} J"
        )
        print(
            f"[RESULT] Idle average power: "
            f"{result['average_power_watts']:.6f} W"
        )

        if result["current_before"] is not None:
            print(
                f"[RESULT] Battery current before: "
                f"{result['current_before'] / 1_000:.1f} mA"
            )

        if result["current_after"] is not None:
            print(
                f"[RESULT] Battery current after: "
                f"{result['current_after'] / 1_000:.1f} mA"
            )

        return {
            "run": run_number,
            "duration_seconds": elapsed,
            **result,
        }
    finally:
        if server_pid is not None:
            stop_llama_servers(
                ssh,
                expected_pid=server_pid,
                description="Post-idle shutdown",
            )


def main():
    args = parse_args()

    if args.duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero.")

    if args.runs <= 0:
        raise ValueError("--runs must be greater than zero.")

    if args.cooldown_seconds < 0:
        raise ValueError("--cooldown-seconds cannot be negative.")

    print("=== Configuration ===")
    print(
        f"Idle measurement duration: "
        f"{args.duration_seconds:.3f} seconds"
    )
    print(f"Number of runs: {args.runs}")
    print(
        f"Cooldown between runs: "
        f"{args.cooldown_seconds:.3f} seconds"
    )
    print(f"Embedding URL: {embedding_url}")
    print(f"Health URL: {health_url}")
    print(f"SSH target: {ssh_username}@{ssh_host}:{ssh_port}")
    print(
        "Baseline definition: llama-server loaded and healthy, "
        "no embedding request."
    )

    ssh = connect_ssh()
    results = []

    try:
        run_remote(ssh, "whoami", "Verify remote Termux user")
        run_remote(ssh, "uname -a", "Show remote system information")

        _, _, battery_tool_status = run_remote(
            ssh,
            "command -v termux-battery-status",
            "Find termux-battery-status",
            check=False,
        )

        if battery_tool_status != 0:
            raise RuntimeError(
                "termux-battery-status is not installed. "
                "Install the Termux:API Android app and run "
                "'pkg install termux-api' inside Termux."
            )

        run_remote(
            ssh,
            "cd ~/qwen3-vl-embedding && "
            "test -x ./llama.cpp/build/bin/llama-server && "
            "ls -lh ./llama.cpp/build/bin/llama-server "
            "./Qwen.Qwen3-VL-Embedding-2B.Q4_K_M.gguf "
            "./mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf",
            "Verify llama-server and model files",
        )

        baseline_battery = read_battery(ssh, "preflight")

        if baseline_battery.get("plugged") != "UNPLUGGED":
            print(
                "[WARNING] The phone reports that it is connected to power. "
                "Disconnect USB and wireless charging before measuring energy."
            )

        for run_number in range(1, args.runs + 1):
            if run_number > 1 and args.cooldown_seconds > 0:
                print("\n=== Cooldown ===")
                print(
                    f"Waiting {args.cooldown_seconds:.3f} seconds "
                    "before the next idle run ..."
                )
                time.sleep(args.cooldown_seconds)

            result = run_idle_measurement(
                ssh,
                run_number,
                args.duration_seconds,
            )
            results.append(result)
    finally:
        print("\n=== Final cleanup ===")
        print("Shutting down llama-server ...")

        try:
            stop_llama_servers(
                ssh,
                expected_pid=None,
                description="Final llama-server shutdown",
            )
        finally:
            ssh.close()
            print("SSH connection closed.")

    print("\n=== Summary ===")

    for result in results:
        print(
            f"Idle run {result['run']}: "
            f"{result['duration_seconds']:.3f} s, "
            f"{result['energy_milliwatt_hours']:.6f} mWh, "
            f"{result['energy_joules']:.6f} J, "
            f"{result['average_power_watts']:.6f} W, "
            f"{result['measurement_method']}"
        )

    powers = [
        result["average_power_watts"]
        for result in results
    ]

    mean_idle_power_watts = statistics.mean(powers)

    if len(powers) > 1:
        median_idle_power_watts = statistics.median(powers)
        stdev_idle_power_watts = statistics.stdev(powers)

        print(
            f"[SUMMARY] Mean idle power: "
            f"{mean_idle_power_watts:.6f} W"
        )
        print(
            f"[SUMMARY] Median idle power: "
            f"{median_idle_power_watts:.6f} W"
        )
        print(
            f"[SUMMARY] Idle-power standard deviation: "
            f"{stdev_idle_power_watts:.6f} W"
        )
    else:
        print(
            f"[SUMMARY] Idle power: "
            f"{mean_idle_power_watts:.6f} W"
        )

    print("\n=== Baseline subtraction ===")
    print(
        "For a pixel.py request with response time t seconds and measured "
        "energy E mWh:"
    )
    print(
        "corrected_energy_mWh = "
        "E - idle_power_W * t / 3.6"
    )
    print(
        "Equivalently, corrected average power is approximately:"
    )
    print(
        "corrected_average_power_W = "
        "measured_average_power_W - idle_power_W"
    )


if __name__ == "__main__":
    main()