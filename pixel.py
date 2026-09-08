import base64
import json
import re
import sys
import time
from pathlib import Path

import paramiko
import requests

from utils import process_json_file


folder = Path(r"C:\Users\Jonathan\Dev\rico-data\data\layouts")
url = "http://192.168.0.52:8080/embeddings"

ssh_host = "192.168.0.52"
ssh_port = 8022
ssh_username = "u0_a285"
ssh_password = "0000"

battery_command = "timeout 20s termux-battery-status"

ssh_command_timeout_seconds = 60
http_connect_timeout_seconds = 30
http_read_timeout_seconds = 900

health_url = url.replace("/embeddings", "/health")

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

run_id = sys.argv[1] if len(sys.argv) > 1 else "2"
rico_id = run_id
image_path = folder / f"{rico_id}.jpg"
json_path = folder / f"{rico_id}.json"

print("=== Configuration ===")
print(f"Run ID: {run_id}")
print(f"RICO ID: {rico_id}")
print(f"Image: {image_path}")
print(f"JSON: {json_path}")
print(f"Embedding URL: {url}")
print(f"SSH target: {ssh_username}@{ssh_host}:{ssh_port}")

if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

if not json_path.is_file():
    raise FileNotFoundError(f"JSON not found: {json_path}")

image_bytes = image_path.read_bytes()
image_base64 = base64.b64encode(image_bytes).decode("ascii")
cleaned_json = process_json_file(json_path, minify=True)

print(f"Image size: {len(image_bytes):,} bytes")
print(f"Base64 image size: {len(image_base64):,} characters")
print(f"Minified JSON size: {len(cleaned_json):,} characters")

requests_to_run = [
    (
        "Image only",
        {
            "content": [
                {
                    "prompt_string": "<__media__>",
                    "multimodal_data": [image_base64],
                }
            ],
            "embd_normalize": 2,
        },
    ),
    (
        "Image + JSON",
        {
            "content": [
                {
                    "prompt_string": f"<__media__>\n{cleaned_json}",
                    "multimodal_data": [image_base64],
                }
            ],
            "embd_normalize": 2,
        },
    ),
    (
        "JSON only",
        {
            "content": [
                {
                    "prompt_string": cleaned_json,
                }
            ],
            "embd_normalize": 2,
        },
    ),
]

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


def run_remote(command, description, check=True, show_output=True):
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


def clean_terminal_output(text):
    text = ansi_escape.sub("", text)
    text = text.replace("\r", "\n")

    return "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )


def list_llama_servers(description):
    stdout_text, _, _ = run_remote(
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


def stop_llama_servers(expected_pid, description):
    print(f"\n=== {description} ===")

    if expected_pid is not None:
        run_remote(
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
        "List llama-server processes after PID-specific SIGTERM"
    )

    if remaining_processes:
        print(
            "[SERVER] Remaining process detected. "
            "Sending SIGTERM by full command-line match."
        )
        run_remote(
            "pids=$(pgrep -f '[l]lama-server' 2>/dev/null || true); "
            "if [ -n \"$pids\" ]; then kill $pids; fi",
            "Stop remaining llama-server processes",
            check=False,
        )

    for attempt in range(1, 11):
        remaining_processes = list_llama_servers(
            f"Verify shutdown, attempt {attempt}/10"
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
            "pids=$(pgrep -f '[l]lama-server' 2>/dev/null || true); "
            "if [ -n \"$pids\" ]; then kill -9 $pids; fi; sleep 1",
            "Force-stop remaining llama-server processes",
            check=False,
        )

        remaining_processes = list_llama_servers(
            "Verify shutdown after SIGKILL"
        )

        if remaining_processes:
            raise RuntimeError(
                "Could not stop all llama-server processes:\n"
                f"{remaining_processes}"
            )

    wait_for_health_endpoint_to_stop()


def verify_server_process(server_pid, description):
    _, _, status = run_remote(
        f"kill -0 {server_pid} 2>/dev/null && "
        f"printf 'PID {server_pid} command: ' && "
        f"tr '\\0' ' ' < /proc/{server_pid}/cmdline && echo",
        description,
        check=False,
    )

    if status != 0:
        run_remote(
            "tail -n 60 ~/qwen3-vl-embedding/llama-server-console.log",
            "Show console log from the failed server process",
            check=False,
        )
        raise RuntimeError(
            f"llama-server PID {server_pid} exited before becoming ready."
        )


def launch_llama_server():
    launch_stdout, _, _ = run_remote(
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
        server_pid,
        "Verify that the newly launched server PID is alive",
    )

    processes = list_llama_servers(
        "List processes after launching the new server"
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


def wait_for_llama_server(server_pid):
    health_started = time.perf_counter()
    last_health_message = None

    for health_attempt in range(1, 181):
        if health_attempt == 1 or health_attempt % 5 == 0:
            verify_server_process(
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


def read_battery(label):
    stdout_text, _, _ = run_remote(
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


def read_process_memory(server_pid):
    """Quietly read this server's kernel-reported resident RAM in MiB.

    VmHWM retains the process-lifetime RSS high-water mark, including model
    startup and short-lived inference allocations, without a polling loop.
    These are OS accounting estimates, not PSS or virtual memory sizes.
    See https://man7.org/linux/man-pages/man5/proc_pid_status.5.html
    """
    channel = None

    try:
        _, stdout, stderr = ssh.exec_command(
            f"cat /proc/{int(server_pid)}/status",
            timeout=5,
        )
        channel = stdout.channel
        status_text = stdout.read().decode("utf-8")
        error_text = stderr.read().decode("utf-8").strip()
        exit_status = channel.recv_exit_status()

        if exit_status != 0:
            raise RuntimeError(error_text or f"Remote exit status {exit_status}")

        values = {
            match.group(1): int(match.group(2)) / 1024
            for match in re.finditer(
                r"^(VmRSS|VmHWM):\s+(\d+)\s+kB\s*$",
                status_text,
                re.MULTILINE,
            )
        }

        return {
            "rss_mib": values.get("VmRSS"),
            "peak_rss_mib": values.get("VmHWM"),
            "error": None,
        }
    except Exception as error:
        # A missing /proc permission or a failed read must not abort the run.
        return {
            "rss_mib": None,
            "peak_rss_mib": None,
            "error": " ".join(f"{type(error).__name__}: {error}".split()),
        }
    finally:
        if channel is not None:
            channel.close()


def print_process_memory(server_pid, memory_before, memory_after):
    for label, snapshot, key in (
        ("Process RSS before request", memory_before, "rss_mib"),
        ("Process RSS after request", memory_after, "rss_mib"),
        (
            "Process peak RSS (startup + request, VmHWM)",
            memory_after,
            "peak_rss_mib",
        ),
    ):
        value = snapshot[key]
        if value is None:
            reason = snapshot["error"] or "Field missing from /proc status"
            formatted = f"unavailable ({reason})"
        else:
            formatted = f"{value:.3f} MiB"
        print(f"[RESULT] {label}: {formatted} [PID {server_pid}]")


class RecoverableRequestError(RuntimeError):
    """A failure confined to one embedding payload, not the whole batch."""


def response_error_detail(response):
    """Return the server's useful error text without assuming a JSON schema."""
    body = response.text.strip()

    if not body:
        return "empty response body"

    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError:
        return " ".join(body.split())

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if error:
            return str(error)

    return " ".join(body.split())


def parse_embedding_response(response):
    """Validate the small part of llama-server's response schema we use."""
    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as error:
        raise RecoverableRequestError(
            "llama-server returned HTTP 200 but not valid JSON: "
            f"{response_error_detail(response)}"
        ) from error

    try:
        embedding = payload[0]["embedding"][0]
    except (IndexError, KeyError, TypeError) as error:
        raise RecoverableRequestError(
            "llama-server returned an unexpected embedding response shape"
        ) from error

    if not isinstance(embedding, list) or not embedding:
        raise RecoverableRequestError(
            "llama-server returned an empty or invalid embedding vector"
        )

    return payload, len(embedding)


def show_request_server_logs():
    run_remote(
        "tail -n 30 ~/qwen3-vl-embedding/llama-server.log",
        "Show final llama-server log lines for this request",
        check=False,
    )
    run_remote(
        "tail -n 30 ~/qwen3-vl-embedding/llama-server-console.log",
        "Show final llama-server console lines for this request",
        check=False,
    )


results = []
request_failures = []
server_pid = None

try:
    run_remote("whoami", "Verify remote Termux user")
    run_remote("uname -a", "Show remote system information")

    _, _, battery_tool_status = run_remote(
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
        "cd ~/qwen3-vl-embedding && "
        "test -x ./llama.cpp/build/bin/llama-server && "
        "ls -lh ./llama.cpp/build/bin/llama-server "
        "./Qwen.Qwen3-VL-Embedding-2B.Q4_K_M.gguf "
        "./mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf",
        "Verify llama-server and model files",
    )

    baseline_battery = read_battery("preflight")

    if baseline_battery.get("plugged") != "UNPLUGGED":
        print(
            "[WARNING] The phone reports that it is connected to power. "
            "Disconnect USB and wireless charging before measuring energy."
        )

    for request_number, (request_name, body) in enumerate(
        requests_to_run,
        start=1,
    ):
        if request_number > 1:
            print("\n=== Cooldown ===")
            print("Waiting 30 seconds before the next request ...")
            time.sleep(30)

        print(f"\n=== Request {request_number}/3: {request_name} ===")

        try:
            stop_llama_servers(
                expected_pid=server_pid,
                description="Pre-launch cleanup",
            )
            server_pid = None

            server_pid = launch_llama_server()
            wait_for_llama_server(server_pid)

            # Keep extra SSH commands outside the battery measurement window.
            memory_before = read_process_memory(server_pid)
            battery_before = read_battery(f"before {request_name}")

            request_size = len(
                json.dumps(body, separators=(",", ":")).encode("utf-8")
            )

            print(f"[HTTP] Sending request to {url}")
            print(f"[HTTP] Request body size: {request_size:,} bytes")
            print(f"[HTTP] Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

            start = time.perf_counter()
            try:
                response = requests.post(
                    url,
                    json=body,
                    timeout=(
                        http_connect_timeout_seconds,
                        http_read_timeout_seconds,
                    ),
                )
            except requests.RequestException as error:
                elapsed = time.perf_counter() - start
                raise RecoverableRequestError(
                    "HTTP request failed after "
                    f"{elapsed:.3f} seconds: {type(error).__name__}: {error}"
                ) from error

            elapsed = time.perf_counter() - start

            print(f"[HTTP] End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"[HTTP] Status code: {response.status_code}")
            print(
                f"[HTTP] Response size: "
                f"{len(response.text.encode('utf-8')):,} bytes"
            )
            print(f"[HTTP] Response time: {elapsed:.3f} seconds")

            battery_after = read_battery(f"after {request_name}")
            memory_after = read_process_memory(server_pid)

            if not response.ok:
                print("[HTTP] Error response body:")
                print(response.text or "<empty>")
                raise RecoverableRequestError(
                    f"llama-server rejected the payload with HTTP "
                    f"{response.status_code}: {response_error_detail(response)}"
                )

            embeddings, embedding_dimensions = parse_embedding_response(response)

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
                raise RecoverableRequestError(
                    "The battery response does not contain enough values to "
                    "calculate energy consumption."
                )

            energy_joules = energy_milliwatt_hours * 3.6
            average_power_watts = energy_joules / elapsed

            print("[HTTP] Full embedding response:")
            print(response.text)
            print(f"[RESULT] Embedding dimensions: {embedding_dimensions}")
            print(f"[RESULT] Response time: {elapsed:.3f} seconds")
            print(f"[RESULT] Measurement method: {measurement_method}")

            if used_microamp_hours is not None:
                print(
                    f"[RESULT] Battery charge used: "
                    f"{used_microamp_hours:.0f} microamp-hours"
                )

            if mean_voltage_volts is not None:
                print(
                    f"[RESULT] Mean battery voltage: "
                    f"{mean_voltage_volts:.3f} V"
                )

            print(
                f"[RESULT] Energy consumption: "
                f"{energy_milliwatt_hours:.6f} mWh"
            )
            print(f"[RESULT] Energy consumption: {energy_joules:.6f} J")
            print(f"[RESULT] Average power: {average_power_watts:.6f} W")

            if current_before is not None:
                print(
                    f"[RESULT] Battery current before: "
                    f"{current_before / 1_000:.1f} mA"
                )

            if current_after is not None:
                print(
                    f"[RESULT] Battery current after: "
                    f"{current_after / 1_000:.1f} mA"
                )

            print_process_memory(server_pid, memory_before, memory_after)
            show_request_server_logs()

            results.append(
                {
                    "request": request_name,
                    "response_time_seconds": elapsed,
                    "measurement_method": measurement_method,
                    "energy_milliwatt_hours": energy_milliwatt_hours,
                    "energy_joules": energy_joules,
                    "average_power_watts": average_power_watts,
                }
            )
        except RecoverableRequestError as error:
            failure = " ".join(str(error).split())
            print(f"[REQUEST SKIPPED] {request_name}: {failure}")
            print(
                f"[RUN INVALID] {request_name} failed; exclude this entire "
                "run from analysis and skip its remaining requests."
            )
            show_request_server_logs()
            request_failures.append(
                {
                    "request": request_name,
                    "error": failure,
                }
            )
            break
        finally:
            if server_pid is not None:
                stop_llama_servers(
                    expected_pid=server_pid,
                    description="Post-request shutdown",
                )
                server_pid = None
finally:
    print("\n=== Final cleanup ===")
    print("Shutting down llama-server ...")

    try:
        stop_llama_servers(
            expected_pid=server_pid,
            description="Final llama-server shutdown",
        )
        server_pid = None
    finally:
        ssh.close()
        print("SSH connection closed.")

print("\n=== Summary ===")

for result in results:
    print(
        f"{result['request']}: "
        f"{result['response_time_seconds']:.3f} s, "
        f"{result['energy_milliwatt_hours']:.6f} mWh, "
        f"{result['energy_joules']:.6f} J, "
        f"{result['average_power_watts']:.6f} W, "
        f"{result['measurement_method']}"
    )

for failure in request_failures:
    print(f"{failure['request']}: SKIPPED - {failure['error']}")

print(
    f"Completed requests: {len(results)}/{len(requests_to_run)}; "
    f"skipped recoverable request failures: {len(request_failures)}"
)
