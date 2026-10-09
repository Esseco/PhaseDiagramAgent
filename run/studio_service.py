"""Own one control service and a loopback-only Studio development server."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import socket
from urllib.request import urlopen


def preflight(config_path, port):
    path = Path(config_path).resolve()
    if not path.is_file():
        raise ValueError(f"Runtime configuration does not exist: {path}")
    for number in (8765, port):
        with socket.socket() as listener:
            try:
                listener.bind(("127.0.0.1", number))
            except OSError:
                raise RuntimeError(f"Port {number} is occupied; close the previous Agent before starting") from None
    cli = Path(sys.executable).parent / "Scripts" / "langgraph.exe"
    if not cli.is_file():
        raise RuntimeError("Install requirements.txt in py1 before starting Studio")
    return path, cli


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--control-port", type=int, default=8765)
    parser.add_argument("--port", type=int, default=2024)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args(argv)
    if args.control_port != 8765 or not 1 <= args.port <= 65535 or args.port == args.control_port:
        parser.error("Studio gateway currently requires control port 8765 and a different valid Studio port")
    config_path, cli = preflight(args.runtime_config, args.port)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["LANGSMITH_TRACING"] = "false"
    env["PHASE_AGENT_RUNTIME_CONFIG"] = str(config_path)
    env["PHASE_STUDIO_RECEIPT_PATH"] = str(Path(args.runtime_config).resolve().parent /
                                           "workflow_state" / "studio_gateway" / "gateway.json")
    child = None
    try:
        child = subprocess.Popen([str(cli), "dev", "--host", "127.0.0.1", "--port", str(args.port),
                                  "--no-browser", "--no-reload"], env=env,
                                 cwd=Path(__file__).resolve().parent.parent,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        import psutil
        deadline = time.monotonic() + 90
        ready = False
        while child.poll() is None:
            if args.parent_pid and not psutil.pid_exists(args.parent_pid):
                break
            if not ready:
                try:
                    with urlopen(f"http://127.0.0.1:{args.port}/info", timeout=.5) as response:
                        ready = response.status == 200
                except OSError:
                    pass
                if ready:
                    print(f"READY: Studio API http://127.0.0.1:{args.port}", flush=True)
                    print(f"Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:{args.port}", flush=True)
                    print("Graph: phase_chat; keep this terminal open. Ctrl+C stops both services.", flush=True)
                elif time.monotonic() > deadline:
                    raise RuntimeError("Studio API did not become ready within 90 seconds; inspect terminal errors")
            time.sleep(.5)
        if child.poll() not in {None, 0}:
            raise RuntimeError("Studio development server exited; inspect agent_server.log")
    except KeyboardInterrupt:
        print("Studio stopped.", flush=True)
    finally:
        if child and child.poll() is None:
            child.terminate()
            child.wait(timeout=15)


if __name__ == "__main__":
    main()
