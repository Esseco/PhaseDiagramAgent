"""Own one control service and a loopback-only Studio development server."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import socket
from urllib.request import ProxyHandler, build_opener


def preflight(config_path, port, control_port=8765):
    path = Path(config_path).resolve()
    if not path.is_file():
        raise ValueError(f"Runtime configuration does not exist: {path}")
    for number in (control_port, port):
        with socket.socket() as listener:
            try:
                listener.bind(("127.0.0.1", number))
            except OSError:
                raise RuntimeError(
                    f"Port {number} is occupied; close the previous Agent before starting"
                ) from None
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
    if (
        not all(1 <= p <= 65535 for p in (args.port, args.control_port))
        or args.port == args.control_port
    ):
        parser.error("Studio 和控制端口必须有效且不同")
    from phase_agent.runtime.project_service import project_lock

    with project_lock(args.runtime_config):
        run_service(args)


def run_service(args):
    from phase_agent.runtime.project_service import prepare_studio_config, write_service
    from phase_agent.runtime.local_project_launcher import validate_project_paths
    import psutil

    config_path, cli = preflight(args.runtime_config, args.port, args.control_port)
    validate_project_paths(config_path)
    graph_config = prepare_studio_config(config_path)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["LANGSMITH_TRACING"] = "false"
    env["PHASE_AGENT_RUNTIME_CONFIG"] = str(config_path)
    env["PHASE_CONTROL_PORT"] = str(args.control_port)
    env["PHASE_STUDIO_PORT"] = str(args.port)
    env["PHASE_STUDIO_RECEIPT_PATH"] = str(
        Path(args.runtime_config).resolve().parent
        / "workflow_state"
        / "studio_gateway"
        / "gateway.json"
    )
    identity = {
        "pid": os.getpid(),
        "process_started": psutil.Process().create_time(),
        "runtime_config": str(config_path),
        "project_name": config_path.parent.name,
        "control_port": args.control_port,
        "studio_port": args.port,
    }
    child = None
    failure = None
    child_log = config_path.parent / "logs" / "studio_server.log"
    child_log.parent.mkdir(parents=True, exist_ok=True)
    write_service(config_path, {**identity, "status": "starting"})
    try:
        stream = child_log.open("a", encoding="utf-8")
        stream.write(f"\nStudio start owner={os.getpid()} port={args.port}\n")
        stream.flush()
        try:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "langgraph_cli",
                    "dev",
                    "--config",
                    str(graph_config),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                    "--no-browser",
                    "--no-reload",
                ],
                env=env,
                cwd=graph_config.parent,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        finally:
            stream.close()
        import psutil

        deadline = time.monotonic() + 90
        ready = False
        while child.poll() is None:
            if args.parent_pid and not psutil.pid_exists(args.parent_pid):
                break
            if not ready:
                try:
                    with build_opener(ProxyHandler({})).open(
                        f"http://127.0.0.1:{args.port}/info", timeout=0.5
                    ) as response:
                        ready = response.status == 200
                    if ready:
                        with build_opener(ProxyHandler({})).open(
                            f"http://127.0.0.1:{args.control_port}/health", timeout=0.5
                        ) as response:
                            ready = response.status == 200
                except OSError:
                    ready = False
                if ready:
                    write_service(
                        config_path,
                        {
                            **identity,
                            "status": "ready",
                            "child_pid": child.pid,
                            "child_log": str(child_log),
                        },
                    )
                    print(f"READY: Studio API http://127.0.0.1:{args.port}", flush=True)
                    print(
                        f"Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:{args.port}",
                        flush=True,
                    )
                    print(
                        "Graph: phase_chat; keep this terminal open. Ctrl+C stops both services.",
                        flush=True,
                    )
                elif time.monotonic() > deadline:
                    raise RuntimeError(
                        "Studio API did not become ready within 90 seconds; inspect terminal errors"
                    )
            time.sleep(0.5)
        if child.poll() is not None:
            raise RuntimeError(f"Studio 子进程退出（退出码 {child.poll()}）；请查看 {child_log}")
    except KeyboardInterrupt:
        print("Studio stopped.", flush=True)
    except Exception as error:
        failure = {
            "error_type": type(error).__name__,
            "error": str(error),
            "child_exit_code": child.poll() if child else None,
            "child_log": str(child_log),
        }
        raise
    finally:
        write_service(config_path, {**identity, "status": "stopping", **(failure or {})})
        from phase_agent.runtime.studio_run_lifecycle import interrupt_studio_runs

        try:
            if child and child.poll() is None:
                interrupt_studio_runs(args.port)
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        finally:
            write_service(
                config_path,
                {**identity, "status": "failed" if failure else "stopped", **(failure or {})},
            )


if __name__ == "__main__":
    main()
