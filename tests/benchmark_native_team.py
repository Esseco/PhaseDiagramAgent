"""Bounded synthetic profiling of the native Codex path; never opens a real project."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
from time import perf_counter, process_time, sleep
import types

import psutil

MODULES = (
    "phase_agent.persistence.memory.knowledge_record",
    "phase_agent.persistence.memory.verify_evidence_refs",
    "phase_agent.persistence.memory.review_queue",
    "phase_agent.persistence.memory.retrieve_relevant_knowledge",
    "phase_agent.persistence.memory.decision_context",
    "phase_agent.persistence.state.field_index",
    "phase_agent.persistence.state.state_view",
    "phase_agent.runtime.context_overview",
    "phase_agent.runtime.codex.context",
    "phase_agent.runtime.local_agent_control",
    "phase_agent.evidence.decision_evidence",
    "phase_agent.evidence.snapshot",
    "phase_agent.models.transport_tables",
    "phase_agent.agents.strategic.codex_team",
)
ROOT = Path(__file__).resolve().parents[1]


def install_baseline(directory):
    for name in MODULES:
        original = importlib.import_module(name)
        module = types.ModuleType(name)
        module.__package__ = name.rpartition(".")[0]
        module.__file__ = str(Path(directory) / (name + ".py"))
        sys.modules[name] = module
        setattr(sys.modules[module.__package__], name.rpartition(".")[2], module)
        exec(
            compile(Path(module.__file__).read_text(encoding="utf-8"), module.__file__, "exec"),
            module.__dict__,
        )


def fixture(path, mib):
    from phase_agent.tools.step_runner.file_protocol import write_json

    rows = max(4, int(mib * 1024 * 1024 / 4600))
    state = {
        "confirmed_config_version": "c1",
        "active_model_version": "m1",
        "confirmed_config": {"budgets": {"total": 1000}},
        "current_status": "awaiting_review",
        "pending_execution_policies": {},
        "tasks": [
            {
                "task_id": f"t{i}",
                "status": "completed",
                "model_version": "m1",
                "stage": "deep_search",
                "branch_id": f"b{i % 8}",
                "batch_id": "batch-1",
                "config_version": "c1",
                "input_sha256": "a" * 64,
                "diagnostic": {"label": str(i), "observations": [float(j) / 7 for j in range(128)]},
            }
            for i in range(max(4, rows))
        ],
        "event_history": [{"event_id": i, "payload": str(i) + "h" * 4096} for i in range(rows)],
        "action_records": [
            {
                "record_id": f"r{i}",
                "status": "completed",
                "final_action": {"tool": "pause_search", "reason": "review"},
                "execution_result": {"status": "completed"},
            }
            for i in range(max(4, rows // 10))
        ],
        "qbc_candidates": [
            {
                "candidate_id": f"c{i}",
                "model_version": "m1",
                "phase": "O3",
                "ehull": i / 1000,
                "D": i / 100,
                "H": i / 200,
                "limitations": ["same-frame only"],
            }
            for i in range(32)
        ],
    }
    write_json(path, state)


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def worker(args):
    sys.stdin.readline()  # Parent attaches the OS memory limit before any large setup.
    # Pin only this owned worker so heterogeneous CPU cores do not skew paired runs.
    owned_process = psutil.Process()
    owned_process.cpu_affinity([owned_process.cpu_affinity()[0]])
    if args.baseline:
        install_baseline(args.baseline)
    from threading import RLock
    from types import SimpleNamespace
    from phase_agent.runtime.local_agent_control import LocalAgentControl
    from phase_agent.agents.strategic import codex_team as team_module
    from phase_agent.evidence import snapshot as snapshot_module
    from phase_agent.evidence import build_decision_context as decision_module
    from phase_agent.models import transport_tables
    from phase_agent.persistence.state import read_versions
    from phase_agent.runtime import turn_process
    from phase_agent.tools.step_runner import file_protocol

    with TemporaryDirectory(prefix="native-case-", dir=args.scratch) as directory:
        path = Path(directory) / "state.json"
        fixture(path, args.mib)
        handler = SimpleNamespace(
            state_path=path,
            lock=RLock(),
            workflow_kwargs={
                "config_session": {"config": {"agent": {"allowed_tools": ["pause_search"]}}}
            },
        )
        control = LocalAgentControl(handler)
        team = team_module.CodexTeam(control)
        stages = defaultdict(lambda: {"calls": 0, "seconds": 0.0})

        def record(name, seconds):
            stages[name]["calls"] += 1
            stages[name]["seconds"] += seconds

        original_call = turn_process.timed_call

        def timing(name, callback, *a, **kw):
            started = perf_counter()
            try:
                return original_call(name, callback, *a, **kw)
            finally:
                record(name, perf_counter() - started)

        turn_process.timed_call = timing

        def wrap(module, name, label, predicate=None):
            original = getattr(module, name)
            depth = 0

            def measured(*a, **kw):
                nonlocal depth
                if depth:
                    return original(*a, **kw)
                top = predicate is None or predicate(*a, **kw)
                depth += 1
                started = perf_counter()
                try:
                    return original(*a, **kw)
                finally:
                    depth -= 1
                    if top:
                        record(label, perf_counter() - started)

            setattr(module, name, measured)
            return measured

        # Count actual source reads (including streamed verification) without
        # copying their buffers. Fixtures are ASCII; character and byte counts agree.
        original_open = Path.open

        class SourceStream:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *exc):
                return self.stream.__exit__(*exc)

            def __getattr__(self, key):
                return getattr(self.stream, key)

            def read(self, *a, **kw):
                begun = perf_counter()
                data = self.stream.read(*a, **kw)
                record("source_io", perf_counter() - begun)
                if isinstance(data, str):
                    assert data.isascii()
                stages["source_io"]["bytes"] = stages["source_io"].get("bytes", 0) + len(data)
                return data

        def profiled_open(source, *a, **kw):
            stream = original_open(source, *a, **kw)
            if source == path:
                stages["source_io"]["opens"] = stages["source_io"].get("opens", 0) + 1
            return SourceStream(stream) if source == path else stream

        Path.open = profiled_open

        from phase_agent.persistence.state.field_index import IndexedState

        wrap(IndexedState, "_verify_content", "index_full_validation")
        original_get = IndexedState.__getitem__
        original_loads = json.loads
        field_scope = [False]

        def field_get(source, key):
            previous = field_scope[0]
            field_scope[0] = source.path == path
            if source.path == path and key in source.fields and key not in source.cache:
                stages["field:" + key]["declared_bytes"] = source.fields[key][1]
            try:
                return original_get(source, key)
            finally:
                field_scope[0] = previous

        def decoded(*a, **kw):
            begun = perf_counter()
            try:
                return original_loads(*a, **kw)
            finally:
                if field_scope[0]:
                    record("indexed_json_decode", perf_counter() - begun)

        IndexedState.__getitem__ = field_get
        json.loads = decoded

        reader = wrap(
            file_protocol, "read_json", "state_read_total", lambda p, *a, **kw: Path(p) == path
        )
        team_module.read_json = reader
        wrap(
            read_versions,
            "read_identity",
            "strong_version",
            lambda p, *a, **kw: kw.get("strong", False),
        )
        wrap(decision_module, "build_decision_context", "decision_evidence")
        snap = wrap(snapshot_module, "evidence_snapshot", "snapshot")
        team_module.evidence_snapshot = snap
        role = wrap(snapshot_module, "role_context", "role_projection")
        team_module.role_context = role
        wrap(transport_tables, "compact_evidence_tables", "table_compression")
        original_lock = team_module.team_lock

        @contextmanager
        def lock(*a, **kw):
            started = perf_counter()
            with original_lock(*a, **kw):
                entered = perf_counter()
                record("file_lock_wait", entered - started)
                try:
                    yield
                finally:
                    record("file_lock_hold", perf_counter() - entered)

        team_module.team_lock = lock

        class MeasuredLock:
            def __init__(self):
                self.lock = RLock()

            def __enter__(self):
                started = perf_counter()
                self.lock.acquire()
                record("handler_lock_wait", perf_counter() - started)
                self.starts = getattr(self, "starts", [])
                self.starts.append(perf_counter())
                return self

            def __exit__(self, *exc):
                entered = self.starts.pop()
                if not self.starts:
                    record("handler_lock_hold", perf_counter() - entered)
                self.lock.release()

        handler.lock = MeasuredLock()
        process = psutil.Process()
        saved_snapshot = None

        def run(label, callback, *, protocol=False):
            nonlocal saved_snapshot
            gc.collect()
            stages.clear()
            memory = process.memory_info()
            emit(
                {
                    "event": "begin",
                    "operation": label,
                    "cpu_affinity": owned_process.cpu_affinity(),
                    "rss": memory.rss,
                    "private": getattr(memory, "private", 0),
                }
            )
            started = perf_counter()
            cpu_started = process_time()
            with turn_process.capture_read_metrics() as read_metrics:
                result = callback()
            before_serialize = perf_counter()
            wire = json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=not protocol,
                separators=None if protocol else (",", ":"),
                allow_nan=False,
                default=str if protocol else None,
            ).encode()
            record("final_serialization", perf_counter() - before_serialize)
            elapsed = perf_counter() - started
            identity = (result.get("snapshot") or {}).get("snapshot_id")
            if identity:
                if saved_snapshot is not None:
                    assert identity == saved_snapshot
                saved_snapshot = identity
            summary = {
                "event": "end",
                "operation": label,
                "seconds": elapsed,
                "cpu_seconds": process_time() - cpu_started,
                "response_bytes": len(wire),
                "input_bytes": path.stat().st_size,
                "stages": dict(stages),
                "read_metrics": read_metrics,
                "model_calls": 0,
                "client_wait_seconds": None,
            }
            del wire
            del result
            gc.collect()
            memory = process.memory_info()
            summary.update(
                rss_end=memory.rss,
                private_end=getattr(memory, "private", 0),
                kernel_process_peak_rss=getattr(memory, "peak_wset", memory.rss),
            )
            emit(summary)
            return identity

        snap_id = run("start", lambda: team.start(question="review synthetic facts"))
        session = team._load()
        for index in range(args.repeats):
            run(f"read-{index + 1}", lambda: team.read(request_role="reliability"))
        opinion = {
            "role": "exploration",
            "snapshot_id": snap_id,
            "assessment": "Evidence incomplete",
            "recommendation": "defer",
            "options": [],
            "evidence_refs": [],
            "expected_outcome": "unknown",
            "limitations": "synthetic only",
        }
        run("advise", lambda: team.advise(session["team_id"], 1, opinion))
        import inspect

        compact = "compact" in inspect.signature(team.advise).parameters
        run(
            "advise_compact",
            lambda: team.advise(
                session["team_id"], 1, opinion, **({"compact": True} if compact else {})
            ),
        )

        from phase_agent.runtime.codex.mcp import CodexMcpServer

        def framed_read():
            body = team.read(request_role="reliability")
            begun = perf_counter()
            content = json.dumps(body, ensure_ascii=False, default=str)
            record("mcp_content_encoding", perf_counter() - begun)
            return CodexMcpServer._result(1, {"content": [{"type": "text", "text": content}]})

        run("mcp_read", framed_read, protocol=True)


def limit_windows_process(process, mib):
    if os.name != "nt":
        raise RuntimeError("This benchmark requires a Windows OS process memory limit")
    import ctypes as c
    from ctypes import wintypes as w

    class Basic(c.Structure):
        _fields_ = [
            ("per_process", c.c_int64),
            ("per_job", c.c_int64),
            ("flags", w.DWORD),
            ("min_working", c.c_size_t),
            ("max_working", c.c_size_t),
            ("active", w.DWORD),
            ("affinity", c.c_size_t),
            ("priority", w.DWORD),
            ("scheduling", w.DWORD),
        ]

    class IO(c.Structure):
        _fields_ = [
            (name, c.c_uint64)
            for name in (
                "read_ops",
                "write_ops",
                "other_ops",
                "read_bytes",
                "write_bytes",
                "other_bytes",
            )
        ]

    class Extended(c.Structure):
        _fields_ = [
            ("basic", Basic),
            ("io", IO),
            ("process_memory", c.c_size_t),
            ("job_memory", c.c_size_t),
            ("peak_process", c.c_size_t),
            ("peak_job", c.c_size_t),
        ]

    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = w.HANDLE
    kernel.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    handle = kernel.CreateJobObjectW(None, None)
    info = Extended()
    info.basic.flags = 0x100 | 0x2000  # Process commit limit and kill owned worker on close.
    info.process_memory = mib * 1024 * 1024
    if (
        not handle
        or not kernel.SetInformationJobObject(handle, 9, c.byref(info), c.sizeof(info))
        or not kernel.AssignProcessToJobObject(handle, w.HANDLE(int(process._handle)))
    ):
        if handle:
            kernel.CloseHandle(handle)
        raise OSError(c.get_last_error(), "Cannot attach benchmark process memory limit")
    return lambda: kernel.CloseHandle(handle)


def measure(args, scratch, baseline=None):
    all_results = []
    for mib in args.sizes:
        command = [
            sys.executable,
            "-B",
            "-m",
            "tests.benchmark_native_team",
            "--worker",
            "--mib",
            str(mib),
            "--repeats",
            str(args.repeats),
            "--scratch",
            scratch,
        ]
        if baseline:
            command += ["--baseline", baseline]
        child = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        close_limit = None
        current = None
        results = []
        errors = []

        def consume():
            nonlocal current
            for line in child.stdout:
                row = json.loads(line)
                if row["event"] == "begin":
                    current = dict(
                        row, sampled_peak_rss=row["rss"], sampled_peak_private=row["private"]
                    )
                else:
                    row.update(
                        {
                            key: value
                            for key, value in (current or {}).items()
                            if key not in {"event", "operation"}
                        }
                    )
                    results.append(row)
                    current = None

        def consume_errors():
            for line in child.stderr:
                errors.append(line)
                if len(errors) > 12:
                    errors.pop(0)

        threads = [
            threading.Thread(target=consume, daemon=True),
            threading.Thread(target=consume_errors, daemon=True),
        ]
        started = perf_counter()
        failure = None
        try:
            close_limit = limit_windows_process(child, args.limit_mib)
            for thread in threads:
                thread.start()
            child.stdin.write("go\n")
            child.stdin.flush()
            child.stdin.close()
            monitored = psutil.Process(child.pid)
            while child.poll() is None:
                if perf_counter() - started > args.timeout:
                    failure = "owned benchmark worker timed out"
                    child.kill()
                    break
                try:
                    memory = monitored.memory_info()
                    row = current
                    if row is not None:
                        row["sampled_peak_rss"] = max(row["sampled_peak_rss"], memory.rss)
                        row["sampled_peak_private"] = max(
                            row["sampled_peak_private"], getattr(memory, "private", 0)
                        )
                except psutil.NoSuchProcess:
                    pass
                sleep(0.005)
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
            for thread in threads:
                if thread.ident:
                    thread.join(timeout=5)
            if close_limit:
                close_limit()
        row = {
            "mib_target": mib,
            "limit_mib": args.limit_mib,
            "returncode": child.returncode,
            "failure": failure,
            "results": results,
            "stderr_tail": "".join(errors) if child.returncode else "",
        }
        all_results.append(row)
        Path(scratch, "profile-baseline.json" if baseline else "profile-after.json").write_text(
            json.dumps(all_results, indent=2), encoding="utf-8"
        )
        emit(
            {
                "mib_target": mib,
                "returncode": child.returncode,
                "failure": failure,
                "stderr_tail": row["stderr_tail"],
                "baseline": bool(baseline),
                "operations": [
                    {
                        key: r[key]
                        for key in (
                            "operation",
                            "seconds",
                            "input_bytes",
                            "response_bytes",
                            "rss",
                            "sampled_peak_rss",
                            "rss_end",
                            "private",
                            "sampled_peak_private",
                            "private_end",
                            "stages",
                        )
                    }
                    for r in results
                ],
            }
        )
        if child.returncode:
            break
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mib", type=int, default=1)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit-mib", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=150)
    parser.add_argument("--scratch")
    parser.add_argument("--baseline")
    parser.add_argument("--hold-baseline", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    with TemporaryDirectory(prefix="native-prof-", dir=args.scratch) as directory:
        baseline = Path(directory) / "baseline"
        baseline.mkdir()
        hashes = {}
        for name in MODULES:
            source = ROOT / (name.replace(".", "/") + ".py")
            target = baseline / (name + ".py")
            shutil.copyfile(source, target)
            hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        emit({"baseline_hashes": hashes, "temporary_root": directory})
        measure(args, directory, str(baseline))
        if args.hold_baseline:
            emit({"baseline_ready": True})
            while True:
                choice = input().strip()
                if choice == "after":
                    measure(args, directory)
                elif choice == "baseline":
                    measure(args, directory, str(baseline))
                elif choice == "done":
                    break
                else:
                    emit({"expected": "after, baseline or done"})
    emit({"temporary_root_removed": not Path(directory).exists()})


if __name__ == "__main__":
    main()
