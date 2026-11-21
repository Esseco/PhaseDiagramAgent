"""Synthetic, isolated benchmark. Run with py1; never binds a real project."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
import tracemalloc

import psutil


def synthetic_state():
    return {
        "active_model_version": "m1",
        "tasks": [
            {"task_id": f"t{i}", "status": "completed", "stage": "deep_search"} for i in range(120)
        ],
        "event_history": [{"record_id": f"e{i}", "note": str(i) + "x" * 4096} for i in range(120)],
        "memory_candidates": [
            {
                "candidate_id": f"old{i}",
                "status": "candidate",
                "facts": {"measurements": list(range(400))},
            }
            for i in range(600)
        ],
        "pending_execution_policies": {},
    }


def worker(case, baseline=False):
    # All imports, setup and every database/project live outside measurement.
    from phase_agent.runtime.runtime_composition import _read_runtime_state
    from phase_agent.runtime.turn_process import record_turn
    from phase_agent.runtime.context_overview import project_overview
    from phase_agent.persistence.state.state_view import read_state_view
    from phase_agent.persistence.models.mark_model_dependent_results_stale import (
        mark_model_dependent_results_stale,
    )
    from phase_agent.tools.step_runner.file_protocol import write_json
    from phase_agent.tools.workflows.run_tool_step import run_tool_step
    from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
    from phase_agent.persistence.compact_checkpoints import CompactSqliteSaver
    from tests.test_event_loop import _session
    from tests.test_compact_checkpoints import checkpoint

    if baseline:
        # Reproduce the pre-change functions, without reverting a shared worktree
        # or keeping a source/project copy. Only the changed paths are replaced.
        from phase_agent.runtime import runtime_composition as composition
        from phase_agent.tools.state import state_manager
        from phase_agent.configuration.schema import state_schema_migrations as migrations
        from copy import deepcopy
        import hashlib

        def old_loader(path, label):
            value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError(label)
            return value

        composition._load_json_object = old_loader
        collect = state_manager.collect_memory_candidates
        state_manager.collect_memory_candidates = lambda state, **_: collect(state)
        migrate = migrations.migrate_state_snapshot
        migrations.migrate_state_snapshot = lambda snapshot, **_: migrate(snapshot)

        def old_fingerprint(state):
            payload = {
                key: value
                for key, value in state.items()
                if key not in {"current_state_snapshot", "state_snapshots", "state_snapshot_index"}
            }
            memory = deepcopy(payload.get("decision_memory") or {})
            memory.pop("short_term", None)
            if "decision_memory" in payload:
                payload["decision_memory"] = memory
            return hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]

        state_manager._state_fingerprint = old_fingerprint

    with tempfile.TemporaryDirectory(prefix="phase-mem-") as directory:
        root = Path(directory)
        path = root / "workflow_state/state.json"
        path.parent.mkdir()
        source = synthetic_state()
        path.write_text(json.dumps(source), encoding="utf-8")
        runtime = root / "agent_runtime.json"
        runtime.write_text(
            json.dumps({"state_path": "workflow_state/state.json"}), encoding="utf-8"
        )
        registry = create_tool_registry({"check_convergence": lambda **_: {"status": "completed"}})

        def decide():
            result = run_tool_step(
                source,
                _session(),
                registry=registry,
                execute=False,
                agent_client=lambda _: {
                    "tool": "check_convergence",
                    "parameters": {},
                    "budget": 0,
                    "reason": "benchmark",
                },
            )
            assert result["status"] == "planned_only", result["status"]

        if case == "cold_start":

            def operation():
                with record_turn(runtime, "benchmark", "synthetic"):
                    value, _ = _read_runtime_state(path)
                    assert len(value["tasks"]) == 120
        elif case == "repeat_query":
            project_overview(path, None, None)

            def operation():
                assert project_overview(path, None, None)["cache_status"] == "hit"
        elif case == "state_view_query":
            write_json(path, source)
            if baseline:

                def operation():
                    assert json.loads(path.read_text(encoding="utf-8"))["tasks"]
            else:

                def operation():
                    assert read_state_view(path)["status"]["task_counts"]
        elif case == "formal_decision":
            operation = decide
        elif case == "activation_copy_on_write":
            activation_state = {
                "active_model_version": "m1",
                "tasks": source["tasks"],
                "phase_diagrams": [{"id": "h", "model_version": "m1", "validity": "valid"}],
                "model_transitions": [],
            }
            operation = lambda: mark_model_dependent_results_stale(
                activation_state,
                old_model_version="m1",
                new_model_version="m2",
                copy_state=baseline,
            )
        else:
            db = str(root / "checkpoints.sqlite")
            value = checkpoint(0)
            value["channel_values"] = {
                "business_frame": {"loaded_state": source, "feedback": {"state": dict(source)}}
            }
            config = {"configurable": {"thread_id": "synthetic", "checkpoint_ns": ""}}
            with CompactSqliteSaver.from_conn_string(db, maintain_history=False) as saver:
                config = saver.put(config, value, {"source": "loop"}, {})
                saver.put_writes(
                    config,
                    [("business_frame", value["channel_values"]["business_frame"])],
                    "pending",
                )

            def operation():
                with CompactSqliteSaver.from_conn_string(db, maintain_history=False) as saver:
                    loaded = saver.get_tuple(config)
                    assert loaded.checkpoint == value
                    assert loaded.pending_writes

        samples = []
        for _ in range(5):
            gc.collect()
            started = perf_counter()
            operation()
            samples.append(perf_counter() - started)
        gc.collect()
        tracemalloc.start()
        operation()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory = psutil.Process().memory_info()
        sizes = {k: len(json.dumps(v).encode()) for k, v in source.items()}
        return {
            "seconds_median": statistics.median(samples),
            "seconds_samples": samples,
            "python_peak_bytes": peak,
            "process_peak_bytes": getattr(memory, "peak_wset", memory.rss),
            "state_bytes": path.stat().st_size,
            "field_bytes": sizes,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker",
        choices=[
            "cold_start",
            "repeat_query",
            "state_view_query",
            "formal_decision",
            "activation_copy_on_write",
            "checkpoint_restore",
        ],
    )
    parser.add_argument("--output")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.baseline)))
        return
    result = {
        "python": sys.version,
        "fixture": "120 tasks; 120 x 4KiB events; 600 x 400-number memory candidates",
        "baseline_reconstructed": args.baseline,
        "cases": {},
    }
    for case in (
        "cold_start",
        "repeat_query",
        "state_view_query",
        "formal_decision",
        "activation_copy_on_write",
        "checkpoint_restore",
    ):
        runs = []
        for _ in range(3):
            command = [sys.executable, "-m", "tests.benchmark_state_memory", "--worker", case]
            if args.baseline:
                command.append("--baseline")
            run = subprocess.run(command, capture_output=True, text=True)
            if run.returncode:
                raise RuntimeError(run.stderr)

            runs.append(json.loads(run.stdout))
        result["cases"][case] = {
            "runs": runs,
            **{
                key: statistics.median(r[key] for r in runs)
                for key in ("seconds_median", "python_peak_bytes", "process_peak_bytes")
            },
        }
    encoded = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
