"""Small change/fallback diagnostics; large unchanged-state timing is benchmark_native_team."""

import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch
import psutil
from phase_agent.runtime.local_agent_control import LocalAgentControl
from phase_agent.runtime.turn_process import capture_read_metrics
from phase_agent.tools.step_runner.file_protocol import write_json
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager


def run():
    results = []
    process = psutil.Process()
    with TemporaryDirectory(prefix="read-cases-") as directory:
        root = Path(directory)
        path = root / "state.json"
        ledger = root / "ledger.json"
        config = root / "session.json"
        source = {
            "active_model_version": "m1",
            "tasks": [
                {"task_id": str(i), "status": "completed", "stage": "relax_and_feature"}
                for i in range(32)
            ],
            "event_history": [{"note": "x" * 1024} for _ in range(2048)],
        }
        write_json(path, source)
        manager = PhaseDataManager(
            {"P": ["O3"], "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]}, "TM_ratio": {"Fe": 1}}
        )
        manager.data["branches"]["synthetic"] = {
            "branch_id": "synthetic",
            "parameters": {"P": "O3"},
        }
        manager.save(ledger)
        session = {"status": "draft", "config": {"agent": {"allowed_tools": []}}}
        config.write_text(json.dumps(session))
        control = LocalAgentControl(
            SimpleNamespace(
                state_path=path,
                lock=RLock(),
                config_session_path=config,
                workflow_kwargs={
                    "manager": manager,
                    "ledger_path": str(ledger),
                    "config_session": session,
                },
            )
        )
        original_open = Path.open
        active = None

        class Stream:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self.stream.__exit__(*exc)

            def __getattr__(self, key):
                return getattr(self.stream, key)

            def read(self, *a, **k):
                value = self.stream.read(*a, **k)
                if active is not None:
                    active["read_bytes"] += len(
                        value.encode("utf8") if isinstance(value, str) else value
                    )
                return value

        def opened(p, *a, **k):
            stream = original_open(p, *a, **k)
            if active is not None and str(p).startswith(str(root)) and p.suffix == ".json":
                active["opens"] += 1
                return Stream(stream)
            return stream

        def measured(label, callback):
            nonlocal active
            gc.collect()
            before = process.memory_info().rss
            io = {"opens": 0, "read_bytes": 0}
            active = io
            started = perf_counter()
            try:
                with capture_read_metrics() as metrics:
                    result = callback()
                encoded = json.dumps(result, default=str).encode()
                elapsed = perf_counter() - started
            finally:
                active = None
            del result, encoded
            gc.collect()
            results.append(
                {
                    "operation": label,
                    "seconds": elapsed,
                    "io": io,
                    "metrics": metrics,
                    "rss_before": before,
                    "rss_end": process.memory_info().rss,
                }
            )

        with patch.object(Path, "open", opened):
            measured("cold_context", control.codex_context)
            measured("unchanged_context", control.codex_context)
            source["current_status"] = "single_field_changed"
            write_json(path, source)
            measured("single_field_changed", control.codex_context)
            config.write_text(json.dumps({**session, "note": "changed"}))
            measured("configuration_changed", control.codex_context)
            data = json.loads(ledger.read_text())
            data["note"] = "external-replacement"
            ledger.write_text(json.dumps(data))
            measured("ledger_changed", control.codex_context)
            control_file = path.with_name("state.json.control.json")
            control_file.write_text(
                json.dumps(
                    {
                        "state_path": str(path),
                        "revision": "r1",
                        "binding": {"active_model_version": "m1"},
                        "patch": {},
                    }
                )
            )
            measured("control_changed", control.codex_context)
            # Restore a normally written state for disposable index failure cases.
            control_file.unlink()
            write_json(path, source)
            path.with_name("state.json.index.json").write_text("broken")
            control._codex_context_cache = None
            measured("index_corrupt", control.codex_context)
            path.with_name("state.json.index.json").unlink()
            control._codex_context_cache = None
            measured("index_missing", control.codex_context)
            write_json(path, source)
            measured("overview_first", control.context_overview)
            measured("overview_repeat", control.context_overview)
            for i in range(20):
                measured("repeat_" + str(i + 1), control.codex_context)
        return {
            "state_bytes": path.stat().st_size,
            "results": results,
            "scope": "synthetic; no model or HPC",
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(run(), indent=2), encoding="utf8")
