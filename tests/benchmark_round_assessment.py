"""Measure the real assessment adapter with synthetic state and saved DFT evidence."""

import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import psutil

from phase_agent.runtime.turn_process import timed_call
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from phase_agent.tools.workflows.lifecycle_recovery import _assess_workflow_round
from phase_agent.analysis.feedback.dft_result_products import record_dft_products
from tests.test_official_dft_products import predict, result


def main():
    process = psutil.Process()
    parent = os.environ.get("PHASE_BENCH_TEMP")
    with TemporaryDirectory(dir=parent, prefix="phase-assess-") as temp:
        root = Path(temp)
        path = root / "state.json"
        state = {
            "active_model_version": "m1",
            "tasks": [],
            "generation_history": [{"synthetic": "x" * 65536, "index": i} for i in range(7200)],
        }
        for i in range(96):
            row = result()
            row["task_id"] = f"T{i}"
            row["structure_id"] = f"S{i}"
            state["tasks"].append({**row, "outputs": {}, "upload_operation_id": f"dft-{i // 24}"})
            record_dft_products(state, row, predict)
        write_json(path, state)
        print(json.dumps({"state_bytes": path.stat().st_size}), flush=True)
        state = read_json(path)
        frame = {
            "feedback": {"state": state},
            "effective_config": {"phase_diagram_directory": str(root / "analysis_outputs")},
            "manager": None,
            "runtime_adapters": {},
            "state_path": path,
        }
        for case in ("cold", "repeat"):
            times = {}

            def capture(name, seconds, **kwargs):
                if name.startswith("round_") or name.startswith("state_"):
                    times[name] = round(seconds, 3)

            with ExitStack() as stack:
                from importlib import import_module

                for name, module in {
                    "post_dft_assessment": "phase_agent.evidence.post_dft_assessment",
                    "refresh_dft_comparisons": "phase_agent.tools.workflows.refresh_dft_comparisons",
                    "export_dft_products": "phase_agent.analysis.feedback.export_dft_products",
                    "record_feedback_cycle": "phase_agent.analysis.convergence.feedback_cycles",
                    "_save_runner_state": "phase_agent.tools.workflows.lifecycle_recovery",
                }.items():
                    target = import_module(module)
                    fn = getattr(target, name)

                    def measure(*args, _fn=fn, _name=name, **kwargs):
                        return timed_call("round_probe_" + _name, _fn, *args, **kwargs)

                    stack.enter_context(patch.object(target, name, measure))
                stack.enter_context(
                    patch("phase_agent.runtime.turn_process.record_timing", capture)
                )
                stack.enter_context(
                    patch("phase_agent.analysis.feedback.physical_layout.organize_physical_layout")
                )
                if "--baseline" in sys.argv:
                    for name, value in (
                        ("report_version", None),
                        ("reusable_metrics", None),
                        ("report_manifest", None),
                    ):
                        stack.enter_context(
                            patch(
                                "phase_agent.analysis.feedback.export_dft_products." + name,
                                return_value=value,
                            )
                        )
                frame = timed_call("round_total", _assess_workflow_round, frame)
            print(
                json.dumps(
                    {
                        "case": case,
                        "seconds": times,
                        "peak_rss_bytes": process.memory_info().peak_wset,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
