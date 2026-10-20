"""Run the recovery release gate with fake backends; never targets live projects."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

TARGETS = [
    "tests/test_agent_retry_guard.py",
    "tests/test_stability_boundaries.py",
    "tests/test_stability_diagnostics.py",
    "tests/test_stability_new_structures.py",
    "tests/test_operation_resolution.py",
    "tests/test_generation_preflight.py",
    "tests/test_generation_runtime_guards.py",
    "tests/test_generation_failure_lifecycle.py",
    "tests/test_generation_contracts.py",
    "tests/test_interrupted_input_recovery.py",
    "tests/test_runtime_diagnostics.py",
    "tests/test_failure_diagnostics.py",
    "tests/test_execution_recovery_scope.py",
    "tests/test_recovery_closure.py",
    "tests/test_memory_recovery_boundaries.py",
    "tests/test_recovery_service_unavailable.py",
    "tests/test_dft_magnetic_diagnostics.py",
    "tests/test_phase_identification_cache.py",
    "tests/test_tiered_closed_loop.py",
    "tests/test_studio_lifecycle_recovery.py",
    "tests/test_collect_only_recovery.py",
    "tests/test_result_recovery_commit.py",
    "tests/test_ui_operations.py",
    "tests/test_unified_recovery_status.py",
    "tests/test_readonly_presentation.py",
    "tests/test_async_control_http.py",
    "tests/test_http_adapter_boundary.py",
    "tests/test_local_agent_control.py",
    "tests/test_control_ui_contracts.py",
    "tests/test_control_request_contracts.py",
    "tests/test_scientific_progress_stream.py",
    "tests/test_status_presentation.py",
    "tests/test_status_approval_binding.py",
    "tests/test_progress_reply.py",
    "tests/test_femn_scheduler_recovery.py",
    "tests/test_long_path_return.py",
    "tests/test_region_phase_contract.py",
    "tests/test_refresh_history_compaction.py",
    "tests/test_lifecycle_checkpoint_resume.py",
    "tests/test_compact_checkpoints.py",
    "tests/test_checkpoint_reader_compatibility.py",
    "tests/test_model_refresh_workflow.py",
    "tests/test_mlip_relaxation_convergence.py",
    "tests/test_readable_review_json.py",
    "tests/test_branch_generation_artifacts.py",
    "tests/test_dft_selection_artifacts.py",
    "tests/test_compact_review_delivery.py",
    "tests/test_reply_artifact_scope.py",
    "tests/test_adaptive_branch_review.py",
    "tests/test_on_demand_branch.py",
    "tests/test_agent_stall_repairs.py",
    "tests/test_relax_mc_batch.py",
    "tests/integration/test_layered_closed_loop.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basetemp", required=True, help="Short writable test directory")
    parser.add_argument("--junitxml")
    args = parser.parse_args()
    base = Path(args.basetemp).resolve()
    root = Path(__file__).resolve().parents[1]
    if not (root / "phase_agent").is_dir():
        raise ValueError("Run from an installed source checkout")
    temporary = base.parent / (base.name + "-temp")
    temporary.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "TEMP": str(temporary),
        "TMP": str(temporary),
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    command = [
        sys.executable,
        "-m",
        "pytest",
        *TARGETS,
        "-q",
        "-p",
        "no:cacheprovider",
        "--tb=short",
        "--basetemp=" + str(base),
    ]
    if args.junitxml:
        command.append("--junitxml=" + str(Path(args.junitxml).resolve()))
    return subprocess.run(command, cwd=root, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
