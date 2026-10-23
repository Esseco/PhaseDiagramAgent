"""Run the recovery release gate with fake backends; never targets live projects."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

TARGETS = [
    "tests/test_selection_reason_reports.py",
    "tests/test_dft_environment_admission.py",
    "tests/test_project_registry_recovery.py",
    "tests/test_new_project_run.py",
    "tests/test_preparation_scope.py",
    "tests/test_relax_input_recovery.py",
    "tests/test_task_wait_cancellation.py",
    "tests/test_upload_task_boundaries.py",
    "tests/test_dialogue_resilience.py",
    "tests/test_legacy_submission_index.py",
    "tests/test_report_inventory.py",
    "tests/test_report_state_reuse.py",
    "tests/test_concrete_interaction_reply.py",
    "tests/test_interaction_efficiency.py",
    "tests/test_compact_action_history.py",
    "tests/test_approval_history_performance.py",
    "tests/test_execution_consistency.py",
    "tests/test_input_admission.py",
    "tests/test_environment_preflight.py",
    "tests/test_project_scaffold_contract.py",
    "tests/test_local_project_launcher.py",
    "tests/test_project_dashboard.py",
    "tests/test_relax_mc_compact_handoff.py",
    "tests/test_remote_input_contract.py",
    "tests/test_graph_architecture_contract.py",
    "tests/test_domain_memory.py",
    "tests/test_decision_memory.py",
    "tests/test_memory_experience.py",
    "tests/test_memory_action_scope.py",
    "tests/test_memory_token_management.py",
    "tests/test_interaction_context.py",
    "tests/test_saved_work_semantics.py",
    "tests/test_conversation_contract.py",
    "tests/test_interaction_continuity.py",
    "tests/test_repeatable_runtime.py",
    "tests/test_runtime_doctor.py",
    "tests/test_dialogue_memory_resilience.py",
    "tests/test_sqlite_paths.py",
    "tests/test_startup_reliability.py",
    "tests/test_scientific_scope.py",
    "tests/test_interaction_markdown.py",
    "tests/test_return_failure_evidence.py",
    "tests/test_training_remote_manifest.py",
    "tests/test_remote_model_references.py",
    "tests/test_result_contract_recovery.py",
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
    parser.add_argument(
        "--basetemp", help="Short writable parent for this run; contents are preserved"
    )
    parser.add_argument("--junitxml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not (root / "phase_agent").is_dir():
        raise ValueError("Run from an installed source checkout")
    parent = Path(args.basetemp).resolve() if args.basetemp else Path(tempfile.gettempdir()) / "pdt"
    parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt" and not str(parent).startswith("\\\\?\\"):
        # TemporaryDirectory cleanup must retain extended paths for long-return fixtures.
        name = str(parent)
        parent = Path("\\\\?\\UNC\\" + name[2:] if name.startswith("\\\\") else "\\\\?\\" + name)
    with tempfile.TemporaryDirectory(prefix="", dir=parent) as directory:
        temporary = Path(directory)
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
            "--basetemp=" + str(temporary / "t"),
        ]
        if args.junitxml:
            command.append("--junitxml=" + str(Path(args.junitxml).resolve()))
        return subprocess.run(command, cwd=root, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
