"""Agent transaction for user-approved independent unknown-GS DFT scope."""

from pathlib import Path
from filelock import FileLock
from phase_agent.persistence.state.read_versions import read_identity
from phase_agent.tools.step_runner.file_protocol import read_json
from phase_agent.tools.workflows.lifecycle_support import _save_runner_state
from phase_agent.evidence.unknown_gs_dft import verify_unknown_gs_frames


def authorize(
    control,
    *,
    candidate_frame_sha256,
    expected_model,
    expected_config,
    expected_hull,
    user_instruction,
):
    if not isinstance(user_instruction, str) or not user_instruction.strip():
        raise ValueError("explicit_unknown_gs_user_instruction_required")
    with (
        control.chat_handler.lock,
        FileLock(
            str(Path(control.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
            timeout=10,
        ),
    ):
        version = read_identity(control.state_path, strong=True)
        state = read_json(control.state_path, {}) or {}
        if state.get("pending_execution_policies"):
            raise ValueError("unknown_gs_pending_execution")
        if (
            state.get("active_model_version") != expected_model
            or state.get("confirmed_config_version") != expected_config
        ):
            raise ValueError("unknown_gs_context_changed")
        diagram = verify_unknown_gs_frames(state, candidate_frame_sha256)
        if diagram.get("version") != expected_hull:
            raise ValueError("unknown_gs_hull_changed")
        record = {
            "status": "approved_for_preparation",
            "candidate_frame_sha256": dict(candidate_frame_sha256),
            "model_version": expected_model,
            "config_version": expected_config,
            "hull_version": expected_hull,
            "user_instruction": user_instruction.strip(),
            "sampling_scope": "independent_unknown_monitor_gs_single_point",
            "submit_authorized": False,
        }
        old = state.get("unknown_gs_dft_authorization")
        if old and old != record:
            raise ValueError("unknown_gs_different_authorization_exists")
        if not old:
            state["unknown_gs_dft_authorization"] = record
            _save_runner_state(state, control.state_path, expected_identity=version)
        elif read_identity(control.state_path, strong=True) != version:
            raise ValueError("stale_state_file_version")
        control._codex_context_cache = None
        return {
            "status": "authorized" if not old else "already_authorized",
            "candidate_count": len(candidate_frame_sha256),
            "submit_authorized": False,
        }
