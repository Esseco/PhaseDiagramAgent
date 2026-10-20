"""A returned-but-unregistered package must not be advertised as a new upload."""

from copy import deepcopy
import json
from unittest.mock import patch

from phase_agent.runtime.local_return_status import pending_dft_return_facts
from phase_agent.runtime.chat_state_presentation import brief_chat_state
from phase_agent.runtime.waiting_files_presentation import waiting_file_lines


def state_for(folder):
    return {
        "tasks": [{"task_id": "d", "stage": "dft_single_point", "status": "pending"}],
        "slurm_batches": [{"task_ids": ["d"], "handoff": {"return_directory": str(folder)}}],
    }


def test_presence_remains_unverified_and_does_not_mutate_state(tmp_path):
    # Even a malformed package is present, pending validation; status must not
    # claim scientific success or ask the user to resubmit the original inputs.
    (tmp_path / "recovered.results.zip").write_bytes(b"not yet validated")
    state = state_for(tmp_path)
    saved = deepcopy(state)
    with patch("phase_agent.tools.remote.legacy_recovery.recovery_handoffs", return_value=[]):
        facts = pending_dft_return_facts(state)
        assert facts[0]["status"] == "awaiting_local_validation"
        assert "待本地校验" in brief_chat_state(state)
        lines = "\n".join(waiting_file_lines(state))
        assert "回传包已到" in lines and "无需重新上传" in lines
        assert "sbatch" not in lines
    assert state == saved


def test_diagnostic_only_empty_or_unregistered_zip_is_not_scientific_return(tmp_path):
    state = state_for(tmp_path)
    (tmp_path / "diagnostics.zip").write_bytes(b"diagnostic")
    (tmp_path / "empty.results.zip").touch()
    other = tmp_path / "unregistered"
    other.mkdir()
    (other / "recovered.results.zip").write_bytes(b"foreign")
    with patch("phase_agent.tools.remote.legacy_recovery.recovery_handoffs", return_value=[]):
        assert pending_dft_return_facts(state) == []
        state["tasks"][0]["status"] = "completed"
        assert pending_dft_return_facts(state) == []


def test_completed_results_do_not_keep_arrival_banner(tmp_path):
    state = state_for(tmp_path)
    state["tasks"][0]["status"] = "completed"
    (tmp_path / "recovered.results.zip").write_bytes(b"existing")
    assert pending_dft_return_facts(state) == []


def test_other_pending_tasks_keep_full_registered_identity_during_discovery(tmp_path):
    state = state_for(tmp_path)
    state["tasks"].append({"task_id": "m", "stage": "deep_search", "status": "pending"})
    (tmp_path / "recovered.results.zip").write_bytes(b"pending validation")

    def deliveries(current):
        assert any(row["task_id"] == "d" for row in current["tasks"])
        return [{"task_ids": ["d"], "return_directory": str(tmp_path)}]

    with patch(
        "phase_agent.tools.remote.legacy_recovery.recovery_handoffs", side_effect=deliveries
    ):
        assert "回传包已到" in "\n".join(waiting_file_lines(state))
