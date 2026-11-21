"""Project control boundary for an explicitly authorized, unsubmitted DFT handoff repair."""

from pathlib import Path
import re
from filelock import FileLock

from phase_agent.tools.step_runner.file_protocol import read_json, write_json_if_version
from phase_agent.persistence.state.read_versions import read_identity


def repair(control, record_id, source_sha256, task_count, partition_counts=None):
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("dft_repair_record_id_required")
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("dft_repair_sha256_required")
    if type(task_count) is not int or not 1 <= task_count <= 1000:
        raise ValueError("dft_repair_invalid_task_count")
    from phase_agent.tools.state.execution_receipts import recovery_report
    from phase_agent.tools.remote.dft_handoff_repair import repair_dft_handoff

    path = Path(control.state_path).resolve()
    with (
        control.chat_handler.lock,
        FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
    ):
        version = read_identity(path, strong=True)
        state = read_json(path, {}) or {}
        if state.get("pending_execution_policies") or recovery_report(path, state)["unsettled"]:
            raise ValueError("dft_repair_pending_or_unsettled_action")
        result = repair_dft_handoff(state, path.parent.parent, record_id, source_sha256, task_count, **({"partition_counts": partition_counts} if partition_counts is not None else {}))
        if result["status"] == "prepared":
            state.setdefault("dft_handoff_repairs", []).append(
                {
                    "record_id": record_id,
                    "source_archive_sha256": source_sha256,
                    "task_count": task_count,
                    "partition_counts": partition_counts,
                    "handoff": result["handoff"],
                    "repair_profile": "one_mpi_rank_per_gpu",
                    "submitted": False,
                }
            )
            write_json_if_version(path, state, version)
            control._codex_context_cache = None
        elif read_identity(path, strong=True) != version:
            raise ValueError("stale_state_file_version; reload state before reporting")
        return result
