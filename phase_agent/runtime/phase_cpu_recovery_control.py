"""Locked registration of a CPU-only refresh recovery package."""

from pathlib import Path, PurePosixPath
import re
from filelock import FileLock

from phase_agent.persistence.state.read_versions import read_identity
from phase_agent.tools.step_runner.file_protocol import read_json, write_json_if_version
from phase_agent.tools.remote.integrity import file_checksum


def prepare(control, source_sha256, spec_sha256, task_count, remote_root):
    """Stage recovery only; scheduler verification and submission remain separate."""
    from phase_agent.tools.remote.phase_cpu_recovery import stage_phase_cpu_recovery
    from phase_agent.tools.state.execution_receipts import recovery_report

    if not all(
        isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x)
        for x in (source_sha256, spec_sha256)
    ):
        raise ValueError("phase_cpu_invalid_digest")
    if type(task_count) is not int or not 1 <= task_count <= 100000:
        raise ValueError("phase_cpu_invalid_count")
    path = Path(control.state_path).resolve()
    project = path.parent.parent
    with (
        control.chat_handler.lock,
        FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
    ):
        version = read_identity(path, strong=True)
        state = read_json(path, {}) or {}
        if state.get("pending_execution_policies") or recovery_report(path, state)["unsettled"]:
            raise ValueError("phase_cpu_pending_or_unsettled_action")
        refresh = state.get("model_refresh") or {}
        ids = refresh.get("task_ids") or []
        if not refresh.get("approved_by"):
            raise ValueError("phase_cpu_approval_missing")
        if (
            refresh.get("status") != "waiting_results"
            or len(ids) != task_count
            or len(set(ids)) != task_count
        ):
            raise ValueError("phase_cpu_refresh_scope_changed")
        wanted = set(ids)
        tasks = [t for t in state.get("tasks") or [] if t.get("task_id") in wanted]
        if len(tasks) != task_count or {t.get("task_id") for t in tasks} != wanted:
            raise ValueError("phase_cpu_task_scope_changed")
        if any(
            t.get("status") != "pending"
            or not t.get("task_checksum")
            or t.get("model_version") != state.get("active_model_version")
            or t.get("config_version") != state.get("confirmed_config_version")
            or t.get("parent_decision_id") != refresh.get("approved_by")
            for t in tasks
        ):
            raise ValueError("phase_cpu_task_binding_changed")
        batches = [
            b
            for b in state.get("slurm_batches") or []
            if wanted.intersection(b.get("task_ids") or [])
        ]
        from collections import Counter

        counts = Counter(tid for b in batches for tid in b.get("task_ids") or [])
        if set(counts) != wanted or any(n != 1 for n in counts.values()):
            raise ValueError("phase_cpu_batch_scope_changed")
        handoff = (batches[0].get("handoff") or {}) if batches else {}
        archive = Path(handoff.get("upload_archive") or "").resolve()
        if not archive.is_relative_to(project / "submissions") or archive.name != "inputs.zip":
            raise ValueError("phase_cpu_archive_outside_project")
        if any((b.get("handoff") or {}).get("upload_archive") != str(archive) for b in batches):
            raise ValueError("phase_cpu_handoffs_differ")
        if file_checksum(archive) != source_sha256:
            raise ValueError("phase_cpu_archive_changed")
        remote = PurePosixPath(remote_root)
        epoch = Path(handoff.get("import_directory") or "").name
        if (
            not remote.is_absolute()
            or ".." in remote.parts
            or remote.name != archive.parent.name
            or remote.parent.name != epoch
        ):
            raise ValueError("phase_cpu_remote_binding_changed")
        history = state.get("phase_cpu_recoveries") or []
        for saved in history:
            if saved.get("source_archive_sha256") == source_sha256:
                if (
                    saved.get("spec_sha256") != spec_sha256
                    or saved.get("task_count") != task_count
                    or saved.get("remote_result_root") != remote_root
                    or file_checksum(saved["upload_archive"]) != saved["upload_archive_sha256"]
                ):
                    raise ValueError("phase_cpu_existing_recovery_changed")
                return {**saved, "status": "reused", "submitted": False}
        destination = archive.parent.parent / (
            archive.parent.name[:-6] + "_PhaseCPU_" + spec_sha256[:12] + "_input"
        )
        result = stage_phase_cpu_recovery(
            archive,
            source_sha256,
            spec_sha256,
            {t["task_id"]: t["task_checksum"] for t in tasks},
            remote_root,
            destination,
        )
        result.update(
            status="prepared",
            submitted=False,
            original_return_directory=handoff.get("return_directory"),
            original_import_directory=handoff.get("import_directory"),
            note="Scheduler status and remote paths must be rechecked before any upload or submission.",
        )
        state.setdefault("phase_cpu_recoveries", []).append(dict(result))
        write_json_if_version(path, state, version)
        control._codex_context_cache = None
        return result
