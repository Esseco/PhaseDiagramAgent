"""Acknowledge a reviewed UI request without replaying any callback or deleting evidence."""

import hashlib
from pathlib import Path
from filelock import FileLock
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


def resolve_operation(control, body):
    from phase_agent.tools.step_runner.build_status_summary import build_status_summary
    from phase_agent.tools.state.execution_receipts import recovery_report

    store = control.operations
    # This endpoint requires direct human attestation, not a model-generated retry decision.
    with control.chat_handler.lock:
        with FileLock(
            str(Path(control.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
            timeout=10,
        ):
            state = read_json(control.state_path, {})
            if (
                build_status_summary(state, config_version=state.get("confirmed_config_version"))[
                    "summary_id"
                ]
                != body["state_version"]
            ):
                raise ValueError("operation_resolution_state_changed")
            if recovery_report(control.state_path, state)["unsettled"]:
                raise ValueError("operation_resolution_requires_execution_reconciliation")
            with FileLock(str(store.directory / ".requests.lock"), timeout=2):
                path = store._path(body["operation_id"])
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != body["record_sha256"]:
                    raise ValueError("operation_resolution_record_changed")
                original = store.get(body["operation_id"])
                if original["status"] == "resolved":
                    return {**original, "idempotent_replay": True}
                if original["status"] not in {"blocked", "unavailable", "interrupted"}:
                    raise ValueError("operation_resolution_requires_inactive_request")
                record = {
                    "operation_id": body["operation_id"],
                    "status": "resolved",
                    "request_fingerprint": original.get("request_fingerprint"),
                    "record_sha256": digest,
                    "message": "已按人工核对记录解除交互阻塞；原请求及科学收据保留，未重放",
                    "attestation": {
                        key: body[key]
                        for key in ("evidence_note", "external_jobs", "output_inventory")
                    },
                    "submitted": False,
                    "replayed": False,
                    "progress_url": store.progress_url(),
                }
                resolution = store.directory / "resolutions" / path.name
                if resolution.is_file():
                    old = resolution.read_bytes()
                    backup = (
                        store.directory
                        / "resolution_backups"
                        / (hashlib.sha256(old).hexdigest() + ".json")
                    )
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_bytes(old)
                    if backup.read_bytes() != old:
                        raise ValueError("operation_resolution_backup_failed")
                write_json(resolution, record)
                return record
