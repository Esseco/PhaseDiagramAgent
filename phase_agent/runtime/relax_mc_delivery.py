"""Admit a verified return into the registered Relax-MC delivery folder."""

import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from filelock import FileLock

from phase_agent.tools.remote.transfer_bundle import checksum, validate_bundle
from phase_agent.tools.relax_mc_batch.completion_gate import completion_error
from phase_agent.tools.relax_mc_batch.handoff import result_archive
from phase_agent.tools.step_runner.file_protocol import read_json


def deliver(control, *, bundle_id, archive_path, archive_sha256):
    source = Path(archive_path).resolve(strict=True)
    if checksum(source) != archive_sha256:
        raise ValueError("return_delivery_checksum_mismatch")
    validate_bundle(source, extra_result_suffixes=(".csv", ".bin"))
    with zipfile.ZipFile(source) as archive:
        summary = json.loads(archive.read("result_summary.json"))
        returned = json.loads(archive.read("workflow_state.json"))
    error = completion_error(summary, returned.get("tasks") or [])
    if error:
        raise ValueError("return_delivery_incomplete:" + error)
    state_path = Path(control.state_path).resolve()
    with control.chat_handler.lock, FileLock(str(state_path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10):
        state = read_json(state_path, {}) or {}
        record = (state.get("relax_mc_batches") or {}).get(bundle_id)
        if not record or record.get("status") in {"cancelled", "superseded"}:
            raise ValueError("return_delivery_unknown_or_cancelled")
        for key in ("bundle_id", "plan_checksum", "model_version", "config_version"):
            expected = bundle_id if key == "bundle_id" else record.get(key)
            if not expected or summary.get(key) != expected:
                raise ValueError("return_delivery_binding_mismatch:" + key)
        if state.get("active_model_version") != record.get("model_version"):
            raise ValueError("return_delivery_inactive_model")
        target = result_archive(record).resolve()
        root = state_path.parent.parent
        if not target.is_relative_to(root) or target.name != "results.zip" or not target.parent.is_dir():
            raise ValueError("return_delivery_invalid_target")
        if target.exists():
            if checksum(target) != archive_sha256:
                raise ValueError("return_delivery_conflicting_target")
            return {"status": "already_present", "archive": str(target), "sha256": archive_sha256}
        fd, name = tempfile.mkstemp(prefix=".results-", suffix=".part", dir=target.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as dst, source.open("rb") as src:
                shutil.copyfileobj(src, dst, 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            if checksum(temporary) != archive_sha256:
                raise ValueError("return_delivery_copy_checksum_mismatch")
            if os.name == "nt":
                os.rename(temporary, target)
            else:
                os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return {"status": "delivered", "archive": str(target), "sha256": archive_sha256}

