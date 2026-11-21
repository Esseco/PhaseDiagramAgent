"""Restore a verified previous publication and atomically record quarantine."""

from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from filelock import FileLock


def quarantine_refresh_publication(state, state_path, scope_id):
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes
    from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory
    from phase_agent.analysis.feedback.versioned_products import (
        _replace_latest,
        validate_bundle,
        write_json,
    )
    from phase_agent.evidence.extrapolation_recovery import quarantine_refresh
    from phase_agent.evidence.model_epoch import model_epoch
    from phase_agent.tools.workflows.lifecycle_support import _save_runner_state

    updated, receipt = quarantine_refresh(state, scope_id)
    root = Path(state["analysis_output_directory"]).resolve()
    if root.name != "analysis_outputs" or root.parent != Path(state_path).resolve().parent.parent:
        raise ValueError("recovery_publication_root_invalid")
    epoch = model_epoch(state, receipt["baseline_model_version"])
    if not epoch:
        raise ValueError("recovery_baseline_epoch_unknown")
    invalid_epoch = model_epoch(state, receipt["invalid_model_version"])
    if not invalid_epoch:
        raise ValueError("recovery_invalid_epoch_unknown")
    baseline = root / epoch / "phase_diagram" / "latest"
    target = root / "latest" / "phase_diagram"
    status_path = root / "latest" / "phase_diagram_status.json"
    invalid_status = root / invalid_epoch / "phase_diagram" / "latest_status.json"
    invalid_notice = root / invalid_epoch / "phase_diagram" / "INVALIDATED.md"
    if receipt.get("policy_version") == "epoch_refresh_v1":
        from hashlib import sha256

        identity = sha256(str(receipt["invalid_model_version"]).encode()).hexdigest()[:10]
        label = f"extrapolation_round_{receipt.get('extrapolation_round', 1):02d}_{identity}"
        invalid_notice = invalid_notice.with_name(label + "_INVALIDATED.md")
    lock_root = analysis_metadata_directory(root)
    lock_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_root / "publication.lock"), timeout=30):
        base_manifest = validate_bundle(baseline)
        identity = base_manifest.get("identity") or {}
        if (
            identity.get("model_version") != receipt["baseline_model_version"]
            or identity.get("hull_version") != receipt["baseline_hull_version"]
        ):
            raise ValueError("recovery_baseline_publication_mismatch")
        prior_manifest = validate_bundle(target)
        prior_identity = prior_manifest.get("identity") or {}
        if not (prior_manifest == base_manifest or
                prior_identity.get("model_version") == receipt["invalid_model_version"]):
            raise ValueError("recovery_latest_publication_mismatch")
        if receipt["status"] == "already_quarantined" and prior_manifest == base_manifest:
            write_json(
                invalid_status,
                {
                    "status": "invalidated",
                    "epoch": invalid_epoch,
                    "model_version": receipt["invalid_model_version"],
                    "hull_version": receipt["invalid_hull_version"],
                    "reason": "大量外推失败；保留历史文件，最新有效相图已回退。",
                },
            )
            _publish_bytes(
                invalid_notice,
                (
                    f"# 相图已作废\n\n该轮凸包 `{receipt['invalid_hull_version']}` 因外推失败作废；原文件保留。"
                    f"当前有效基线为 `{receipt['baseline_hull_version']}`。\n"
                ).encode("utf8"),
            )
            return receipt
        old_status = status_path.read_bytes() if status_path.is_file() else None
        old_invalid_status = invalid_status.read_bytes() if invalid_status.is_file() else None
        old_notice = invalid_notice.read_bytes() if invalid_notice.is_file() else None
        with TemporaryDirectory(prefix=".quarantine-", dir=root / "latest") as temporary:
            previous = Path(temporary) / "previous"
            shutil.copytree(target, previous)
            validate_bundle(previous)
            _replace_latest(baseline, target)
            try:
                write_json(
                    status_path,
                    {
                        "status": "completed",
                        "epoch": epoch,
                        "model_version": receipt["baseline_model_version"],
                        "version": base_manifest["version"],
                        "directory": str(baseline),
                        "latest_directory": str(target),
                        "reason": "本轮外推失败，已保留历史并恢复上一有效相图。",
                        "invalid_hull_version": receipt["invalid_hull_version"],
                    },
                )
                write_json(
                    invalid_status,
                    {
                        "status": "invalidated",
                        "epoch": invalid_epoch,
                        "model_version": receipt["invalid_model_version"],
                        "hull_version": receipt["invalid_hull_version"],
                        "reason": "大量外推失败；保留历史文件，最新有效相图已回退。",
                    },
                )
                _publish_bytes(
                    invalid_notice,
                    (
                        f"# 相图已作废\n\n该轮凸包 `{receipt['invalid_hull_version']}` 因外推失败作废；原文件保留。"
                        f"当前有效基线为 `{receipt['baseline_hull_version']}`。\n"
                    ).encode("utf8"),
                )
                _save_runner_state(updated, state_path)
            except BaseException:
                _replace_latest(previous, target)
                if old_status is None:
                    status_path.unlink(missing_ok=True)
                else:
                    _publish_bytes(status_path, old_status)
                for path, previous_data in (
                    (invalid_status, old_invalid_status),
                    (invalid_notice, old_notice),
                ):
                    if previous_data is None:
                        path.unlink(missing_ok=True)
                    else:
                        _publish_bytes(path, previous_data)
                raise
    return receipt
