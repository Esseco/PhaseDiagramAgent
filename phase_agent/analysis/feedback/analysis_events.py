"""Publish user-facing analysis events from verified, already registered evidence."""

from contextvars import ContextVar
from hashlib import sha256
import json
from pathlib import Path

from phase_agent.analysis.feedback.analysis_steps import (
    publish_step,
    refresh_derived_files,
    _validate_step,
)
from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory
from phase_agent.analysis.feedback.versioned_products import validate_bundle
from phase_agent.evidence.model_epoch import model_epoch


ACTIVE_RELAX = ContextVar("analysis_active_relax", default=None)


def _existing(root, epoch, identity):
    index = analysis_metadata_directory(root) / "steps" / (epoch + ".json")
    if not index.is_file():
        return None
    data = json.loads(index.read_text(encoding="utf8"))
    rows = [row for row in data.get("events", []) if row.get("event_id") == identity]
    if len(rows) > 1:
        raise ValueError("analysis_step_identity_duplicate")
    if rows:
        _validate_step(Path(root) / epoch / "phase_diagram" / rows[0]["directory"], rows[0])
        return rows[0]
    return None


def phase_event(state, root, publication=None):
    """Record a hull once; older pre-adoption history remains separately accessible."""
    root = Path(root).resolve()
    model = state.get("active_model_version")
    epoch = model_epoch(state, model)
    if not epoch:
        raise ValueError("analysis_step_epoch_unavailable")
    directory = (
        Path(publication["directory"]) if publication else root / epoch / "phase_diagram" / "latest"
    )
    manifest = validate_bundle(directory)
    if publication is None:
        status = json.loads(
            (root / "latest" / "phase_diagram_status.json").read_text(encoding="utf8")
        )
        if status.get("status") != "completed" or status.get("version") != manifest.get("version"):
            raise ValueError("analysis_step_current_phase_not_ready")
    identity = manifest.get("identity") or {}
    hull = identity.get("hull_version")
    snapshot = json.loads((directory / "phase_diagram.json").read_text(encoding="utf8"))
    if (
        identity.get("epoch") != epoch
        or identity.get("model_version") != model
        or not hull
        or snapshot.get("version") != hull
        or snapshot.get("model_version", model) != model
    ):
        raise ValueError("analysis_step_hull_binding_invalid")
    event_id = "phase:" + str(model) + ":" + str(hull)
    existing = _existing(root, epoch, event_id)
    if existing:
        derived = {
            name: directory / name
            for name in ("voltage_curve.png", "voltage_curve.csv", "voltage_curve_status.json")
            if (directory / name).is_file()
        }
        if derived:
            try:
                return refresh_derived_files(root, epoch, existing, derived)
            except (OSError, ValueError, KeyError, TypeError):
                # Preserve the original durable event when a derived refresh cannot
                # be completed; the next formal analysis refresh can retry it.
                pass
        return existing
    index = analysis_metadata_directory(root) / "steps" / (epoch + ".json")
    if index.is_file():
        events = json.loads(index.read_text(encoding="utf8")).get("events") or []
        for row in reversed(events):
            info = row.get("information") or {}
            if (
                row.get("kind") == "relaxmc"
                and info.get("model_version") == model
                and info.get("hull_version") == hull
            ):
                _validate_step(root / epoch / "phase_diagram" / row["directory"], row)
                return row
    files = {
        name: directory / name
        for name in ("main.png", "hd_cross.png", "phase_diagram.json", "report.md")
    }
    if (directory / "phase_diagram.csv").is_file():
        files["phase_diagram.csv"] = directory / "phase_diagram.csv"
    for name in ("voltage_curve.png", "voltage_curve.csv", "voltage_curve_status.json"):
        if (directory / name).is_file():
            files[name] = directory / name
    return publish_step(
        root,
        epoch,
        kind="update_phasediagram",
        event_id=event_id,
        files=files,
        information={
            "model_version": model,
            "hull_version": hull,
            "sequence_scope": "Numbered history begins when this publisher was adopted; earlier records remain in the legacy archive.",
        },
    )


def relax_event(state, root, bundle_id, receipt):
    model = receipt.get("model_version")
    epoch = model_epoch(state, model)
    batch = (state.get("relax_mc_batches") or {}).get(bundle_id) or {}
    accepted_early_stop = False
    if batch.get("formal_recovery_status") == "accepted" and batch.get("status") == "paused":
        from phase_agent.tools.relax_mc_batch.completion_gate import completion_error

        tasks = [t for t in state.get("tasks") or [] if t.get("relax_mc_bundle_id") == bundle_id]
        accepted_early_stop = completion_error(batch.get("result_summary"), tasks) is None
    if (
        receipt.get("status") != "completed"
        or model != state.get("active_model_version")
        or not epoch
        or not receipt.get("import_checksum")
        or not receipt.get("hull_version")
        or (batch.get("status") != "completed" and not accepted_early_stop)
        or batch.get("model_version") != model
        or batch.get("import_checksum") != receipt.get("import_checksum")
    ):
        raise ValueError("analysis_step_relax_binding_invalid")
    identity = "relaxmc:" + bundle_id
    existing = _existing(root, epoch, identity)
    if existing:
        information = existing.get("information") or {}
        if (
            information.get("import_checksum") != receipt["import_checksum"]
            or information.get("hull_version") != receipt["hull_version"]
        ):
            raise ValueError("analysis_step_relax_identity_changed")
        return existing
    index = analysis_metadata_directory(root) / "steps" / (epoch + ".json")
    if not index.is_file():
        phase_event(state, root)
    plots = receipt.get("plot_paths") or []
    if len(plots) != 2:
        raise ValueError("analysis_step_relax_figures_missing")
    files = {"main.png": plots[0], "hd_cross.png": plots[1]}
    publication_directory = Path(plots[0]).resolve().parent
    for name in ("voltage_curve.png", "voltage_curve.csv", "voltage_curve_status.json"):
        candidate = publication_directory / name
        if candidate.is_file():
            files[name] = candidate
    for key, name in (
        ("markdown_path", "round_report.md"),
        ("review_report_path", "quality_report.json"),
    ):
        if receipt.get(key):
            files[name] = receipt[key]
    return publish_step(
        root,
        epoch,
        kind="relaxmc",
        event_id=identity,
        files=files,
        information={
            "model_version": model,
            "hull_version": receipt["hull_version"],
            "bundle_id": bundle_id,
            "import_checksum": receipt["import_checksum"],
        },
    )


def selection_event(state, root, record_path, rows):
    model = rows[0].get("model_version") if rows else None
    epoch = model_epoch(state, model)
    path = Path(record_path).resolve()
    if not epoch or model != state.get("active_model_version") or not path.is_file():
        raise ValueError("analysis_step_selection_binding_invalid")
    digest = sha256(path.read_bytes()).hexdigest()
    event_id = "dft-selection:" + sha256((str(path) + ":" + digest).encode()).hexdigest()
    existing = _existing(root, epoch, event_id)
    if existing:
        return existing
    index = analysis_metadata_directory(root) / "steps" / (epoch + ".json")
    if not index.is_file():
        phase_event(state, root)
    files = {}
    for number, row in enumerate(rows, 1):
        if row.get("status") != "completed" or row.get("model_version") != model:
            raise ValueError("analysis_step_selection_unverified")
        image = Path(row["path"])
        files[f"selection_{number:02d}.png"] = image
        if image.with_suffix(".json").is_file():
            files[f"selection_{number:02d}.json"] = image.with_suffix(".json")
    return publish_step(
        root,
        epoch,
        kind="dft-selection",
        event_id=event_id,
        files=files,
        information={
            "model_version": model,
            "hull_versions": sorted({row["hull_version"] for row in rows}),
            "selected_count": sum(row["selected_count"] for row in rows),
            "record_sha256": digest,
            "record_path": str(path),
        },
    )


def direct_event(state, root, receipt):
    from phase_agent.tools.workflows.direct_extrapolation_sampling import valid_direct_receipt

    model = receipt.get("model_version")
    epoch = model_epoch(state, model)
    if not epoch or model != state.get("active_model_version") or not valid_direct_receipt(receipt):
        raise ValueError("analysis_step_direct_binding_invalid")
    event_id = "direct:" + receipt["source_digest"]
    existing = _existing(root, epoch, event_id)
    if existing:
        if (existing.get("information") or {}).get("manifest_sha256") != receipt["manifest_sha256"]:
            raise ValueError("analysis_step_direct_identity_changed")
        return existing
    index = analysis_metadata_directory(root) / "steps" / (epoch + ".json")
    phase_note = "Previous verified phase event retained."
    if not index.is_file():
        try:
            phase_event(state, root)
        except (OSError, ValueError, KeyError, TypeError):
            phase_note = (
                "No verified current phase publication was available when DIRECT was recorded."
            )
    plan = Path(receipt["plan_path"])
    return publish_step(
        root,
        epoch,
        kind="direct",
        event_id=event_id,
        files={"selection.md": plan, "direct_result.json": plan.parent / "direct_result.json"},
        information={
            "model_version": model,
            "scope_id": receipt["scope_id"],
            "selected_count": receipt["selected_count"],
            "manifest_sha256": receipt["manifest_sha256"],
            "phase_context": phase_note,
        },
    )
