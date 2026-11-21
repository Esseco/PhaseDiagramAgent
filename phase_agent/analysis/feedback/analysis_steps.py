"""Publish traceable per-epoch phase-diagram steps with stable sequence numbers."""

from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from filelock import FileLock

from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory


KINDS = {"update_phasediagram", "relaxmc", "dft-selection", "direct"}


def _hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _validate_step(directory, expected):
    manifest = _read(Path(directory) / "manifest.json")
    if manifest != expected:
        raise ValueError("analysis_step_manifest_mismatch")
    for name, digest in manifest["files"].items():
        path = Path(directory) / name
        if not path.is_file() or _hash(path) != digest:
            raise ValueError("analysis_step_file_changed:" + name)
        if path.suffix.lower() == ".png":
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
    return manifest


def refresh_derived_files(root, epoch, row, files):
    """Backfill derived files in an existing step without changing its event identity.

    Historical scientific inputs and the event row remain unchanged.  This is used
    when a newly adopted derived product (for example the voltage curve) was not
    available or was not ready when an older step was first published.
    """
    if not isinstance(files, dict) or not files:
        return row
    root = Path(root).resolve()
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

    phase = root / epoch / "phase_diagram"
    internal = analysis_metadata_directory(root) / "steps"
    index_path = internal / (epoch + ".json")
    target = phase / row["directory"]
    with FileLock(str(internal / (epoch + ".lock")), timeout=30):
        index = _read(index_path)
        events = index.get("events")
        matches = [item for item in events or [] if item.get("event_id") == row.get("event_id")]
        if len(matches) != 1:
            raise ValueError("analysis_step_identity_missing")
        current = matches[0]
        _validate_step(target, current)
        checksums = dict(current["files"])
        changed = False
        for name, source in files.items():
            source = Path(source).resolve()
            if not source.is_file() or not source.is_relative_to(root.parent):
                raise ValueError("analysis_step_source_invalid:" + str(name))
            digest = _hash(source)
            if checksums.get(name) != digest:
                checksums[name] = digest
                changed = True
        if not changed:
            return {**current, "reused": True}

        refreshed = {**current, "files": checksums}
        phase.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".refresh-", dir=phase) as temporary:
            staged = Path(temporary) / "step"
            shutil.copytree(target, staged)
            for name, source in files.items():
                shutil.copy2(source, staged / name)
            (staged / "manifest.json").write_text(
                json.dumps(refreshed, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
            )
            _validate_step(staged, refreshed)
            previous = Path(temporary) / "previous"
            target.rename(previous)
            try:
                staged.rename(target)
            except BaseException:
                previous.rename(target)
                raise
            updated_events = [
                refreshed if item.get("event_id") == current.get("event_id") else item
                for item in events
            ]
            try:
                _publish_bytes(
                    index_path,
                    (
                        json.dumps({**index, "events": updated_events}, ensure_ascii=False, indent=2)
                        + "\n"
                    ).encode("utf8"),
                )
            except BaseException:
                target.rename(staged)
                previous.rename(target)
                raise
        return {**refreshed, "reused": True, "derived_refreshed": True}


def publish_step(root, epoch, *, kind, event_id, files, information, expected_ordinal=None):
    """Append or reuse one evidence-bound step; never renumber an existing step."""
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

    root = Path(root).resolve()
    if not re.fullmatch(r"epoch\d+", str(epoch)) or kind not in KINDS:
        raise ValueError("analysis_step_epoch_or_kind_invalid")
    if not isinstance(event_id, str) or not event_id or len(event_id) > 512:
        raise ValueError("analysis_step_identity_invalid")
    if not isinstance(information, dict) or not isinstance(files, dict):
        raise ValueError("analysis_step_content_invalid")
    if kind in {"update_phasediagram", "relaxmc"} and not {"main.png", "hd_cross.png"} <= set(
        files
    ):
        raise ValueError("analysis_step_phase_figures_required")
    if kind == "dft-selection" and not any(name.endswith(".png") for name in files):
        raise ValueError("analysis_step_selection_figure_required")
    project = root.parent.resolve()
    checksums = {}
    for name, source in files.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in {"README.md", "manifest.json"}
            or name in {".", ".."}
            or "\\" in name
            or "/" in name
        ):
            raise ValueError("analysis_step_filename_invalid")
        path = Path(source).resolve()
        if not path.is_relative_to(project) or not path.is_file():
            raise ValueError("analysis_step_source_invalid:" + name)
        checksums[name] = _hash(path)
    phase = root / epoch / "phase_diagram"
    internal = analysis_metadata_directory(root) / "steps"
    internal.mkdir(parents=True, exist_ok=True)
    index_path = internal / (epoch + ".json")
    with FileLock(str(internal / (epoch + ".lock")), timeout=30):
        index = (
            _read(index_path)
            if index_path.is_file()
            else {"schema": 1, "epoch": epoch, "events": []}
        )
        events = index.get("events")
        if index.get("epoch") != epoch or not isinstance(events, list):
            raise ValueError("analysis_step_index_invalid")
        if [item.get("ordinal") for item in events] != list(range(len(events))):
            raise ValueError("analysis_step_index_noncontiguous")
        matches = [item for item in events if item.get("event_id") == event_id]
        if len(matches) > 1:
            raise ValueError("analysis_step_identity_duplicate")
        if matches:
            row = matches[0]
            if (
                row["kind"] != kind
                or row["files"] != checksums
                or row["information"] != information
            ):
                raise ValueError("analysis_step_identity_changed")
            _validate_step(phase / row["directory"], row)
            return {**row, "reused": True}
        ordinal = len(events)
        if expected_ordinal is not None and expected_ordinal != ordinal:
            raise ValueError("analysis_step_order_changed")
        if ordinal == 0 and kind not in {"update_phasediagram", "direct"}:
            raise ValueError("analysis_step_initial_hull_required")
        count = 1 + sum(row.get("kind") == kind for row in events)
        label = "update_phasediagram" if kind == "update_phasediagram" else f"{kind}-{count}"
        name = f"{ordinal}_{label}"
        target = phase / name
        if target.exists():
            raise ValueError("analysis_step_destination_conflict")
        phase.mkdir(parents=True, exist_ok=True)
        row = {
            "ordinal": ordinal,
            "kind": kind,
            "event_id": event_id,
            "directory": name,
            "files": checksums,
            "information": information,
        }
        with TemporaryDirectory(prefix=".step-", dir=phase) as temporary:
            staged = Path(temporary) / "step"
            staged.mkdir()
            for filename, source in files.items():
                shutil.copy2(source, staged / filename)
                if _hash(staged / filename) != checksums[filename]:
                    raise ValueError("analysis_step_source_changed:" + filename)
            lines = [f"# {ordinal}: {label}", ""]
            lines += [f"- {key}: {value}" for key, value in sorted(information.items())]
            lines += ["", "## Files", ""] + [f"- [{name}]({name})" for name in sorted(files)] + [""]
            (staged / "README.md").write_text("\n".join(lines), encoding="utf8")
            (staged / "manifest.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
            )
            _validate_step(staged, row)
            staged.rename(target)
            try:
                _publish_bytes(
                    index_path,
                    (
                        json.dumps(
                            {**index, "events": [*events, row]}, ensure_ascii=False, indent=2
                        )
                        + "\n"
                    ).encode("utf8"),
                )
            except BaseException:
                target.rename(staged)
                raise
    return row
