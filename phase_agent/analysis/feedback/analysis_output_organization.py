"""Organize and describe saved analysis products without changing scientific decisions."""

from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory


def _hash(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _record(path):
    from phase_agent.tools.workflows.branch_review_view import raw_record
    return raw_record(_load(path))


def _snapshot(epoch, model, version):
    """Select the exact saved hull, never the current hull of another version."""
    paths = [epoch / "phase_diagram" / "latest" / "phase_diagram.json"]
    paths += sorted((epoch / "phase_diagram" / "versions").glob("*/phase_diagram.json"))
    paths += sorted((epoch / "phase_diagram" / "_internal" / "mlip" / "history").glob("*.json"))
    paths += sorted((epoch / "phase_diagrams" / "mlip" / "history").glob("*.json"))
    recovered = (epoch.parent.parent / "submissions" / epoch.name / "recovered_results" / "analysis"
                 / epoch.name / "phase_diagrams" / "mlip" / "history" / f"phase_diagram_mlip_{version}.json")
    if recovered.is_file():
        paths.append(recovered)
    matches = []
    for path in paths:
        try:
            value = _load(path)
            if value.get("model_version") == model and value.get("version") == version:
                matches.append(value)
        except (OSError, ValueError, TypeError):
            continue
    if not matches:
        raise ValueError("saved_hull_version_unavailable")
    # Delivery migration can change structure_path without changing a frame.
    # Accept that difference only when its recorded frame hashes match exactly.
    def same_entry(left, right):
        if left == right:
            return True
        return (left.get("structure_sha256") and left.get("structure_sha256") == right.get("structure_sha256")
                and {k: v for k, v in left.items() if k != "structure_path"}
                == {k: v for k, v in right.items() if k != "structure_path"})
    reference = matches[0].get("entries") or []
    for value in matches[1:]:
        entries = value.get("entries") or []
        if len(entries) != len(reference) or any(not same_entry(a, b) for a, b in zip(reference, entries)):
            raise ValueError("conflicting_saved_hull_versions")
    return matches[0]


def _selection(state, root, epoch, record, target):
    from phase_agent.tools.workflows.dft_selection_plot import plot_dft_selection
    from phase_agent.tools.remote.transfer_bundle import local_path

    tasks = (record.get("actual") or {}).get("tasks") or []
    if not tasks:
        raise ValueError("prepared_task_inventory_unavailable")
    selected = [x for x in record.get("decisions") or []
                if x.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}]
    if not selected:
        raise ValueError("selected_frames_unavailable")
    task_pairs = {(x.get("structure_id"), str(x.get("structure_path") or "")) for x in tasks}
    selected = [x for x in selected if
                ((x.get("candidate_id") or "").split(":frame:", 1)[0],
                 str(x.get("structure_path") or "")) in task_pairs]
    if len(selected) != len(tasks):
        raise ValueError("prepared_tasks_and_selected_frames_differ")
    versions = {x.get("phase_diagram_version") for x in selected if x.get("phase_diagram_version")}
    hull = record.get("hull_version") or (next(iter(versions)) if len(versions) == 1 else None)
    if not hull or (versions and versions != {hull}):
        raise ValueError("selection_hull_binding_unavailable")
    model = record.get("model_version")
    snapshot = _snapshot(epoch, model, hull)
    moves = []
    receipt = root.parent / "submissions" / ".layout-migration.json"
    if receipt.is_file():
        migration = _load(receipt)
        if (migration.get("project") == root.parent.name and migration.get("status") == "migrated"
            and isinstance(migration.get("immutable_files_verified"), int)
            and migration["immutable_files_verified"] > 0):
            moves = migration.get("moves") or []
    candidates = []
    for row in selected:
        path = Path(row.get("structure_path") or "")
        try:
            if not path.is_file() and moves:
                from phase_agent.configuration.session.relocate_workspace_paths import relocate_workspace_paths

                path = Path(relocate_workspace_paths(str(path), moves, root.parent))
            if not path.is_file() or not local_path(path).resolve().is_relative_to(local_path(root.parent).resolve()):
                raise ValueError("selected_structure_unavailable")
            digest = _hash(path)
        except OSError as error:
            raise ValueError("selected_structure_unavailable") from error
        candidate_id = row.get("candidate_id") or ""
        suffix = candidate_id.split(":frame:", 1)[1] if ":frame:" in candidate_id else None
        expected = row.get("structure_sha256") or suffix
        if expected and expected != digest:
            raise ValueError("selected_structure_hash_changed")
        candidates.append({"candidate_id": row["candidate_id"], "structure_path": str(path),
                           "structure_sha256": digest,
                           "extrapolation_failure": row.get("extrapolation_failure") is True})
    view = {"phase_diagrams": {"mlip": snapshot}, "active_model_version": model,
            "qbc_candidates": candidates}
    # Reuse only an unchanged, complete figure with the exact model, hull and frames.
    try:
        meta = _load(target.with_suffix(".json"))
        hull_hashes = {entry.get("structure_sha256") for entry in snapshot.get("entries") or []}
        if (target.is_file() and meta.get("figure_sha256") == _hash(target)
            and meta.get("model_version") == model and meta.get("hull_version") == hull
            and meta.get("selected_count") == len(selected) and not meta.get("unplotted_selected")
            and set(meta.get("marked_candidates") or []) == {row["candidate_id"] for row in selected}
            and meta.get("candidate_frame_sha256") == {row["candidate_id"]: row["structure_sha256"] for row in candidates}
            and all(row["structure_sha256"] in hull_hashes and not row.get("extrapolation_failure") for row in candidates)):
            return {"status": "completed", "model_version": model, "hull_version": hull,
                    "selected_count": len(selected), "path": str(target), "reused": True}
    except (OSError, ValueError, TypeError, KeyError):
        pass
    # Render into an isolated temporary folder; do not publish incomplete selections.
    with TemporaryDirectory(prefix="selection-", dir=root / "latest") as temporary:
        result = plot_dft_selection(view, {"model_version": model, "hull_version": hull,
                                           "decisions": selected}, Path(temporary) / "selection.png")
        if result["unplotted_count"]:
            raise ValueError("selected_frames_missing_from_saved_hull:" + str(result["unplotted_count"]))
        metadata = _load(result["metadata"])
        metadata["candidate_frame_sha256"] = {row["candidate_id"]: row["structure_sha256"] for row in candidates}
        Path(result["metadata"]).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result["figure"], target)
        shutil.copy2(result["metadata"], target.with_suffix(".json"))
    return {"status": "completed", "model_version": model, "hull_version": hull,
            "selected_count": len(selected), "path": str(target)}




def _reconstruct_history_selection(state, root, history):
    """Rebuild a display-only record from uniquely registered tasks and their exact frames."""
    from phase_agent.analysis.feedback.versioned_products import validate_bundle
    from phase_agent.evidence.model_epoch import model_epoch

    accepted = [row for row in history.get("accepted") or []
                if row.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}]
    if not accepted:
        raise ValueError("selection_record_unavailable")
    keys = {row.get("task_key") for row in accepted}
    if None in keys or len(keys) != len(accepted):
        raise ValueError("selection_task_identity_ambiguous")
    by_key = {}
    for task in state.get("tasks") or []:
        key = task.get("task_key")
        if key in keys:
            if key in by_key:
                raise ValueError("selection_task_identity_ambiguous")
            by_key[key] = task
    if set(by_key) != keys:
        raise ValueError("registered_selection_tasks_missing")
    tasks = [by_key[row["task_key"]] for row in accepted]
    versions = {task.get("model_version") for task in tasks}
    if len(versions) != 1 or None in versions:
        raise ValueError("selection_model_binding_unavailable")
    model = next(iter(versions))
    epoch = model_epoch(state, model)
    if not epoch:
        raise ValueError("selection_epoch_unavailable")
    publication = root / epoch / "phase_diagram" / "latest"
    manifest = validate_bundle(publication)
    status = _load(root / epoch / "phase_diagram" / "latest_status.json")
    if (status.get("status") != "completed" or status.get("version") != manifest.get("version")
        or (manifest.get("identity") or {}).get("model_version") != model):
        raise ValueError("selection_epoch_hull_not_ready")
    hull = (manifest.get("identity") or {}).get("hull_version")
    if not hull:
        raise ValueError("selection_hull_binding_unavailable")
    decisions, actual = [], []
    for row, task in zip(accepted, tasks):
        candidate = row.get("candidate_id")
        if (candidate or "").split(":frame:", 1)[0] != task.get("structure_id"):
            raise ValueError("selection_task_structure_mismatch")
        expected_stage = "dft_single_point" if row["action"] == "DFT_SINGLE_POINT" else "dft_relax"
        if task.get("stage") != expected_stage or not task.get("structure_path"):
            raise ValueError("selection_task_structure_mismatch")
        decisions.append({**row, "structure_path": task["structure_path"], "phase_diagram_version": hull})
        actual.append({"task_id": task.get("task_id"), "structure_id": task["structure_id"],
                       "structure_path": task["structure_path"]})
    return {"model_version": model, "hull_version": hull, "decisions": decisions,
            "actual": {"tasks": actual}, "display_basis": "reconstructed_on_saved_epoch_hull_selection_time_hull_unrecorded"}


def _selection_groups(record):
    """Split a frozen selection only on its explicitly recorded hull versions."""
    selected = [row for row in record.get("decisions") or []
                if row.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}]
    tasks = (record.get("actual") or {}).get("tasks") or []
    if not selected or not tasks:
        raise ValueError("prepared_task_inventory_unavailable")
    task_by_frame = {}
    for task in tasks:
        key = (task.get("structure_id"), str(task.get("structure_path") or ""))
        if key in task_by_frame:
            raise ValueError("ambiguous_prepared_task_frame")
        task_by_frame[key] = task
    groups = {}
    for row in selected:
        key = ((row.get("candidate_id") or "").split(":frame:", 1)[0], str(row.get("structure_path") or ""))
        task = task_by_frame.get(key)
        if task is None:
            raise ValueError("prepared_tasks_and_selected_frames_differ")
        version = row.get("phase_diagram_version") or record.get("hull_version")
        if not version or (record.get("hull_version") and record["hull_version"] != version):
            raise ValueError("selection_hull_binding_unavailable")
        group = groups.setdefault(version, {"decisions": [], "tasks": []})
        group["decisions"].append(row)
        group["tasks"].append(task)
    if sum(len(row["tasks"]) for row in groups.values()) != len(tasks):
        raise ValueError("prepared_tasks_and_selected_frames_differ")
    return [{**record, "hull_version": version, "decisions": group["decisions"],
             "actual": {**(record.get("actual") or {}), "tasks": group["tasks"]}}
            for version, group in groups.items()]


def _move(source, destination):
    """Never replace differing data; retain every unique historical artifact."""
    if source.is_symlink() or source.is_junction():
        raise ValueError("analysis_symlink_not_supported:" + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.is_file() and destination.is_file() and _hash(source) == _hash(destination):
            source.unlink()
            return {"source": str(source), "target": str(destination), "kind": "identical_duplicate"}
        base, suffix, index = destination.stem, destination.suffix, 2
        while destination.exists():
            destination = destination.with_name(f"{base}_{index:02d}{suffix}")
            index += 1
    source.rename(destination)
    return {"source": str(source), "target": str(destination), "kind": "preserved"}


def write_epoch_guide(epoch, selections=None):
    """Render an English index from verified saved publications and selection records."""
    from phase_agent.analysis.feedback.versioned_products import validate_bundle
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

    epoch = Path(epoch)
    publication = epoch / "phase_diagram" / "latest"
    status_file = epoch / "phase_diagram" / "latest_status.json"
    try:
        status = _load(status_file)
        manifest = validate_bundle(publication)
        ready = status.get("status") == "completed" and status.get("version") == manifest.get("version")
    except (OSError, ValueError, KeyError, TypeError):
        try:
            status = _load(status_file)
        except (OSError, ValueError, TypeError):
            status = {}
        manifest, ready = {}, False
    unavailable = None
    lines = [f"# {epoch.name}", "", "## Phase diagram", ""]
    if ready:
        ident = manifest.get("identity") or {}
        lines += [f"Model: `{ident.get('model_version', 'unknown')}`  ",
                  f"Hull: `{ident.get('hull_version', 'unknown')}`", "",
                  "- [Main phase diagram](phase_diagram/latest/main.png)",
                  "- [D/H review](phase_diagram/latest/hd_cross.png)",
                  "- [Phase data and report](phase_diagram/latest/report.md)"]
    else:
        unavailable = {"epoch": epoch.name, "reason": status.get("reason", "publication or status missing")}
        lines += ["Main and D/H diagrams are not available as a verified complete publication.",
                  f"Reason: {status.get('reason', 'publication or status missing')}"]
    lines += ["", "## DFT selections", ""]
    selection = selections or []
    if not selection:
        lines.append("No registered DFT selection record was found for this epoch.")
    for n, row in enumerate(selection, 1):
        if row["status"] == "completed" and Path(row.get("path", "")).is_file() and (Path(row["path"]).resolve().is_relative_to(epoch.resolve()) or Path(row["path"]).resolve().is_relative_to((epoch.parent / "history" / epoch.name).resolve())):
            from os.path import relpath
            relative = Path(relpath(Path(row["path"]).resolve(), epoch)).as_posix()
            lines.append(f"- [DFT selection {n:02d}]({relative}): {row['selected_count']} frames; hull `{row['hull_version']}`.")
            if row.get("display_basis"):
                lines.append("  Reconstructed from registered task frames on the saved epoch hull; the hull used at selection time was not recorded.")
        elif row["status"] == "completed":
            lines.append(f"- DFT selection {n:02d} unavailable: `saved_image_missing`.")
        else:
            lines.append(f"- DFT selection {n:02d} unavailable: `{row['reason']}`.")
    lines += ["", "## Numbered analysis steps", ""]
    from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory
    index = analysis_metadata_directory(epoch.parent) / "steps" / (epoch.name + ".json")
    try:
        events = _load(index).get("events") or []
        for event in events:
            target = epoch / "phase_diagram" / event["directory"] / "README.md"
            if target.is_file():
                lines.append(f"- [{event['directory']}](phase_diagram/{event['directory']}/README.md)")
        if not events:
            lines.append("No numbered steps have been recorded yet.")
    except (OSError, ValueError, KeyError, TypeError):
        lines.append("No numbered steps have been recorded yet.")
    lines += ["", "Historical reports and internal versions are retained in [history](../history/README.md); original paths remain readable through compatibility entries.", ""]
    _publish_bytes(epoch / "README.md", "\n".join(lines).encode("utf8"))
    return unavailable


def write_latest_guide(root, state, results):
    """Keep latest links and selected-image copy bound to the current verified hull."""
    from phase_agent.analysis.feedback.versioned_products import validate_bundle
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes
    from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory
    from os.path import relpath

    root = Path(root).resolve()
    latest = root / "latest"
    latest_lines = ["# Latest analysis", ""]
    latest_status = latest / "phase_diagram_status.json"
    current = state.get("active_model_version")
    try:
        status = _load(latest_status)
        manifest = validate_bundle(latest / "phase_diagram")
        identity = manifest.get("identity") or {}
        ready = (status.get("status") == "completed" and status.get("version") == manifest.get("version")
                 and identity.get("model_version") == current)
    except (OSError, ValueError, KeyError, TypeError):
        try:
            status = _load(latest_status)
        except (OSError, ValueError, TypeError):
            status = {}
        identity, ready = {}, False
    if ready:
        latest_lines += [f"Current epoch: `{identity.get('epoch')}`  ",
                         f"Model: `{current}`  ", f"Hull: `{identity.get('hull_version')}`", "",
                         "- [Main phase diagram](phase_diagram/main.png)",
                         "- [D/H review](phase_diagram/hd_cross.png)"]
    else:
        latest_lines += ["Current phase diagrams are not ready.",
                         f"Reason: {status.get('reason', 'publication or status missing')}"]
    dft_status_path = latest / "dft_error_status.json"
    try:
        dft_status = _load(dft_status_path)
        dft_manifest = validate_bundle(latest / "dft_error")
        dft_ready = (dft_status.get("status") == "completed" and dft_status.get("version") == dft_manifest.get("version")
                     and (dft_manifest.get("identity") or {}).get("model_version") == current)
    except (OSError, ValueError, KeyError, TypeError):
        try:
            dft_status = _load(dft_status_path)
        except (OSError, ValueError, TypeError):
            dft_status = {}
        dft_ready = False
    # The latest round view is model-bound; retain an old view in history when
    # the current model or verified publication changes.
    round_view = latest / "Round_Error"
    round_identity = round_view / "identity.json"
    try:
        previous = _load(round_identity)
    except (OSError, ValueError, TypeError):
        previous = {}
    identity_key = {"model_version": current, "version": dft_status.get("version") if dft_ready else None}
    if round_view.exists() and previous != identity_key:
        import uuid
        history = root / "history" / "compatibility" / "round_error_latest"
        history.mkdir(parents=True, exist_ok=True)
        round_view.rename(history / uuid.uuid4().hex)
    round_view.mkdir(parents=True, exist_ok=True)
    _publish_bytes(round_identity, (json.dumps(identity_key, ensure_ascii=False, indent=2) + "\n").encode("utf8"))
    if dft_ready:
        for figure in ("energy_parity.png", "force_parity.png"):
            _publish_bytes(round_view / figure, (latest / "dft_error" / figure).read_bytes())
        description = "当前模型下已验证的同帧 DFT 结构汇总，可包含多个本轮 DFT 批次。\n"
    else:
        description = "当前模型尚无可靠同帧 DFT 对比：" + str(dft_status.get("reason", "缺少同帧数据")) + "\n"
    _publish_bytes(round_view / "README.md", ("# 本轮模型与 DFT 同帧误差图\n\n" + description + "\n[逐批次结果](../../round_dft_parity.md)\n").encode("utf8"))
    latest_lines += ["", "## Current-model DFT error", "", "[Round_Error](Round_Error/README.md) · [Per-round selected DFT structures and model comparisons](../round_dft_parity.md)"]
    if (latest / "Round_Error_E_MAE_by_epoch.png").is_file() and (latest / "Round_Error_F_MAE_by_epoch.png").is_file():
        latest_lines += ["- [Energy MAE by epoch](Round_Error_E_MAE_by_epoch.png) · [Force MAE by epoch](Round_Error_F_MAE_by_epoch.png) · [Values and unavailable reasons](Round_Error_MAE_by_epoch.csv)"]
    if dft_ready:
        latest_lines += ["- [Energy parity](dft_error/energy_parity.png)",
                         "- [Force parity](dft_error/force_parity.png)",
                         "- [DFT error information](dft_error/metrics.json)"]
    else:
        latest_lines += ["Current-model DFT error is not ready.",
                         f"Reason: {dft_status.get('reason', 'same-frame data not available')}"]
    latest_lines += ["", "## K-fold fine-tuning (out-of-fold)", "", "[All training rounds and unavailable reasons](../training_parity.md)"]
    if (latest / "KFold_Error_E_MAE_by_epoch.png").is_file() and (latest / "KFold_Error_F_MAE_by_epoch.png").is_file():
        latest_lines += ["- [K-fold energy MAE by model epoch](KFold_Error_E_MAE_by_epoch.png) · [K-fold force MAE by model epoch](KFold_Error_F_MAE_by_epoch.png) · [Values and unavailable reasons](KFold_Error_MAE_by_epoch.csv)"]
    try:
        training = _load(latest / "training_parity_status.json")
        if training.get("status") == "completed" and training.get("model_version") == current:
            latest_lines += ["K-fold out-of-fold evaluation; not an independent test of the final model.", "- [Energy](KFold_Error/energy_parity.png)", "- [Forces](KFold_Error/force_parity.png)"]
        else:
            latest_lines.append("Unavailable: " + str(training.get("reason", "model mismatch")))
    except (OSError, ValueError, TypeError):
        latest_lines.append("No verified fine-tuning evaluation for the current model.")
    metadata = analysis_metadata_directory(root)
    index = Path(relpath(metadata / "output_index.md", latest)).as_posix()
    report = Path(relpath(metadata / "analysis_organization.json", latest)).as_posix()
    latest_lines += ["", f"- [Project analysis index]({index})",
                     f"- [Organization report]({report})", ""]
    latest_lines += ["- [全部分析目录](../README.md)"]
    if (latest / "out_data").is_dir():
        latest_lines.append("- [当前导出表格](../out_data/)")
    _publish_bytes(latest / "README.md", "\n".join(latest_lines).encode("utf8"))




def publish_prepared_selection(state, record, record_path):
    """Publish one newly approved selection without reanalyzing older epochs."""
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes
    from phase_agent.evidence.model_epoch import model_epoch

    directory = state.get("analysis_output_directory")
    if not directory:
        return {"status": "unavailable", "reason": "analysis_output_directory_unavailable"}
    root = Path(directory).resolve()
    from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory

    report_path = analysis_metadata_directory(root) / "analysis_organization.json"
    if root.name != "analysis_outputs" or not root.is_dir():
        return {"status": "unavailable", "reason": "analysis_organization_unavailable"}
    report = _load(report_path) if report_path.is_file() else {"migrations": [], "dft_selections": {}}
    epoch = model_epoch(state, record.get("model_version"))
    if not epoch or not (root / epoch).is_dir():
        raise ValueError("selection_epoch_unavailable")
    selections = report.setdefault("dft_selections", {})
    rows = selections.setdefault(epoch, [])
    # Idempotent retries replace only this exact frozen plan's own views.
    old = [row for row in rows if row.get("record") == str(record_path)]
    if old:
        try:
            from phase_agent.analysis.feedback.analysis_events import selection_event
            step = selection_event(state, root, record_path, old)
        except (OSError, ValueError, KeyError, TypeError) as error:
            step = {"status": "unavailable", "reason": str(error)}
        return {"status": "already_registered", "plots": [row.get("path") for row in old],
                "analysis_step": step}
    created, temporary_targets = [], []
    try:
        for group in _selection_groups(record):
            target = root / epoch / "phase_diagram" / "_internal" / "selection" / f"DFT_Selection_{len(rows)+1:02d}.png"
            if target.exists() or target.with_suffix(".json").exists():
                raise ValueError("selection_presentation_path_conflict")
            temporary_targets.extend((target, target.with_suffix(".json")))
            outcome = _selection(state, root, root / epoch, group, target)
            outcome["record"] = str(record_path)
            rows.append(outcome)
            created.append(outcome)
    except Exception:
        for target in temporary_targets:
            target.unlink(missing_ok=True)
        raise
    _publish_bytes(report_path, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf8"))
    write_epoch_guide(root / epoch, rows)
    write_latest_guide(root, state, selections)
    try:
        from phase_agent.analysis.feedback.analysis_events import selection_event
        step = selection_event(state, root, record_path, created)
    except (OSError, ValueError, KeyError, TypeError) as error:
        step = {"status": "unavailable", "reason": str(error)}
    write_epoch_guide(root / epoch, rows)
    return {"status": "completed", "plots": [row["path"] for row in created], "analysis_step": step}



def _referenced_root_entries(state, root, candidates):
    """Find saved paths before moving old outputs; ignore only managed navigation metadata."""
    names = {p.name.casefold(): p.name for p in candidates
             if p.name not in {"output_index.md", "round_summary.csv"}}
    if not names:
        return {}

    def normalize(value):
        value = str(value).replace("\\", "/").casefold()
        if value.startswith("//?/unc/"):
            value = "//" + value[8:]
        elif value.startswith("//?/"):
            value = value[4:]
        return value.rstrip("/")

    prefix = normalize(root) + "/"
    found, stack = {}, [state]
    while stack and len(found) < len(names):
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif isinstance(value, str):
            normalized = normalize(value)
            if normalized.startswith(prefix):
                name = normalized[len(prefix):].split("/", 1)[0]
                if name in names:
                    found[names[name]] = value
    return found


def organize_analysis_outputs(state, directory):
    """Called by Agent under its project lock; never changes task or scientific state."""
    from phase_agent.analysis.feedback.versioned_products import validate_bundle
    from phase_agent.evidence.model_epoch import model_epoch
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

    root = Path(directory).resolve()
    if root.name != "analysis_outputs" or not root.is_dir():
        raise ValueError("analysis_output_root_must_be_registered_project_directory")
    latest = root / "latest"
    latest.mkdir(exist_ok=True)
    if latest.is_symlink() or latest.is_junction():
        raise ValueError("analysis_latest_must_not_be_a_link")
    from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory

    metadata = analysis_metadata_directory(root)
    metadata.mkdir(parents=True, exist_ok=True)
    report_path = metadata / "analysis_organization.json"
    legacy_report = latest / "analysis_organization.json"
    try:
        previous_report = _load(report_path if report_path.is_file() else legacy_report)
        previous = previous_report.get("migrations") or []
        previous_selections = previous_report.get("dft_selections") or {}
    except (OSError, ValueError, TypeError, AttributeError):
        previous, previous_selections = [], {}
    migrations = list(previous)
    from phase_agent.analysis.feedback.physical_layout import managed_alias, organize_physical_layout

    sources = sorted(root.iterdir())
    candidates = [source for source in sources if source.name not in {"latest", "out_data", "README.md", "history", "training_parity_index.json", "training_parity.md", "round_dft_parity.md"}
                  and not (source.is_dir() and re.fullmatch(r"epoch\d+(?:-extrapolation_failed)?", source.name))]
    referenced = _referenced_root_entries(state, root, candidates)
    protected = []
    # Never move a saved scientific or historical path without a reference migration.
    for source in sources:
        if managed_alias(root, source):
            continue
        if source.is_symlink() or source.is_junction():
            raise ValueError("analysis_symlink_not_supported:" + str(source))
        if source.name in {"latest", "out_data", "README.md", "history", "training_parity_index.json", "training_parity.md", "round_dft_parity.md"} or (source.is_dir() and re.fullmatch(r"epoch\d+(?:-extrapolation_failed)?", source.name)):
            continue
        if source.name in referenced:
            protected.append({"path": str(source), "reference": referenced[source.name],
                              "reason": "referenced_by_saved_state"})
            continue
        match = re.match(r"(epoch\d+)(?:_|$)", source.name)
        if match and (root / match.group(1)).is_dir():
            destination = root / match.group(1) / "Historical_Reports" / source.name
        elif source.name in {"gpu_usage.csv", "round_summary.csv"} and not (metadata / source.name).exists():
            destination = metadata / source.name
        else:
            destination = metadata / "Historical_Reports" / source.name
        if not (destination.resolve().is_relative_to(root) or destination.resolve().is_relative_to(metadata)):
            raise ValueError("analysis_destination_outside_root")
        migrations.append(_move(source, destination))
        _publish_bytes(report_path, (json.dumps({"migrations": migrations}, ensure_ascii=False, indent=2) + "\n").encode("utf8"))
    # Keep latest for current MLIP and current-model DFT error only.
    keep_latest = {"README.md", "phase_diagram", "phase_diagram_status.json", "dft_error", "dft_error_status.json", "out_data", "training_parity", "training_parity_status.json", "KFold_Error", "Round_Error", "Round_Error_E_MAE_by_epoch.png", "Round_Error_E_MAE_by_epoch.pdf", "Round_Error_F_MAE_by_epoch.png", "Round_Error_F_MAE_by_epoch.pdf", "Round_Error_MAE_by_epoch.csv", "Round_Error_MAE_by_epoch.json", "KFold_Error_E_MAE_by_epoch.png", "KFold_Error_F_MAE_by_epoch.png", "KFold_Error_MAE_by_epoch.csv", "KFold_Error_MAE_by_epoch.json"}
    latest_sources = [item for item in sorted(latest.iterdir()) if item.name not in keep_latest and not managed_alias(root, item)]
    latest_references = _referenced_root_entries(state, latest, latest_sources)
    for item in latest_sources:
        if item.is_symlink() or item.is_junction():
            raise ValueError("analysis_symlink_not_supported:" + str(item))
        if item.name in latest_references:
            protected.append({"path": str(item), "reference": latest_references[item.name],
                              "reason": "referenced_by_saved_state"})
            continue
        destination = metadata / item.name if item.name in {"output_index.md", "round_summary.csv", "gpu_usage.csv"} else metadata / "Previous_Latest" / item.name
        if not (destination.resolve().is_relative_to(metadata.resolve()) or destination.resolve().is_relative_to((root / "history").resolve())):
            raise ValueError("analysis_destination_outside_metadata")
        migrations.append(_move(item, destination))
        _publish_bytes(report_path, (json.dumps({"migrations": migrations}, ensure_ascii=False, indent=2) + "\n").encode("utf8"))
    # Do not leave an old model's complete bundle in the current-model view.
    # Preserve its bytes in internal history, unless a saved path still refers to it.
    for kind in ("phase_diagram", "dft_error"):
        item = latest / kind
        if not item.exists():
            continue
        try:
            status = _load(latest / (kind + "_status.json"))
            manifest = validate_bundle(item)
            current = (status.get("status") == "completed" and status.get("version") == manifest.get("version")
                       and (manifest.get("identity") or {}).get("model_version") == state.get("active_model_version"))
        except (OSError, ValueError, KeyError, TypeError):
            current = False
        if current:
            continue
        old_reference = _referenced_root_entries(state, latest, [item])
        if item.name in old_reference:
            protected.append({"path": str(item), "reference": old_reference[item.name],
                              "reason": "stale_publication_referenced_by_saved_state"})
            continue
        migrations.append(_move(item, metadata / "Previous_Latest" / ("stale_" + kind)))
        _publish_bytes(report_path, (json.dumps({"migrations": migrations}, ensure_ascii=False, indent=2) + "\n").encode("utf8"))
    # Preserve user-authored guide contents and update only the generated index link.
    for guide in (root.parent / "README.md", root.parent / "documentation" / "FILE_LAYOUT.md"):
        if guide.is_file():
            text = guide.read_text(encoding="utf8")
            updated = text.replace("analysis_outputs/latest/output_index.md", "workflow_state/analysis/output_index.md").replace("analysis_outputs/output_index.md", "workflow_state/analysis/output_index.md")
            if updated != text:
                _publish_bytes(guide, updated.encode("utf8"))
    results = {}
    discarded = []
    tasks_by_id = {row.get("task_id"): row for row in state.get("tasks") or [] if row.get("task_id")}
    histories = state.get("qbc_selection_history") or []
    for history in histories:
        name, image = None, None
        path = (history.get("artifacts") or {}).get("json")
        try:
            if path:
                plan = Path(path)
                if not plan.is_file() or not plan.resolve().is_relative_to(root.parent):
                    raise ValueError("registered_selection_record_unavailable")
                record = _record(plan)
            else:
                record = _reconstruct_history_selection(state, root, history)
                path = str(history.get("task_key") or "registered task history")
            model = record.get("model_version")
            name = model_epoch(state, model)
            if not name or not (root / name).is_dir():
                raise ValueError("selection_epoch_unavailable")
            actual = (record.get("actual") or {}).get("tasks") or []
            task_ids = [row.get("task_id") for row in actual]
            if task_ids and all(t in tasks_by_id and tasks_by_id[t].get("status") == "cancelled" for t in task_ids):
                discarded.append({"epoch": name, "record": str(path), "reason": "all_prepared_tasks_cancelled"})
                continue
            groups = _selection_groups(record)
            for group in groups:
                entries = results.setdefault(name, [])
                image = root / name / "DFT_Selection" / f"DFT_Selection_{len(entries)+1:02d}.png"
                # Reuse selections produced by the current event publisher. Legacy
                # paths remain in place for historical references.
                internal = (root / name / "phase_diagram" / "_internal" / "selection").resolve()
                for saved in previous_selections.get(name) or []:
                    saved_path = Path(saved.get("path") or "")
                    if (saved.get("record") == str(path) and saved.get("hull_version") == group.get("hull_version")
                        and saved_path.is_file() and saved_path.resolve().is_relative_to(internal)):
                        image = saved_path
                        break
                # Never overwrite another recorded selection merely because its index changed.
                if image.is_file():
                    sidecar = image.with_suffix(".json")
                    existing = _load(sidecar) if sidecar.is_file() else {}
                    ids = {row.get("candidate_id") for row in group.get("decisions") or []}
                    if set(existing.get("marked_candidates") or []) != ids:
                        if not existing.get("candidate_frame_sha256"):
                            raise ValueError("existing_selection_image_unmanaged")
                        _move(image, root / name / "Historical_Reports" / "DFT_Selection" / image.name)
                        _move(sidecar, root / name / "Historical_Reports" / "DFT_Selection" / sidecar.name)
                try:
                    outcome = _selection(state, root, root / name, group, image)
                    outcome["record"] = str(path)
                    if group.get("display_basis"):
                        outcome["display_basis"] = group["display_basis"]
                        metadata_path = image.with_suffix(".json")
                        selection_metadata = _load(metadata_path)
                        selection_metadata["display_basis"] = group["display_basis"]
                        _publish_bytes(metadata_path, (json.dumps(selection_metadata, ensure_ascii=False, indent=2) + "\n").encode("utf8"))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    if image.is_file():
                        _move(image, root / name / "Historical_Reports" / "DFT_Selection" / image.name)
                        sidecar = image.with_suffix(".json")
                        if sidecar.is_file():
                            _move(sidecar, root / name / "Historical_Reports" / "DFT_Selection" / sidecar.name)
                    outcome = {"status": "unavailable", "reason": str(error), "record": str(path),
                               "hull_version": group.get("hull_version")}
                results.setdefault(name, []).append(outcome)
        except (OSError, ValueError, KeyError, TypeError) as error:
            outcome = {"status": "unavailable", "reason": str(error), "record": str(path)}
            results.setdefault(name or "unassigned", []).append(outcome)
    # Previous managed images no longer in the current plans are retained as history.
    for epoch_name, entries in results.items():
        if epoch_name == "unassigned":
            continue
        allowed = {Path(row["path"]).resolve() for row in entries if row.get("status") == "completed"}
        for old in (root / epoch_name / "DFT_Selection").glob("DFT_Selection_*.png"):
            sidecar = old.with_suffix(".json")
            if old.resolve() in allowed or not sidecar.is_file():
                continue
            try:
                selection_metadata = _load(sidecar)
            except (OSError, ValueError, TypeError):
                continue
            if selection_metadata.get("candidate_frame_sha256"):
                _move(old, root / epoch_name / "Historical_Reports" / "DFT_Selection" / old.name)
                _move(sidecar, root / epoch_name / "Historical_Reports" / "DFT_Selection" / sidecar.name)
    epochs = sorted((p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"epoch\d+", p.name)),
                    key=lambda p: int(p.name[5:]))
    # Begin the new ordered history at the current verified publication only.
    # Earlier history is preserved without inventing event ordering.
    current_step = None
    try:
        current_status = _load(latest / "phase_diagram_status.json")
        if current_status.get("status") == "completed":
            current_manifest = validate_bundle(latest / "phase_diagram")
            if current_status.get("version") != current_manifest.get("version"):
                raise ValueError("current_phase_status_version_mismatch")
            from phase_agent.analysis.feedback.analysis_events import phase_event
            current_step = phase_event(state, root)
        else:
            current_step = {"status": "unavailable", "reason": current_status.get("reason", "current_phase_not_ready")}
    except (OSError, ValueError, KeyError, TypeError) as error:
        current_step = {"status": "unavailable", "reason": str(error)}
    # Route legacy DFT selection figures into the numbered event folder while
    # preserving the old image path for compatibility and auditability.
    for epoch_name, entries in results.items():
        if epoch_name == "unassigned":
            continue
        grouped = {}
        for row in entries:
            if row.get("status") == "completed" and row.get("record"):
                grouped.setdefault(row["record"], []).append(row)
        for record_path, rows in grouped.items():
            if all(Path(row.get("path", "")).resolve().is_relative_to(
                (root / epoch_name / "phase_diagram").resolve()
            ) for row in rows):
                continue
            try:
                from phase_agent.analysis.feedback.analysis_events import selection_event
                step = selection_event(state, root, record_path, rows)
                for number, row in enumerate(rows, 1):
                    target = root / epoch_name / "phase_diagram" / step["directory"] / f"selection_{number:02d}.png"
                    if target.is_file():
                        row["path"] = str(target)
                        row["numbered_step"] = step["directory"]
            except (OSError, ValueError, KeyError, TypeError):
                # Keep the verified legacy path when provenance is insufficient.
                continue
    unavailable_publications = []
    for epoch in epochs:
        missing = write_epoch_guide(epoch, results.get(epoch.name))
        if missing:
            unavailable_publications.append(missing)
    from phase_agent.analysis.feedback.output_catalog import output_catalog, catalog_markdown

    physical_layout = organize_physical_layout(root)
    _publish_bytes(metadata / "output_index.md",
                   catalog_markdown(output_catalog(state), root, state=state, index_directory=metadata).encode("utf8"))
    write_latest_guide(root, state, results)
    from phase_agent.analysis.feedback.folder_navigation import publish_folder_navigation
    publish_folder_navigation(root)
    protected = [row for row in protected if not managed_alias(root, Path(row["path"]))]
    legacy_phase_directories = [str(epoch / "phase_diagrams") for epoch in epochs
                                if (epoch / "phase_diagrams").exists() and not managed_alias(root, epoch / "phase_diagrams")]
    _publish_bytes(report_path, (json.dumps({"migrations": migrations,
        "dft_selections": results, "discarded_selections": discarded, "protected_root_entries": protected,
        "unavailable_publications": unavailable_publications, "legacy_phase_directories": legacy_phase_directories, "current_step": current_step}, ensure_ascii=False, indent=2) + "\n").encode("utf8"))
    incomplete = protected or unavailable_publications or any(
        row.get("status") != "completed" for rows in results.values() for row in rows
    )
    status = "incomplete" if incomplete else ("completed_with_legacy" if legacy_phase_directories else "completed")
    return {"status": status, "physical_layout": physical_layout,
            "unavailable_publications": unavailable_publications, "protected_root_entries": protected, "legacy_phase_directories": legacy_phase_directories, "current_step": current_step, "discarded_selections": discarded, "epochs": [p.name for p in epochs], "migrations": migrations, "dft_selections": results,
            "report": str(report_path)}

