"""Export compact per-epoch DFT and Relax/MC data products."""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

from phase_agent.evidence.model_epoch import model_epoch
from phase_agent.evidence.summarize_model_rounds import summarize_model_rounds

DFT_FIELDS = (
    "epoch", "model_version", "metric_scope", "fold", "structures",
    "force_components", "energy_mae_meV_per_atom", "energy_rmse_meV_per_atom",
    "force_mae_meV_per_A", "force_rmse_meV_per_A", "status", "source_path", "notes",
)

STRUCT_FIELDS = (
    "name", "epoch", "model_version", "task_id", "task_key", "stage", "structure_id",
    "branch_id", "composition", "phase", "na_content", "na_per_o2", "ehull_eV_per_atom",
    "energy_eV_per_atom", "qbc_error", "qbc_energy_std", "qbc_force_std_max",
    "m3gnet_features", "m3gnet_feature_status", "m3gnet_feature_error", "dh_extrapolation", "source",
    "structure_path", "structure_checksum",
)


def _json_cell(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _first(*values):
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _epoch(state, version):
    return model_epoch(state, version) or "unassigned"


def _atomic_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent,
                               prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_file() and path.read_bytes() == temporary.read_bytes():
            return
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _snapshot_name(value):
    value = str(value or "refresh")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "refresh"


def _snapshot_csv(path, fields, rows):
    """Write a versioned copy without making the current export depend on it."""
    _atomic_csv(path, fields, rows)


def _write_versioned_snapshots(root, dft_rows, structure_rows, *, action="refresh", epoch=None):
    """Persist action/epoch snapshots and a stable latest view."""
    out = Path(root)
    action_name = _snapshot_name(action)
    epochs = sorted(set(structure_rows) | {str(row.get("epoch") or "unassigned") for row in dft_rows})
    if epoch is not None:
        epochs = [str(epoch)]
    versioned = []
    for epoch_name in epochs:
        version_root = out / "versions" / epoch_name / action_name
        dft_for_epoch = [row for row in dft_rows if str(row.get("epoch") or "unassigned") == epoch_name]
        structs_for_epoch = structure_rows.get(epoch_name, [])
        if action_name in {"dft", "dft_feedback"} or dft_for_epoch:
            path = version_root / "DFT_err_data.csv"
            _snapshot_csv(path, DFT_FIELDS, dft_for_epoch)
            versioned.append(str(path))
        if action_name in {"relax_mc", "relax-mc", "refresh"} or structs_for_epoch:
            path = version_root / f"{epoch_name}_structs_features.csv"
            _snapshot_csv(path, STRUCT_FIELDS, structs_for_epoch)
            versioned.append(str(path))

    latest = out / "latest"
    _snapshot_csv(latest / "DFT_err_data.csv", DFT_FIELDS, dft_rows)
    for epoch_name, rows in structure_rows.items():
        _snapshot_csv(latest / f"{epoch_name}_structs_features.csv", STRUCT_FIELDS, rows)

    # Keep the user-facing analysis_outputs/latest view in sync as well.
    analysis_latest = out.parent / "latest" / "out_data"
    _snapshot_csv(analysis_latest / "DFT_err_data.csv", DFT_FIELDS, dft_rows)
    for epoch_name, rows in structure_rows.items():
        _snapshot_csv(analysis_latest / f"{epoch_name}_structs_features.csv", STRUCT_FIELDS, rows)
    manifest = {
        "action": action_name,
        "epoch": str(epoch) if epoch is not None else "all",
        "dft_error_path": str(latest / "DFT_err_data.csv"),
        "structure_paths": {name: str(latest / f"{name}_structs_features.csv") for name in structure_rows},
        "versioned_paths": versioned,
    }
    manifest_path = latest / "snapshot_manifest.json"
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

    _publish_bytes(
        manifest_path,
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return {"action": action_name, "epoch": str(epoch) if epoch is not None else "all", "paths": versioned}


def _training_rows(state):
    rows = []
    jobs = state.get("remote_finetune_jobs") or {}
    for job_id, job in jobs.items():
        if job.get("status") == "abandoned" or not job.get("directory"):
            continue
        results = Path(job["directory"]) / "results"
        csv_path = results / "kfold_metrics.csv"
        report = job.get("report") or {}
        source_model = job.get("original_model_version") or report.get("original_model_version")
        epoch_name = report.get("epoch") or _epoch(state, source_model)
        if not csv_path.is_file():
            metric = report.get("kfold_metrics") or {}
            if not metric:
                continue
            metric_rows = [dict(metric, fold="all_out_of_fold")]
            source_path = str(results)
            status = "reported_without_csv"
        else:
            try:
                with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                    metric_rows = list(csv.DictReader(handle))
            except (OSError, ValueError, csv.Error):
                continue
            source_path = str(csv_path)
            status = "completed"
        for metric in metric_rows:
            fold = metric.get("fold") or ""
            if fold == "all_out_of_fold":
                scope = "kfold_all_out_of_fold"
            elif fold == "current_round_out_of_fold":
                scope = "kfold_current_round_out_of_fold"
            else:
                scope = "kfold_fold"
            row = {
                "epoch": epoch_name,
                "model_version": source_model or "",
                "metric_scope": scope,
                "fold": fold,
                "structures": metric.get("structures", ""),
                "force_components": metric.get("force_components", ""),
                "energy_mae_meV_per_atom": metric.get("energy_MAE_meV_per_atom", ""),
                "energy_rmse_meV_per_atom": metric.get("energy_RMSE_meV_per_atom", ""),
                "force_mae_meV_per_A": metric.get("force_MAE_meV_per_A", ""),
                "force_rmse_meV_per_A": metric.get("force_RMSE_meV_per_A", ""),
                "status": status,
                "source_path": source_path,
                "notes": "job_id=" + str(job_id),
            }
            rows.append(row)
    return rows


def _dft_feedback_rows(state, root):
    rows = []
    for summary in summarize_model_rounds(state):
        rows.append({
            "epoch": summary.get("model_epoch") or _epoch(state, summary.get("model_version")),
            "model_version": summary.get("model_version") or "",
            "metric_scope": "dft_feedback_epoch_cumulative",
            "fold": "",
            "structures": summary.get("matched_dft_tasks", ""),
            "force_components": summary.get("force_component_count", ""),
            "energy_mae_meV_per_atom": _mev(summary.get("energy_mae_eV_per_atom")),
            "energy_rmse_meV_per_atom": _mev(summary.get("energy_rmse_eV_per_atom")),
            "force_mae_meV_per_A": _mev(summary.get("force_mae_eV_per_A")),
            "force_rmse_meV_per_A": _mev(summary.get("force_rmse_eV_per_A")),
            "status": summary.get("error_status") or "unknown",
            "source_path": summary.get("model_version") or "",
            "notes": "summarize_model_rounds; epoch cumulative same-frame feedback",
        })
    epochs = {row.get("model_epoch") or _epoch(state, row.get("model_version")): row.get("model_version") for row in rows}
    for epoch, version in epochs.items():
        path = Path(root) / str(epoch) / "round_metrics.csv"
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                metric_rows = list(csv.DictReader(handle))
        except (OSError, csv.Error, ValueError):
            continue
        grouped = {}
        for metric in metric_rows:
            if metric.get("model_version") and version and metric.get("model_version") != version:
                continue
            key = (metric.get("dft_round") or "", metric.get("search_group_index") or "")
            grouped.setdefault(key, {}).update(metric)
        for key, metric in sorted(grouped.items()):
            energy = next((item for item in metric_rows if (item.get("dft_round") or "", item.get("search_group_index") or "") == key and item.get("metric") == "energy_per_atom"), None)
            forces = next((item for item in metric_rows if (item.get("dft_round") or "", item.get("search_group_index") or "") == key and item.get("metric") == "forces"), None)
            rows.append({
                "epoch": epoch,
                "model_version": version or metric.get("model_version") or "",
                "metric_scope": "dft_feedback_round",
                "fold": f"{key[0]}@group{key[1]}",
                "structures": (energy or {}).get("matched_structures", ""),
                "force_components": (forces or {}).get("sample_count", ""),
                "energy_mae_meV_per_atom": _mev(_float((energy or {}).get("mae"))),
                "energy_rmse_meV_per_atom": _mev(_float((energy or {}).get("rmse"))),
                "force_mae_meV_per_A": _mev(_float((forces or {}).get("mae"))),
                "force_rmse_meV_per_A": _mev(_float((forces or {}).get("rmse"))),
                "status": metric.get("status") or "unknown",
                "source_path": str(path),
                "notes": "search_group_index=" + str(key[1]),
            })
    return rows


def _float(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _mev(value):
    value = _number(value)
    return None if value is None else value * 1000.0


def _phase_entries(state):
    entries = {}
    snapshots = dict(state.get("phase_diagrams_by_model") or {})
    for key, snapshot in (state.get("phase_diagrams") or {}).items():
        if not isinstance(snapshot, dict) or key == "dft":
            continue
        version = snapshot.get("model_version") or key
        snapshots[version] = snapshot
    for version, snapshot in snapshots.items():
        if not isinstance(snapshot, dict):
            continue
        for entry in snapshot.get("entries") or []:
            keys = [entry.get("structure_id"), entry.get("record_id"), entry.get("structure_sha256"), entry.get("structure_path")]
            for key in keys:
                if key:
                    entries[(version, str(key))] = entry
    return entries


def _candidate_entries(state):
    entries = {}
    for collection in ("qbc_candidates", "branch_candidates", "candidates"):
        for entry in state.get(collection) or []:
            if not isinstance(entry, dict):
                continue
            for key in (entry.get("structure_id"), entry.get("candidate_id"), entry.get("structure_sha256"), entry.get("structure_path")):
                if key:
                    entries[str(key)] = entry
    return entries


def _feature_entries(state):
    entries = {}
    for entry in state.get("feature_records") or []:
        if not isinstance(entry, dict):
            continue
        for key in (entry.get("structure_id"), entry.get("task_id"), entry.get("structure_sha256"), entry.get("structure_path")):
            if key:
                entries[str(key)] = entry
    return entries


def _lookup(mapping, *keys):
    for key in keys:
        if key is None:
            continue
        if key in mapping:
            return mapping[key]
        if str(key) in mapping:
            return mapping[str(key)]
    return {}


def _structure_rows(state):
    phase_entries = _phase_entries(state)
    candidate_entries = _candidate_entries(state)
    feature_entries = _feature_entries(state)
    selected = {}
    for task in state.get("tasks") or []:
        if task.get("stage") not in {"relax_and_feature", "deep_search"} or task.get("status") != "completed":
            continue
        version = task.get("model_version") or state.get("active_model_version")
        output = task.get("outputs") or {}
        path = _first(output.get("structure_path"), output.get("final_structure_path"), task.get("structure_path"))
        checksum = _first(output.get("structure_checksum"), task.get("structure_checksum"))
        structure_id = _first(task.get("structure_id"), output.get("structure_id"), task.get("task_id"))
        entry = _lookup(phase_entries, (version, structure_id), (version, checksum), (version, path))
        candidate = _lookup(candidate_entries, structure_id, checksum, path)
        feature = _lookup(feature_entries, structure_id, task.get("task_id"), checksum, path)
        composition = _first(output.get("composition"), task.get("composition"), entry.get("composition"), candidate.get("composition")) or {}
        phase_identification = output.get("phase_identification") or task.get("phase_identification") or {}
        phase = _first(output.get("actual_phase"), task.get("actual_phase"), phase_identification.get("phase"), entry.get("phase"), candidate.get("actual_phase"))
        qbc = output.get("qbc") or task.get("qbc") or candidate.get("qbc") or {}
        energy = _first(output.get("energy_per_atom"), task.get("energy_per_atom"), entry.get("energy_per_atom"), candidate.get("energy_per_atom"))
        if energy is None:
            total = _first(output.get("energy"), task.get("energy"))
            atoms = _first(output.get("atom_count"), task.get("atom_count"))
            if _number(total) is not None and _number(atoms) not in (None, 0):
                energy = float(total) / float(atoms)
        ehull = _first(task.get("relaxed_ehull"), output.get("relaxed_ehull"), entry.get("ehull"), candidate.get("predicted_Ehull"), candidate.get("ehull"))
        features = _first(output.get("m3gnet_features"), output.get("features"), output.get("feature_vector"), feature.get("m3gnet_features"), feature.get("features"))
        monitor = output.get("extrapolation_monitor") or task.get("extrapolation_monitor") or {}
        dh = _first(monitor.get("final_feature_difference"), monitor.get("d_h"), output.get("dh_extrapolation"), task.get("dh_extrapolation"), entry.get("dh_extrapolation"))
        qbc_force = _first(qbc.get("f_std_max"), qbc.get("force_max_atom_disagreement"), qbc.get("force_uncertainty_max"))
        qbc_energy = _first(qbc.get("energy_per_atom_std"), qbc.get("energy_std"), qbc.get("energy_uncertainty"))
        qbc_error = _first(qbc_force, qbc_energy)
        na = composition.get("Na") if isinstance(composition, dict) else None
        oxygen = composition.get("O") if isinstance(composition, dict) else None
        na_per_o2 = None
        if _number(na) is not None and _number(oxygen) not in (None, 0):
            na_per_o2 = float(na) / (float(oxygen) / 2.0)
        key = (version, task.get("task_id") or structure_id, checksum or path or "")
        selected[key] = {
            "epoch": _epoch(state, version),
            "model_version": version or "",
            "task_id": task.get("task_id") or "",
            "task_key": task.get("task_key") or "",
            "stage": task.get("stage") or "",
            "structure_id": structure_id or "",
            "branch_id": task.get("branch_id") or "",
            "composition": _json_cell(composition),
            "phase": phase or "",
            "na_content": na if na is not None else "",
            "na_per_o2": na_per_o2 if na_per_o2 is not None else "",
            "ehull_eV_per_atom": ehull if ehull is not None else "",
            "energy_eV_per_atom": energy if energy is not None else "",
            "qbc_error": qbc_error if qbc_error is not None else "",
            "qbc_energy_std": qbc_energy if qbc_energy is not None else "",
            "qbc_force_std_max": qbc_force if qbc_force is not None else "",
            "m3gnet_features": _json_cell(features),
            "m3gnet_feature_status": "available" if features is not None else "missing",
            "m3gnet_feature_error": "",
            "dh_extrapolation": dh if dh is not None else "",
            "source": _first(task.get("source"), task.get("selection_source"), task.get("generation_source"), task.get("task_key"), task.get("task_id")) or "",
            "structure_path": path or "",
            "structure_checksum": checksum or "",
        }
    stage_order = {"relax_and_feature": 0, "deep_search": 1}
    ordered = sorted(selected.values(), key=lambda row: (str(row["epoch"]), stage_order.get(row["stage"], 9), str(row["task_id"]), str(row["structure_id"])))
    by_epoch = {}
    counters = {}
    for row in ordered:
        epoch = row["epoch"]
        base = str(row["structure_id"] or row["task_id"] or "structure")
        counters[(epoch, base)] = counters.get((epoch, base), 0) + 1
        suffix = counters[(epoch, base)]
        row["name"] = base if suffix == 1 else f"{base}__{suffix:02d}"
        by_epoch.setdefault(epoch, []).append(row)
    return by_epoch


def export_out_data(state, directory, *, action="refresh", epoch=None):
    """Write out_data products from already recorded facts only."""
    root = Path(directory).resolve()
    out = root / "out_data"
    out.mkdir(parents=True, exist_ok=True)
    dft_rows = _dft_feedback_rows(state, root) + _training_rows(state)
    dft_rows.sort(key=lambda row: (str(row.get("epoch")), str(row.get("model_version")), str(row.get("metric_scope")), str(row.get("fold"))))
    dft_path = out / "DFT_err_data.csv"
    _atomic_csv(dft_path, DFT_FIELDS, dft_rows)
    structure_paths = {}
    structure_counts = {}
    structure_rows = _structure_rows(state)
    for epoch, rows in structure_rows.items():
        path = out / f"{epoch}_structs_features.csv"
        _atomic_csv(path, STRUCT_FIELDS, rows)
        structure_paths[epoch] = str(path)
        structure_counts[epoch] = len(rows)
    snapshots = _write_versioned_snapshots(
        out, dft_rows, structure_rows, action=action, epoch=epoch
    )
    return {
        "status": "completed",
        "directory": str(out),
        "dft_error_path": str(dft_path),
        "dft_error_rows": len(dft_rows),
        "structure_paths": structure_paths,
        "structure_counts": structure_counts,
        "missing_feature_rows": sum(1 for rows in structure_rows.values() for row in rows if row.get("m3gnet_feature_status") == "missing"),
        "snapshots": snapshots,
    }

