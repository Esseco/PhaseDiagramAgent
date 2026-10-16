"""Content-addressed DFT labels; diagram state and MLIP never enter the store."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

DFT_STAGES = {"dft_single_point", "dft_relax"}
LOCAL_OUTPUTS = {
    "remote_mlip_prediction",
    "mlip_result_file",
    "mlip_result_checksum",
    "ehull",
    "hull_reference_version",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def geometry(value):
    from pymatgen.core import Structure

    structure = (
        Structure.from_file(str(value))
        if isinstance(value, (str, Path))
        else Structure.from_dict(value)
    )
    # Retain atom ordering and magnetic site properties; reuse is conservative.
    return {
        "lattice": structure.lattice.matrix.round(10).tolist(),
        "species": [str(s.species) for s in structure],
        "fractional": structure.frac_coords.round(10).tolist(),
        "properties": structure.site_properties,
    }


def settings(state, config=None):
    options = {**(state.get("shared_dft") or {}), **((config or {}).get("shared_dft") or {})}
    if not options or options.get("enabled") is not True or not options.get("source_projects"):
        return None
    return options


@contextmanager
def database(options):
    root = Path(options["directory"])
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root / "records.sqlite3", timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, system TEXT NOT NULL, protocol TEXT NOT NULL, stage TEXT NOT NULL, input_geometry TEXT, parameters TEXT NOT NULL, payload TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS lookup ON records(system,protocol,stage,input_geometry,parameters)"
    )
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def identity(config):
    ratios = (config.get("system") or {}).get("boundary", {}).get("TM_ratio") or {}
    total = sum(float(v) for v in ratios.values())
    if not total:
        raise ValueError("共享 DFT 需要明确体系 TM_ratio")
    return digest({k: float(v) / total for k, v in sorted(ratios.items())})


def protocol(config):
    return digest(config.get("dft") or {})


def publish(state, manager, config):
    options = settings(state, config)
    if not options:
        return 0
    from phase_agent.science.dft.spin_acceptance import spin_standard_passed

    by_task = {}
    for frame in state.get("dft_training_records", []):
        if (
            frame.get("checks_passed") is True
            and spin_standard_passed(frame)
            and frame.get("structure")
        ):
            by_task.setdefault(frame.get("task_id"), []).append(frame)
    phases = {
        r.get("source_task_id"): r
        for r in state.get("phase_records", [])
        if r.get("energy_method") == "dft"
    }
    inserted = 0
    with database(options) as db:
        for task in state.get("tasks", []):
            task_id = task.get("task_id")
            if (
                task.get("stage") not in DFT_STAGES
                or task.get("status") != "completed"
                or task.get("checks_passed") is not True
                or task.get("shared_dft_record_id")
            ):
                continue
            frames = by_task.get(task_id)
            if not frames:
                continue
            structure = manager.data.get("structures", {}).get(task.get("structure_id"), {})
            branch = manager.data.get("branches", {}).get(
                task.get("branch_id") or structure.get("branch_id"), {}
            )
            source = task.get("structure_path") or structure.get("source_path")
            input_geometry = None
            if source and Path(source).is_file():
                input_geometry = digest(geometry(source))
            outputs = {
                k: deepcopy(v)
                for k, v in (task.get("outputs") or {}).items()
                if k not in LOCAL_OUTPUTS
            }
            # Remote files belong to the producer; consumers use embedded labels only.
            for key in ("source_vasprun", "training_file", "training_checksum", "source_phase"):
                outputs.pop(key, None)
            payload = {
                "source_diagram": options["diagram_id"],
                "source_task_id": task_id,
                "branch": {k: deepcopy(branch.get(k)) for k in ("P", "H", "x")},
                "outputs": _pure_label(outputs),
                "frames": [_pure_label(f) for f in frames],
                "phase_record": _pure_label(phases[task_id]) if phases.get(task_id) else None,
                "converged": task.get("converged"),
            }
            record_id = digest(
                {
                    "system": identity(config),
                    "protocol": protocol(config),
                    "stage": task["stage"],
                    "input": input_geometry,
                    "parameters": task.get("parameters") or {},
                    "frames": [
                        {k: f.get(k) for k in ("structure", "energy", "forces", "stress")}
                        for f in frames
                    ],
                }
            )
            before = db.total_changes
            db.execute(
                "INSERT OR IGNORE INTO records VALUES (?,?,?,?,?,?,?)",
                (
                    record_id,
                    identity(config),
                    protocol(config),
                    task["stage"],
                    input_geometry,
                    digest(task.get("parameters") or {}),
                    json.dumps(payload, sort_keys=True, allow_nan=False),
                ),
            )
            inserted += db.total_changes - before
    return inserted


def in_boundary(payload, config):
    from phase_agent.science.structures.boundary_utils import (
        allowed_phases_at_x,
        allowed_supercells,
        normalize_H,
    )

    boundary = config["system"]["boundary"]
    branch = payload["branch"]
    frame = payload["frames"][-1]
    composition = frame.get("composition") or {}
    tm = boundary["TM_ratio"]
    denominator = sum(float(composition.get(k, 0)) for k in tm)
    if not denominator:
        return False
    expected = sum(float(v) for v in tm.values())
    if any(
        abs(float(composition.get(k, 0)) / denominator - float(v) / expected) > 1e-8
        for k, v in tm.items()
    ):
        return False
    x = float(composition.get("Na", 0)) / denominator
    phase = frame.get("actual_phase") or branch.get("P")
    if not phase or phase.upper() not in allowed_phases_at_x(boundary["P"], x):
        return False
    bounds = boundary.get("x_range")
    if bounds and not float(bounds[0]) <= x <= float(bounds[1]):
        return False
    if branch.get("H") is None:
        return False  # Unknown legacy scope is preserved, but not silently imported.
    return normalize_H(branch["H"]) in [
        normalize_H(h) for h in allowed_supercells(boundary["H"], phase)
    ]


def select(state, config, *, available=False):
    options = settings(state, config)
    if not options:
        return []
    path = Path(options["directory"]) / "records.sqlite3"
    if not path.is_file():
        return []
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT id,payload FROM records WHERE system=? AND protocol=? ORDER BY id",
            (identity(config), protocol(config)),
        ).fetchall()
    snapshot = state.get("shared_dft_snapshot") or {}
    ids = set(snapshot.get("record_ids") or [])
    sources = set(options["source_projects"])
    if not available and set(snapshot.get("source_projects") or []) != sources:
        return []
    selected = []
    for key, raw in rows:
        payload = json.loads(raw)
        if (
            payload["source_diagram"] in sources
            and (available or key in ids)
            and in_boundary(payload, config)
        ):
            selected.append((key, payload))
    return selected


def capture_snapshot(state, config):
    options = settings(state, config)
    if not options:
        return None
    value = {
        "source_projects": sorted(set(options["source_projects"])),
        "record_ids": [key for key, _ in select(state, config, available=True)],
    }
    value["fingerprint"] = digest(value)
    return value


def accept_snapshot(state, config, snapshot):
    """Called only after the existing approval gate; never expands saved IDs."""
    if snapshot is None:
        return
    options = settings(state, config)
    if not options or sorted(set(options["source_projects"])) != snapshot["source_projects"]:
        raise ValueError("shared_dft_sources_changed_since_approval")
    expected = digest({key: snapshot[key] for key in ("source_projects", "record_ids")})
    if expected != snapshot.get("fingerprint"):
        raise ValueError("shared_dft_snapshot_identity_mismatch")
    state["shared_dft_snapshot"] = deepcopy(snapshot)
    import_records(state, config)


def _pure_label(value):
    row = deepcopy(value)
    for key in (
        "model_version",
        "source_version",
        "remote_mlip_prediction",
        "mlip_result_file",
        "mlip_result_checksum",
        "ehull",
        "hull_reference_version",
        "round_scope",
        "upload_operation_id",
        "batch_id",
        "parent_relax_round",
        "search_group_index",
        "source_vasprun",
        "training_file",
        "training_checksum",
    ):
        row.pop(key, None)
    if row.get("structure"):
        for key in ("structure_path", "final_structure_path"):
            row.pop(key, None)
    return row


def import_records(state, config):
    options = settings(state, config)
    if not options:
        return 0
    count = 0
    known = {r.get("data_id") for r in state.get("dft_training_records", [])}
    used = state.setdefault("shared_dft_data_ids", [])
    phase_ids = {r.get("record_id") for r in state.get("phase_records", [])}
    for key, payload in select(state, config):
        if payload["source_diagram"] == options["diagram_id"]:
            continue
        for index, original in enumerate(payload["frames"]):
            data_id = f"shared-dft:{key}:frame:{index}"
            if data_id in known:
                continue
            frame = _pure_label(original)
            for local in (
                "model_version",
                "round_scope",
                "upload_operation_id",
                "batch_id",
                "parent_relax_round",
                "search_group_index",
            ):
                frame.pop(local, None)
            frame.update(
                data_id=data_id,
                task_id=f"shared-dft:{key}",
                branch_id=f"shared-branch:{digest(payload['branch'])}",
                shared_dft_record_id=key,
                shared_dft_source=payload["source_diagram"],
            )
            frame.update(
                shared_dft_source_task_id=payload["source_task_id"], dft_data_role="reference"
            )
            state.setdefault("dft_training_records", []).append(frame)
            state.setdefault("new_dft_records", []).append(deepcopy(frame))
            known.add(data_id)
            count += 1
        phase = _pure_label(payload["phase_record"]) if payload.get("phase_record") else None
        if phase and f"shared-dft:{key}" not in phase_ids:
            phase.update(
                record_id=f"shared-dft:{key}",
                structure_id=f"shared-dft:{key}",
                source_task_id=f"shared-dft:{key}",
                branch_id=None,
                shared_dft_record_id=key,
            )
            phase.update(
                shared_dft_source=payload["source_diagram"],
                shared_dft_source_task_id=payload["source_task_id"],
                dft_data_role="reference",
            )
            phase.pop("model_version", None)
            state.setdefault("phase_records", []).append(phase)
        if key not in used:
            used.append(key)
    state["shared_dft_dataset_version"] = digest(sorted(used))
    return count


def reusable_results(state, manager, config, *, task_ids=None):
    options = settings(state, config)
    if not options:
        return []
    eligible = dict(select(state, config))
    results = []
    with database(options) as db:
        for task in state.get("tasks", []):
            if (
                task.get("stage") not in DFT_STAGES
                or task.get("status") != "pending"
                or task.get("slurm_batch_id")
                or (task_ids is not None and task.get("task_id") not in task_ids)
            ):
                continue
            row = manager.data.get("structures", {}).get(task.get("structure_id"), {})
            source = task.get("structure_path") or row.get("source_path")
            if not source or not Path(source).is_file():
                continue
            matches = db.execute(
                "SELECT id FROM records WHERE system=? AND protocol=? AND stage=? AND input_geometry=? AND parameters=? ORDER BY id",
                (
                    identity(config),
                    protocol(config),
                    task["stage"],
                    digest(geometry(source)),
                    digest(task.get("parameters") or {}),
                ),
            ).fetchall()
            matches = [key for (key,) in matches if key in eligible]
            if len(matches) != 1:
                continue  # Conflicting labels require an explicit scientific decision.
            key = matches[0]
            payload = eligible[key]
            outputs = deepcopy(payload["outputs"])
            result = {
                **deepcopy(task),
                "status": "completed",
                "checks_passed": True,
                "converged": payload["converged"],
                "actual_cost": 0.0,
                "actual_gpu_core_hours": 0.0,
                "outputs": outputs,
                "shared_dft_record_id": key,
                "shared_dft_source": payload["source_diagram"],
            }
            result.update(
                shared_dft_source_task_id=payload["source_task_id"], dft_data_role="reused_result"
            )
            results.append(result)
    return results
