"""Training provenance and geometric validation identities, never inferred errors."""

import hashlib
import json


def geometry_key(record):
    if record.get("structure"):
        from phase_agent.persistence.shared_dft import geometry

        try:
            payload = geometry(record["structure"])
        except (ValueError, TypeError, KeyError, OSError):
            return None
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return record.get("structure_sha256")


def training_snapshot(records, base=None):
    """Preserve inherited training identities when finetuning a previous model."""
    base = base or {}
    return {
        "training_provenance_complete": base.get(
            "training_provenance_complete", not base.get("dataset_version")
        )
        and all(geometry_key(r) for r in records),
        "training_data_ids": sorted(
            set(base.get("training_data_ids") or [])
            | {r["data_id"] for r in records if r.get("data_id")}
        ),
        "training_structure_hashes": sorted(
            set(base.get("training_structure_hashes") or [])
            | {key for r in records if (key := geometry_key(r))}
        ),
    }


def independent_records(state, version):
    model = (state.get("model_registry", {}).get(version) or {}).get("model") or {}
    if (state.get("active_model") or {}).get("version") == version:
        model = state["active_model"]
    from phase_agent.analysis.convergence.training_provenance import model_with_training_provenance

    model = model_with_training_provenance(state, model)
    known = "training_data_ids" in model and "training_structure_hashes" in model
    if (model.get("dataset_version") or state.get("last_trained_dataset_version")) and (
        not known or model.get("training_provenance_complete") is False
    ):
        return {"status": "missing_training_provenance", "records": [], "excluded": []}
    trained_ids = set(model.get("training_data_ids") or [])
    trained_geometry = set(model.get("training_structure_hashes") or [])
    selected, excluded = [], []
    for row in state.get("dft_dataset_records") or []:
        if row.get("model_version") != version or row.get("stage") != "dft_single_point":
            continue
        key = geometry_key(row)
        if (
            row.get("data_id") in trained_ids
            or key in trained_geometry
            or row.get("dft_data_role") == "reference"
            or not key
        ):
            excluded.append(row.get("task_id"))
        else:
            selected.append(row)
    return {"status": "completed", "records": selected, "excluded": excluded}


def ground_state_validation(state, version):
    from phase_agent.analysis.feedback.dft_comparison_tables import build_comparison_tables

    snapshot = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if snapshot.get("model_version") != version or snapshot.get("status") != "completed":
        return {"fraction": None, "validated": 0, "total": 0}
    stable = {geometry_key(r): r for r in snapshot.get("entries") or [] if r.get("is_stable")}
    # Missing geometry cannot silently shrink the denominator.
    total = (
        len(stable)
        if None not in stable
        else sum(bool(r.get("is_stable")) for r in snapshot.get("entries", []))
    )
    records = [r for r in state.get("dft_dataset_records") or [] if r.get("checks_passed") is True]
    energies, _ = build_comparison_tables(
        records, state.get("dft_mlip_comparisons") or [], round_name="ground_state"
    )
    valid = {r["task_id"] for r in energies if r["comparison_status"] == "completed"}
    checked = {geometry_key(r) for r in records if r.get("task_id") in valid}
    count = len((set(stable) - {None}) & checked)
    return {"fraction": count / total if total else None, "validated": count, "total": total}
