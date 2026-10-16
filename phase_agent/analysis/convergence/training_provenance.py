"""Recover legacy training identities from matching, retained training artifacts."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path


def model_with_training_provenance(state, model, *, _seen=None):
    """Return a derived model copy; never amend historical model or approval records."""
    current = deepcopy(model)
    if "training_data_ids" in current and "training_structure_hashes" in current:
        return current
    if not current.get("dataset_version"):
        return current
    version = current.get("version")
    seen = set(_seen or ())
    if not version or version in seen or len(seen) >= 8:
        return current
    seen.add(version)
    from phase_agent.analysis.convergence.independent_evidence import (
        geometry_key,
        training_snapshot,
    )

    matches = []
    for key, job in (state.get("remote_finetune_jobs") or {}).items():
        result = job.get("returned_results") or {}
        handoff = job.get("training_handoff") or {}
        fingerprint = result.get("fingerprint") or ""
        if (
            job.get("status") == "abandoned"
            or len(fingerprint) != 64
            or result.get("issues")
            or handoff.get("candidate_model_version") != version
            or version != f"remote-{key}-{fingerprint[:12]}"
        ):
            continue
        main = [m for m in result.get("models") or [] if m.get("is_main_model") is True]
        if (
            len(main) != 1
            or len(current.get("sha256") or "") != 64
            or main[0].get("sha256") != current["sha256"]
        ):
            continue
        report = job.get("report") or {}
        if report.get("training_round") != current["dataset_version"]:
            continue
        directory = job.get("directory")
        if not directory:
            continue
        path = Path(directory) / "inputs" / "training_plan.json"
        try:
            raw = path.read_bytes()
            plan = json.loads(raw.decode("utf-8-sig"))
        except (OSError, UnicodeError, ValueError):
            continue
        scope = plan.get("dataset_scope") or {}
        ids = scope.get("data_ids") or []
        if (
            scope != report.get("dataset_scope")
            or not isinstance(ids, list)
            or not ids
            or any(not isinstance(i, str) for i in ids)
            or len(ids) != len(set(ids))
            or scope.get("cumulative_records") != len(ids)
            or plan.get("original_model_version") != job.get("original_model_version")
            or plan.get("training_round") != current["dataset_version"]
        ):
            continue
        selected = {}
        selected_geometry = {}
        id_set = set(ids)
        ambiguous = False
        for row in [
            *(state.get("dft_training_records") or []),
            *(state.get("dft_dataset_records") or []),
        ]:
            identity = row.get("data_id")
            if identity not in id_set:
                continue
            geometry = geometry_key(row)
            if not geometry or row.get("checks_passed") is not True:
                ambiguous = True
                break
            if identity in selected_geometry and selected_geometry[identity] != geometry:
                ambiguous = True
                break
            selected[identity] = row
            selected_geometry[identity] = geometry
        if ambiguous or set(selected) != set(ids):
            continue
        base_version = job.get("original_model_version")
        base = ((state.get("model_registry") or {}).get(base_version) or {}).get("model")
        if not base:
            continue
        inherited = model_with_training_provenance(state, base, _seen=seen)
        if (
            inherited.get("dataset_version")
            and inherited.get("training_provenance_complete") is not True
        ):
            continue
        if not inherited.get("dataset_version"):
            config = state.get("confirmed_config") or {}
            initial = config.get("mlip") or {}
            foundation_versions = {
                initial.get("version"),
                initial.get("name"),
                (config.get("calculation") or {}).get("mlip_version"),
            }
            if base_version not in foundation_versions:
                continue
        snapshot = training_snapshot(list(selected.values()), inherited)
        if snapshot["training_provenance_complete"]:
            matches.append(
                {
                    **snapshot,
                    "training_provenance_source": {
                        "job_key": key,
                        "training_plan_sha256": hashlib.sha256(raw).hexdigest(),
                        "model_sha256": current["sha256"],
                        "method": "matched_legacy_training_plan",
                    },
                }
            )
    if len(matches) == 1:
        current.update(matches[0])
    return current
