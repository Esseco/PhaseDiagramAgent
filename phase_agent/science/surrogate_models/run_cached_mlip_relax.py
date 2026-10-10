"""Run/reuse MLIP relaxation using a content-addressed cache."""

import hashlib
import json
from pathlib import Path

from phase_agent.science.mlip.run_relax import run_mlip_relax
from phase_agent.science.features.build_cost_record import build_cost_record
from phase_agent.science.structures.boundary_utils import compact_json, load_structure


def run_cached_mlip_relax(
    structure_record: dict, *, relax_config: dict, cache_directory, backend, family_checker=None
) -> dict:
    source = structure_record.get("source_path") or structure_record.get("structure")
    source_id = structure_record.get("structure_id")
    fingerprint = _source_fingerprint(source, source_id)
    identity = {
        "source": fingerprint,
        "model_version": relax_config.get("model_version"),
        "model_path": str(relax_config.get("model_path")),
        "parameters": relax_config.get("parameters") or {},
    }
    cache_key = hashlib.sha256(compact_json(identity).encode()).hexdigest()[:20]
    root = Path(cache_directory) / cache_key
    record_path = root / "relax_record.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("cache_identity") == identity:
            record["cache_hit"] = True
            record["charged_cost"] = 0.0
            record["charged_proxy_cost"] = 0.0
            return record
    if not relax_config.get("model_version") or not relax_config.get("model_path"):
        return {
            "status": "not_configured",
            "cache_key": cache_key,
            "cache_hit": False,
            "error": "model_path/model_version missing",
        }
    initial = load_structure(source)
    result = run_mlip_relax(
        initial,
        backend=backend,
        model_path=relax_config["model_path"],
        parameters=relax_config.get("parameters"),
        work_directory=root,
    )
    relaxed = result.get("structure") or (result.get("outputs") or {}).get("structure")
    relaxed_path = None
    if relaxed is not None and hasattr(relaxed, "to"):
        root.mkdir(parents=True, exist_ok=True)
        relaxed_path = root / "relaxed.vasp"
        relaxed.to(filename=relaxed_path)
    changed = (
        family_checker(initial, relaxed)
        if callable(family_checker) and relaxed is not None
        else None
    )
    outputs = result.get("outputs") or {}
    evaluations = outputs.get("force_evaluation_count", outputs.get("steps"))
    cost_record = build_cost_record(
        atom_count=len(initial),
        evaluation_count=evaluations,
        proxy_config=relax_config.get("cost_model")
        or {"reference_atoms": 1, "atom_exponent": 1, "scale": 1},
        measured_value=result.get("actual_cost"),
        measured_unit=outputs.get("cost_unit"),
    )
    record = {
        "cache_identity": identity,
        "cache_key": cache_key,
        "cache_hit": False,
        "original_structure_id": source_id,
        "relaxed_structure_id": f"SR-{cache_key}",
        "branch_id": structure_record.get("branch_id"),
        "model_version": relax_config.get("model_version"),
        "relax_parameters": relax_config.get("parameters") or {},
        "status": result.get("status"),
        "converged": result.get("converged"),
        "atom_count": len(initial),
        "force_evaluation_count": evaluations,
        "actual_cost": result.get("actual_cost"),
        "charged_cost": result.get("actual_cost"),
        "charged_proxy_cost": cost_record["proxy"].get("value"),
        "cost_record": cost_record,
        "relaxed_structure_path": str(relaxed_path) if relaxed_path else None,
        "structure_family_changed": changed,
        "error": result.get("error"),
    }
    root.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def _source_fingerprint(source, source_id):
    if isinstance(source, (str, Path)) and Path(source).exists():
        return hashlib.sha256(Path(source).read_bytes()).hexdigest()
    return str(source_id or source)
