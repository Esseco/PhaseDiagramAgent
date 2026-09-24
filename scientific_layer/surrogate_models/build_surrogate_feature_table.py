"""Join fixed dataset splits with pre-search relaxed features."""

from scientific_layer.structures.boundary_utils import load_structure
from scientific_layer.surrogate_models.extract_mlip_embedding import extract_mlip_embedding
from scientific_layer.surrogate_models.extract_registered_features import extract_registered_features
from scientific_layer.surrogate_models.run_cached_mlip_relax import run_cached_mlip_relax


def build_surrogate_feature_table(dataset: dict, *, config: dict, registry: dict, relax_backend, cache_directory, family_checker=None) -> dict:
    rows = []
    for branch_row in dataset.get("rows", []):
        records = []
        for initial in branch_row.get("initial_structures", []):
            item = {**initial, "branch_id": branch_row["branch_id"]}
            relax = run_cached_mlip_relax(item, relax_config=config["relax"], cache_directory=cache_directory, backend=relax_backend, family_checker=family_checker)
            manual = {"status": "not_run", "values": {}, "definitions": {}, "failures": {}}
            embedding = {"status": "not_run", "values": None}
            if relax.get("status") == "completed" and relax.get("converged") is True and relax.get("relaxed_structure_path"):
                structure = load_structure(relax["relaxed_structure_path"])
                manual = extract_registered_features(structure, registry=registry, enabled=config.get("enabled_features", []), context={"branch": branch_row["branch"]})
                embedding = extract_mlip_embedding(structure, config=config.get("embedding") or {}, context={"branch": branch_row["branch"]})
            records.append({"initial": initial, "relax": relax, "manual": manual, "embedding": embedding})
        rows.append({"branch_id": branch_row["branch_id"], "split": branch_row.get("split"), "split_group": branch_row.get("split_group"), "target": branch_row.get("target"), "structure_records": records})
    return {"schema_version": 1, "dataset_task": dataset.get("task"), "feature_set_version": config.get("feature_set_version"), "rows": rows}
