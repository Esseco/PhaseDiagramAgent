"""Build comparable method records on one target and one fixed split."""


def compare_surrogate_methods(feature_table: dict, *, config: dict) -> dict:
    methods = {}
    for name in config.get("methods", []):
        if name == "mattertune":
            from scientific_layer.surrogate_models.create_mattertune_surrogate import create_mattertune_surrogate
            methods[name] = create_mattertune_surrogate(config.get("mattertune") or {})
            continue
        required = ["manual"] if name == "manual_rf" else ["embedding"] if name == "embedding_rf" else ["manual", "embedding"]
        usable = 0
        relax_cost = 0.0
        for row in feature_table["rows"]:
            for record in row["structure_records"]:
                relax_cost += record["relax"].get("charged_cost") or 0.0
                if all(record[key].get("status") == "completed" for key in required):
                    usable += 1
        methods[name] = {"status": "ready" if usable else "not_configured", "usable_structure_count": usable, "cost": {"relax": relax_cost, "feature": None, "model_fit": None}, "note": "unknown cost fields are not imputed"}
    return {"dataset_task": feature_table.get("dataset_task"), "feature_set_version": feature_table.get("feature_set_version"), "methods": methods}
