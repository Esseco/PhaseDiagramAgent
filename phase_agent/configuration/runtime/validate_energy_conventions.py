"""Validate explicit energy/reference units without judging source reliability."""


def validate_energy_conventions(config):
    conventions = dict(config or {})
    errors = []
    if conventions.get("oxygen_reference_unit") != "eV/O2":
        errors.append("oxygen_reference_unit_must_be_eV/O2")
    if conventions.get("model_error_unit") != "eV/atom":
        errors.append("model_error_unit_must_be_eV/atom")
    references = {}
    for name, row in (conventions.get("voltage_references") or {}).items():
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("value"), (int, float))
            or not row.get("source")
        ):
            errors.append(f"voltage_reference_requires_value_and_source:{name}")
            continue
        references[name] = {
            "value": float(row["value"]),
            "unit": row.get("unit", "eV/atom"),
            "source": str(row["source"]),
            "reliability_reviewed_by_program": False,
        }
    return {
        "valid": not errors,
        "errors": errors,
        "oxygen_reference_unit": "eV/O2",
        "model_error_unit": "eV/atom",
        "voltage_references": references,
    }
