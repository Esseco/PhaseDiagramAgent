"""Validate frozen values on both submitted inputs and returned outputs."""


def validate_frozen_parameters_io(
    *,
    frozen_values: dict,
    submitted_inputs: dict,
    calculated_outputs: dict,
    output_constraint_fields=None,
    reidentify_fields=None,
) -> dict:
    input_violations = [
        key for key, value in frozen_values.items() if submitted_inputs.get(key) != value
    ]
    constraints = set(output_constraint_fields or frozen_values)
    output_violations = [
        key
        for key in constraints
        if key in frozen_values and calculated_outputs.get(key) != frozen_values[key]
    ]
    reidentified = {
        key: calculated_outputs.get(key)
        for key in (reidentify_fields or [])
        if calculated_outputs.get(key) != frozen_values.get(key)
    }
    return {
        "valid": not input_violations and not output_violations,
        "input_violations": input_violations,
        "output_violations": output_violations,
        "reidentified_properties": reidentified,
        "original_frozen_values": dict(frozen_values),
        "actual_outputs": dict(calculated_outputs),
    }
