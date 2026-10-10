"""Keep measured and proxy costs separate instead of inventing conversions."""

from phase_agent.science.features.calculate_size_aware_cost import calculate_size_aware_cost


def build_cost_record(
    *,
    atom_count,
    evaluation_count,
    proxy_config: dict,
    measured_value=None,
    measured_unit=None,
    initial_state_count=1,
) -> dict:
    measured = {"status": "unknown", "value": None, "unit": measured_unit}
    if measured_value is not None:
        measured = {
            "status": "measured",
            "value": float(measured_value),
            "unit": measured_unit or "unspecified",
        }
    return {
        "measured": measured,
        "proxy": calculate_size_aware_cost(
            atom_count=atom_count,
            evaluation_count=evaluation_count,
            initial_state_count=initial_state_count,
            config=proxy_config,
        ),
    }
