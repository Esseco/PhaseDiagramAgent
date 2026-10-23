"""Validate result values before mutating scientific ledgers; no model evaluation."""

from math import isfinite, isclose
from numbers import Real


class InvalidScientificResult(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def validate_scientific_result(result):
    if not isinstance(result, dict):
        raise InvalidScientificResult("result_object_required")
    for key in (
        "task_id",
        "task_key",
        "structure_id",
        "stage",
        "status",
        "model_version",
        "config_version",
    ):
        if result.get(key) is not None and not isinstance(result[key], str):
            raise InvalidScientificResult("result_identifier_required:" + key)
    for key in (
        "runtime_observation",
        "parameters",
        "worker_job",
        "job_accounting",
        "scheduler_accounting",
    ):
        if result.get(key) is not None and not isinstance(result[key], dict):
            raise InvalidScientificResult("result_object_required:" + key)
    for key in ("actual_mc_steps", "requested_max_mc_steps", "max_mc_steps"):
        value = result.get(key)
        if value is not None:
            try:
                valid = not isinstance(value, bool) and isfinite(float(value)) and float(value) >= 0
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise InvalidScientificResult("result_nonnegative_steps_required:" + key)
    for key in ("actual_cost", "estimated_cost"):
        value = result.get(key)
        if isinstance(value, Real) and (not isfinite(value) or value < 0):
            raise InvalidScientificResult("result_finite_nonnegative_cost_required:" + key)
    outputs = result.get("outputs")
    if outputs is not None and not isinstance(outputs, dict):
        raise InvalidScientificResult("result_outputs_object_required")
    outputs = outputs or {}
    for key in (
        "phase_identification",
        "qbc",
        "spin_state_check",
        "magnetic_check",
    ):
        if outputs.get(key) is not None and not isinstance(outputs[key], dict):
            raise InvalidScientificResult("result_output_object_required:" + key)
    summary = outputs.get("pool_summary")
    if summary is not None and (
        not isinstance(summary, list) or any(not isinstance(row, dict) for row in summary)
    ):
        raise InvalidScientificResult("result_output_object_list_required:pool_summary")
    for key in ("energy", "energy_per_atom"):
        value = outputs.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
        ):
            raise InvalidScientificResult("result_finite_" + key + "_required")
    if (
        outputs.get("energy") is not None
        and result.get("stage")
        in {"relax_and_feature", "deep_search", "dft_single_point", "dft_relax"}
        and outputs.get("energy_unit") != "eV"
    ):
        raise InvalidScientificResult("result_energy_unit_eV_required")
    composition = outputs.get("composition")
    if composition is not None:
        if (
            not isinstance(composition, dict)
            or not composition
            or any(
                isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) or v < 0
                for v in composition.values()
            )
            or not any(v > 0 for v in composition.values())
        ):
            raise InvalidScientificResult("result_valid_composition_required")
        from pymatgen.core import Element

        try:
            for symbol in composition:
                Element(symbol)
        except (ValueError, TypeError, KeyError):
            raise InvalidScientificResult("result_valid_elements_required") from None
        if (
            outputs.get("energy") is not None
            and outputs.get("energy_per_atom") is not None
            and not isclose(
                float(outputs["energy"]) / sum(composition.values()),
                float(outputs["energy_per_atom"]),
                abs_tol=1e-6,
                rel_tol=1e-8,
            )
        ):
            raise InvalidScientificResult("result_energy_per_atom_mismatch")


def validate_result_binding(result, prior):
    validate_scientific_result(result)
    for key in ("task_key", "structure_id", "stage", "config_version", "model_version"):
        if prior.get(key) is not None and key in result and prior[key] != result[key]:
            raise InvalidScientificResult("result_binding_mismatch:" + key)
    validate_scientific_result({**prior, **result})
