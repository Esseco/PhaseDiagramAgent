"""检查结构是否满足搜索边界和基本几何条件。"""

from __future__ import annotations

from typing import Any

from Process_AL_MC.structure_check import check_structure

from phase_agent.science.structures.boundary_utils import load_structure
from phase_agent.science.structures.identify_branch import identify_branch_parameters
from phase_agent.science.structures.validate_structure_transition import (
    validate_structure_transition,
)


def run_structure_check(
    structure: Any,
    boundary: dict[str, Any],
    phase_references: dict[str, Any],
    *,
    expected_branch: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行可重复的边界、组成和最小原子距离检查。"""
    config = {"minimum_distance": 0.6}
    config.update(parameters or {})
    try:
        present = load_structure(structure)
        basic = check_structure(present, minimum_distance=config["minimum_distance"])
        if basic["status"] == "invalid":
            return {
                "stage": "simple_check",
                "status": "completed",
                "converged": False,
                "structure": present,
                "outputs": {
                    "checks_passed": False,
                    "failed_checks": basic["failed_checks"],
                    "minimum_distance": basic["minimum_distance"],
                    "structure_check": basic,
                },
                "actual_cost": None,
                "error": None,
            }
        identified = identify_branch_parameters(
            present, boundary, phase_references=phase_references
        )
        transition = validate_structure_transition(
            expected_branch or {}, identified, policy=config.get("structure_transition_policy")
        )
        mismatches = list(transition["constraint_violations"])
        minimum = basic["minimum_distance"]
        passed = not mismatches
        return {
            "stage": "simple_check",
            "status": "completed",
            "converged": passed,
            "structure": present,
            "outputs": {
                "identified_branch": identified,
                "minimum_distance": minimum,
                "checks_passed": passed,
                "failed_checks": mismatches,
                "structure_transition": transition,
                "structure_check": basic,
            },
            "actual_cost": None,
            "error": None,
        }
    except Exception as error:
        return {
            "stage": "simple_check",
            "status": "failed",
            "converged": False,
            "outputs": {},
            "actual_cost": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
