"""Join current-model results to branch IDs; report facts without choosing actions."""

from collections import Counter, defaultdict
from fractions import Fraction
import math

from phase_agent.tools.state.retired_generations import retired_branch_ids


def existing_branch_evidence(state, branches=None):
    if branches is None:
        branches = {
            row["branch_id"]: row
            for row in state.get("branch_candidates") or state.get("available_branches") or []
        }
    retired = retired_branch_ids(state)
    protected_gap = ((state.get("confirmed_config") or {}).get("convergence") or {}).get(
        "phase_competition_gap_ev_per_atom", 0.003
    )
    if protected_gap is None:
        protected_gap = 0.003
    version = state.get("active_model_version")
    tasks = state.get("tasks") or []
    by_task = {row.get("task_id"): row for row in tasks}
    branch_by_structure = {
        row.get("structure_id"): row.get("branch_id") for row in tasks if row.get("branch_id")
    }
    branch_by_record = {
        row.get("record_id"): (by_task.get(row.get("source_task_id")) or {}).get("branch_id")
        for row in state.get("phase_records") or []
    }
    diagram = (state.get("phase_diagrams") or {}).get("mlip") or {}
    valid_diagram = diagram.get("status") == "completed" and diagram.get("model_version") == version
    entries = defaultdict(list)
    if valid_diagram:
        for row in diagram.get("entries") or []:
            gap = row.get("ehull")
            if (
                row.get("source_version") != version
                or row.get("phase_identification_status") != "identified"
                or row.get("ehull_unit") != "eV/atom"
                or isinstance(gap, bool)
                or not isinstance(gap, (int, float))
                or not math.isfinite(gap)
                or gap < -1e-8
            ):
                continue
            bid = branch_by_record.get(row.get("record_id")) or branch_by_structure.get(
                row.get("structure_id")
            )
            if bid:
                entries[bid].append(row)
    mc = defaultdict(Counter)
    for row in tasks:
        if row.get("stage") == "deep_search" and row.get("branch_id"):
            key = "current_model" if row.get("model_version") == version else "other_model"
            mc[row["branch_id"]][key + ":" + str(row.get("status"))] += 1
    output = []
    for bid, branch in sorted(branches.items()):
        if bid in retired:
            continue
        matches = entries.get(bid) or []
        best = min(matches, key=lambda row: row["ehull"]) if matches else {}
        row = {
            "branch_id": bid,
            "phase": branch.get("P"),
            "x": str(branch.get("x")),
            "det_H": branch.get("det_H"),
            "current_model_result_count": len(matches),
            "current_model_ehull_ev_per_atom": best.get("ehull"),
            "result_phase": best.get("phase"),
            "is_hull_ground_state": any(item.get("is_stable") for item in matches),
            "near_degenerate": any(item["ehull"] <= float(protected_gap) for item in matches),
            "result_record_ids": [item["record_id"] for item in matches],
            "mc_task_states": dict(mc[bid]),
        }
        try:
            row["has_mobile_ions"] = Fraction(str(branch.get("x"))) > 0
        except (ValueError, ZeroDivisionError):
            row["has_mobile_ions"] = None
        output.append(row)
    return output
