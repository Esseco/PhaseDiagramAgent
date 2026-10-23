from copy import deepcopy
import json

import pytest

from phase_agent.decisions.agent.transport_tables import (
    compact_evidence_tables,
    expand_evidence_tables,
)


@pytest.mark.parametrize(
    "field",
    [
        "existing_branches",
        "available_allocations",
        "lowest_ehull_entries",
        "recent_action_history",
        "selection_purposes",
    ],
)
def test_branch_and_related_scientific_rows_are_losslessly_encoded(field):
    rows = [
        {
            "branch_index": i,
            "branch_id": f"B-{i:012d}",
            "model_version": "m2",
            "ehull": i / 1000,
            "near_degenerate": i < 3,
            "result_record_ids": [f"r-{i}"],
            "mc_task_states": {"old:completed": 1},
            "phase": "O3",
        }
        for i in range(24)
    ]
    rows[-1].pop("ehull")
    rows[-2]["ehull"] = None
    source = {"decision_context": {"branch_pool_facts": {field: rows}}}
    before = deepcopy(source)
    encoded = compact_evidence_tables(source)
    assert expand_evidence_tables(encoded) == source == before
    assert len(json.dumps(encoded)) < len(json.dumps(source)) * 0.65


def test_small_report_tables_keep_json_types_and_exact_values():
    from phase_agent.decisions.agent.transport_tables import (
        compact_evidence_tables,
        expand_evidence_tables,
    )
    import json

    reports = [
        {"id": i, "measured": value, "explanation": "unchanged scientific observation " * 20}
        for i, value in enumerate([True, 1, 1.0, 0.12345678901234567])
    ]
    source = {"reports": reports}
    encoded = compact_evidence_tables(source)
    assert len(json.dumps(encoded)) < len(json.dumps(source))
    assert json.dumps(expand_evidence_tables(encoded), sort_keys=True) == json.dumps(
        source, sort_keys=True
    )
