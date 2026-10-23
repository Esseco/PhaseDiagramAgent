from phase_agent.decisions.qbc_selection.select_dft_candidates import select_dft_candidates
from phase_agent.tools.workflows.preview_dft_inputs import preview_dft_inputs
from phase_agent.configuration.defaults.default_dft_decision_config import (
    default_dft_decision_config,
)
from phase_agent.runtime.agent_api import format_workflow_reply


def test_phase_coverage_includes_competing_phase():
    candidates = [
        {
            "candidate_id": f"s{i}",
            "phase": "O3",
            "hull_impact": 1,
            "estimated_cost": 30,
            "x_Na_per_O2": i / 10,
        }
        for i in range(8)
    ]
    candidates += [
        {
            "candidate_id": "p",
            "phase": "P3",
            "hull_impact": 0.1,
            "estimated_cost": 30,
            "qbc": {"status": "not_available_single_model"},
        }
    ]
    result = select_dft_candidates(
        candidates, batch_size=3, seed=0, cover_phases=True, audit_fraction=0, cost_budget=90
    )
    assert {row["phase"] for row in result["selected_candidates"]} == {"O3", "P3"}
    assert result["summary"]["relative_cost"] <= 90


def test_qbc_when_available_affects_rank():
    candidates = [
        {"candidate_id": "high", "hull_impact": 1, "qbc": {"f_std_max": 1.0}},
        {"candidate_id": "low", "hull_impact": 1, "qbc": {"f_std_max": 0.01}},
    ]
    result = select_dft_candidates(candidates, batch_size=1, seed=0, audit_fraction=0)
    assert result["selected_candidates"][0]["candidate_id"] == "high"


def test_legacy_round_cost_cap_is_advisory():
    config = {
        "qbc": default_dft_decision_config(),
        "dft": {"selection": {"single_point_cost_per_round": 10}},
    }
    state = {"qbc_candidates": [{"candidate_id": "s", "atom_count": 40}]}
    preview, error = preview_dft_inputs(
        {
            "parameters": {
                "decisions": [
                    {"candidate_id": "s", "action": "DFT_SINGLE_POINT", "reason": "near hull"}
                ]
            }
        },
        state,
        config,
    )
    assert error is None
    assert preview["task_count"] == 1
    assert preview["relative_cost"] == 30


def test_saved_inventory_keeps_grounded_values_and_chat_short(tmp_path):
    import json
    from pathlib import Path
    from phase_agent.tools.workflows.branch_review_view import raw_record
    from phase_agent.tools.workflows.dft_selection_artifacts import proposal_artifacts

    candidate = {
        "candidate_id": "S-x",
        "x_Na_per_O2": 0.5,
        "phase": "O3",
        "predicted_Ehull": 0.001,
        "ehull_unit": "eV/atom",
    }
    preview = {
        "task_count": 1,
        "single_point_count": 1,
        "relax_count": 0,
        "selected_structures": [{**candidate, "ehull": 0.001}],
        "reasons": ["long explanation"],
    }
    state = {"workspace_layout": {"workspace_root": str(tmp_path)}, "qbc_candidates": [candidate]}
    proposal = {
        "recommended_action": "select_dft_candidates",
        "raw_action": {
            "tool": "select_dft_candidates",
            "task_key": "selection",
            "parameters": {
                "dft_input_preview": preview,
                "decisions": [
                    {"candidate_id": "S-x", "action": "DFT_SINGLE_POINT", "reason": "near hull"}
                ],
            },
        },
    }
    text = format_workflow_reply(
        {"status": "awaiting_approval", "agent_proposal": proposal, "state": state},
        tmp_path / "state.json",
    )
    artifacts = proposal_artifacts(proposal, state)
    assert artifacts["text"] in text and "long explanation" not in text
    saved = raw_record(json.loads(Path(artifacts["json"]).read_text(encoding="utf-8")))
    row = saved["decisions"][0]
    assert (
        row["candidate_id"],
        row["phase"],
        row["Na_fraction"],
        row["Ehull"],
        row["Ehull_unit"],
    ) == ("S-x", "O3", 0.5, 0.001, "eV/atom")
    assert "S-x | O3 | 0.5" in Path(artifacts["text"]).read_text(encoding="utf-8")
