from collections import Counter
from types import SimpleNamespace
import importlib
import pytest


def test_empty_history_covers_all_five_phases_within_first_five(monkeypatch):
    module = importlib.import_module(
        "phase_agent.science.structures.generate_coverage_branches"
    )
    phases = ["O1", "O3", "OP2", "P2", "P3"]
    manager = SimpleNamespace(data={"branches": {}}, boundary={})
    frameworks = [
        {
            "P": p,
            "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "allowed_x": [str(i) for i in range(30)],
            "det_H": 1,
        }
        for p in phases
    ]
    monkeypatch.setattr(module, "generate_branch_structure", lambda *a, **kw: kw)
    monkeypatch.setattr(
        module,
        "identify_branch_parameters",
        lambda structure, *a, **kw: {
            "P": structure["phase"],
            "H": structure["H"],
            "x": structure["x"],
            "det_H": 1,
            "T": [],
            "composition": {},
        },
    )
    rows = module.generate_coverage_branches(
        manager, frameworks, {}, quota=25, seed=1, preserve_reference_tm=True
    )
    assert {r["P"] for r in rows[:5]} == set(phases)
    assert Counter(r["P"] for r in rows) == dict.fromkeys(phases, 5)
    assert manager.data["branches"] == {}


def test_empty_parent_strategy_is_reported_without_changing_quota(
    monkeypatch, tmp_path
):
    module = importlib.import_module(
        "phase_agent.tools.workflows.run_branch_generation"
    )
    manager = SimpleNamespace(
        data={"branches": {}, "structures": {}}, boundary={"P": ["O1", "O3"]}
    )
    monkeypatch.setattr(module, "propose_branches", lambda *a, **kw: [])
    monkeypatch.setattr(
        module, "deduplicate_candidates", lambda rows, **kw: {"unique_candidates": rows}
    )
    monkeypatch.setattr(module, "estimate_candidate_cost", lambda rows, **kw: rows)
    monkeypatch.setattr(
        module,
        "select_candidates",
        lambda rows, **kw: {"selected_candidates": rows, "rejected_candidates": []},
    )
    monkeypatch.setattr(module, "register_candidate_batch", lambda *a, **kw: [])
    result = module.run_branch_generation(
        manager,
        {},
        structure_directory=tmp_path,
        quotas={"composition": 12},
        batch_size=12,
        initial_states_per_branch=3,
        seed=1,
        generation_plan=[
            {"phase": "all", "strategy": "composition", "quota": 12, "reason": "test"}
        ],
    )
    assert result["summary"]["unfilled_quota"] == 12
    assert (
        result["summary"]["allocation_reports"][0]["reason"]
        == "parent_branches_required"
    )
    assert result["summary"]["phase_coverage"]["missing"] == ["O1", "O3"]


@pytest.mark.parametrize("use_summary", [False, True])
def test_package_preview_blocks_missing_phase_and_allows_focused_batch(tmp_path, use_summary):
    from tests.test_relax_mc_batch import fixture
    from phase_agent.tools.relax_mc_batch.plan import attach_preview

    manager, state, config, context, action = fixture(tmp_path)
    manager.boundary["P"] = ["O3", "P2"]
    state["generation_history"] = [
        {
            "model_version": state.get("active_model_version"),
            "registered_ids": list(manager.data["structures"]),
            "generation_plan": [{"phase": "all", "strategy": "coverage", "quota": 30}],
        }
    ]
    if use_summary:
        state["generation_history"][0]["generation_plan"] = None
        state["generation_history"][0]["summary"] = {"phase_coverage": {"expected": ["O3", "P2"]}}
    with pytest.raises(ValueError, match="generation_phase_coverage_incomplete:P2"):
        attach_preview(action, state, context, config)
    state["generation_history"][0]["registered_ids"] = ["unrelated_previous_batch"]
    assert attach_preview(action, state, context, config)["parameters"][
        "relax_mc_preview"
    ]
