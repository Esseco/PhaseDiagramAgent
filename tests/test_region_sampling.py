from copy import deepcopy

import pytest

from phase_agent.tools.policy.region_sampling import (
    CONTROLS,
    MODE,
    errors,
    policy,
    preview,
)
from phase_agent.tools.workflows.region_controller import sample_regions
from phase_agent.tools.policy.workflow_policy import branch_batch_errors
from phase_agent.decisions.agent.propose_tool_action import _apply_generation_defaults
from phase_agent.decisions.agent.proposal_validation import proposal_errors
from phase_agent.tools.workflows.region_review import write_proposal, write_result
from phase_agent.tools.policy.execution_policy import (
    build_agent_proposal,
    apply_execution_policy,
)

H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def params():
    return {
        "sampling_mode": MODE,
        "regions": [
            {
                "region_id": "r1",
                "priority": 1,
                "strategy": "coverage",
                "phase": "O3",
                "min_det_H": 1,
                "max_det_H": 4,
                "na_min": 0.5,
                "na_max": 1,
                "reason": "探索假设，尚无能量证据",
                "evidence_refs": [],
            }
        ],
    }


def action():
    return {
        "tool": "generate_branches",
        "task_key": "region-test",
        "parameters": params(),
        "budget": 0,
        "target_ids": [],
        "reason": "探索",
        "expected_purpose": "生成后统计",
        "decision_context": {"branch_count_policy": policy()},
    }


class Manager:
    def __init__(self):
        self.data = {"branches": {}, "structures": {}}

    def add_branch(self, **row):
        self.data["branches"][str(len(self.data["branches"]))] = row


def candidate(label, x="1"):
    return {"P": "O3", "H": H, "x": x, "T": [label], "det_H": 1}


def test_no_target_defaults_or_minimum():
    a = _apply_generation_defaults(action(), {}, {"run": {"batch_size": 96, "total_quota": 600}})
    assert not {"total_quota", "quotas", "batch_size", "generation_plan"} & a["parameters"].keys()
    assert a["parameters"]["sampling_controls"] == CONTROLS
    assert branch_batch_errors(a, {}) == []
    assert (
        proposal_errors(
            a,
            {
                "allowed_tools": ["generate_branches"],
                "decision_context": {"branch_count_policy": policy()},
            },
        )
        == []
    )


@pytest.mark.parametrize(
    "field", ["quota", "quotas", "total_quota", "batch_size", "generation_plan"]
)
def test_legacy_count_rejected(field):
    p = params()
    p[field] = 600
    assert any(field in e for e in errors(p))


def test_stop_on_novelty_stagnation_and_preserve_real_ledger():
    manager = Manager()
    before = deepcopy(manager.data)
    count = 0

    def producer(*args, **kwargs):
        nonlocal count
        count += 1
        return [candidate("Fe")]

    chosen, report = sample_regions(
        manager,
        {},
        params(),
        producer=producer,
        descriptor=lambda r: (r["P"], tuple(r["T"])),
    )
    assert len(chosen) == 1  # no minimum100, no fill600
    assert count == 4
    assert report["status"] == "completed"
    assert manager.data == before
    assert report["regions"][0]["duplicates"] == 3


def test_redundant_descriptors_do_not_drive_more_generation():
    manager = Manager()
    count = 0

    def producer(*args, **kwargs):
        nonlocal count
        count += 1
        return [candidate(str(count))]

    chosen, report = sample_regions(
        manager, {}, params(), producer=producer, descriptor=lambda r: "same"
    )
    assert len(chosen) == 1
    assert report["regions"][0]["redundant_descriptor"] == 3


def test_guard_is_pause_and_regions_get_first_pass(monkeypatch):
    p = params()
    p["regions"].append({**p["regions"][0], "region_id": "r2", "priority": 2})
    calls = []

    def producer(*args, **kwargs):
        calls.append(kwargs["strategy_options"]["_focus"]["region_id"])
        return [candidate(f"{len(calls)}-{i}") for i in range(16)]

    chosen, report = sample_regions(
        Manager(), {}, p, producer=producer, descriptor=lambda r: tuple(r["T"])
    )
    assert calls[:2] == ["r1", "r2"]
    assert len(chosen) == CONTROLS["review_after_new_branches"]
    assert report["status"] == "paused_for_review"
    assert "guard" in report["stop_reason"]


def test_empty_regions_explicit_defer():
    p = {"sampling_mode": MODE, "regions": [], "defer_reason": "已有新branch待评价"}
    chosen, report = sample_regions(Manager(), {}, p, descriptor=lambda r: None)
    assert chosen == []
    assert report["attempts"] == 0


def test_review_has_no_quota_and_actual_counts_after_registration(tmp_path):
    a = _apply_generation_defaults(action(), {}, {})
    state = {
        "workspace_layout": {"workspace_root": str(tmp_path)},
        "active_model_version": "m",
        "confirmed_config_version": "c",
    }
    proposal = build_agent_proposal(a, {})
    assert not apply_execution_policy(proposal, execution_mode="interactive")["execute"]
    path = write_proposal(proposal, state)["json"]
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["本次新增branch数"] is None
    assert "配额" not in payload["本次选择的区域"][0]
    manager = Manager()
    manager.add_branch(**candidate("Fe"))
    history = {
        "new_branch_ids": ["0"],
        "summary": {"sampling_report": {"status": "completed"}},
    }
    write_result(history, manager, [{"branch_id": "0", "structure_id": "s"}], state, a)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["本次新增branch数"] == 1
    assert payload["实际分布"][0]["branch数"] == 1
    # Refreshing a proposal cannot overwrite the completed inventory.
    original = Path(path).read_bytes()
    write_proposal(proposal, state)
    assert Path(path).read_bytes() == original


def test_metadata_preview_never_expands(monkeypatch):
    from pymatgen.core import Lattice, Structure
    import phase_agent.science.structures.boundary_utils as utils

    reference = Structure(
        Lattice.cubic(5),
        ["Na", "Fe", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2], [0.8, 0.8, 0.8]],
    )
    monkeypatch.setattr(utils, "load_structure", lambda value: reference)
    monkeypatch.setattr(utils, "allowed_phases", lambda value: {"O3"})
    monkeypatch.setattr(utils, "allowed_phases_at_x", lambda p, x: {"O3"})
    monkeypatch.setattr(utils, "allowed_supercells", lambda h, p: [H])
    monkeypatch.setattr(
        Structure, "make_supercell", lambda *a, **k: pytest.fail("preview expanded")
    )
    manager = Manager()
    manager.boundary = {"P": {}, "H": {}}
    report = preview(params(), manager, {"O3": reference}, {}, {})
    assert report["regions"][0]["legal_cells"] == 1
    assert report["expected_new_branch_count"] is None


def test_changed_review_rejected_before_generator(monkeypatch):
    import phase_agent.tools.workflows.run_region_generation as workflow

    monkeypatch.setattr(workflow, "preview", lambda *a: {"model_version": "new"})
    monkeypatch.setattr(
        workflow, "sample_regions", lambda *a, **k: pytest.fail("generated stale plan")
    )
    a = action()
    a["parameters"]["sampling_controls"] = CONTROLS
    a["parameters"]["region_preview"] = {"model_version": "old"}
    result = workflow.run_region_generation(
        action=a,
        context={
            "effective_config": {},
            "event_state": {},
            "manager": Manager(),
            "phase_references": {},
        },
    )
    assert result["status"] == "rejected"


def test_online_agent_gets_only_region_contract(monkeypatch):
    import phase_agent.decisions.agent.propose_tool_action as module

    seen = []

    def fake(client, payload):
        seen.append(payload)
        return action()

    monkeypatch.setattr(module, "request_validated_action", fake)
    result = module.propose_agent_tool_action(
        {},
        agent_client=object(),
        allowed_tools=["generate_branches"],
        config={"run": {"total_quota": 600, "batch_size": 96}},
    )
    assert result["decision_source"] == "llm_agent"
    assert "generation_allocation_contract" not in seen[0]["decision_context"]
    assert "early_branch_planning" not in seen[0]["decision_context"]
    assert seen[0]["branch_count_policy"]["count_is_result"] is True
    assert "total_quota" not in result["parameters"]


def test_service_failure_cannot_bootstrap_legacy_quota(monkeypatch):
    import phase_agent.decisions.agent.propose_tool_action as module

    def failed(*args, **kwargs):
        raise RuntimeError("503")

    monkeypatch.setattr(module, "request_validated_action", failed)
    result = module.propose_agent_tool_action(
        {}, agent_client=object(), allowed_tools=["generate_branches"]
    )
    assert result["tool"] != "generate_branches"
    assert "503" in result["fallback_reason"]


def test_approved_region_route_counts_registered_only_and_stops(tmp_path, monkeypatch):
    import phase_agent.tools.workflows.run_region_generation as workflow
    import phase_agent.tools.workflows.scientific_module_scope as downstream

    manager = Manager()
    a = _apply_generation_defaults(action(), {}, {})
    a["parameters"]["region_preview"] = {"ledger_fingerprint": "same"}
    monkeypatch.setattr(workflow, "preview", lambda *args: {"ledger_fingerprint": "same"})
    monkeypatch.setattr(
        workflow,
        "sample_regions",
        lambda *args, **kw: (
            [candidate("Fe"), candidate("Mn")],
            {"status": "completed"},
        ),
    )
    monkeypatch.setattr(
        downstream,
        "prepare_generated_module",
        lambda *a: pytest.fail("prepared downstream inputs"),
    )

    def fake_pipeline(mgr, refs, **kw):
        assert len(kw["prepared_candidates"]) == 2
        assert kw["quotas"] == {}
        assert kw["batch_size"] == 2
        assert kw["selection_config"]["max_per_framework"] == 2
        mgr.add_branch(**candidate("Fe"))
        mgr.data["structures"]["s1"] = {}
        return {
            "registered": [{"branch_id": "0", "structure_id": "s1"}],
            "coverage": [],
            "summary": {
                "sampling_report": kw["sampling_report"],
                "initialization_failed_branches": 1,
            },
        }

    monkeypatch.setattr(workflow, "run_branch_generation", fake_pipeline)
    state = {
        "workspace_layout": {"workspace_root": str(tmp_path)},
        "active_model_version": "m",
    }
    result = workflow.run_region_generation(
        action=a,
        context={
            "manager": manager,
            "phase_references": {},
            "event_state": state,
            "effective_config": {"structure_directory": str(tmp_path / "structures")},
        },
    )
    assert result["status"] == "completed"
    assert result["state"]["generation_history"][-1]["new_branch_ids"] == ["0"]
    import json
    from pathlib import Path

    payload = json.loads(Path(result["generation_artifacts"]["json"]).read_text(encoding="utf-8"))
    assert payload["本次新增branch数"] == 1
    assert payload["本次登记结构数"] == 1
