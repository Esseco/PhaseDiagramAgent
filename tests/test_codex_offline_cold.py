from copy import deepcopy

import pytest

from ase import Atoms
from ase.io import write

from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)
from phase_agent.configuration.session.confirm_config_snapshot import confirm_config_snapshot
from phase_agent.configuration.session.create_config_draft import create_config_draft
from phase_agent.configuration.session.resolve_workspace_paths import default_workspace_storage
from phase_agent.configuration.session.save_config_session import save_config_session
from phase_agent.runtime.codex.offline import CodexOffline
from phase_agent.runtime.local_project_launcher import create_project


@pytest.mark.parametrize(
    "native_team,auto_authorized", [(False, False), (True, False), (True, True)]
)
def test_cold_project_codex_proposal_is_registered_without_project_model(
    tmp_path, native_team, auto_authorized
):
    root = tmp_path / "project"
    runtime = create_project(root, registry=tmp_path / "registry.json")
    ref = root / "structures/reference_structures/O3.vasp"
    write(
        ref,
        Atoms(
            "NaFeO2",
            positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
            cell=[6, 6, 6],
            pbc=True,
        ),
        format="vasp",
    )
    boundary = {
        "P": ["O3"],
        "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]},
        "TM_ratio": {"Fe": 1},
    }
    cfg = default_layered_search_config(boundary=boundary, phase_references={"O3": str(ref)})
    cfg["storage"] = default_workspace_storage(root)
    cfg["mlip"]["model_path"] = "/fake/model"
    cfg["calculation"]["mlip_version"] = "m0"
    cfg["mlip"]["name"] = "m0"
    if auto_authorized:
        cfg["workflow"]["auto_approved_operations"] = ["generate_branches"]
        cfg["generation_actions"]["first_round_max_det_H"] = 1
    session = confirm_config_snapshot(create_config_draft(cfg), user_confirmed=True)
    save_config_session(session, root / "parameters/config_session.json")
    agent = CodexOffline(runtime)

    def forbidden(*args, **kwargs):
        raise AssertionError("Project model must not be called")

    agent.handler.workflow_kwargs["agent_client"] = forbidden
    context = agent.context()
    assert context["config_version"] == session["confirmed_snapshot"]["config_version"]
    assert context["model_version"] is None
    assert context["decision_context"]["initial_coverage"]["allowed_strategies"] == ["coverage"]
    action = {
        "tool": "generate_branches",
        "task_key": "cold-first",
        "target_ids": [],
        "budget": 0,
        "reason": "初始覆盖",
        "expected_purpose": "生成初始候选",
        "parameters": {
            "sampling_mode": "region_adaptive",
            "initial_states_per_branch": 1,
            "regions": [
                {
                    "region_id": "cold-r1",
                    "priority": 1,
                    "strategy": "coverage",
                    "phase": "O3",
                    "min_det_H": 1,
                    "max_det_H": 1,
                    "na_min": 0.0,
                    "na_max": 1.0,
                    "reason": "初始覆盖",
                    "evidence_refs": [],
                }
            ],
        },
        "scientific_decision": {
            "point": "branch_sampling",
            "config_version": context["config_version"],
            "model_version": "m0",
            "bottleneck": "初始候选池为空",
            "rationale": "探索相区",
            "alternatives": [{"route": "等待", "reason": "无任务等待"}],
            "expected_outcome": "得到初始候选",
            "success_criterion": "登记可追溯",
            "limitations": "暂无能量数据",
            "selection_purposes": [],
        },
    }
    if auto_authorized:
        action["operation"] = "generate_branches"
    incomplete = deepcopy(action)
    incomplete["parameters"]["regions"][0]["na_min"] = 0.5
    with pytest.raises(ValueError, match="initial_coverage_missing_cells"):
        agent.propose(incomplete)
    parent_dependent = deepcopy(action)
    parent_dependent["parameters"]["regions"][0]["strategy"] = "composition"
    with pytest.raises(ValueError, match="initial_coverage_requires_coverage"):
        agent.propose(parent_dependent)
    assert not list((root / "structures/candidate_structures").glob("**/*.vasp"))
    if native_team:
        from phase_agent.agents.strategic.codex_team import CodexTeam

        team = CodexTeam(agent.control)
        view = team.start(
            question="比较初始覆盖与模型可靠性", consult=["exploration", "reliability"]
        )
        for role in view["consult"]:
            team.advise(
                view["team_id"],
                1,
                {
                    "role": role,
                    "snapshot_id": view["snapshot"]["snapshot_id"],
                    "assessment": "初始池为空，需要先建立覆盖"
                    if role == "exploration"
                    else "缺少预测或DFT，模型可靠性尚不能评估",
                    "recommendation": "explore"
                    if role == "exploration"
                    else "insufficient_evidence",
                    "options": [{"tool": "generate_branches", "reason": "初始覆盖"}]
                    if role == "exploration"
                    else [],
                    "evidence_refs": [],
                    "expected_outcome": "建立可追溯候选",
                    "limitations": "没有预测或DFT结果，不能判断收敛",
                },
            )
        draft = team.draft(view["team_id"], 1, action, rationale="先覆盖再核验，不宣称模型可靠")
        from tests.test_simulation_agent import SimulationModel

        planned = team.simulate(
            view["team_id"], 1, SimulationModel()(draft["requests"]["simulation"])
        )
        assert not agent.pending()["pending"]
        if auto_authorized:
            assert planned["status"] == "ready_to_propose"
        else:
            team.feedback(view["team_id"], 1, "accept", "接受初始覆盖方向，仍需单独审批动作")
        result = team.propose(view["team_id"], 1)
    else:
        result = agent.propose(action)
    if auto_authorized:
        from phase_agent.tools.step_runner.file_protocol import read_json

        assert result["reply"]["status"] in {"completed", "tasks_in_progress"}, result
        assert not agent.pending()["pending"]
        state = read_json(agent.handler.state_path)
        record = state["action_records"][-1]
        assert record["human_feedback"] is None
        assert record["execution_result"]["execution_authorization"]["source"] == "confirmed_policy"
        assert (
            record["execution_result"]["simulation_report"]["scientific_completion_claimed"]
            is False
        )
        assert list((root / "structures/candidate_structures").glob("**/*.vasp"))
        assert CodexTeam(agent.control).read()["status"] == "submitted"
        return
    assert result["reply"]["status"] == "awaiting_approval", result
    assert result["reply"]["submitted"] is False
    assert result["pending"]["count"] == 1
    plan = result["pending"]["pending"][0]
    assert plan["recommended_action"] == "generate_branches"
    assert plan["review_card"]["proposal_hash"]
    assert plan["state_version"]
    assert "Relax" in plan["review_card"]["approval_boundary"]
    assert "report_path" in result
    assert not list((root / "structures/candidate_structures").glob("**/*.vasp"))
    response = agent.decide(
        "approve",
        plan_id=plan["plan_id"],
        state_version=plan["state_version"],
        proposal_hash=plan["review_card"]["proposal_hash"],
    )
    assert response["status"] in {"completed", "tasks_in_progress"}
    result = response["result"]
    assert result["submitted"] is False
    record = result["state"]["action_records"][-1]
    assert record["agent_proposal"]["decision_owner"] == "codex"
    assert record["agent_proposal"]["raw_action"]["decision_source"] == "codex"
    assert list((root / "structures/candidate_structures").glob("**/*.vasp"))
    if native_team:
        trace = record["agent_proposal"]["raw_action"]["_decision_team"]
        assert (
            record["execution_result"]["simulation_report"]["scientific_completion_claimed"]
            is False
        )
        assert trace["backend"] == "codex_native"
        assert {row["role"] for row in trace["advice"]} == {"exploration", "reliability"}
        assert trace["human_review"][-1]["kind"] == "human_accept"
        restored = CodexOffline(runtime)
        assert CodexTeam(restored.control).read()["status"] == "submitted"
