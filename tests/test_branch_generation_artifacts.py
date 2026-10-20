from phase_agent.tools.workflows.branch_review_view import raw_record
from types import SimpleNamespace
from pathlib import Path
import json

from phase_agent.tools.workflows.generation_artifacts import proposal_artifacts, result_artifacts
from phase_agent.runtime.concise_dialogue import approval_reply


def test_single_json_plan_becomes_actual_inventory(tmp_path):
    state = {
        "workspace_layout": {"workspace_root": str(tmp_path)},
        "active_model_version": "m1",
        "confirmed_config": {"boundary": {"P": ["O3"], "x": [0, 1]}},
    }
    action = {
        "tool": "generate_branches",
        "task_key": "g1",
        "parameters": {
            "total_quota": 10,
            "batch_size": 4,
            "max_det_H": 12,
            "initial_states_per_branch": 3,
            "generation_plan": [{"phase": "O3", "strategy": "coverage", "quota": 10}],
        },
    }
    proposal = {"raw_action": action}
    artifacts = proposal_artifacts(proposal, state)
    assert set(artifacts) == {"json", "text"}
    path = Path(artifacts["json"])
    before = raw_record(json.loads(path.read_text(encoding="utf-8")))
    assert before["actual"] is None and before["plan"]["candidate_quota"] == 10
    assert artifacts["text"] in approval_reply(proposal, state)
    assert "候选配额：10" in Path(artifacts["text"]).read_text(encoding="utf-8")
    manager = SimpleNamespace(
        data={
            "branches": {"b1": {"P": "O3", "H": [[2, 0, 0], [0, 1, 0], [0, 0, 1]], "x": "1/2"}},
            "structures": {
                "s1": {"source_path": "s1.vasp", "composition": {"Na": 1, "Fe": 2, "O": 4}}
            },
        }
    )
    result = result_artifacts(
        tmp_path,
        {"new_branch_ids": ["b1"]},
        manager,
        [{"branch_id": "b1", "structure_id": "s1"}],
        state,
        action,
    )
    assert result == artifacts
    actual = raw_record(json.loads(path.read_text(encoding="utf-8")))
    assert actual["actual"]["branches"][0]["supercell_det"] == 2
    assert actual["actual"]["branches"][0]["Na_fraction"] == "1/2"
    assert proposal_artifacts(proposal, state) == result
    assert raw_record(json.loads(path.read_text(encoding="utf-8")))["status"] == "completed"
    assert len(list(path.parent.iterdir())) == 2
    assert "O3 | 1/2 | 2 | 1" in Path(result["text"]).read_text(encoding="utf-8")


def test_scope_is_visible_without_verbose_generation_details(tmp_path):
    state = {"workspace_layout": {"workspace_root": str(tmp_path)}}
    proposal = {
        "recommended_action": "generate_branches",
        "action_parameters": {"module_execution_scope": {"submit_authorized": False}},
    }
    state["pending_execution_policies"] = {"p": {"agent_proposal": proposal}}
    text = approval_reply(proposal, state)
    assert "准备 Relax+MC 输入" in text and "不提交超算" in text
    assert "Branch 分布与构成文件：" in text and ".txt`" in text
    assert "审批范围记录不完整" in text
