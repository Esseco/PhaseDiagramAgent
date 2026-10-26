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


def test_relax_mc_approval_shows_generation_scope_before_approval(tmp_path):
    generation_json = tmp_path / "generation.json"
    generation_text = tmp_path / "generation.txt"
    generation_json.write_text(json.dumps({"actual": {"branches": []}}), encoding="utf-8")
    generation_text.write_text("新增 branch：2\nO3 | 1/2 | 4 | 2\n", encoding="utf-8")
    state = {
        "active_model_version": "m1",
        "generation_history": [
            {
                "task_key": "g1",
                "model_version": "m1",
                "new_branch_ids": ["new-1", "new-2"],
                "registered_ids": ["s1", "s2", "s3"],
                "summary": {
                    "proposed_branches": 2,
                    "registered_branches": 2,
                    "registered_structures": 3,
                    "phase_counts": {"O3": 2},
                    "det_H_range": [4, 8],
                    "sampling_report": {
                        "status": "paused_for_review",
                        "stop_reason": "paused_time_guard",
                        "attempts": 4,
                        "regions": [
                            {
                                "region_id": "r1",
                                "attempts": 4,
                                "accepted": 2,
                                "duplicates": 2,
                            }
                        ],
                    },
                },
                "artifacts": {"json": str(generation_json), "text": str(generation_text)},
            }
        ],
        "confirmed_config": {},
    }
    proposal = {
        "recommended_action": "prepare_local_batch_files",
        "expected_purpose": "准备本轮 Relax+MC 输入",
        "raw_action": {
            "tool": "prepare_local_batch_files",
            "task_key": "relax1",
            "reason": "为本轮新branch准备 Relax+MC",
            "target_ids": ["old-1"],
            "parameters": {
                "mode": "relax_mc_inputs",
                "relax_mc_preview": {
                    "model_version": "m1",
                    "checksum": "sha256:preview",
                    "branch_ids": ["old-1"],
                    "structures": [{"structure_id": "s1"}, {"structure_id": "s2"}],
                    "workload": {
                        "relax_structure_tasks": 2,
                        "gpu_jobs_upper_by_wave": [1],
                        "mc_task_upper": [1],
                    },
                },
            },
        },
        "estimated_cost": {},
    }
    state["pending_execution_policies"] = {"p": {"agent_proposal": proposal}}
    artifacts = proposal_artifacts(proposal, state)
    assert artifacts.get("scope_mismatch") is True
    report = Path(artifacts["text"]).read_text(encoding="utf-8")
    assert "交集：0" in report
    reply = approval_reply(proposal, state)
    assert reply.index("Branch 分布与构成文件：") < reply.index("下一步：")
    assert "先修正 branch 范围后再审批" in reply
