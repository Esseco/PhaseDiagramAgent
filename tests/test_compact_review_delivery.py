from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from phase_agent.runtime.concise_dialogue import concise_body
from phase_agent.runtime.studio_reply_presentation import format_turn_reply
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
from phase_agent.tools.workflows.compact_review import review_artifacts
from phase_agent.tools.workflows.generation_artifacts import proposal_artifacts, result_artifacts
from phase_agent.tools.workflows.propose_followup import propose_after_generation


def generated(tmp_path):
    state = {
        "workspace_layout": {"workspace_root": str(tmp_path)},
        "active_model_version": "m",
        "confirmed_config": {"system": {"boundary": {"P": ["O3"], "TM_ratio": {"Fe": 1, "Mn": 1}}}},
    }
    action = {"tool": "generate_branches", "task_key": "g", "parameters": {"total_quota": 10}}
    proposal_artifacts({"raw_action": action}, state)
    manager = SimpleNamespace(
        data={
            "branches": {"b": {"P": "O3", "H": [[2, 0, 0], [0, 1, 0], [0, 0, 1]], "x": "1/2"}},
            "structures": {"s": {"source_path": "s.vasp"}},
        }
    )
    artifacts = result_artifacts(
        tmp_path,
        {"new_branch_ids": ["b"]},
        manager,
        [{"branch_id": "b", "structure_id": "s"}],
        state,
        action,
    )
    state["generation_history"] = [{"task_key": "g", "artifacts": artifacts}]
    return state, action, artifacts


def test_real_followup_proposal_keeps_completed_generation_and_full_path(tmp_path):
    state, action, artifacts = generated(tmp_path)
    summary = {"registered_branches": 1, "registered_structures": 1}
    result = {
        "status": "completed",
        "state": state,
        "events": [
            {"action": action, "execution": {"status": "completed", "result": {"summary": summary}}}
        ],
    }
    following = {
        "raw_action": {"tool": "prepare_local_batch_files", "task_key": "next", "parameters": {}},
        "recommended_action": "prepare_local_batch_files",
    }
    calls = []

    def next_proposal(current, session, **kwargs):
        calls.append(kwargs)
        return {
            "status": "awaiting_approval",
            "state": {
                **current,
                "pending_execution_policies": {"p": {"agent_proposal": following}},
            },
            "agent_proposal": following,
        }

    combined = propose_after_generation(
        {
            "execution_mode": "interactive",
            "agent_client": lambda _: None,
            "config_session": {},
            "effective_registry": {},
            "effective_config": {},
        },
        result,
        event_loop=next_proposal,
    )
    before = deepcopy(combined)
    text = format_workflow_reply(combined, tmp_path / "workflow_state/state.json")
    text = format_turn_reply(concise_body(text, combined["state"]), combined["state"])
    assert "已生成 1 个 branch" in text and artifacts["text"] in text
    assert text.count(artifacts["text"]) == 1
    assert combined == before and len(calls) == 1 and calls[0]["human_feedback"] is None
    brief = Path(artifacts["text"]).read_text(encoding="utf-8")
    assert "O3 | 1/2 | 2 | 1" in brief and "Fe:1" in brief and "Mn:1" in brief
    assert len(brief.splitlines()) <= 6 and "H矩阵" not in brief and "s.vasp" not in brief


def test_completed_inventory_does_not_leak_into_unrelated_stage(tmp_path):
    state, action, artifacts = generated(tmp_path)
    reply = format_workflow_reply(
        {
            "status": "completed",
            "state": state,
            "final_action": {"tool": "prepare_local_batch_files", "task_key": "other"},
        },
        tmp_path / "state.json",
    )
    assert artifacts["text"] not in reply and artifacts["json"] not in reply


def test_legacy_projection_preserves_json_and_does_not_rewrite_identical_text(tmp_path):
    state, action, artifacts = generated(tmp_path)
    path = Path(artifacts["json"])
    before = path.read_bytes()
    text = Path(artifacts["text"])
    modified = text.stat().st_mtime_ns
    assert review_artifacts(path, kind="branch", state=state) == artifacts
    assert path.read_bytes() == before and text.stat().st_mtime_ns == modified


def test_actual_distribution_counts_only_new_branches(tmp_path):
    from phase_agent.tools.workflows.compact_review import branch_lines

    record = {
        "actual": {
            "new_branch_count": 1,
            "registered_structure_count": 2,
            "branches": [
                {"phase": "O3", "Na_fraction": 0, "supercell_det": 2, "is_new": True},
                {"phase": "O1", "Na_fraction": 1, "supercell_det": 4, "is_new": False},
            ],
        }
    }
    text = "\n".join(branch_lines(record))
    assert "O3 | 0 | 2 | 1" in text and "O1" not in text


def test_studio_handoff_replacement_keeps_current_inventory_address(monkeypatch):
    monkeypatch.setattr(
        "phase_agent.runtime.waiting_files_presentation.waiting_file_lines",
        lambda _: ["本轮提交文件夹：input"],
    )
    state = {"tasks": []}
    text = "Relax+MC 上传包：input\nBranch 分布与构成文件：`C:\\p\\branch.txt`"
    result = format_turn_reply(text, state)
    assert "本轮提交文件夹：input" in result and "C:\\p\\branch.txt" in result


def test_proposal_lists_scope_and_file_before_generating_any_structures(tmp_path):
    from phase_agent.runtime.concise_dialogue import approval_reply

    state = {
        "workspace_layout": {"workspace_root": str(tmp_path)},
        "tasks": [],
        "generation_history": [],
    }
    action = {
        "tool": "generate_branches",
        "task_key": "before",
        "parameters": {
            "total_quota": 100,
            "batch_size": 100,
            "generation_plan": [
                {
                    "phase": "O3",
                    "strategy": "coverage",
                    "quota": 100,
                    "na_min": 0.25,
                    "na_max": 0.75,
                    "max_det_H": 8,
                }
            ],
        },
    }
    original = deepcopy(state)
    text = approval_reply({"raw_action": action}, state)
    path = next(tmp_path.rglob("*.txt"))
    assert str(path) in text and "同意" in text
    assert "尚未生成" in path.read_text(encoding="utf-8")
    assert "O3 | 0.25–0.75 | ≤8 | 100" in path.read_text(encoding="utf-8")
    assert state == original and not list(tmp_path.rglob("*.vasp"))
