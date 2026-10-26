from copy import deepcopy
import json
from pathlib import Path
from phase_agent.tools.workflows.dft_selection_artifacts import proposal_artifacts
from phase_agent.runtime.concise_dialogue import approval_reply
from phase_agent.runtime.review_presentation import review_card


def test_storage_root_and_empty_response_use_original_pending(tmp_path):
    action = {
        "tool": "select_dft_candidates",
        "task_key": "existing",
        "target_ids": ["s1"],
        "parameters": {
            "decisions": [
                {"candidate_id": "s1", "action": "DFT_SINGLE_POINT", "reason": "recorded"}
            ]
        },
    }
    proposal = {
        "recommended_action": "select_dft_candidates",
        "expected_purpose": "原选点用途",
        "raw_action": action,
    }
    state = {
        "confirmed_config": {"storage": {"workspace_root": str(tmp_path)}},
        "active_model_version": "m1",
        "pending_execution_policies": {"p": {"agent_proposal": proposal}},
    }
    before = deepcopy(state)
    proposal_artifacts(proposal, state)
    text = approval_reply({}, state)
    assert "已指定 0" not in text and "待审阅动作" not in text
    assert "DFT单点 1" in text and "DFT 选点文件：" in text and ".txt`" in text
    assert "完成本批输入准备后停止" in text
    assert state == before
    paths = list((tmp_path / "workflow_state/dft_plans").glob("*.json"))
    assert len(paths) == 1
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    assert data["数量"]["选中结构数"] == 1 and data["状态"] == "待审批，尚未准备输入"


def test_metadata_reads_owned_ledger_without_creating_tasks(tmp_path):
    p = tmp_path / "workflow_state/ledgers/phase_data.json"
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps(
            {
                "branches": {"b": {"H": [[1, 0, 0], [0, 2, 0], [0, 0, 1]], "P": "O3", "x": "1/2"}},
                "structures": {"s": {"branch_id": "b", "source_path": "saved.vasp"}},
            }
        )
    )
    state = {"confirmed_config": {"storage": {"workspace_root": str(tmp_path)}}, "tasks": []}
    proposal = {
        "raw_action": {
            "tool": "select_dft_candidates",
            "parameters": {"decisions": [{"candidate_id": "s", "action": "DFT_SINGLE_POINT"}]},
        }
    }
    before = deepcopy(state)
    old = p.read_bytes()
    artifact = proposal_artifacts(proposal, state)
    data = json.loads(Path(artifact["json"]).read_text(encoding="utf-8"))
    row = data["选中结构清单"][0]
    assert row["超胞大小_detH"] == 2 and row["相"] == "O3" and row["钠含量_x"] == "1/2"
    assert state == before and p.read_bytes() == old


def test_dft_card_counts_selected_decisions_only():
    proposal = {
        "raw_action": {
            "tool": "select_dft_candidates",
            "target_ids": ["s", "defer"],
            "parameters": {
                "decisions": [
                    {"candidate_id": "s", "action": "DFT_SINGLE_POINT"},
                    {"candidate_id": "defer", "action": "DEFER"},
                ]
            },
        }
    }
    card = review_card("p", {"agent_proposal": proposal})
    assert card["target_count"] == 1 and card["target_ids"] == ["s"]
