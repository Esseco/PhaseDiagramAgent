"""Report rationale is recorded evidence; chat and small TXT inventories stay concise."""

from copy import deepcopy
from pathlib import Path

from phase_agent.runtime.interaction_reports import save_interaction
from phase_agent.tools.workflows.branch_review_view import markdown_rationale
from phase_agent.tools.workflows.dft_review_view import markdown_view


def test_region_reason_and_evidence_appear_in_markdown_only(tmp_path):
    from phase_agent.tools.workflows.region_review import write_proposal

    state = {
        "workspace_layout": {"workspace_root": str(tmp_path)},
        "active_model_version": "m1",
        "confirmed_config_version": "v1",
    }
    action = {
        "tool": "generate_branches",
        "task_key": "g1",
        "reason": "本轮补充低钠区覆盖",
        "branch_region_review_notes": "已有高钠区域本轮复用",
        "branch_pool_review": {
            "rationale": "低钠覆盖不足",
            "alternative_reason": "现有样本不能代替该区域",
            "evidence_refs": ["coverage:low-na"],
        },
        "parameters": {
            "regions": [
                {
                    "region_id": "r1",
                    "priority": 1,
                    "phase": "O3",
                    "strategy": "coverage",
                    "min_det_H": 4,
                    "max_det_H": 8,
                    "na_min": 0.2,
                    "na_max": 0.4,
                    "reason": "本区域样本缺口",
                    "evidence_refs": ["coverage:low-na"],
                }
            ]
        },
    }
    before = deepcopy((state, action))
    artifacts = write_proposal({"raw_action": action}, state)
    compact = Path(artifacts["text"])
    short_before = compact.read_bytes()
    full = (
        f"结果：Branch 方案待确认。\nBranch 分布：`{artifacts['text']}`\n下一步：核对后回复同意。"
    )
    reply = save_interaction(tmp_path, "继续", full, full, state)
    report = next((tmp_path / "交互记录").glob("*.md")).read_text(encoding="utf-8")
    for text in (
        "本轮 branch 选点理由",
        "本轮补充低钠区覆盖",
        "本区域样本缺口",
        "coverage:low-na",
        "现有样本不能代替该区域",
        "已有高钠区域本轮复用",
    ):
        assert text in report
    assert "本区域样本缺口" not in reply and "本区域样本缺口" not in compact.read_text(
        encoding="utf-8"
    )
    assert len(reply.splitlines()) <= 5 and compact.read_bytes() == short_before
    assert (state, action) == before


def test_dft_round_and_individual_reason_survive_saved_projection(tmp_path):
    from phase_agent.tools.workflows.dft_selection_artifacts import selection_record

    action = {
        "tool": "select_dft_candidates",
        "task_key": "d1",
        "expected_purpose": "校验近凸包区域的模型误差",
        "reason": "本轮近凸包误差需要DFT证据",
        "scientific_decision": {
            "rationale": "旧branch已提供候选",
            "alternatives": ["暂缓新增branch"],
            "evidence_refs": ["hull:current"],
        },
        "parameters": {
            "decisions": [
                {
                    "candidate_id": "s",
                    "action": "DFT_SINGLE_POINT",
                    "reason": "该结构接近凸包且尚待校验",
                }
            ]
        },
    }
    state = {
        "active_model_version": "m",
        "confirmed_config_version": "v",
        "qbc_candidates": [{"candidate_id": "s", "phase": "O3"}],
    }
    record = selection_record({"raw_action": action}, state)
    before = deepcopy(record)
    report = markdown_view(record)
    for text in (
        "本轮 DFT 选点理由",
        "本轮近凸包误差需要DFT证据",
        "校验近凸包区域的模型误差",
        "该结构接近凸包且尚待校验",
        "旧branch已提供候选",
        "hull:current",
    ):
        assert text in report
    assert record == before


def test_completed_branch_retains_proposed_reason_with_actual_inventory():
    record = {
        "quantity_reason": "验证未覆盖组成",
        "plan": {
            "allocations": [
                {
                    "phase": "O3",
                    "strategy": "composition",
                    "reason": "该组成缺口",
                    "evidence_refs": ["coverage:x"],
                }
            ]
        },
        "actual": {"new_branch_count": 2},
    }
    before = deepcopy(record)
    text = markdown_rationale(record)
    assert "验证未覆盖组成" in text and "该组成缺口" in text and "coverage:x" in text
    assert record == before


def test_missing_reasons_are_explicit_without_inventing_science():
    branch = markdown_rationale({"parameters": {"regions": [], "defer_reason": "等待已有结果"}})
    dft = markdown_view({"decisions": [{"candidate_id": "s", "action": "DFT_SINGLE_POINT"}]})
    assert "等待已有结果" in branch and "未记录" in branch and "未记录" in dft
    assert "本轮 DFT 选点理由" in dft
