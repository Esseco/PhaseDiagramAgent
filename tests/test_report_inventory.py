from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json

import pytest

from phase_agent.runtime.interaction_reports import save_interaction
from phase_agent.runtime.interaction_report_paths import report_stem, reserve_report
from phase_agent.tools.workflows.dft_selection_artifacts import selection_record


def report(reply):
    return Path(reply.split("交互记录地址：", 1)[1])


def saved_dft():
    proposal = {
        "raw_action": {
            "tool": "select_dft_candidates",
            "task_key": "saved-plan",
            "parameters": {
                "decisions": [
                    {"candidate_id": "s1", "action": "DFT_SINGLE_POINT", "reason": "近凸包核对"},
                    {"candidate_id": "s2", "action": "DFT_RELAX", "reason": "核对大胞"},
                    {"candidate_id": "s3", "action": "DEFER", "reason": "已有标签"},
                ],
                "dft_input_preview": {
                    "relative_cost": 12,
                    "selected_structures": [
                        {
                            "candidate_id": "s2",
                            "phase": "P3",
                            "x_Na_per_O2": 0.5,
                            "ehull": 0.02,
                            "ehull_unit": "eV/atom",
                        },
                    ],
                },
            },
        }
    }
    return {
        "active_model_version": "m1",
        "confirmed_config_version": "c1",
        "upload_layout": {"model_rounds": {"m1": 2}},
        "pending_execution_policies": {
            "p1": {
                "agent_proposal": proposal,
                "project_identity": {"model_version": "m1", "config_version": "c1"},
            }
        },
        "qbc_candidates": [
            {
                "candidate_id": "s1",
                "phase": "OP2",
                "x_Na_per_O2": 0,
                "atom_count": 12,
                "composition": {"Fe": 2, "Mn": 2, "O": 8},
                "predicted_Ehull": 0,
                "ehull_unit": "eV/atom",
                "structure_path": "results/mc/rank_000.vasp",
            }
        ],
    }


def test_saved_plan_without_link_has_concrete_inventory_and_no_plan_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "phase_agent.tools.workflows.dft_selection_artifacts.read",
        lambda *a, **k: pytest.fail("extra ledger read"),
    )
    state = saved_dft()
    before = deepcopy(state)
    receipt = save_interaction(
        tmp_path, "状态", "结果：DFT方案待确认。", "结果：DFT方案待确认。", state
    )
    path = report(receipt)
    assert path.parent == tmp_path / "interaction_records/epoch1"
    text = path.read_text(encoding="utf-8")
    assert "DFT选点_" in path.name and "方案待确认" not in path.name
    assert "| s1 | OP2 | 0 | 12 | 未记录 | DFT单点 | 0 eV/atom | 近凸包核对 |" in text
    assert "| s2 | P3 | 0.5 | 未记录 | 未记录 | DFT结构优化 |" in text
    assert "results/mc/rank_000.vasp" in text and "不是 DFT 结果" in text
    assert "不提交超算" in text and "已有标签" in text and "m1" in text and "c1" in text
    assert state == before and not (tmp_path / "workflow_state").exists()
    assert "s1" not in receipt and len(receipt.splitlines()) <= 3


def test_projection_enriches_only_recorded_ledger_facts_and_does_not_mutate():
    state = saved_dft()
    proposal = state["pending_execution_policies"]["p1"]["agent_proposal"]
    ledger = {
        "structures": {"s1": {"branch_id": "b1"}},
        "branches": {"b1": {"H": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]}},
    }
    before = deepcopy((state, ledger))
    result = selection_record(proposal, state, ledger=ledger)
    assert result["decisions"][0]["supercell_det"] == 2
    assert result["decisions"][1]["H"] is None
    assert (state, ledger) == before


def test_old_model_proposal_is_not_enriched_from_current_candidates(tmp_path):
    state = saved_dft()
    state["pending_execution_policies"]["p1"]["project_identity"]["model_version"] = "old"
    text = report(save_interaction(tmp_path, "状态", "待核对", "待核对", state)).read_text(
        encoding="utf-8"
    )
    assert "方案版本与当前项目不同" in text and "rank_000.vasp" not in text


def test_bad_optional_artifact_does_not_drop_answer_or_saved_dft_inventory(tmp_path):
    path = tmp_path / "workflow_state/dft_plans/broken.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    text = report(
        save_interaction(
            tmp_path, "状态", f"结果：已有方案。\n文件：`{path}`", "待确认", saved_dft()
        )
    ).read_text(encoding="utf-8")
    assert "展示文件读取受阻" in text and "已有方案" in text and "| s1 | OP2 |" in text


def test_names_are_factual_short_collision_safe_and_record_retry_reuses_path(tmp_path):
    now = datetime(2026, 10, 20, 8, 0, tzinfo=timezone.utc)
    stem = report_stem(saved_dft(), {}, now)
    assert stem == "DFT选点_" + now.astimezone().strftime("%Y%m%d")
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: reserve_report(tmp_path, stem, {})[0], range(12)))
    assert len(set(paths)) == 12 and all(path.is_file() for path in paths)
    state = saved_dft()
    record = {"id": "a" * 32}
    one = report(save_interaction(tmp_path, "状态", "待确认", "待确认", state, record=record))
    two = report(save_interaction(tmp_path, "状态", "更新说明", "更新说明", state, record=record))
    assert one == two and "更新说明" in two.read_text(encoding="utf-8")
    assert "a" * 12 not in one.name and "a" * 32 in two.read_text(encoding="utf-8")


def test_report_publish_failure_leaves_no_empty_file_or_temp(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        save_interaction(tmp_path, "状态", "待确认", "待确认", {})
    assert not list((tmp_path / "interaction_records/epoch0").iterdir())


def test_historical_controller_error_and_missing_tasks_are_explicit_without_claiming_failure(
    tmp_path,
):
    state = {
        "active_model_version": "m1",
        "tasks": [
            {
                "task_id": "mc1",
                "status": "pending",
                "stage": "deep_search",
                "relax_mc_bundle_id": "b",
            }
        ],
        "relax_mc_batches": {
            "b": {
                "model_version": "m1",
                "import_checksum": "verified",
                "result_summary": {"reason": "squeue returned exit status 1"},
            }
        },
    }
    text = report(save_interaction(tmp_path, "状态", "待核验", "待核验", state)).read_text(
        encoding="utf-8"
    )
    assert (
        "squeue returned exit status 1" in text and "不等同于计算失败" in text and "`mc1`" in text
    )


@pytest.mark.parametrize("change", ["state", "control", "none"])
def test_presentation_reuses_only_unchanged_facts_once_then_releases(tmp_path, change):
    from phase_agent.runtime.turn_process import (
        record_turn,
        presentation_signature,
        remember_presentation_state,
        read_presentation_state,
        _current,
    )
    from phase_agent.tools.step_runner.file_protocol import read_json

    path = tmp_path / "workflow_state/state.json"
    path.parent.mkdir()
    path.write_text('{"tasks":[],"value":1}')
    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    with record_turn(config, "状态", "thread"):
        signature = presentation_signature(path)
        facts = read_json(path)
        remember_presentation_state(path, facts, signature)
        if change == "state":
            path.write_text('{"tasks":[],"value":222}')
        elif change == "control":
            control = {
                "state_path": str(path.resolve()),
                "revision": "new",
                "binding": {},
                "patch": {"values": {"workflow_stage": "updated"}},
            }
            path.with_name("state.json.control.json").write_text(json.dumps(control))
        rendered = read_presentation_state(path)
        assert (rendered is facts) == (change == "none")
        assert rendered["value"] == (222 if change == "state" else 1)
        if change == "control":
            assert rendered["workflow_stage"] == "updated"
        assert "_presentation_state" not in _current.get()
        assert read_json(path) is not facts
    saved = json.loads(
        next((tmp_path / "workflow_state/dialogue_process").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert "_presentation_state" not in saved
