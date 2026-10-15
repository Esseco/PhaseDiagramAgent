import json
from pathlib import Path

from phase_agent.tools.workflows.branch_review_view import readable_view as branch_view, raw_record, RAW_KEY
from phase_agent.tools.workflows.dft_review_view import readable_view as dft_view
from phase_agent.tools.relax_mc_batch.files import write


def test_branch_matches_approved_chinese_example_and_preserves_source(tmp_path):
    raw = {"status": "awaiting_approval", "plan": {
        "candidate_quota": 300, "selected_branch_upper": 96, "initial_states_per_branch": 3,
        "phases": {"at_x": {"0": ["O1"], "1": ["O3"]}, "intermediate": ["O3", "P3", "OP2"]},
        "max_det_H": 12,
        "supercells": [{"phase": phase, "det_H": size, "H": [[size, offset, 0], [0, 1, 0], [0, 0, 1]]}
                       for phase in ["O1", "O3", "OP2", "P3"]
                       for size, count in [(4,2),(6,1),(8,3),(10,1),(12,5)] for offset in range(count)]}}
    view = branch_view(raw)
    assert view["状态"] == "待审批，尚未生成"
    assert view["数量"]["候选采样配额"] == 300
    assert view["数量"]["去重后选取branch上限"] == 96
    assert "尚未生成" in view["实际生成结果"]["状态"]
    groups = view["超胞"]["允许选项（相同矩阵合并）"]
    assert [g["超胞大小_detH"] for g in groups] == [4, 6, 8, 10, 12]
    assert sum(g["不同H矩阵数量"] for g in groups) == 12
    assert raw_record(view) == raw
    path = tmp_path / "review.json"
    write(path, view, sort_keys=False)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert list(data)[0] == "状态" and list(data)[-1] == RAW_KEY


def test_dft_is_clear_without_fabricating_missing_values():
    raw = {"status": "awaiting_approval", "summary": {"selected_count": 1, "counts_by_action": {"DFT_SINGLE_POINT": 1, "DEFER": 1}},
           "decisions": [{"candidate_id": "s1", "action": "DFT_SINGLE_POINT", "phase": "O3", "Na_fraction": 0,
                          "reason": "近凸包", "Ehull": 0.01, "Ehull_unit": "eV/atom", "structure_path": "s1.vasp"},
                         {"candidate_id": "s2", "action": "DEFER", "reason": "已有标签"}],
           "input_preview": {"relative_cost": 0}, "actual": None}
    view = dft_view(raw)
    assert view["数量"]["DFT单点数"] == 1
    row = view["选中结构清单"][0]
    assert row["计算类型"] == "DFT单点" and row["钠含量_x"] == 0
    assert row["H矩阵"] == "未记录" and row["训练集关系"] == "未确认"
    assert view["计算预算"]["估计相对成本"] == 0
    assert len(view["暂缓或不选清单"]) == 1
    assert raw_record(view) == raw


def test_dft_prepared_means_inputs_not_completed_calculation():
    view = dft_view({"status": "prepared", "actual": {"task_count": 1, "tasks": [{"task_id": "t1", "stage": "dft_relax"}],
                   "input_preparation": {"status": "prepared", "directory": "upload"}}})
    assert "尚不代表计算完成" in view["状态"]
    assert view["实际输入任务"]["任务清单"][0]["计算类型"] == "DFT结构优化"


def test_completed_branch_groups_actual_counts():
    raw = {"status": "completed", "actual": {"new_branch_count": 1, "registered_branch_count": 1,
           "registered_structure_count": 2, "branches": [{"phase": "O3", "supercell_det": 4, "Na_fraction": "1/2", "structure_count": 2}]}}
    view = branch_view(raw)
    grouped = view["实际生成结果"]["按相_超胞_钠含量汇总"]
    assert grouped[0]["branch数量"] == 1 and grouped[0]["初态数量"] == 2
