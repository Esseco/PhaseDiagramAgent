"""Project summary and direct saved-product navigation do not calculate science."""

from copy import deepcopy

from phase_agent.analysis.feedback.output_catalog import output_catalog, catalog_markdown
from phase_agent.analysis.feedback.workspace_guide import publish_workspace_guide


def test_existing_epoch_images_and_project_summary_have_direct_links(tmp_path):
    root = tmp_path / "analysis_outputs"
    csv = root / "epoch1/phase_diagrams/mlip/phase_diagram.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("saved data")
    images = csv.parent / "plots"
    images.mkdir()
    (images / "saved.png").write_bytes(b"existing synthetic image")
    (root / "round_summary.csv").write_text("epoch,round\nepoch1,1")
    for name in ("reference_structures", "candidate_structures"):
        (tmp_path / "structures" / name).mkdir(parents=True)
    state = {
        "active_model_version": "m2",
        "upload_layout": {"model_rounds": {"m1": 1, "m2": 2}},
        "tasks": [
            {"task_id": "one", "status": "pending"},
            {"task_id": "two", "status": "completed"},
        ],
        "pending_execution_policies": {"preserved": {"status": "awaiting_approval"}},
        "phase_diagrams": {"mlip": {"model_version": "m2", "csv_path": str(csv)}},
    }
    before = deepcopy(state)
    rows = output_catalog(state)
    assert any(row["path"] == str(images) for row in rows)
    text = catalog_markdown(rows, root, state=state)
    assert "## 项目总览" in text and "当前轮：epoch1" in text
    assert "本地任务共 2 项" in text and "待审批方案：1" in text
    assert "[跨 epoch summary](round_summary.csv)" in text
    assert "epoch1/phase_diagrams/mlip/plots" in text
    assert "../structures/reference_structures" in text
    assert "../structures/candidate_structures" in text
    assert state == before
    assert csv.read_text() == "saved data"


def test_missing_image_folder_does_not_invent_plot_output(tmp_path):
    root = tmp_path / "analysis_outputs"
    csv = root / "epoch1/phase_diagrams/mlip/phase_diagram.csv"
    csv.parent.mkdir(parents=True)
    state = {"active_model_version": "m", "phase_diagrams": {"mlip": {"csv_path": str(csv)}}}
    rows = output_catalog(state)
    assert not any("相图图片目录" in row["product"] for row in rows)
    text = catalog_markdown(rows, root, state=state)
    assert "`round_summary.csv`：尚未生成" in text
    assert "[跨 epoch summary]" not in text
    assert not (root / "round_summary.csv").exists()


def test_workspace_structure_links_only_link_existing_directories(tmp_path):
    publish_workspace_guide(tmp_path)
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "母结构（初始化后生成）" in text
    assert "候选结构（初始化后生成）" in text
    assert "[母结构]" not in text and "[候选结构]" not in text
    (tmp_path / "structures/reference_structures").mkdir(parents=True)
    publish_workspace_guide(tmp_path)
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "[母结构](structures/reference_structures/)" in text
    assert "候选结构（初始化后生成）" in text
