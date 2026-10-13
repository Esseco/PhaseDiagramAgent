import pytest
from phase_agent.tools.remote.submission_layout import write_submission_index, index_operation_path


def test_new_index_accepts_same_project_old_physical_records(tmp_path):
    project = tmp_path / "project"
    root = project / "diagrams/diagram_default/submissions"
    root.mkdir(parents=True)
    old = project / "submissions/epoch0_model/G0001_Relax-0001/inputs/batch"
    old.mkdir(parents=True)
    new = root / "epoch0_model/G0001_Relax-0001_MC-round-0002/inputs/batch"
    new.mkdir(parents=True)
    state = {"slurm_batches": [{"upload_directory": str(old)}, {"upload_directory": str(new)}]}
    path = write_submission_index(root, state)
    text = path.read_text(encoding="utf-8")
    assert "G0001_Relax-0001_MC-round-0002" in text and "../../../submissions/" in text
    assert old.is_dir() and new.is_dir()


def test_other_project_or_other_diagram_is_rejected(tmp_path):
    root = tmp_path / "project/diagrams/diagram_default/submissions"
    root.mkdir(parents=True)
    for other in [
        tmp_path / "other/submissions/job",
        tmp_path / "project/diagrams/second/submissions/job",
    ]:
        other.mkdir(parents=True)
        with pytest.raises(ValueError, match="outside"):
            index_operation_path(root, other)
