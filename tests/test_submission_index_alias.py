from pathlib import Path

import pytest

from phase_agent.tools.remote.submission_layout import write_submission_index


def test_historical_alias_and_canonical_batch_share_index(tmp_path, monkeypatch):
    root = tmp_path / "diagrams/diagram_default/submissions"
    root.mkdir(parents=True)
    alias = tmp_path / "submissions"
    operation = root / "epoch0_m1/G0001_Relax-0001"
    operation.mkdir(parents=True)
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        if path.is_relative_to(alias):
            return original(root / path.relative_to(alias), *args, **kwargs)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    state = {
        "slurm_batches": [
            {"upload_directory": str(base / "epoch0_m1/G0001_Relax-0001/inputs/batch")}
            for base in (alias, root)
        ]
    }
    result = write_submission_index(root, state)
    text = result.read_text(encoding="utf-8")
    assert text.count("| [epoch0_m1/G0001_Relax-0001]") == 1
    assert "| 2 |" in text
    write_submission_index(alias, state)


def test_outside_operation_rejected_before_writing(tmp_path):
    root = tmp_path / "submissions"
    root.mkdir()
    state = {"slurm_batches": [{"upload_directory": str(tmp_path / "outside/inputs/batch")}]}
    with pytest.raises(ValueError):
        write_submission_index(root, state)
    assert not (root / "README.md").exists()
