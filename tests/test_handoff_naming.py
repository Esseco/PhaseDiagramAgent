import json
from pathlib import Path

from phase_agent.tools.remote.handoff_naming import handoff_label
from phase_agent.tools.remote.submission_handoff import prepare_handoff


def test_english_label_describes_refresh(tmp_path):
    root = tmp_path / "O3-2ele-NaFeMn/submissions/epoch1_model"
    scripts = []
    for op, count in [("Single-point", 2), ("Relax", 1)]:
        batch = root / f"Model-refresh-0001_{op}/inputs/batch"
        batch.mkdir(parents=True)
        script = batch / "GPU.sh"
        script.write_text("#!/bin/bash\ntrue\n")
        (batch / "manifest.json").write_text(
            json.dumps([{"stage": "relax_and_feature"}] * count)
        )
        scripts.append(script)
    relative = [p.relative_to(root).as_posix() for p in scripts]
    assert handoff_label(root, relative) == "epoch1_ModelRefresh"
    first = prepare_handoff(root, scripts)
    second = prepare_handoff(root, scripts)
    assert first["folder"] == second["folder"]
    assert Path(first["folder"]).name == "epoch1_ModelRefresh_Batch01"
    (scripts[0].parent / "extra.json").write_text("{}")
    third = prepare_handoff(root, scripts)
    assert Path(third["folder"]).name == "epoch1_ModelRefresh_Batch02"
    assert Path(first["upload_archive"]).is_file()


def test_return_directory_is_physical_without_proxy(tmp_path):
    root = tmp_path / "project/submissions/epoch0_m1"
    batch = root / "G0001_Relax-0001/inputs/batch"
    batch.mkdir(parents=True)
    script = batch / "GPU.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    (batch / "manifest.json").write_text(json.dumps([{"stage": "relax_and_feature"}]))
    result = prepare_handoff(root, [script])
    returned = Path(result["return_directory"])
    assert returned == root / "return-results"
    assert (
        returned.is_dir() and not returned.is_junction() and not returned.is_symlink()
    )
    assert not (Path(result["folder"]) / "return-results").exists()
