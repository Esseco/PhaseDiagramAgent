import json
from pathlib import Path
import pytest
from execution_layer.local.compact_pending_finetune import compact_pending_finetune


def fixture(tmp_path):
    root = tmp_path
    old = root / "upload_batches/epoch0_m1/MLIP-finetune-round-0001"
    live = old.with_name("MLIP-finetune-round-0002")
    live.mkdir(parents=True)
    report = {"training_round": live.name, "directory": str(live), "energy": 1.234}
    for name in ("training_plan.json", "finetune_report.json"):
        (live / name).write_text(json.dumps(report))
    (live / "train.xyz").write_bytes(b"unchanged scientific input")
    jobs = {key: {"directory": str(path), "status": "inputs_prepared", "submitted": False,
                  "original_model_version": "m1", "report": report}
            for key, path in (("old", old), ("live", live))}
    (root / "runtime").mkdir()
    (root / "runtime/state.json").write_text(json.dumps({"remote_finetune_jobs": jobs}))
    return root, old, live


def test_compact_pending_only_and_keep_data(tmp_path):
    root, old, live = fixture(tmp_path)
    result = compact_pending_finetune(root, apply=True, agent_port=None)
    assert result["removed_duplicates"] == 1
    assert old.is_dir() and not live.exists()
    assert (old / "train.xyz").read_bytes() == b"unchanged scientific input"
    state = json.loads((root / "runtime/state.json").read_text())
    assert list(state["remote_finetune_jobs"]) == ["live"]
    assert state["remote_finetune_jobs"]["live"]["directory"] == str(old)
    assert json.loads((old / "training_plan.json").read_text())["training_round"] == old.name
    assert compact_pending_finetune(root)["status"] == "unchanged"
    assert (Path(result["backup"]) / "retired_finetune_drafts.json").exists()


def test_submitted_is_not_renumbered(tmp_path):
    root, old, live = fixture(tmp_path)
    path = root / "runtime/state.json"
    state = json.loads(path.read_text())
    state["remote_finetune_jobs"]["old"]["submitted"] = True
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="提交"):
        compact_pending_finetune(root, apply=True, agent_port=None)
    assert live.exists() and not old.exists()
