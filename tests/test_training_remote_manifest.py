import hashlib
import json
from phase_agent.tools.remote.collect_training_results import write_model_manifest


def test_models_remain_in_training_directories_and_hashes_are_recorded(tmp_path):
    for name in ("com_1", "com_2"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / f"{name}_ema.model").write_bytes(name.encode())
    plan = {"main_model": {"final_model": "com_1"},
            "committees": [{"model_id": "com_1"}, {"model_id": "com_2"}],
            "original_model_version": "m1", "training_round": "round1"}
    output = tmp_path / "results"
    rows = write_model_manifest(tmp_path, plan, output)
    assert len(rows) == 2
    assert rows[0]["sha256"] == hashlib.sha256(b"com_1").hexdigest()
    assert rows[0]["size_bytes"] == 5
    assert rows[0]["is_main_model"] and not rows[1]["is_main_model"]
    assert rows[0]["model_path"] == rows[0]["remote_model_path"]
    assert not (output / "models").exists()
    assert json.loads((output / "models.json").read_text()) == rows
    assert (tmp_path / "com_1/com_1_ema.model").read_bytes() == b"com_1"
