from pathlib import Path
from phase_agent.runtime.isolated_run_paths import isolate_runtime_outputs


def test_all_standard_outputs_are_inside_new_run(tmp_path):
    original = {"state_path": "old/state.json", "qbc": {"output_path": "old/qbc.json"},
                "mlip": {"model": "shared-model"}}
    config, paths = isolate_runtime_outputs(original, tmp_path / "new")
    root = (tmp_path / "new").resolve()
    for key, value in config.items():
        if key.endswith("_path") or key.endswith("_directory"):
            assert Path(value).is_relative_to(root)
    assert Path(config["qbc"]["output_path"]).is_relative_to(root)
    assert original["qbc"]["output_path"] == "old/qbc.json"
    assert config["mlip"] == original["mlip"]
