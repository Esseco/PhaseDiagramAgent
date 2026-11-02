"""Validate the local factory contract without loading any scientific model."""

from pathlib import Path
import pytest

from phase_agent.science.mlip.model_interface import create_calculator


@pytest.mark.parametrize("framework,preset", [
    ("mace", "mace-mh1"), ("sevennet", "sevennet-omni"),
    ("chgnet", "chgnet"), ("orb", "orb-v3"), ("nep", "nep"),
])
def test_selected_weights_reach_real_local_factory_config(tmp_path, monkeypatch, framework, preset):
    from Process_MLIP import calculators
    from Process_MLIP.model_config import _resolve_model_config

    weights = tmp_path / "weights.model"
    weights.write_bytes(b"synthetic; no model loaded")
    selected = []

    def factory(config):
        resolved = _resolve_model_config(config)
        selected.append(resolved)
        return resolved

    monkeypatch.setattr(calculators, "create_mlip_calculator", factory)
    root = Path(calculators.__file__).resolve().parents[1]
    result = create_calculator({"framework": framework, "model_name": preset,
                                "local_repo_path": str(root), "device": "cpu"}, model_path=weights)
    assert result["type"] == framework
    assert result["model_path"] == str(weights)
    assert len(selected) == 1


def test_mismatched_framework_and_model_is_rejected(tmp_path, monkeypatch):
    from Process_MLIP import calculators

    def forbidden(config):
        raise AssertionError("A mismatched model must not be loaded")

    monkeypatch.setattr(calculators, "create_mlip_calculator", forbidden)
    with pytest.raises(ValueError, match="framework_model_mismatch"):
        create_calculator({"framework": "orb", "model_name": "chgnet"})
