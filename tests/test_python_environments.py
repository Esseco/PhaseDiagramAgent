import pytest
from config_layer.schema.python_environments import python_environment, remote_comparison_model
from config_layer.defaults.default_layered_search_config import default_layered_search_config


def test_remote_not_inferred_from_local_or_model():
    config = default_layered_search_config()
    assert python_environment(config, "local") == "py1"
    assert python_environment(config, "local", mlip=True) == "py-mace"
    with pytest.raises(ValueError, match="remote_mlip"):
        remote_comparison_model({"environment": "py-mace"}, config)


def test_remote_override_preserves_model_version():
    config = {"python_environments": {"remote_mlip": "hpc-mace"}}
    model = remote_comparison_model({"version": "round-2", "environment": "py-mace"}, config)
    assert model["environment"] == "hpc-mace"
    assert model["version"] == "round-2"
