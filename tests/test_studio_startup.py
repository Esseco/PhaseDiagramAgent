import pytest
from phase_agent.runtime.studio_service import preflight


def test_missing_config_fails_before_binding(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        preflight(tmp_path / "missing.json", 2024)


def test_occupied_port_fails_before_runtime(tmp_path):
    from unittest.mock import patch, MagicMock
    path = tmp_path / "runtime.json"
    path.touch()
    listener = MagicMock()
    listener.bind.side_effect = OSError("occupied")
    listener.__enter__.return_value = listener
    with patch("phase_agent.runtime.studio_service.socket.socket", return_value=listener):
        with pytest.raises(RuntimeError, match="occupied"):
            preflight(path, 2024)
