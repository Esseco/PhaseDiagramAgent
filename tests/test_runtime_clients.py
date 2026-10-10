from unittest.mock import Mock
import pytest
from phase_agent.runtime.runtime_clients import create_optional_runtime_client


def test_no_key_does_not_create_client():
    factory = Mock()
    assert create_optional_runtime_client({}, credential_loader=lambda: None,
                                          client_factory=factory) is None
    factory.assert_not_called()


def test_config_error_is_not_treated_as_missing_key():
    with pytest.raises(ValueError, match="thinking"):
        create_optional_runtime_client({"thinking": "bad"}, credential_loader=lambda: None)


def test_factory_error_is_not_swallowed():
    with pytest.raises(ValueError, match="invalid client"):
        create_optional_runtime_client({}, credential_loader=lambda: "test-key",
            client_factory=Mock(side_effect=ValueError("invalid client")))


def test_credential_error_is_not_swallowed():
    with pytest.raises(OSError, match="credential unavailable"):
        create_optional_runtime_client({},
            credential_loader=Mock(side_effect=OSError("credential unavailable")))
