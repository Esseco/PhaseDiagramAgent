from types import SimpleNamespace
import pytest
from phase_agent.runtime.control_request_contracts import validate_control_request
from phase_agent.runtime.local_agent_control import LocalAgentControl


@pytest.mark.parametrize("path,body", [
    ("/phase/config/patch", {"patch": []}), ("/phase/config/patch", {"patch": {"a..b": 1}}),
    ("/phase/config/patch", {"patch": {}, "reasons": []}),
    ("/phase/config/confirm", {"explicit": "true"}),
    ("/phase/memory/review", {"proposal_id": "p1", "approved": 1}),
    ("/phase/memory/propose", {"record": "private-data"}),
    ("/phase/memory/skills/publish", {"draft_directory": "path", "approved": "yes"}),
    ("/phase/memory/skills/import", {"approved": True})])
def test_bad_config_memory_input_rejected(path, body):
    with pytest.raises(ValueError) as caught:
        validate_control_request(path, body)
    assert "private-data" not in str(caught.value)


def test_invalid_direct_requests_do_not_read_or_write_state(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{}")
    before = path.read_bytes()
    control = LocalAgentControl(SimpleNamespace(state_path=path))
    for call in [lambda: control.confirm_config(explicit="yes"),
                 lambda: control.review_memory("p1", approved=1),
                 lambda: control.propose_memory([]),
                 lambda: control.patch_config({"a..b": 1})]:
        with pytest.raises(ValueError):
            call()
    assert path.read_bytes() == before


def test_missing_confirmation_flags_are_not_approval():
    from phase_agent.runtime.control_request_contracts import ConfigConfirmRequest, MemoryReviewRequest, SkillPublishRequest
    assert ConfigConfirmRequest.model_validate({}).explicit is False
    assert MemoryReviewRequest.model_validate({"proposal_id": "p1"}).approved is False
    assert SkillPublishRequest.model_validate({"draft_directory": "path"}).approved is False
