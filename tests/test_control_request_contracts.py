from types import SimpleNamespace
import pytest
from run.control_request_contracts import validate_control_request
from run.local_agent_control import LocalAgentControl


@pytest.mark.parametrize("path,body", [("/phase/pause", []),
    ("/phase/propose", {"instruction": {"secret": "private"}}),
    ("/phase/decision", {"decision": "approve"}),
    ("/phase/pause", {"reason": 1}), ("/phase/pause", {"approved": True}),
    ("/phase/propose", {"instruction": " "})])
def test_bad_control_request_diagnostics_do_not_echo_body(path, body):
    with pytest.raises(ValueError) as caught:
        validate_control_request(path, body)
    assert "private" not in str(caught.value)


def test_invalid_approval_never_reaches_review(tmp_path):
    calls = []
    handler = SimpleNamespace(state_path=tmp_path / "state.json",
        review_pending=lambda *args, **kwargs: calls.append((args, kwargs)))
    control = LocalAgentControl(handler)
    with pytest.raises(ValueError):
        control.decide("approve", plan_id="p1", expected_state_version=None, expected_proposal_hash="hash")
    assert calls == []
    control.decide("reject", plan_id="p1", expected_state_version="state", expected_proposal_hash="hash")
    assert len(calls) == 1
    assert calls[0][1]["expected_proposal_hash"] == "hash"


def test_invalid_propose_never_calls_chat(tmp_path):
    calls = []
    class Handler:
        state_path = tmp_path / "state.json"
        def __call__(self, *args, **kwargs):
            calls.append(args)
    control = LocalAgentControl(Handler())
    with pytest.raises(ValueError):
        control.propose({"action": "approve"})
    with pytest.raises(ValueError):
        control.pause(True)
    assert calls == []
