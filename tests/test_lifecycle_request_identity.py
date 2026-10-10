from phase_agent.graphs.invocation_context import lifecycle_request_id, scientific_message


def test_review_has_new_request_but_action_identity_stays_stable():
    feedback = {"decision": "approve", "comment": "同意"}
    assert lifecycle_request_id("action") == "action"
    reviewed = lifecycle_request_id("action", feedback)
    assert reviewed != "action"
    assert lifecycle_request_id("action", feedback) == reviewed
    with scientific_message("thread:message2"):
        assert lifecycle_request_id("action", feedback) == "webui-thread:message2"
    assert lifecycle_request_id("action") == "action"
