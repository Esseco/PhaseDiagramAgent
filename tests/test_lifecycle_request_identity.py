from phase_agent.graphs.invocation_context import lifecycle_request_id, scientific_message


def test_review_keeps_the_approved_action_identity():
    feedback = {"decision": "approve", "comment": "同意"}
    assert lifecycle_request_id("action") == "action"
    reviewed = lifecycle_request_id("action", feedback)
    assert reviewed == "action"
    assert lifecycle_request_id("action", feedback) == reviewed
    with scientific_message("thread:message2"):
        assert lifecycle_request_id("action", feedback) == "action"
    assert lifecycle_request_id("action") == "action"
