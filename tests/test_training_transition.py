import pytest

from phase_agent.policies.training_transition import (
    ACTIVATE_THEN_REFRESH_APPROVAL,
    resolve_training_transition,
)


def test_training_transition_defaults_to_legacy_manual_activation():
    assert resolve_training_transition({"mlip_finetune": {}}) is None


def test_training_transition_requires_an_allowed_mode():
    with pytest.raises(ValueError, match="unsupported_post_training_transition"):
        resolve_training_transition(
            {"mlip_finetune": {}}, requested="activate_without_refresh_gate"
        )


def test_training_transition_records_explicit_audit_reason():
    result = resolve_training_transition(
        {"mlip_finetune": {}},
        requested=ACTIVATE_THEN_REFRESH_APPROVAL,
        approval_reason="训练审批明确选择自动切换",
    )
    assert result == {
        "mode": ACTIVATE_THEN_REFRESH_APPROVAL,
        "approval_reason": "训练审批明确选择自动切换",
    }
