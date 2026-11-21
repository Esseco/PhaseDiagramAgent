"""Explicit post-training transition policy for portable MLIP handoffs."""

from copy import deepcopy


MANUAL_ACTIVATION = "manual_activation"
ACTIVATE_THEN_REFRESH_APPROVAL = "activate_then_refresh_approval"
_ALLOWED = {MANUAL_ACTIVATION, ACTIVATE_THEN_REFRESH_APPROVAL}


def resolve_training_transition(config, *, requested=None, approval_reason=None):
    """Return an immutable transition record, or ``None`` for legacy behavior.

    The manual path remains the default.  The automatic path is only enabled by
    an explicit action parameter or the corresponding reviewed configuration;
    it never prepares or submits the subsequent refresh calculation.
    """

    activation = ((config or {}).get("mlip_finetune") or {}).get("activation") or {}
    mode = requested if requested is not None else activation.get(
        "post_training_transition", MANUAL_ACTIVATION
    )
    if mode is None:
        mode = MANUAL_ACTIVATION
    if mode not in _ALLOWED:
        raise ValueError("unsupported_post_training_transition")
    if mode == MANUAL_ACTIVATION:
        return None
    reason = str(
        approval_reason
        or activation.get("post_training_approval_reason")
        or "已批准训练交接；验证通过后自动切换模型，并单独审批刷新"
    ).strip()
    if not reason:
        raise ValueError("post_training_transition_approval_reason_required")
    return {"mode": mode, "approval_reason": reason}


def same_training_transition(left, right):
    """Compare persisted transition records without accepting mutable extras."""

    return deepcopy(left or None) == deepcopy(right or None)
