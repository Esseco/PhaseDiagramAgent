"""Preserve model capability metadata across client adapters."""

_MODEL_CAPABILITIES = (
    "_decision_source",
    "decision_timeout_seconds",
    "decision_repair_limit",
    "_codex_structured",
    "_frozen_generation_draft",
    "_frozen_mc_action",
    "_frozen_dft_action",
    "_frozen_training_action",
    "_frozen_exact_relax_action",
    "_validates_original_facts",
    "_explicit_direct_scope",
    "_semantic_dialogue",
    "supports_expert_advice",
    "supports_concurrent_experts",
    "expert_client",
    "supports_simulation",
    "supports_strategic_review",
)


def copy_model_capabilities(source, target):
    """Copy declared capabilities without inventing defaults for missing ones."""
    for name in _MODEL_CAPABILITIES:
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))
    return target
