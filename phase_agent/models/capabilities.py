"""Preserve model capability metadata across client adapters."""

_MODEL_CAPABILITIES = (
    "_decision_source",
    "_codex_structured",
    "_validates_original_facts",
    "_semantic_dialogue",
    "supports_expert_advice",
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
