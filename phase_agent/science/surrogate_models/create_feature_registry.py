"""Create the configurable registry of independent feature functions."""

from phase_agent.science.surrogate_models.feature_atom_count import FEATURE_DEFINITION as ATOM_COUNT
from phase_agent.science.surrogate_models.feature_volume_per_atom import (
    FEATURE_DEFINITION as VOLUME_PER_ATOM,
)


def create_feature_registry(extra_features=None) -> dict:
    registry = {item["name"]: dict(item) for item in (ATOM_COUNT, VOLUME_PER_ATOM)}
    for item in extra_features or []:
        name = item.get("name")
        if not name or not callable(item.get("function")) or not item.get("version"):
            raise ValueError("新增特征必须包含 name、version 和 callable function")
        if name in registry:
            raise ValueError(f"重复特征：{name}")
        registry[name] = dict(item)
    return registry
