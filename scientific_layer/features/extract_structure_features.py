"""提取基础物理描述符或由外部后端计算 embedding。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from scientific_layer.structures.boundary_utils import load_structure


def extract_structure_features(
    structure: Any,
    *,
    representation: str = "physical",
    backend: Callable[..., Any] | None = None,
    model_path: str | Path | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """physical 可直接运行；embedding 必须提供实际提取后端和模型。"""
    present = load_structure(structure)
    if representation == "physical":
        composition = present.composition.get_el_amt_dict()
        features = {
            "atom_count": len(present),
            "volume": float(present.volume),
            "volume_per_atom": float(present.volume / len(present)),
            "density": float(present.density),
            **{
                f"fraction_{key}": float(value / len(present))
                for key, value in sorted(composition.items())
            },
        }
    elif representation == "embedding":
        if backend is None or model_path is None:
            return {
                "stage": "relax_and_feature",
                "status": "not_configured",
                "representation": representation,
                "features": None,
                "model_path": str(model_path) if model_path else None,
                "error": "embedding backend 或 model_path 未配置",
            }
        try:
            values = backend(
                structure=present,
                model_path=Path(model_path),
                parameters=dict(parameters or {}),
            )
            features = values.tolist() if hasattr(values, "tolist") else values
        except Exception as error:
            return {
                "stage": "relax_and_feature",
                "status": "failed",
                "representation": representation,
                "features": None,
                "model_path": str(model_path),
                "error": f"{type(error).__name__}: {error}",
            }
    else:
        raise ValueError("representation 必须是 physical 或 embedding")
    return {
        "stage": "relax_and_feature",
        "status": "completed",
        "representation": representation,
        "features": features,
        "model_path": str(model_path) if model_path else None,
        "error": None,
    }
