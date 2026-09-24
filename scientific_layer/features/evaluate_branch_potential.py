"""使用显式提供的代理模型评估 branch 潜力。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def evaluate_branch_potential(
    features: Any,
    *,
    model: Any | None = None,
    backend: Callable[..., Any] | None = None,
    model_path: str | Path | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """调用 backend 或模型 predict；不内置虚假评分。"""
    if backend is None and model is None:
        return {
            "stage": "relax_and_feature",
            "status": "not_configured",
            "score": None,
            "error": "代理模型未配置",
        }
    try:
        if backend is not None:
            prediction = backend(
                features=features,
                model_path=model_path,
                parameters=dict(parameters or {}),
            )
        else:
            prediction = model.predict(features)
        if hasattr(prediction, "tolist"):
            prediction = prediction.tolist()
        return {
            "stage": "relax_and_feature",
            "status": "completed",
            "score": prediction,
            "model_path": str(model_path) if model_path else None,
            "error": None,
        }
    except Exception as error:
        return {
            "stage": "relax_and_feature",
            "status": "failed",
            "score": None,
            "error": f"{type(error).__name__}: {error}",
        }
