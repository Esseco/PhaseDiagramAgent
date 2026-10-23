"""Explicit host-scoped environments; never infer a remote name from local setup."""


def python_environment(config, host, *, mlip=False):
    key = host + ("_mlip" if mlip else "_python")
    environments = config.get("python_environments") or {}
    if not isinstance(environments, dict):
        raise ValueError("python_environments 必须是按本地/超算分别设置的对象。")
    value = environments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"python_environments.{key} 未配置；请确认{host}端实际环境名（已激活环境可填 current）。"
        )
    return value.strip()


def remote_comparison_model(model, config):
    return {**model, "environment": python_environment(config, "remote", mlip=True)}


def merge_python_environments(confirmed, supplied=None):
    """Fill missing operational facts, never replace a confirmed environment."""
    if not isinstance(confirmed or {}, dict) or not isinstance(supplied or {}, dict):
        raise ValueError("python_environments 必须是按本地/超算分别设置的对象。")
    result = dict(supplied or {})
    for key, value in (confirmed or {}).items():
        if value is not None and value != "" and not (isinstance(value, str) and not value.strip()):
            result[key] = value
    return result


def dft_environment_error(action, config):
    """Cheap admission check for remote DFT preparation; no model or filesystem IO."""
    if action.get("tool") != "select_dft_candidates" or not config.get("upload_batches_directory"):
        return None
    decisions = (action.get("parameters") or {}).get("decisions") or []
    if not any(
        isinstance(row, dict) and row.get("action") in {"DFT_RELAX", "DFT_SINGLE_POINT"}
        for row in decisions
    ):
        return None
    try:
        python_environment(config, "remote", mlip=True)
    except ValueError as error:
        return str(error) + " 补齐项目环境配置后复用原选点；未生成输入，无需重新选点。"
    return None
