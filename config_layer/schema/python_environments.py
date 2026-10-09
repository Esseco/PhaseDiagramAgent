"""Explicit host-scoped environments; never infer a remote name from local setup."""


def python_environment(config, host, *, mlip=False):
    key = host + ("_mlip" if mlip else "_python")
    value = (config.get("python_environments") or {}).get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"python_environments.{key} 未配置；请确认{host}端实际环境名（已激活环境可填 current）。")
    return value.strip()


def remote_comparison_model(model, config):
    return {**model, "environment": python_environment(config, "remote", mlip=True)}
