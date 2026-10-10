"""Describe existing parameter paths without exposing their additional values."""


def build_config_field_catalog(config):
    meanings = {
        "run.initial_states_per_branch": "每个branch初态数量上限；静电能前10随机选，实际最多3个；不足时保留实际数量",
        "run.batch_size": "本轮入选branch数量上限",
        "run.total_quota": "本轮生成候选branch总配额",
        "run.seed": "随机种子",
        "budgets.total_relative_cost": "总相对成本预算",
        "mlip.model_path": "超算端MLIP模型路径",
        "calculation.mlip_version": "科学计算MLIP版本，与对话模型不同",
    }
    meanings.update(
        {
            "system.boundary.P": "用户允许的相边界；列表表示所有Na含量使用同一相列表，对象表示按Na端点/区间分相",
            "system.boundary.TM_ratio": "用户指定的过渡金属元素及正比例，完整替换元素映射",
            "system.H_generation.size_max": "生成H的尺寸上限",
            "python_environments.remote_python": "超算DFT/提取环境",
            "python_environments.remote_mlip": "超算Relax/MC/MACE环境",
        }
    )
    derived = {
        "system.constraints.phases": "system.boundary.P",
        "system.constraints.TM_ratio": "system.boundary.TM_ratio",
        "system.species.substitutional": "system.boundary.TM_ratio",
        "system.occupancy_rules.substitutional_sites": "system.boundary.TM_ratio",
    }
    atomic = {"system.boundary.P", "system.boundary.TM_ratio", *derived}
    fields = {}

    def visit(value, path):
        if isinstance(value, dict) and value and path not in atomic:
            for key, child in value.items():
                if any(
                    word in str(key).upper() for word in ("API_KEY", "PASSWORD", "SECRET", "TOKEN")
                ):
                    continue
                if path == "system.boundary.H":
                    continue
                visit(child, f"{path}.{key}" if path else str(key))
        else:
            fields[path] = {"type": type(value).__name__, "editable": path not in derived}
            if path in derived:
                fields[path]["derived_from"] = derived[path]
            if path == "system.boundary.P":
                fields[path]["replace_types"] = ["list", "dict"]
            elif path == "system.boundary.TM_ratio":
                fields[path]["replace_types"] = ["dict"]
            if path in meanings:
                fields[path]["meaning"] = meanings[path]

    visit(config, "")
    return fields


def config_patch_field_errors(response, catalog):
    """Reject derived fields before a patch reaches filesystem mutation."""
    if not isinstance(response, dict) or not isinstance(response.get("patch", {}), dict):
        return ["patch must be an object"]

    def paths(patch, prefix=""):
        for key, value in patch.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path
            if isinstance(value, dict):
                yield from paths(value, path)

    errors = []
    for path in paths(response.get("patch", {})):
        for field, contract in catalog.items():
            if not contract["editable"] and (path == field or path.startswith(field + ".")):
                errors.append(
                    f"{path}: read-only derived field; write source {contract['derived_from']}"
                )
                break
    return errors
