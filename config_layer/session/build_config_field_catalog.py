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
    fields = {}

    def visit(value, path):
        if isinstance(value, dict) and value:
            for key, child in value.items():
                if any(word in str(key).upper() for word in ("API_KEY", "PASSWORD", "SECRET", "TOKEN")):
                    continue
                if path == "system.boundary.H":
                    continue
                visit(child, f"{path}.{key}" if path else str(key))
        else:
            fields[path] = {"type": type(value).__name__}
            if path in meanings:
                fields[path]["meaning"] = meanings[path]

    visit(config, "")
    return fields
