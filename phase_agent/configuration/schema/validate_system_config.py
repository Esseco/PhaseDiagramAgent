"""验证体系配置的最小公共协议。"""


def validate_system_config(config: dict) -> dict:
    required = {
        "system_id",
        "species",
        "occupancy_rules",
        "constraints",
        "branch_schema",
        "generation",
        "calculation_workflow",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"体系配置缺少字段：{sorted(missing)}")
    fields = config["branch_schema"].get("fields", [])
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("branch_schema.fields 必须非空且不能重复")
    from phase_agent.configuration.schema.validate_configuration_space import (
        validate_configuration_space,
    )

    roles = validate_configuration_space(config)
    if not roles["valid"]:
        raise ValueError("；".join(roles["errors"]))
    stages = config["calculation_workflow"].get("stages", [])
    names = [item.get("name") for item in stages]
    if not names or None in names or len(names) != len(set(names)):
        raise ValueError("计算阶段名称必须非空且唯一")
    return config
