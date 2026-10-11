"""Read saved configuration without creating a draft or preparing science."""

import json


def configuration_view(handler, facts):
    from phase_agent.runtime.project_context import handler_project_context, current_config_session
    from phase_agent.analysis.state.project_authority import confirmed_configuration

    card = handler_project_context(handler, facts)
    config = confirmed_configuration(facts, current_config_session(handler))
    label = "当前生效配置 " + str(card["configuration"]["version"] or "未确认")
    source = (card.get("editable_file") or {}).get("path")
    system = config.get("system") or {}
    fields = {
        "相边界": (system.get("boundary") or {}).get("P"),
        "TM比例": (system.get("boundary") or {}).get("TM_ratio"),
        "H生成设置": system.get("H_generation"),
        "Python环境": config.get("python_environments"),
        "当前运行模型": (card.get("active_model") or {}).get("version"),
        "当前模型路径": (card.get("active_model") or {}).get("model_path"),
        "配置中的初始模型": card.get("initial_model_version"),
        "母结构目录": system.get("phase_reference_directory"),
    }
    # Matrices and full runtime defaults are available in the source, not dumped into chat.
    h = fields["H生成设置"] or {}
    fields["H生成设置"] = {
        key: h[key]
        for key in ("enabled", "size_min", "size_max", "size_step", "first_round_max_det_H")
        if key in h
    }
    lines = [
        label + "：",
        *[
            f"- {name}：{json.dumps(value, ensure_ascii=False)}"
            for name, value in fields.items()
            if value is not None
        ],
    ]
    if source:
        lines.append(f"设置文件（未确认修改不影响运行）：{source}")
    draft = card["draft"]
    if draft["status"] != "no_changes":
        lines.append(
            f"未生效草稿：版本 {draft['revision']}；修改项："
            + "、".join(
                draft.get("changed_paths")
                or (card.get("editable_file") or {}).get("changed_paths")
                or []
            )
        )
    if (card.get("editable_file") or {}).get("error"):
        lines.append("设置文件读取问题：" + card["editable_file"]["error"])
    lines.extend(card.get("consistency_issues") or [])
    lines.append("本次仅查看，未进入配置修订、生成结构或批准待审批方案。")
    return "\n".join(lines)
