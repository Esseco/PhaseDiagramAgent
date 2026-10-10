"""Read saved configuration without creating a draft or preparing science."""

from pathlib import Path
import json


def configuration_view(handler, facts):
    session = handler.workflow_kwargs.get("config_session") or {}
    confirmed = (
        facts.get("confirmed_config")
        or (session.get("confirmed_snapshot") or {}).get("config")
        or session.get("config")
        or {}
    )
    state_path = Path(handler.state_path).resolve()
    root = (
        state_path.parent.parent
        if state_path.parent.name == "workflow_state"
        else state_path.parent
    )
    source = root / "parameters/search_config.project.json"
    if not source.is_file():
        source = root / "search_config.project.json"
    config = confirmed
    label = "已确认配置"
    if source.is_file():
        from phase_agent.configuration.session.load_editable_config_json import (
            load_editable_config_json,
        )

        try:
            config = load_editable_config_json(source, confirmed)
            label = "当前设置文件（是否用于运行以确认快照为准）"
        except (OSError, TypeError, ValueError, KeyError) as error:
            return f"设置文件读取失败：{type(error).__name__}: {error}；未修改文件或待审批方案。"
    system = config.get("system") or {}
    fields = {
        "相边界": (system.get("boundary") or {}).get("P"),
        "TM比例": (system.get("boundary") or {}).get("TM_ratio"),
        "H生成设置": system.get("H_generation"),
        "Python环境": config.get("python_environments"),
        "远端模型路径": (config.get("mlip") or {}).get("model_path"),
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
    if source.is_file():
        lines.append(f"设置文件：{source}")
    lines.append("本次仅查看，未进入配置修订、生成结构或批准待审批方案。")
    return "\n".join(lines)
