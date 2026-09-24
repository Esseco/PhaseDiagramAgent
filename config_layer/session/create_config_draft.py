"""Create an editable configuration draft and dialogue session."""

from copy import deepcopy
from datetime import datetime, timezone
import uuid


def create_config_draft(config: dict, *, require_workspace_path: bool = False) -> dict:
    defaults = deepcopy(config)
    if require_workspace_path:
        question = (
            "开始配置前，请先发送本地工作区根目录的绝对路径。"
            "我会先展示配置文件及各类结果的保存位置；你回复“确认存储路径”后，"
            "才会在该目录生成带注释的设置 JSON。"
        )
        dialogue_type = "workspace_path_question"
        setup_stage = "awaiting_storage_path"
        default_prompt_status = "skipped"
    else:
        question = "是否采用默认参数？选择是后会输出完整参数合集，你只需修改需要覆盖的项目。"
        dialogue_type = "default_parameter_question"
        setup_stage = None
        default_prompt_status = "awaiting_response"

    session = {
        "session_id": f"config-{uuid.uuid4().hex[:12]}",
        "status": "draft",
        "draft_revision": 1,
        "config": defaults,
        "dialogue": [{"type": dialogue_type, "role": "assistant", "message": question}],
        "default_parameter_prompt": {
            "status": default_prompt_status,
            "question": question,
            "default_parameters": defaults,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "confirmed_snapshot": None,
    }
    if setup_stage:
        session["setup_stage"] = setup_stage
    return session
