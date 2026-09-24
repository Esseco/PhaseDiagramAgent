"""First-run, config-only Open WebUI dialogue.

This handler can only revise an unconfirmed JSON draft. It never calls the
scientific workflow. An exact user confirmation creates the versioned snapshot;
the next chat turn then switches to the regular workflow handler.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from config_layer.schema.validate_search_config import validate_search_config
from config_layer.session.apply_config_revision import apply_config_revision
from config_layer.session.answer_default_parameter_prompt import answer_default_parameter_prompt
from config_layer.session.confirm_config_snapshot import confirm_config_snapshot
from config_layer.session.record_config_dialogue import record_config_dialogue
from config_layer.session.resolve_workspace_paths import (
    default_workspace_storage, resolve_workspace_paths,
)
from config_layer.session.save_config_session import save_config_session


CONFIG_AGENT_SYSTEM_PROMPT = """你是材料相图搜索项目的首次配置助手，不是计算执行器。
只讨论并检查配置草稿。不得提交任务、调用科学计算、创建结构、运行命令或声称计算已完成。
仅当用户明确提供或确认信息时，才返回对应配置修改；不确定时提出问题，不猜测边界矩阵、预算、DFT 设置或超算命令。
只返回 JSON：{"reply":"给用户的中文答复","patch":{"点分配置路径": JSON值},"reasons":{"点分配置路径":"修改原因"},"questions":["待用户回答的问题"],"ready_for_search":false}。
patch 只修改草稿，不会确认配置或启动运行；每项 patch 都会向用户展示。区分硬约束与可调整建议。
当 mode 为 configuration_json_review 时，只检查并解释用户刚导入的 JSON 配置，patch 必须为空；列出重要缺项、冲突、歧义和需要用户确认的内容，不要擅自填值。只有在没有阻止开始搜索的缺项、冲突或歧义时才令 ready_for_search=true，否则为 false。
本项目初次配置的重点是本地母结构路径、体系 boundary、远端 MACE 模型路径、预算、DFT/atomate 参数和收敛标准。
首次配置先收集工作区根路径和 Agent 模型版本（DeepSeek V4.1 Flash 或 V4 Pro）；允许同一条消息或分别提供。只展示这两项和设置 JSON 路径，等待用户回复“确认”。确认前不得创建设置 JSON/工作区目录或修改运行时模型。确认后更新本地运行时模型（若用户选择切换）、生成带注释的 JSON，并直接列出需要填写的内容。用户编辑 JSON 后发送“读取配置 JSON”，由 Agent 审核；程序检查也通过后，等待用户回复“同意”，再保存配置快照并进入搜索 run。不得在此之前派发科学计算。DeepSeek 模型属于本地 Open WebUI 运行时，不写入科学搜索配置。
配置 JSON 输出后说明工作区和主要文件位置，并列出用户需要填写的字段；用户可调整 storage.paths。Agent 审核、程序检查与用户“同意”均完成后才开始搜索流程；不会因此自动提交计算。
本地绝不能尝试打开/加载超算上的 MACE 模型。超算模型路径仅作为远端配置元数据。
用户提供的“已知候选信息”只是可核对的建议值；不要假定已写入草稿，先向用户说明并征求确认。"""


BOOTSTRAP_HINTS = {
    "local_initial_structure_directory": r"E:\0-FM-PhaseDiagram\InitFile\Struct",
    "expected_phase_files": ["O3.vasp", "O1.vasp", "P3.vasp", "OP2.vasp"],
    "remote_only_mlip_model_path": "/data/home/lichaoyue/Py-lzy/MLIP_Model/mace-mh-1.model",
    "notes": [
        "以上是待用户在 Agent 对话中核对的候选值，不会自动写入或确认配置。",
        "母结构位于本地电脑；MACE 模型路径属于超算，只能记录为远端元数据。",
    ],
}


def configuration_readiness(session: dict, *, base_directory, phase_references_path=None,
                            workspace_root_default=None) -> dict:
    """Return missing/conflicting facts required before confirming first-run config."""
    config = session.get("config") or {}
    try:
        audit = validate_search_config(config)
    except (TypeError, ValueError, KeyError, AttributeError) as error:
        audit = {"missing": [], "conflicts": [f"配置字段格式无效：{error}"], "ambiguities": []}
    missing = list(audit.get("missing") or [])
    conflicts = list(audit.get("conflicts") or [])
    ambiguities = list(audit.get("ambiguities") or [])
    system = config.get("system") or {}
    awaiting_workspace = session.get("setup_stage") in {
        "awaiting_storage_path", "awaiting_storage_confirmation"
    }
    if awaiting_workspace:
        missing.append("storage.workspace_root (需要先选择并确认本地工作区目录)")
        storage_paths = {}
    else:
        storage_config = config.get("storage") or default_workspace_storage(
            workspace_root_default or (Path(base_directory) / "../local")
        )
        try:
            resolved_storage = resolve_workspace_paths(
                {**config, "storage": storage_config}, base_directory=base_directory
            )
            storage_paths = {key: str(value) for key, value in resolved_storage.items()}
        except (TypeError, ValueError, OSError) as error:
            conflicts.append(f"storage 保存路径无效：{error}")
            storage_paths = {}
    boundary = system.get("boundary")
    if not isinstance(boundary, dict):
        missing.append("system.boundary (需要 P、H、TM_ratio)")
        phases = []
    else:
        raw_phases = boundary.get("P")
        phases = ([str(phase) for phase in raw_phases]
                  if isinstance(raw_phases, list) and raw_phases
                  and all(isinstance(phase, str) and phase.strip() for phase in raw_phases) else [])
        if not phases:
            missing.append("system.boundary.P (非空相列表)")
        h_by_phase = boundary.get("H")
        if not isinstance(h_by_phase, dict):
            missing.append("system.boundary.H (按相列出允许的超胞矩阵)")
        else:
            if set(h_by_phase) - set(phases):
                conflicts.append("system.boundary.H 包含 P 之外的相")
            for phase in phases:
                matrices = h_by_phase.get(phase)
                if not isinstance(matrices, list) or not matrices:
                    missing.append(f"system.boundary.H.{phase}")
                    continue
                for index, matrix in enumerate(matrices):
                    valid_matrix = (
                        isinstance(matrix, list) and len(matrix) in {2, 3}
                        and all(isinstance(row, list) and len(row) == len(matrix)
                                and all(isinstance(value, int) and not isinstance(value, bool)
                                        for value in row) for row in matrix)
                    )
                    if not valid_matrix:
                        conflicts.append(f"system.boundary.H.{phase}[{index}] 必须是 2×2 或 3×3 整数矩阵")
        tm_ratio = boundary.get("TM_ratio")
        if (not isinstance(tm_ratio, dict) or not tm_ratio
                or any(not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
                       for value in tm_ratio.values())):
            missing.append("system.boundary.TM_ratio (元素及正比例值)")
        configured_phases = [str(phase) for phase in (system.get("constraints") or {}).get("phases") or []]
        if configured_phases and phases and set(configured_phases) != set(phases):
            conflicts.append("system.boundary.P 与 system.constraints.phases 不一致")
        configured_ratio = (system.get("constraints") or {}).get("TM_ratio")
        if configured_ratio and isinstance(tm_ratio, dict) and configured_ratio != tm_ratio:
            conflicts.append("system.boundary.TM_ratio 与 system.constraints.TM_ratio 不一致")

    references = system.get("phase_references") or {}
    reference_file = Path(phase_references_path) if phase_references_path else None
    if not references and reference_file and reference_file.is_file():
        try:
            references = json.loads(reference_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            references = {}
    if not isinstance(references, dict) or not references:
        missing.append("system.phase_references (各相母结构文件)")
        references = {}
    for phase in phases:
        if phase not in references:
            missing.append(f"system.phase_references.{phase}")
    for phase, value in references.items():
        path_value = value.get("path") if isinstance(value, dict) else value
        if not isinstance(path_value, str) or not path_value.strip():
            missing.append(f"system.phase_references.{phase} (必须是本地结构文件路径)")
            continue
        path = Path(path_value)
        if not path.is_absolute():
            path = Path(base_directory) / path
        if not path.is_file():
            missing.append(f"system.phase_references.{phase} 文件不存在：{path}")

    mlip = config.get("mlip") or {}
    remote_model = (mlip.get("model_path")
                    or ((config.get("supercomputer") or {}).get("paths") or {}).get("mlip_model"))
    if not remote_model:
        missing.append("mlip.model_path (超算端 MACE 模型路径)")
    return {
        "ready": not missing and not conflicts and not ambiguities,
        "missing": sorted(set(missing)),
        "conflicts": sorted(set(conflicts)),
        "ambiguities": sorted(set(ambiguities)),
        "audit": audit,
        "storage_paths": storage_paths,
    }


class ConfigurationChatHandler:
    """Interactive configuration stage that cannot execute search actions."""

    def __init__(self, workflow_kwargs, *, config_session_path, base_directory,
                 phase_references_path=None, editable_config_path=None,
                 editable_config_filename="search_config.draft.json",
                 workspace_root_default=None, agent_client=None, runtime_factory=None,
                 current_deepseek_model="deepseek-v4-pro", deepseek_model_switcher=None):
        self.workflow_kwargs = dict(workflow_kwargs)
        self.state_path = Path(workflow_kwargs["state_path"])
        self.config_session_path = Path(config_session_path)
        self.base_directory = Path(base_directory)
        self.phase_references_path = Path(phase_references_path) if phase_references_path else None
        self.editable_config_path = Path(editable_config_path) if editable_config_path else None
        self.editable_config_filename = safe_config_filename(editable_config_filename)
        session = workflow_kwargs.get("config_session") or {}
        selected_root = (
            session.get("pending_workspace_root")
            or ((session.get("config") or {}).get("storage") or {}).get("workspace_root")
        )
        try:
            root_path = normalize_workspace_path(
                selected_root or workspace_root_default or self.config_session_path.parent,
                base_directory=self.base_directory,
            )
        except (TypeError, ValueError, OSError):
            root_path = self.config_session_path.parent.resolve()
        self.workspace_root_default = root_path
        self.agent_client = agent_client
        self.current_deepseek_model = current_deepseek_model
        self.deepseek_model_switcher = deepseek_model_switcher
        self.runtime_factory = runtime_factory
        self.delegate = None

    def configuration_readiness(self, session=None):
        return configuration_readiness(
            session or self.workflow_kwargs.get("config_session") or {},
            base_directory=self.base_directory,
            phase_references_path=self.phase_references_path,
            workspace_root_default=self.workspace_root_default,
        )

    def __call__(self, messages, *, conversation_id=None):
        session = self.workflow_kwargs.get("config_session") or {}
        if session.get("status") == "confirmed":
            if self.delegate is None:
                if not callable(self.runtime_factory):
                    return "配置快照已确认。请重启本地 Agent 进入搜索对话模式；本次没有启动计算。"
                self.delegate = self.runtime_factory()
            return self.delegate(messages, conversation_id=conversation_id)

        message = _latest_user_message(messages)
        setup_stage = session.get("setup_stage")
        if setup_stage in {"awaiting_storage_path", "awaiting_storage_confirmation"}:
            return self._handle_workspace_setup(session, message)
        from run.resolve_deepseek_model_request import resolve_deepseek_model_request
        requested_model = resolve_deepseek_model_request(message)
        if requested_model:
            if not callable(self.deepseek_model_switcher):
                return "本地运行时未配置 DeepSeek 模型切换接口；运行时设置未更改。"
            try:
                selected_model, client = self.deepseek_model_switcher(requested_model)
            except (OSError, TypeError, ValueError) as error:
                return f"DeepSeek 模型切换失败：{type(error).__name__}: {error}。运行时设置未更改。"
            self.current_deepseek_model = selected_model
            self.agent_client = client
            reply = _deepseek_model_change_reply(selected_model)
            return self._save_and_report(self._with_turn(session, message, reply), reply)
        if message.strip() in {"同意", "同意开始", "确认配置", "confirm configuration"}:
            readiness = self.configuration_readiness(session)
            reviewed_revision = session.get("agent_reviewed_revision")
            reviewed_hash = session.get("agent_reviewed_config_hash")
            current_hash = self._editable_config_hash()
            if (readiness["ready"] and reviewed_revision
                    and reviewed_revision == session.get("draft_revision")
                    and reviewed_hash and reviewed_hash == current_hash):
                return self._confirm_and_start_search(
                    session, user_message=message, conversation_id=conversation_id,
                )
            return self._save_and_report(
                self._with_turn(session, message, ""),
                "还不能开始：请先检查配置 JSON 是否在审核后又被修改；若已修改，请重新发送“读取配置 JSON”，"
                "等待 Agent 审核通过后，再回复“同意”。",
            )
        if _is_config_json_import_command(message):
            return self._import_and_review_config_json(session, message)
        if session.get("default_parameter_prompt", {}).get("status") == "awaiting_response":
            choice = _default_choice(message)
            if choice is not None:
                updated = answer_default_parameter_prompt(session, use_defaults=choice)
                path_note = (f"完整可编辑参数 JSON：{self.editable_config_path}。"
                             "你可以直接修改并发送“读取配置 JSON”导入检查。"
                             if self.editable_config_path else "")
                return self._save_and_report(
                    updated, "已记录默认参数选择。配置仍是草稿，尚未确认或运行。" + path_note
                )

        if not self.agent_client:
            return self._unavailable_message()
        try:
            result = self.agent_client({
                "mode": "configuration_dialogue",
                "instruction": message,
                "draft_config": session.get("config") or {},
                "draft_revision": session.get("draft_revision"),
                "bootstrap_hints": session.get("bootstrap_hints") or BOOTSTRAP_HINTS,
                "editable_config_json": str(self.editable_config_path) if self.editable_config_path else None,
                "readiness": self.configuration_readiness(session),
            })
        except Exception as error:
            diagnostic = getattr(error, "safe_message", type(error).__name__)
            self._record(session, "user", message)
            self._record(session, "assistant", f"配置助手暂时不可用（{diagnostic}）；草稿未更改。")
            return f"配置助手暂时不可用：{diagnostic}\n配置草稿未更改；请修正提示的问题后重试。"
        return self._apply_agent_response(session, message, result)

    def _handle_workspace_setup(self, session, user_message):
        normalized = " ".join(str(user_message).strip().lower().split())
        from run.resolve_deepseek_model_request import resolve_deepseek_model_request

        root_value, requested_model = _parse_workspace_setup_values(user_message)
        requested_model = requested_model or resolve_deepseek_model_request(user_message)
        if requested_model:
            updated = dict(session)
            updated["pending_deepseek_model"] = requested_model
            if root_value:
                return self._preview_workspace_root(
                    updated, user_message, path_value=root_value,
                )
            root_value = session.get("pending_workspace_root")
            if root_value:
                try:
                    root = normalize_workspace_path(root_value, base_directory=self.base_directory)
                except (TypeError, ValueError, OSError):
                    root = None
                if root is not None:
                    return self._save_workspace_preview(updated, user_message, root)
            label = _deepseek_model_label(requested_model)
            reply = (
                f"已暂存可选模型：{label}（`{requested_model}`）。"
                "还没有修改运行时设置。请发送本地工作区根目录路径；"
                "路径预览后，你可以一起确认工作区和模型。"
            )
            return self._save_and_report(self._with_turn(updated, user_message, reply), reply)
        if normalized in {"保持当前模型", "沿用当前模型", "不切换模型", "keep current model"}:
            updated = dict(session)
            updated.pop("pending_deepseek_model", None)
            root_value = session.get("pending_workspace_root")
            if root_value:
                try:
                    root = normalize_workspace_path(root_value, base_directory=self.base_directory)
                except (TypeError, ValueError, OSError):
                    root = None
                if root is not None:
                    return self._save_workspace_preview(updated, user_message, root)
            reply = f"将沿用当前模型 {_deepseek_model_label(self.current_deepseek_model)}。请发送工作区根目录路径。"
            return self._save_and_report(self._with_turn(updated, user_message, reply), reply)
        if normalized in {"确认配置", "confirm configuration", "读取配置 json", "读取配置json"}:
            return self._save_and_report(
                session,
                "请先提供工作区根路径和 DeepSeek 模型版本。确认这两项后才生成设置 JSON；"
                "随后进入 Agent 配置流程，边界等问题配置完成并确认后再开始搜索 run。",
            )
        if normalized in {"取消", "重新选择", "修改路径", "重新输入路径"}:
            updated = dict(session)
            updated.pop("pending_workspace_root", None)
            updated["setup_stage"] = "awaiting_storage_path"
            return self._save_and_report(
                updated,
                "好的，请重新发送本地工作区根目录路径。确认前不会创建设置 JSON 或工作区目录。",
            )
        if (session.get("setup_stage") == "awaiting_storage_confirmation"
                and normalized in {
                    "确认", "确认并开始配置", "确认并开始", "开始配置", "确认存储路径",
                    "确认工作区", "确认路径", "confirm workspace",
                }):
            return self._confirm_workspace_root(session, user_message)
        return self._preview_workspace_root(
            session, user_message, path_value=root_value,
        )

    def _preview_workspace_root(self, session, user_message, *, path_value=None):
        try:
            root = normalize_workspace_path(
                path_value if path_value is not None else _extract_workspace_path(user_message),
                base_directory=self.base_directory,
            )
        except (TypeError, ValueError, OSError):
            model_choice = session.get("pending_deepseek_model") or self.current_deepseek_model
            message = (
                "首次只需提供两项：工作区根路径，以及 Agent 模型版本。"
                "请按两行发送，例如：`工作区根路径：E:\\PhaseSearch`、"
                "`Agent版本：V4.1 Flash`。模型可选 V4.1 Flash（`deepseek-flash`）"
                "或 V4 Pro（`deepseek-v4-pro`）；不写模型则沿用当前版本 "
                f"{_deepseek_model_label(model_choice)}。"
                "收到后我会只回显路径和版本供你确认。"
            )
            self._save(self._with_turn(session, user_message, message))
            return message
        if root.exists() and not root.is_dir():
            message = f"该路径已存在但不是文件夹：{root}。请换一个工作区目录。"
            return self._save_and_report(self._with_turn(session, user_message, message), message)

        return self._save_workspace_preview(session, user_message, root)

    def _save_workspace_preview(self, session, user_message, root):

        from config_layer.session.resolve_workspace_paths import default_workspace_storage

        self.workspace_root_default = root
        storage = default_workspace_storage(root)
        resolve_workspace_paths({"storage": storage}, base_directory=self.base_directory)
        updated = dict(session)
        updated["setup_stage"] = "awaiting_storage_confirmation"
        updated["pending_workspace_root"] = str(root)
        model = session.get("pending_deepseek_model") or self.current_deepseek_model
        lines = [
            f"工作区根目录：{root}",
            f"Agent 模型版本：{_deepseek_model_label(model)}（`{model}`）",
            f"设置 JSON：{root / self.editable_config_filename}",
        ]
        lines.append(
            "确认这两项后回复“确认”；如需修改，请重新发送正确的路径或模型版本。"
        )
        reply = "\n".join(lines)
        updated = self._with_turn(updated, user_message, reply)
        self._save(updated)
        return reply

    def _confirm_workspace_root(self, session, user_message):
        root_value = session.get("pending_workspace_root")
        if not root_value:
            return self._preview_workspace_root(session, "")
        try:
            root = normalize_workspace_path(root_value, base_directory=self.base_directory)
        except (TypeError, ValueError, OSError):
            updated = dict(session)
            updated.pop("pending_workspace_root", None)
            updated.pop("editable_config_json_path", None)
            updated["setup_stage"] = "awaiting_storage_path"
            reply = "保存路径无效，尚未创建设置文件。请重新发送一行本地工作区目录路径。"
            return self._save_and_report(self._with_turn(updated, user_message, reply), reply)
        if root.exists() and not root.is_dir():
            updated = dict(session)
            updated.pop("pending_workspace_root", None)
            updated["setup_stage"] = "awaiting_storage_path"
            reply = f"该路径已存在但不是文件夹：{root}。尚未创建设置文件，请重新选择工作区目录。"
            return self._save_and_report(self._with_turn(updated, user_message, reply), reply)
        from config_layer.session.resolve_workspace_paths import default_workspace_storage

        storage = default_workspace_storage(root)
        draft_path = (root / self.editable_config_filename).resolve()
        selected_model = session.get("pending_deepseek_model")
        if selected_model and selected_model != self.current_deepseek_model:
            if not callable(self.deepseek_model_switcher):
                reply = (
                    "工作区路径有效，但本地运行时未配置 DeepSeek 模型切换接口；"
                    "设置 JSON 尚未创建。请检查 run/open_webui_runtime.json 后重试。"
                )
                return self._save_and_report(self._with_turn(session, user_message, reply), reply)
            try:
                selected_model, client = self.deepseek_model_switcher(selected_model)
            except (OSError, TypeError, ValueError) as error:
                reply = (
                    f"DeepSeek 模型设置未能保存：{type(error).__name__}: {error}。"
                    "工作区设置尚未提交；请检查本地 API Key 与运行时配置后重试。"
                )
                return self._save_and_report(self._with_turn(session, user_message, reply), reply)
            self.current_deepseek_model = selected_model
            self.agent_client = client

        from config_layer.session.create_editable_config_json import create_editable_config_json

        try:
            created = create_editable_config_json(
                draft_path,
                session.get("config") or {},
                bootstrap_hints=session.get("bootstrap_hints"),
                workspace_defaults=storage,
            )
        except (OSError, TypeError, ValueError) as error:
            reply = (
                f"设置 JSON 写入失败：{type(error).__name__}: {error}。"
                "未确认或保存该工作区设置，也没有启动计算。请检查目录权限后重新确认，"
                "或发送“重新选择”改用其他目录。"
            )
            pending = dict(session)
            pending["setup_stage"] = "awaiting_storage_confirmation"
            pending["pending_workspace_root"] = str(root)
            return self._save_and_report(self._with_turn(pending, user_message, reply), reply)

        # Commit the chosen root only after the JSON file has been created (or
        # an existing user file has been safely preserved).
        updated = apply_config_revision(
            session,
            {"storage": storage},
            reasons={"storage": "用户明确确认了本地工作区及其派生保存目录。"},
            author="user_confirmed_storage_path",
        )
        updated["setup_stage"] = "json_ready"
        updated["editable_config_json_path"] = str(draft_path)
        updated.pop("pending_workspace_root", None)
        updated.pop("pending_deepseek_model", None)
        updated = self._with_turn(updated, user_message, "")
        try:
            self._save(updated)
        except OSError as error:
            reply = (
                f"设置 JSON 已保留在 {draft_path}，但配置会话状态未能保存：{error}。"
                "没有启动计算；再次确认同一路径可安全续上，不会覆盖现有 JSON。"
            )
            return reply
        self.workspace_root_default = root
        self.editable_config_path = draft_path

        if created:
            status = f"已在确认的工作区生成带注释的设置 JSON：{draft_path}。"
        else:
            status = f"该位置已有设置文件，未覆盖：{draft_path}。"
        reply = (
            f"已确认工作区：{root}\nAgent 模型版本：{_deepseek_model_label(self.current_deepseek_model)}"
            f"（`{self.current_deepseek_model}`）\n{status}\n"
            "请按 JSON 注释填写或核对体系边界、母结构、MLIP 路径、预算、DFT 参数和收敛标准。"
        )
        readiness = self.configuration_readiness(updated)
        checklist = readiness.get("missing") or []
        if checklist:
            reply += "\n\n需要补齐：\n" + "\n".join(f"- {item}" for item in checklist)
        reply += "\n\n完成后发送“读取配置 JSON”，由 Agent 审核；Agent 通过后你回复“同意”即可开始搜索流程。"
        self._save(record_config_dialogue(updated, role="assistant", message=reply))
        return reply

    def _import_and_review_config_json(self, session, user_message):
        if self.editable_config_path is None:
            return "未配置可编辑 JSON 草稿路径；配置草稿未更改。"
        from config_layer.schema.validate_search_config import validate_search_config
        from config_layer.session.apply_config_revision import apply_config_revision
        from config_layer.session.load_editable_config_json import (
            config_leaf_patch, load_editable_config_json,
        )
        from config_layer.session.resolve_phase_reference_directory import (
            resolve_phase_reference_directory,
        )

        try:
            config = load_editable_config_json(
                self.editable_config_path, session.get("config") or {},
                default_storage=default_workspace_storage(self.workspace_root_default),
            )
            config = resolve_phase_reference_directory(
                config, base_directory=self.base_directory
            )
            audit = validate_search_config(config)
            imported_config_hash = self._editable_config_hash()
        except (OSError, TypeError, ValueError, KeyError) as error:
            return f"配置 JSON 未导入：{error}\n当前草稿未更改。"

        patch = config_leaf_patch(session.get("config") or {}, config)
        updated = session
        if patch:
            updated = apply_config_revision(
                updated, patch,
                reasons={path: "用户编辑本地配置 JSON 后导入。" for path in patch},
                author="user_json_draft",
            )
        updated = self._with_turn(updated, user_message, "")
        updated = self._save(updated)
        agent_reply = ""
        questions = []
        agent_note = ""
        agent_passed = False
        readiness = self.configuration_readiness(updated)
        if callable(self.agent_client):
            try:
                result = self.agent_client({
                    "mode": "configuration_json_review",
                    "instruction": (
                        "用户刚从本地 JSON 文件导入配置草稿。请检查缺项、冲突、歧义和参数口径，"
                        "解释必要的修订并提出集中问题。只审核，不返回或应用 patch。必须返回布尔字段 "
                        "ready_for_search；仅当没有阻止开始搜索的问题时为 true。不要确认配置或启动计算。"
                    ),
                    "draft_config": updated.get("config") or {},
                    "draft_revision": updated.get("draft_revision"),
                    "readiness": readiness,
                    "validation": audit,
                })
                if isinstance(result, dict):
                    agent_reply = str(result.get("reply") or "配置 JSON 已检查。")
                    questions = result.get("questions") or []
                    if not isinstance(questions, list):
                        questions = []
                    if result.get("patch"):
                        agent_note = "Agent 返回了修改建议，但本次审查不会自动应用；请编辑 JSON 后再次读取。"
                    agent_passed = (
                        result.get("ready_for_search") is True
                        and readiness["ready"]
                        and not result.get("patch")
                        and imported_config_hash is not None
                        and self._editable_config_hash() == imported_config_hash
                    )
                    if agent_passed:
                        updated["agent_reviewed_revision"] = updated.get("draft_revision")
                        updated["agent_reviewed_config_hash"] = imported_config_hash
                    else:
                        updated.pop("agent_reviewed_revision", None)
                        updated.pop("agent_reviewed_config_hash", None)
                        if readiness["ready"]:
                            agent_note = (
                                agent_note
                                or "Agent 尚未通过配置审查，或审核期间文件发生变化；请确认后重新读取 JSON。"
                            )
                else:
                    agent_note = "Agent 返回格式无效；已导入的草稿仍保留，不能开始。"
            except Exception as error:
                diagnostic = getattr(error, "safe_message", type(error).__name__)
                updated.pop("agent_reviewed_revision", None)
                updated.pop("agent_reviewed_config_hash", None)
                agent_note = f"Agent 暂时无法完成审核（{diagnostic}）；已导入的草稿仍保留，不能开始。"
        else:
            updated.pop("agent_reviewed_revision", None)
            updated.pop("agent_reviewed_config_hash", None)
            agent_note = "DeepSeek Agent 当前不可用；文件已导入，可稍后重新发送“读取配置 JSON”进行审核。"

        reply = "配置 JSON 已交给 Agent 审核，并通过程序字段校验；此次读取不会自动确认或开始搜索。"
        if agent_reply:
            reply += "\n\n" + agent_reply
        if agent_note:
            reply += "\n\n" + agent_note
        if agent_passed:
            reply += "\n\nAgent 与程序检查均通过。你核对无误后回复“同意”，即可保存配置版本并进入搜索 Agent；不会自动提交计算作业。"
        elif not readiness["ready"]:
            reply += "\n\n当前仍有必填项或冲突，按下方清单修改 JSON 后重新发送“读取配置 JSON”。"
        updated = record_config_dialogue(updated, role="assistant", message=reply)
        updated = self._save(updated)
        readiness = self.configuration_readiness(updated)
        formatted = _format_agent_reply(
            reply, [], readiness, questions, agent_review_passed=agent_passed,
        )
        return f"文件：{self.editable_config_path}\n草稿版本：{updated.get('draft_revision')}\n\n{formatted}"

    def _apply_agent_response(self, session, user_message, result):
        if not isinstance(result, dict):
            return self._save_and_report(
                self._with_turn(session, user_message, "Agent 返回格式无效，草稿未更改。"),
                "Agent 返回格式无效，草稿未更改。请重试或使用本地配置控制接口。",
            )
        patch = result.get("patch") or {}
        if not isinstance(patch, dict) or len(patch) > 40:
            patch = None
        updated = self._with_turn(session, user_message, "")
        changes = []
        if patch:
            try:
                _validate_patch(patch)
                updated = apply_config_revision(
                    updated, patch, reasons=result.get("reasons") or {},
                    author="agent_draft_suggestion",
                )
                changes = updated["dialogue"][-1].get("changes") or []
            except (TypeError, ValueError) as error:
                patch = None
                result = {**result, "reply": f"{result.get('reply', '')}\n\n草稿修改被安全检查拒绝：{error}"}
        reply = str(result.get("reply") or "我已检查当前草稿。")
        questions = result.get("questions") or []
        if not isinstance(questions, list):
            questions = []
        updated["dialogue"].append({"type": "message", "role": "assistant", "message": reply})
        updated = self._save(updated)
        readiness = self.configuration_readiness(updated)
        return _format_agent_reply(reply, changes, readiness, questions)

    def _confirm_and_start_search(self, session, *, user_message, conversation_id):
        if not callable(self.runtime_factory):
            return (
                "Agent 与程序检查已通过，但当前没有配置搜索运行入口；"
                "配置尚未确认，请检查 runtime_factory 后重试。"
            )
        confirmation = self._confirm(self._with_turn(session, user_message, ""))
        confirmed = self.workflow_kwargs.get("config_session") or {}
        if confirmed.get("status") != "confirmed":
            return confirmation
        try:
            self.delegate = self.runtime_factory()
            first_turn = self.delegate(
                [{"role": "user", "content": (
                    "配置已由用户确认。开始首轮搜索分析并给出建议；只生成待审批建议，"
                    "不得直接执行、提交或派发任何计算任务。"
                )}],
                conversation_id=conversation_id,
            )
        except Exception as error:
            diagnostic = getattr(error, "safe_message", type(error).__name__)
            return (
                f"{confirmation}\n\n配置已确认，但搜索 Agent 暂时未能启动（{diagnostic}）。"
                "配置快照已保存；重启本地服务后可继续，不会自动提交计算。"
            )
        return f"{confirmation}\n\n搜索 Agent 已启动首轮分析：\n{first_turn}"

    def _editable_config_hash(self):
        if self.editable_config_path is None:
            return None
        try:
            return hashlib.sha256(self.editable_config_path.read_bytes()).hexdigest()
        except OSError:
            return None

    def _confirm(self, session):
        readiness = self.configuration_readiness(session)
        if not readiness["ready"]:
            return _format_agent_reply(
                "当前配置还不能开始。请先补齐缺项并解决冲突，再让 Agent 重新审核。",
                [], readiness, [],
            )
        if "storage" not in (session.get("config") or {}):
            session = apply_config_revision(
                session,
                {"storage": default_workspace_storage(self.workspace_root_default)},
                reasons={"storage": "采用当前本地工作区路径作为默认值；用户可在确认前调整。"},
                author="workspace_storage_default",
            )
        confirmed = confirm_config_snapshot(session, user_confirmed=True)
        if confirmed.get("confirmation_error"):
            return _format_agent_reply(
                "配置检查未通过，尚未生成快照。", [], self.configuration_readiness(confirmed),
                (confirmed.get("audit") or {}).get("missing") or [],
            )
        try:
            from config_layer.session.save_confirmed_config_file import save_confirmed_config_file
            snapshot_path = save_confirmed_config_file(
                confirmed["confirmed_snapshot"], base_directory=self.base_directory
            )
        except (OSError, TypeError, ValueError, FileExistsError) as error:
            return (f"配置检查通过，但版本快照未写入工作区：{error}。"
                    "未提交或启动任何计算；请检查 storage.workspace_root 和 storage.paths 后重试。")
        self._save(confirmed)
        version = confirmed["confirmed_snapshot"]["config_version"]
        return (f"配置已确认并保存为版本 {version}。配置快照：{snapshot_path}\n"
                "配置现已冻结；搜索 Agent 开始提出本轮建议，具体计算仍需按审批规则确认后才会派发。")

    def _with_turn(self, session, user_message, assistant_message):
        updated = record_config_dialogue(session, role="user", message=user_message)
        if assistant_message:
            updated = record_config_dialogue(updated, role="assistant", message=assistant_message)
        return updated

    def _record(self, session, role, message):
        self._save(record_config_dialogue(session, role=role, message=message))

    def _save_and_report(self, session, message):
        updated = self._save(session)
        return _format_agent_reply(message, [], self.configuration_readiness(updated), [])

    def _save(self, session):
        save_config_session(session, self.config_session_path)
        self.workflow_kwargs["config_session"] = session
        return session

    @staticmethod
    def _unavailable_message():
        return ("当前是配置对话模式；尚未确认配置，也不会运行计算。"
                "本地 DeepSeek API 尚不可用，请设置本机 DEEPSEEK_API_KEY 后重启服务；"
                "草稿仍可通过 /phase/config/patch 修改。")


def _latest_user_message(messages):
    for item in reversed(messages or []):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    raise ValueError("messages must contain a user message")


def _default_choice(message):
    normalized = str(message).strip().lower()
    if normalized in {"采用默认参数", "使用默认参数", "是", "yes", "use defaults"}:
        return True
    if normalized in {"不采用默认参数", "不用默认参数", "否", "no", "do not use defaults"}:
        return False
    return None


def _is_config_json_import_command(message):
    """Only an explicit short command imports the edited local JSON draft."""
    normalized = " ".join(str(message).strip().lower().split())
    return normalized in {
        "读取配置 json", "读取配置json", "检查配置 json", "检查配置json",
        "审查配置 json", "审查配置json", "read config json", "review config json",
    }


def _extract_workspace_path(message):
    """Accept only a single-line path reply, never arbitrary prompt text."""
    candidate = str(message or "").strip()
    if any(ord(char) < 32 or ord(char) == 127 for char in candidate):
        return None
    if len(candidate) > 512 or "###" in candidate or "<chat_history" in candidate.lower():
        return None
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate[3:-3].strip()
        if any(ord(char) < 32 or ord(char) == 127 for char in candidate):
            return None
    candidate = re.sub(
        r"^(?:工作区根目录|工作区根路径|根路径|工作区路径|存储路径|保存路径)\s*[:：]\s*",
        "",
        candidate,
    ).strip()
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "\"'`":
        candidate = candidate[1:-1].strip()
    return candidate


def _parse_workspace_setup_values(message):
    """Parse only a path and an optional labeled/model selection from setup input."""
    from run.resolve_deepseek_model_request import resolve_deepseek_model_request

    raw = str(message or "").strip()
    if not raw or len(raw) > 1024:
        return None, None
    if any(ord(char) < 32 and char not in "\r\n" or ord(char) == 127 for char in raw):
        return None, None
    if "\n" in raw or "\r" in raw:
        segments = [part.strip() for part in raw.splitlines() if part.strip()]
        if len(segments) > 2:
            return None, None
    elif any(separator in raw for separator in (";", "；", "|", ",", "，")):
        segments = [part.strip() for part in re.split(r"[;；|,，]", raw) if part.strip()]
    else:
        segments = [raw]

    root_value = None
    model_value = None
    path_label = re.compile(r"^(?:工作区根目录|工作区根路径|根路径|工作区路径|workspace(?:\s+root)?)\s*[:：]\s*(.+)$", re.I)
    model_label = re.compile(r"^(?:agent(?:\s*模型)?(?:\s*版本)?|模型(?:版本)?|deepseek(?:\s*模型)?)\s*[:：]\s*(.+)$", re.I)
    for segment in segments:
        path_match = path_label.fullmatch(segment)
        if path_match:
            value = _extract_workspace_path(path_match.group(1))
            if not value or root_value is not None:
                return None, None
            root_value = value
            continue

        model_match = model_label.fullmatch(segment)
        if model_match:
            selected = resolve_deepseek_model_request(model_match.group(1))
            if not selected or (model_value is not None and model_value != selected):
                return None, None
            model_value = selected
            continue

        selected = resolve_deepseek_model_request(segment)
        if selected:
            if model_value is not None and model_value != selected:
                return None, None
            model_value = selected
            continue

        value = _extract_workspace_path(segment)
        looks_absolute = bool(value) and (
            Path(value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value)
        )
        looks_relative = (value or "").startswith(("./", "../", ".\\", "..\\"))
        if value and (looks_absolute or looks_relative) and root_value is None:
            root_value = value
            continue
        return None, None
    return root_value, model_value


def normalize_workspace_path(value, *, base_directory):
    """Validate and resolve a user-selected local workspace path."""
    candidate = _extract_workspace_path(value)
    if not candidate:
        raise ValueError("工作区路径必须是单行路径")
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        if not candidate.startswith(("./", "../", ".\\", "..\\")):
            raise ValueError("相对路径必须显式以 ./、../、.\\ 或 ..\\ 开头")
        path = Path(base_directory) / path
    return path.resolve()


def safe_config_filename(value):
    """Restrict the editable draft setting to a plain JSON filename."""
    default = "search_config.draft.json"
    candidate = str(value or default).strip()
    if (not candidate or len(candidate) > 128
            or any(ord(char) < 32 or ord(char) == 127 for char in candidate)
            or "/" in candidate or "\\" in candidate
            or candidate in {".", ".."} or not candidate.lower().endswith(".json")):
        return default
    return candidate


def _deepseek_model_label(model):
    if model == "deepseek-flash":
        return "DeepSeek V4.1 Flash"
    if model == "deepseek-v4-pro":
        return "DeepSeek V4 Pro"
    return f"DeepSeek ({model})"


def _deepseek_model_change_reply(model):
    return (
        f"已将本地 Agent 切换为 {_deepseek_model_label(model)}（`{model}`），"
        "并保存到本地 Open WebUI 运行时配置；从下一条消息起生效。"
        "搜索配置、API Key 和计算任务未修改。"
    )


def _validate_patch(patch):
    for path, value in patch.items():
        if not isinstance(path, str) or not path or len(path) > 180:
            raise ValueError("配置路径无效")
        parts = path.split(".")
        if any(not part or part.startswith("_") for part in parts):
            raise ValueError(f"配置路径不允许：{path}")
        if any(any(word in part.lower() for word in ("api_key", "token", "password", "secret", "private_key"))
               for part in parts):
            raise ValueError(f"密钥不能写入配置草稿：{path}")
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path} 不是合法 JSON 值") from error


def _format_agent_reply(reply, changes, readiness, questions, *, agent_review_passed=False):
    lines = [reply]
    if changes:
        lines.append("\n草稿修改（仅草稿，尚未确认）：")
        for change in changes:
            value = json.dumps(change.get("new"), ensure_ascii=False, default=str)
            if len(value) > 500:
                value = value[:497] + "..."
            lines.append(
                f"- {change['path']} [{change.get('constraint_type', 'unclassified')}]: "
                f"{value}；原因：{change.get('reason') or '未提供'}"
            )
    if questions:
        lines.append("\n待回答：")
        lines.extend(f"- {str(question)}" for question in questions[:12])
    missing = readiness.get("missing") or []
    conflicts = readiness.get("conflicts") or []
    ambiguities = readiness.get("ambiguities") or []
    if missing or conflicts or ambiguities:
        lines.append("\n当前配置检查：")
        for label, values in (("缺项", missing), ("冲突", conflicts), ("待明确", ambiguities)):
            if values:
                lines.append(f"- {label}：" + "；".join(map(str, values[:20])))
    elif agent_review_passed:
        lines.append("\nAgent 与程序检查通过；你确认无误后回复“同意”即可进入搜索流程。")
    else:
        lines.append("\n程序字段检查通过；等待 Agent 审核。Agent 通过后再回复“同意”。")
    paths = readiness.get("storage_paths") or {}
    if paths:
        lines.append(f"\n工作区：{paths.get('workspace_root', '未知')}")
    return "\n".join(lines)
