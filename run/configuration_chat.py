"""First-run, config-only Open WebUI dialogue.

This handler can only revise an unconfirmed JSON draft. It never calls the
scientific workflow. An exact user confirmation creates the versioned snapshot;
the next chat turn then switches to the regular workflow handler.
"""

from __future__ import annotations

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
只返回 JSON：{"reply":"给用户的中文答复","patch":{"点分配置路径": JSON值},"reasons":{"点分配置路径":"修改原因"},"questions":["待用户回答的问题"]}。
patch 只修改草稿，不会确认配置或启动运行；每项 patch 都会向用户展示。区分硬约束与可调整建议。
当 mode 为 configuration_json_review 时，只检查并解释用户刚导入的 JSON 配置，patch 必须为空；列出重要缺项、冲突、歧义和需要用户确认的内容，不要擅自填值。
本项目初次配置的重点是本地母结构路径、体系 boundary、远端 MACE 模型路径、预算、DFT/atomate 参数和收敛标准。
首次配置必须先询问本地工作区根目录。用户给出路径后，先展示派生保存目录并等待用户明确回复“确认存储路径”；确认前不得创建设置 JSON 或创建工作区目录。确认后才在所选工作区生成带注释的 JSON 设置文件，用户编辑后发送“读取配置 JSON”，再审核缺项和冲突。
配置 JSON 输出后展示 workspace_root 及其下配置快照、台账、结构、相图、QBC、审批和任务目录；用户可逐项调整 storage.paths。确认配置时仅保存快照，不启动计算。
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
                 workspace_root_default=None, agent_client=None, runtime_factory=None):
        self.workflow_kwargs = dict(workflow_kwargs)
        self.state_path = Path(workflow_kwargs["state_path"])
        self.config_session_path = Path(config_session_path)
        self.base_directory = Path(base_directory)
        self.phase_references_path = Path(phase_references_path) if phase_references_path else None
        self.editable_config_path = Path(editable_config_path) if editable_config_path else None
        self.editable_config_filename = Path(editable_config_filename).name or "search_config.draft.json"
        session = workflow_kwargs.get("config_session") or {}
        selected_root = (
            session.get("pending_workspace_root")
            or ((session.get("config") or {}).get("storage") or {}).get("workspace_root")
        )
        root_path = Path(
            selected_root or workspace_root_default or self.config_session_path.parent
        ).expanduser()
        if not root_path.is_absolute():
            root_path = self.base_directory / root_path
        self.workspace_root_default = root_path.resolve()
        self.agent_client = agent_client
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
        if _is_config_json_import_command(message):
            return self._import_and_review_config_json(session, message)
        if message.strip() in {"确认配置", "confirm configuration"}:
            return self._confirm(session)
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
        if normalized in {"确认配置", "confirm configuration", "读取配置 json", "读取配置json"}:
            return self._save_and_report(
                session,
                "请先确定本地工作区。设置 JSON 尚未生成；先发送工作区根目录路径，"
                "查看目录预览后再回复“确认存储路径”。",
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
                and normalized in {"确认存储路径", "确认工作区", "确认路径", "confirm workspace"}):
            return self._confirm_workspace_root(session, user_message)
        return self._preview_workspace_root(session, user_message)

    def _preview_workspace_root(self, session, user_message):
        raw_path = _extract_workspace_path(user_message)
        if not raw_path:
            message = (
                "请发送一个本地工作区路径，例如 `E:\\PhaseSearch`。"
                "路径可以是尚不存在的新目录；我会先预览，不会立即创建文件。"
            )
            return self._save_and_report(self._with_turn(session, user_message, message), message)

        root = Path(raw_path).expanduser()
        if not root.is_absolute():
            if raw_path.startswith((".", "~")) or "/" in raw_path or "\\" in raw_path:
                root = self.base_directory / root
            else:
                message = "请提供绝对路径，或包含 `./`、`../` 的相对路径；我不会猜测保存位置。"
                return self._save_and_report(self._with_turn(session, user_message, message), message)
        root = root.resolve()
        if root.exists() and not root.is_dir():
            message = f"该路径已存在但不是文件夹：{root}。请换一个工作区目录。"
            return self._save_and_report(self._with_turn(session, user_message, message), message)

        from config_layer.session.resolve_workspace_paths import default_workspace_storage

        self.workspace_root_default = root
        storage = default_workspace_storage(root)
        resolved = resolve_workspace_paths({"storage": storage}, base_directory=self.base_directory)
        updated = dict(session)
        updated["setup_stage"] = "awaiting_storage_confirmation"
        updated["pending_workspace_root"] = str(root)
        lines = [
            f"工作区根目录：{root}",
            f"设置 JSON：{root / self.editable_config_filename}",
            "其他保存位置：",
        ]
        labels = {
            "config_snapshots": "配置快照", "state": "运行状态", "ledger": "主台账",
            "branch_energy_pool_ledger": "Branch 能量池", "structures": "结构",
            "phase_diagrams": "相图", "qbc_results": "QBC 结果", "work": "任务工作目录",
            "approvals": "审批记录", "approved_batches": "已批准批次",
            "upload_batches": "待上传批次", "new_runs": "新运行目录",
        }
        lines.extend(f"- {labels[key]}：{value}" for key, value in resolved.items() if key in labels)
        lines.append(
            "确认后才会创建工作区和设置 JSON。回复“确认存储路径”确认，"
            "或回复“重新选择”修改路径。"
        )
        reply = "\n".join(lines)
        updated = self._with_turn(updated, user_message, reply)
        return self._save_and_report(updated, reply)

    def _confirm_workspace_root(self, session, user_message):
        root_value = session.get("pending_workspace_root")
        if not root_value:
            return self._preview_workspace_root(session, "")
        root = Path(root_value)
        from config_layer.session.resolve_workspace_paths import default_workspace_storage

        storage = default_workspace_storage(root)
        self.workspace_root_default = root
        updated = apply_config_revision(
            session,
            {"storage": storage},
            reasons={"storage": "用户明确确认了本地工作区及其派生保存目录。"},
            author="user_confirmed_storage_path",
        )
        draft_path = (root / self.editable_config_filename).resolve()
        updated["setup_stage"] = "json_ready"
        updated["editable_config_json_path"] = str(draft_path)
        updated.pop("pending_workspace_root", None)
        updated = self._with_turn(updated, user_message, "")
        self.editable_config_path = draft_path
        self._save(updated)

        from config_layer.session.create_editable_config_json import create_editable_config_json

        try:
            created = create_editable_config_json(
                draft_path,
                updated.get("config") or {},
                bootstrap_hints=updated.get("bootstrap_hints"),
                workspace_defaults=storage,
            )
        except (OSError, TypeError, ValueError) as error:
            return (
                f"已记录你确认的工作区路径，但设置 JSON 写入失败：{error}。"
                f"路径：{draft_path}。修正目录权限后重启配置助手即可重试；没有启动计算。"
            )

        if created:
            status = f"已在确认的工作区生成带注释的设置 JSON：{draft_path}。"
        else:
            status = f"该位置已有设置文件，未覆盖：{draft_path}。"
        reply = (
            f"已确认工作区：{root}\n{status}\n"
            "接下来可编辑 JSON 中的参数，然后发送“读取配置 JSON”；"
            "导入只检查和更新草稿，仍需另行确认配置才会保存版本快照。"
            "确认工作区和生成文件本身不会启动计算。"
        )
        return self._save_and_report(
            record_config_dialogue(updated, role="assistant", message=reply), reply
        )

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
        if callable(self.agent_client):
            try:
                result = self.agent_client({
                    "mode": "configuration_json_review",
                    "instruction": (
                        "用户刚从本地 JSON 文件导入配置草稿。请检查缺项、冲突、歧义和参数口径，"
                        "解释必要的修订并提出集中问题。只审核，不返回或应用 patch；不要确认配置或启动计算。"
                    ),
                    "draft_config": updated.get("config") or {},
                    "draft_revision": updated.get("draft_revision"),
                    "readiness": self.configuration_readiness(updated),
                    "validation": audit,
                })
                if isinstance(result, dict):
                    agent_reply = str(result.get("reply") or "配置 JSON 已检查。")
                    questions = result.get("questions") or []
                    if not isinstance(questions, list):
                        questions = []
                    if result.get("patch"):
                        agent_note = "Agent 返回了修改建议，但本次审查不会自动应用；请编辑 JSON 后再次读取。"
                else:
                    agent_note = "Agent 返回格式无效；已导入的草稿仍保留，未确认。"
            except Exception as error:
                diagnostic = getattr(error, "safe_message", type(error).__name__)
                agent_note = f"Agent 暂时无法完成审核（{diagnostic}）；已导入的草稿仍保留，未确认。"
        else:
            agent_note = "DeepSeek Agent 当前不可用；文件已导入，可稍后重新发送“读取配置 JSON”进行审核。"

        reply = "配置 JSON 已读取并写入未确认草稿。不会自动确认快照或启动计算。"
        if agent_reply:
            reply += "\n\n" + agent_reply
        if agent_note:
            reply += "\n\n" + agent_note
        updated = record_config_dialogue(updated, role="assistant", message=reply)
        updated = self._save(updated)
        readiness = self.configuration_readiness(updated)
        formatted = _format_agent_reply(reply, [], readiness, questions)
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

    def _confirm(self, session):
        readiness = self.configuration_readiness(session)
        if not readiness["ready"]:
            return _format_agent_reply(
                "当前配置还不能确认。请先补齐缺项并解决冲突；确认配置只保存版本快照，不会启动任何计算。",
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
        paths = readiness.get("storage_paths") or {}
        path_lines = [f"- {key}: {value}" for key, value in paths.items()]
        return (f"配置已确认并保存为版本 {version}。配置快照：{snapshot_path}\n"
                "本轮配置已冻结；确认本身不会提交或启动计算。后续状态、台账和分析输出使用以下路径：\n"
                + "\n".join(path_lines) + "\n下一条消息才会进入搜索决策对话。")

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
    """Accept a path-only reply, optionally prefixed or wrapped in quotes."""
    candidate = str(message or "").strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        candidate = candidate[3:-3].strip()
        if "\n" in candidate:
            candidate = candidate.split("\n", 1)[1].strip()
    candidate = re.sub(
        r"^(?:工作区根目录|工作区路径|存储路径|保存路径)\s*[:：]\s*",
        "",
        candidate,
    ).strip()
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "\"'`":
        candidate = candidate[1:-1].strip()
    return candidate


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


def _format_agent_reply(reply, changes, readiness, questions):
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
    else:
        lines.append("\n配置已满足快照检查；如内容无误，可单独发送“确认配置”。这不会启动计算。")
    paths = readiness.get("storage_paths") or {}
    if paths:
        lines.append("\n保存位置预览（尚未确认，可在配置 JSON 的 storage.paths 中调整）：")
        lines.append(f"- 工作区根目录：{paths.get('workspace_root', '未知')}")
        labels = {
            "config_snapshots": "版本配置", "state": "运行状态", "ledger": "主台账",
            "branch_energy_pool_ledger": "Branch 能量池", "structures": "候选结构",
            "phase_diagrams": "相图", "qbc_results": "QBC 结果", "work": "任务工作目录",
            "approvals": "审批记录", "approved_batches": "已批准任务批次",
            "upload_batches": "待上传批次", "new_runs": "新运行目录",
        }
        for key, label in labels.items():
            if key in paths:
                lines.append(f"- {label}：{paths[key]}")
    return "\n".join(lines)
