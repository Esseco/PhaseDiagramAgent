"""Open WebUI configuration dialogue and versioned draft revisions."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
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
from scientific_layer.structures.boundary_utils import allowed_phases


CONFIG_AGENT_SYSTEM_PROMPT = """你是材料相图搜索项目的首次配置助手，不是计算执行器。
Relax每个提交作业最多100个结构，MC每个提交作业最多10个模拟（每个branch的MC结果仍独立），DFT一个作业一个结构；这是supercomputer.batch_sizes或运行时stage_batch_sizes的分组限制，超出就拆分作业。budgets.stage_limits.max_tasks是独立的累计预算限制，不能把它解释为每个作业上限，也不能因入选branch增加自动建议增改它。用户说每个作业/每个任务弛豫数量时，应修改分组大小。generation_actions.max_det_H不存在，不得新增该字段。
只修改当前用户明确指定的参数，其他参数原值保留。不要为满足数量而擅自调整候选配额、预算、策略、相边界或全局结构限制；关联调整只能提出建议。没有“候选/生成量/总配额”等限定时，“300个branch”按入选上限run.batch_size理解，不是run.total_quota。本轮det(H)上限属于生成动作max_det_H，不得写入budgets.structure_limits.max_det_H；只有明确首轮配置才用system.H_generation.first_round_max_det_H，明确全局边界才改全局字段。读取配置不能恢复旧默认覆盖用户已确认值。
editable_field_catalog 是程序提供的真实配置字段目录。用户只需说中文含义和目标值，你负责从目录定位字段，不得要求用户提供内部字段路径。目录包含被精简摘要省略的参数；摘要未展示不等于字段不存在。初态数量对应 run.initial_states_per_branch，入选上限对应 run.batch_size，候选生成量对应 run.total_quota；初态规则不属于DFT。用户明确要求修改且值明确时直接返回patch和write_requested=true。确有多个不同语义字段时只集中问一个必要问题。不得声称已写入，写入是否成功由程序返回。
只讨论并检查配置草稿。不得提交任务、调用科学计算、创建结构、运行命令或声称计算已完成。
仅当用户明确提供或确认信息时，才返回对应配置修改；不确定时提出问题，不猜测边界矩阵、预算、DFT 设置或超算命令。
只返回 JSON：{"reply":"给用户的中文答复","patch":{"点分配置路径": JSON值},"reasons":{"点分配置路径":"修改原因"},"questions":["待用户回答的问题"],"write_requested":false,"ready_for_search":false}。
使用短配置文件时，只要用户当前这条消息明确要求实际修改项目参数，就返回 write_requested=true 和对应 patch；不要求出现固定关键词。比如“入选上限改为500”“把 Relax 任务上限提高到900”属于直接修改；询问参数含义、征求建议或只改当前这轮 action 不属于配置写入。write_requested 只能依据当前用户原话，不能从历史对话推断。程序仍会校验字段、值和文件版本，再原子写入。缺少明确数值时先追问，不猜值。密钥、未明确值和已确认快照绝不能改写。旧版无文件会话仍按原草稿规则处理。区分硬约束与可调整建议。
当 mode 为 configuration_json_review 时，只检查并解释用户刚导入的 JSON 配置，patch 必须为空；列出有具体字段和值作为证据的缺项、冲突或歧义。readiness 是程序检查的权威结果；若它 ready=true 且你没有发现可定位的新问题，应令 ready_for_search=true。可选阈值、空配额、尚未接通的远端提交接口不阻止仅生成搜索建议。不要猜测问题、重复索要已确认信息或擅自填值。
程序已经读取了 verified_config_source.source_path，并在导入时解析母结构、生成合法 H、计算允许相并集和校验结果。你只能依据程序提供的 verified_config_source 和本次 draft_config 讨论事实；不得引用旧对话中的配置值，不得声称程序无法读取文件，不得要求用户手抄已生成的 H 矩阵。生成失败时程序会给出具体错误，你不要猜测生成结果。boundary.P 的允许相是端点与中间相的并集；H 的四相键属于这个并集。超算调度器未接通不会阻止只生成搜索建议，但不能提交作业。
以下均是已定义的非阻塞口径，不得要求用户逐项重复确认：parameter_source=atomate_defaults 时空 dft.parameters 表示采用 atomate 默认值；空 generation_actions.quotas 和 focus_regions 表示由调度策略动态决定；阶段 max_cost 是各阶段独立安全上限，不要求求和小于总预算，实际累计仍受 total_relative_cost 约束；用户写入配置文件的收敛阈值视为已选择值；空 scheduler 命令只表示暂不能远端提交，不阻止配置确认和搜索建议。generated_H=false 且 readiness.status=file_not_imported 只表示尚未执行导入，不能称为 H 缺失或要求再次确认生成参数。
本项目初次配置的重点是本地母结构路径、体系 boundary、远端 MACE 模型路径、预算、DFT/atomate 参数和收敛标准。科学计算 MLIP 默认是 mace-mh-1；calculation.mlip_version、mlip.name 和 bohb.scope.mlip_version 默认保持一致。不要询问用户选择 MLIP 版本，也不要与 DeepSeek Agent 模型混淆。用户只需提供超算端 mlip.model_path；路径缺失时只问该路径。仅当用户明确提出更换 MLIP 时才讨论其他版本。
首次配置先收集工作区根路径和 Agent 模型版本（DeepSeek V4.1 Flash 或 V4 Pro）；允许同一条消息或分别提供。只展示这两项和设置 JSON 路径，等待用户回复“确认”。确认后将工作区根路径视为已确认的锁定事实；后续必须从 setup_facts 读取，绝不能再次索要或要求重复确认该路径。确认前不得创建设置 JSON/工作区目录或修改运行时模型。确认后更新本地运行时模型（若用户选择切换）、生成带注释的 JSON，并区分列出必填项和建议检查项。用户编辑 JSON 后可发送“读取配置 JSON”进行审核；仅审核模式通过后，等待用户简短回复“同意”，再保存配置快照并进入搜索 run。用户也可发送“读取配置 JSON 并继续”，这表示仅在 Agent 与程序检查均通过时条件确认继续；此时直接保存快照并进入搜索 Agent，不再要求第二次同意。两种方式都只进入搜索建议阶段，不派发科学计算。不得在此之前派发科学计算。DeepSeek 模型属于本地 Open WebUI 运行时，不写入科学搜索配置。
每次配置对话都必须结合 conversation_context 中最近的问答理解短答（例如“是的”是对紧邻前一条助手问题的回答），不可丢失上下文、重复询问已确认字段或把一个确认泛化成其他问题。若紧邻问题是在确认 calculation.mlip_version 与 bohb.scope.mlip_version 使用 mace-mh-1，用户回答肯定，则将两字段记为 mace-mh-1 并告知已记录；不得转而重问工作区路径。
配置 JSON 输出后说明工作区和主要文件位置，并分别列出必填项与建议检查项；用户可调整 storage.paths。只有 Agent 和程序检查通过，且用户已明确回复“同意”或使用“读取配置 JSON 并继续”作条件确认后，才进入搜索建议流程；不会因此自动提交计算。
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

RECOMMENDED_CONFIG_CHECKS = (
    "system.boundary.P 的 Na 含量→允许相映射、H_generation 尺寸与包含条件、TM_ratio 是否准确。",
    "母结构目录是否包含与相名一致的文件（例如 O3.vasp）；文件内容及相名映射是否正确。",
    "mlip.model_path 是否是超算端实际可访问的模型路径；本地 Agent 不会加载该模型。",
    "总预算及 MC、DFT 子预算是否使用一致的项目成本单位，且子预算没有超过总预算。",
    "DFT 单点与弛豫的 user_incar_settings、赝势和计算参数来源是否符合你的既定流程。",
    "收敛阈值及单位是否符合预期（能量误差为 eV/atom；相图变化按配置口径）；覆盖率是否仅作为证据而非硬门槛。",
)


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
        try:
            phases = sorted(allowed_phases(raw_phases)) if raw_phases else []
        except (TypeError, ValueError):
            phases = []
        if not phases:
            missing.append("system.boundary.P (非空相列表或 Na 含量→相映射)")
        if isinstance(raw_phases, dict) and ("at_x" in raw_phases or "intermediate" in raw_phases):
            from scientific_layer.structures.boundary_utils import allowed_phases_at_x
            try:
                allowed_phases_at_x(raw_phases, 0)
                allowed_phases_at_x(raw_phases, 1)
                allowed_phases_at_x(raw_phases, "1/2")
            except (TypeError, ValueError, ZeroDivisionError) as error:
                conflicts.append(f"system.boundary.P 组分规则无效：{error}")
        h_by_phase = boundary.get("H")
        if not isinstance(h_by_phase, dict):
            missing.append("system.boundary.H (按相列出允许的超胞矩阵)")
        elif not h_by_phase and (system.get("H_generation") or {}).get("enabled"):
            missing.append("system.boundary.H (读取配置 JSON 时由程序从各相母结构生成)")
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
                 editable_config_filename="search_config.project.json",
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

    def revise_mc_budget_limit(self, steps: int, *, conversation_id=None):
        """Apply an already-disambiguated MC round limit without another LLM guess."""
        session = self.workflow_kwargs.get("config_session") or {}
        if session.get("status") != "draft" or session.get("setup_stage") != "json_ready":
            return "当前没有可编辑的配置草稿；MC 预算未修改。"
        try:
            source = self._project_source_context(session)
            if source is None:
                raise ValueError("当前设置文件不是项目短配置")
            from config_layer.session.project_config_json import write_project_config_patch
            write_project_config_patch(
                self.editable_config_path,
                {"round_strategy.maximum_mc_budget": int(steps)},
                expected_hash=source["hash"], baseline_config=session.get("config") or {},
            )
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            return f"MC 单轮预算上限未修改：{error}"
        updated = deepcopy(session)
        _clear_config_review(updated)
        self._save(updated)
        review = self._import_and_review_config_json(
            updated, f"将 MC 单轮预算上限设为 {steps} 步", conversation_id=conversation_id,
        )
        return (f"已只将 MC 单轮预算上限设为 {steps} 步；其他参数未改。"
                "新配置仍需你确认，旧任务不会自动执行。\n" + review)

    def _setup_facts(self, session, *, config=None):
        config = config if config is not None else session.get("config") or {}
        storage = config.get("storage") or {}
        calculation = config.get("calculation") or {}
        mlip = config.get("mlip") or {}
        bohb_scope = ((config.get("bohb") or {}).get("scope") or {})
        workspace_root = storage.get("workspace_root") or session.get("pending_workspace_root")
        return {
            "workspace_root": str(workspace_root) if workspace_root else None,
            "workspace_root_confirmed": bool(storage.get("workspace_root")),
            "setup_stage": session.get("setup_stage"),
            "agent_model": self.current_deepseek_model,
            "default_mlip_version": "mace-mh-1",
            "calculation_mlip_version": calculation.get("mlip_version") or "mace-mh-1",
            "bohb_mlip_version": bohb_scope.get("mlip_version") or "mace-mh-1",
            "mlip_model_path": mlip.get("model_path"),
        }

    def __call__(self, messages, *, conversation_id=None):
        metadata_reply = _open_webui_metadata_reply(_latest_user_message(messages))
        if metadata_reply is not None:
            return metadata_reply
        session = self.workflow_kwargs.get("config_session") or {}
        message = _latest_user_message(messages)
        if session.get("status") != "confirmed" and session.get("setup_stage") == "json_ready":
            session = self._sync_editable_source(session)
        if session.get("status") == "confirmed":
            if _is_config_revision_request(message):
                updated = deepcopy(session)
                updated["status"] = "draft"
                updated["setup_stage"] = "json_ready"
                updated["draft_revision"] = int(updated.get("draft_revision", 0)) + 1
                updated["previous_confirmed_snapshot"] = deepcopy(session.get("confirmed_snapshot"))
                for key in ("agent_reviewed_revision", "agent_reviewed_config_hash",
                            "agent_reviewed_config_digest", "agent_reviewed_mother_digest"):
                    updated.pop(key, None)
                updated.setdefault("dialogue", []).append({
                    "type": "configuration_reopened", "reason": message,
                    "previous_config_version": (session.get("confirmed_snapshot") or {}).get("config_version"),
                })
                session = self._save(updated)
                self.delegate = None
            else:
                if self.delegate is None:
                    if not callable(self.runtime_factory):
                        return "配置快照已确认。请重启本地 Agent 进入搜索对话模式；本次没有启动计算。"
                    self.delegate = self.runtime_factory()
                return self.delegate(messages, conversation_id=conversation_id)
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
            reviewed_config_digest = session.get("agent_reviewed_config_digest")
            reviewed_mother_digest = session.get("agent_reviewed_mother_digest")
            current_hash = self._editable_config_hash()
            from config_layer.session.replace_config_from_json import config_digest, mother_structure_digest
            if (readiness["ready"] and reviewed_revision
                    and reviewed_revision == session.get("draft_revision")
                    and reviewed_hash and reviewed_hash == current_hash
                    and reviewed_config_digest == config_digest(session.get("config") or {})
                    and reviewed_mother_digest == mother_structure_digest(session.get("config") or {})):
                return self._confirm_and_start_search(
                    session, user_message=message, conversation_id=conversation_id,
                )
            return self._save_and_report(
                self._with_turn(session, message, ""),
                "还不能开始：请先检查配置 JSON 是否在审核后又被修改；若已修改，请重新发送“读取配置 JSON”，"
                "等待 Agent 审核通过后，再回复“同意”。",
            )
        if message.strip() in {"开始", "开始搜索", "继续"} and setup_stage == "json_ready":
            # Navigation commands are not parameter edits. In particular, do
            # not send "继续" with old budget dialogue back to the LLM.
            return self._import_and_review_config_json(
                session, message, continue_if_ready=message.strip() != "继续",
                conversation_id=conversation_id,
            )
        if _is_config_json_import_command(message):
            return self._import_and_review_config_json(
                session, message,
                continue_if_ready=_config_json_command_mode(message) == "continue",
                conversation_id=conversation_id,
            )
        if _is_pending_config_write_command(message):
            return self._write_pending_config_patch(session, message)
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
            source_context = self._project_source_context(session)
        except (OSError, TypeError, ValueError, KeyError) as error:
            return (f"当前设置文件 {self.editable_config_path} 读取失败：{error}。"
                    "请修正该文件后发送“读取配置 JSON”；本次没有使用旧会话配置作答。")
        working_config = (source_context["config"] if source_context else session.get("config") or {})
        from config_layer.session.build_config_field_catalog import build_config_field_catalog
        try:
            result = self.agent_client({
                "mode": "configuration_dialogue",
                "instruction": message,
                "draft_config": _compact_agent_config(working_config),
                "editable_field_catalog": build_config_field_catalog(working_config),
                "draft_revision": session.get("draft_revision"),
                "conversation_context": _recent_dialogue_context(session),
                "setup_facts": self._setup_facts(session, config=working_config),
                "bootstrap_hints": session.get("bootstrap_hints") or BOOTSTRAP_HINTS,
                "editable_config_json": str(self.editable_config_path) if self.editable_config_path else None,
                "verified_config_source": source_context["facts"] if source_context else None,
                "readiness": (self.configuration_readiness(session) if not source_context or source_context["imported"]
                              else {"ready": False, "status": "file_not_imported"}),
            })
        except Exception as error:
            diagnostic = getattr(error, "safe_message", type(error).__name__)
            self._record(session, "user", message)
            self._record(session, "assistant", f"配置助手暂时不可用（{diagnostic}）；草稿未更改。")
            return f"配置助手暂时不可用：{diagnostic}\n配置草稿未更改；请修正提示的问题后重试。"
        return self._apply_agent_response(session, message, result, source_context=source_context)

    def _sync_editable_source(self, session):
        """Follow the selected workspace's configured short file after migration."""
        if not self.editable_config_filename.endswith(".project.json"):
            return session
        preferred = self.workspace_root_default / self.editable_config_filename
        if not preferred.is_file() or self.editable_config_path == preferred:
            return session
        updated = deepcopy(session)
        old_path = str(self.editable_config_path) if self.editable_config_path else None
        self.editable_config_path = preferred
        updated["editable_config_json_path"] = str(preferred)
        updated.pop("agent_reviewed_revision", None)
        updated.pop("agent_reviewed_config_hash", None)
        updated.pop("agent_reviewed_config_digest", None)
        updated.pop("agent_reviewed_mother_digest", None)
        updated.setdefault("dialogue", []).append({
            "type": "editable_config_source_changed", "old_path": old_path,
            "new_path": str(preferred),
            "reason": "已按工作区运行时配置切换到现存短配置；等待重新导入。",
        })
        return self._save(updated)

    def _project_source_context(self, session):
        if self.editable_config_path is None or not self.editable_config_path.name.endswith(".project.json"):
            return None
        from config_layer.session.load_editable_config_json import load_editable_config_json
        from config_layer.session.replace_config_from_json import config_digest, mother_structure_digest
        from config_layer.session.resolve_phase_reference_directory import resolve_phase_reference_directory
        from config_layer.session.summarize_config_for_agent import summarize_config_for_agent

        source_hash = self._editable_config_hash()
        if not source_hash:
            raise ValueError("文件不存在或不可读")
        imported = (session.get("last_imported_config_hash") == source_hash
                    and session.get("last_imported_config_path") == str(self.editable_config_path)
                    and session.get("last_imported_config_digest") == config_digest(session.get("config") or {})
                    and session.get("last_imported_mother_digest") == mother_structure_digest(session.get("config") or {}))
        if imported:
            config = session.get("config") or {}
        else:
            config = load_editable_config_json(
                self.editable_config_path, session.get("config") or {},
                default_storage=default_workspace_storage(self.workspace_root_default),
            )
            config = resolve_phase_reference_directory(config, base_directory=self.base_directory)
            if self._editable_config_hash() != source_hash:
                raise ValueError("配置文件在读取过程中发生变化，请保存后重新读取")
        return {
            "hash": source_hash, "imported": imported, "config": config,
            "facts": summarize_config_for_agent(
                config, source_path=self.editable_config_path, source_hash=source_hash,
                generated_h=imported,
            ),
        }

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
        from config_layer.session.project_config_json import create_project_config_json

        try:
            creator = create_project_config_json if draft_path.name.endswith(".project.json") else create_editable_config_json
            created = creator(
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
            "请直接编辑该文件；下面分开列出必须补齐项和建议核对项。"
        )
        readiness = self.configuration_readiness(updated)
        required_items = [
            *(readiness.get("missing") or []),
            *(readiness.get("conflicts") or []),
            *(readiness.get("ambiguities") or []),
        ]
        if required_items:
            reply += "\n\n必须补齐或解决（否则不能继续）：\n" + "\n".join(
                f"- {item}" for item in required_items
            )
        else:
            reply += "\n\n必填项：程序暂未发现缺项；仍需在导入时由 Agent 和程序复核。"
        reply += "\n\n建议检查项（用于避免边界或成本口径误设）：\n" + "\n".join(
            f"- {item}" for item in RECOMMENDED_CONFIG_CHECKS
        )
        reply += (
            "\n\n改好后，想先审核再决定可发送“读取配置 JSON”；"
            "若检查通过就直接进入搜索 Agent，可发送“读取配置 JSON 并继续”。"
            "后者只在 Agent 和程序都通过时生效，也不会提交计算作业。"
        )
        self._save(record_config_dialogue(updated, role="assistant", message=reply))
        return reply

    def _import_and_review_config_json(
        self, session, user_message, *, continue_if_ready=False, conversation_id=None
    ):
        if self.editable_config_path is None:
            return "未配置可编辑 JSON 草稿路径；配置草稿未更改。"
        from config_layer.schema.validate_search_config import validate_search_config
        from config_layer.session.load_editable_config_json import load_editable_config_json
        from config_layer.session.replace_config_from_json import replace_config_from_json
        from config_layer.session.resolve_phase_reference_directory import (
            resolve_phase_reference_directory,
        )
        from config_layer.session.summarize_config_for_agent import summarize_config_for_agent

        try:
            from config_layer.session.load_editable_config_json import _strip_jsonc_comments
            from config_layer.session.project_config_json import FORMAT_ID as PROJECT_FORMAT_ID
            source_hash_before = self._editable_config_hash()
            document = json.loads(_strip_jsonc_comments(
                self.editable_config_path.read_text(encoding="utf-8")))
            config = load_editable_config_json(
                self.editable_config_path, session.get("config") or {},
                default_storage=default_workspace_storage(self.workspace_root_default),
            )
            config, _ = _fill_default_mlip_versions(config)
            config = resolve_phase_reference_directory(
                config, base_directory=self.base_directory
            )
            from config_layer.session.replace_config_from_json import mother_structure_digest
            mother_digest_before = mother_structure_digest(config)
            from config_layer.session.materialize_layered_h import materialize_layered_h
            config = materialize_layered_h(config)
            if mother_structure_digest(config) != mother_digest_before:
                raise ValueError("母结构文件在生成 H 期间发生变化，请确认文件稳定后重新读取")
            audit = validate_search_config(config)
            imported_config_hash = self._editable_config_hash()
            if source_hash_before != imported_config_hash:
                raise ValueError("配置文件在读取或生成 H 期间发生变化，请保存后重新读取")
        except (OSError, TypeError, ValueError, KeyError) as error:
            return f"配置 JSON 未导入：{error}\n当前草稿未更改。"

        updated = replace_config_from_json(
            session, config, source_path=str(self.editable_config_path),
            source_hash=imported_config_hash,
        )
        if document.get("_format") == PROJECT_FORMAT_ID:
            updated = dict(updated)
            updated["config_profile"] = {
                "name": document["profile"], "digest": document["profile_digest"],
            }
        updated = self._with_turn(updated, user_message, "")
        updated = self._save(updated)
        agent_reply = ""
        questions = []
        agent_note = ""
        agent_passed = False
        readiness = self.configuration_readiness(updated)
        verified_source = summarize_config_for_agent(
            config, source_path=self.editable_config_path,
            source_hash=imported_config_hash, generated_h=bool(
                (config.get("system") or {}).get("H_generation", {}).get("enabled")
            ),
        )
        if callable(self.agent_client):
            try:
                result = self.agent_client({
                    "mode": "configuration_json_review",
                    "instruction": (
                        "用户刚从本地 JSON 文件导入配置草稿。请检查缺项、冲突、歧义和参数口径，"
                        "解释必要的修订并提出集中问题。只审核，不返回或应用 patch。必须返回布尔字段 "
                        "ready_for_search；仅当没有阻止开始搜索的问题时为 true。不要确认配置或启动计算。"
                        "verified_config_source 是程序已读取、生成和校验的事实；不得否认文件可读，"
                        "不得要求用户手工填写已生成的 H，也不得引用旧对话配置。"
                        "atomate_defaults 的空参数、空动态配额、空 focus_regions、未接通 scheduler、"
                        "已写入文件的收敛阈值以及阶段上限之和超过总预算都不是配置阻塞项，"
                        "不得要求用户逐项重复确认。只报告 readiness 或 validation 中有证据的阻塞项。"
                    ),
                    "draft_config": _compact_agent_config(updated.get("config") or {}),
                    "draft_revision": updated.get("draft_revision"),
                    "readiness": readiness,
                    "validation": audit,
                    "verified_config_source": verified_source,
                })
                if isinstance(result, dict):
                    contradiction = _agent_source_contradiction(result, verified_source)
                    agent_reply = (f"Agent 本次回答与程序核实结果矛盾，已忽略：{contradiction}。"
                                   if contradiction else str(result.get("reply") or "配置 JSON 已检查。"))
                    questions = [] if contradiction else result.get("questions") or []
                    if not isinstance(questions, list):
                        questions = []
                    if result.get("patch"):
                        agent_note = "Agent 返回了修改建议，但本次审查不会自动应用；请编辑 JSON 后再次读取。"
                    agent_passed = (
                        result.get("ready_for_search") is True
                        and not contradiction
                        and readiness["ready"]
                        and not result.get("patch")
                        and imported_config_hash is not None
                        and self._editable_config_hash() == imported_config_hash
                    )
                    if agent_passed:
                        from config_layer.session.replace_config_from_json import config_digest, mother_structure_digest
                        updated["agent_reviewed_revision"] = updated.get("draft_revision")
                        updated["agent_reviewed_config_hash"] = imported_config_hash
                        updated["agent_reviewed_config_digest"] = config_digest(updated.get("config") or {})
                        updated["agent_reviewed_mother_digest"] = mother_structure_digest(updated.get("config") or {})
                    else:
                        updated.pop("agent_reviewed_revision", None)
                        updated.pop("agent_reviewed_config_hash", None)
                        updated.pop("agent_reviewed_config_digest", None)
                        updated.pop("agent_reviewed_mother_digest", None)
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

        reply = "配置 JSON 已读取，程序检查结果已交给 Agent 审核；此次读取不会自动确认或开始搜索。"
        h_counts = ", ".join(
            f"{phase}={row['count']}" for phase, row in verified_source["H_by_phase"].items()
        ) or "无"
        reply += (f"\n程序核实：来源={self.editable_config_path}；"
                  f"允许相={','.join(verified_source['allowed_phases_union'])}；"
                  f"实际 H 数量：{h_counts}。")
        if document.get("_format") == PROJECT_FORMAT_ID:
            selected = [f"{section}.{key}" for section, values in document.get("config", {}).items()
                        if isinstance(values, dict) for key in values]
            overrides = list(document.get("overrides", {}))
            reply += (f"\n默认模板：{document['profile']} ({document['profile_digest']})。"
                      f"\n项目填写项：{', '.join(selected) or '无'}。"
                      f"\n高级覆盖章节：{', '.join(overrides) or '无'}。"
                      f"\n生效关键值：MLIP={config['calculation']['mlip_version']}，"
                      f"总预算={config['budgets']['total_relative_cost']}，"
                      f"初态数={config['run']['initial_states_per_branch']}。")
        if continue_if_ready:
            reply = reply.replace("此次读取不会自动确认或开始搜索。", "按你的条件确认指令继续检查。")
        if agent_reply:
            reply += "\n\n" + agent_reply
        if agent_note:
            reply += "\n\n" + agent_note
        if agent_passed:
            if continue_if_ready:
                reply += "\n\nAgent 与程序检查均通过，按你本条指令继续；不会自动提交计算作业。"
            else:
                reply += "\n\nAgent 与程序检查均通过。你核对无误后回复“同意”，即可保存配置版本并进入搜索 Agent；不会自动提交计算作业。"
        elif not readiness["ready"]:
            reply += "\n\n当前仍有必填项或冲突，按下方清单修改 JSON 后重新发送“读取配置 JSON”。"
        updated = record_config_dialogue(updated, role="assistant", message=reply)
        updated = self._save(updated)
        readiness = self.configuration_readiness(updated)
        formatted = _format_agent_reply(
            reply, [], readiness, questions, agent_review_passed=agent_passed,
        )
        report = f"文件：{self.editable_config_path}\n草稿版本：{updated.get('draft_revision')}\n\n{formatted}"
        if continue_if_ready and agent_passed:
            continuation = self._confirm_and_start_search(
                updated, user_message=None, conversation_id=conversation_id,
            )
            return f"{report}\n\n{continuation}"
        return report

    def _apply_agent_response(self, session, user_message, result, *, source_context=None):
        if not isinstance(result, dict):
            return self._save_and_report(
                self._with_turn(session, user_message, "Agent 返回格式无效，草稿未更改。"),
                "Agent 返回格式无效，草稿未更改。请重试或使用本地配置控制接口。",
            )
        if source_context:
            contradiction = _agent_source_contradiction(result, source_context["facts"])
            if contradiction:
                result = {
                    "reply": f"Agent 本次说法与本地程序已核实的配置矛盾，已忽略：{contradiction}。",
                    "patch": {}, "questions": [],
                }
        patch = result.get("patch") or {}
        if not isinstance(patch, dict) or len(patch) > 40:
            patch = None
        updated = self._with_turn(session, user_message, "")
        changes = []
        write_authorized = (_explicit_config_write_request(user_message)
                            or result.get("write_requested") is True)
        if patch and source_context and write_authorized:
            try:
                from config_layer.session.validate_requested_config_patch import validate_requested_config_patch
                patch = validate_requested_config_patch(user_message, patch)
                _validate_patch(patch)
                from config_layer.session.project_config_json import write_project_config_patch
                changes = write_project_config_patch(
                    self.editable_config_path, patch, expected_hash=source_context["hash"],
                    baseline_config=session.get("config") or {},
                )
                updated.pop("pending_config_patch", None)
                _clear_config_review(updated)
                fields = "、".join(change["path"] for change in changes)
                self._save(updated)
                review = self._import_and_review_config_json(
                    updated, user_message, continue_if_ready=False,
                )
                result = {**result, "reply": (
                    f"{_project_config_migration_note(self.editable_config_path)}"
                    f"{'已写入' if changes else '原值已生效，无需重复写入'} {fields or '所请求参数'}。"
                    f"\n{review}"
                )}
                return result["reply"]
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                patch = None
                result = {**result, "reply": f"{result.get('reply', '')}\n\n配置文件写入被安全检查拒绝：{error}"}
        elif patch and source_context:
            updated["pending_config_patch"] = {
                "patch": deepcopy(patch), "source_hash": source_context["hash"],
                "reasons": deepcopy(result.get("reasons") or {}),
            }
            fields = "、".join(sorted(patch))
            result = {**result, "reply": (
                f"已识别配置修改建议：{fields}。尚未写入文件；回复“写入”即可执行。"
            )}
        elif patch:
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
        if source_context and not source_context["imported"]:
            facts = source_context["facts"]
            return (f"当前设置文件：{self.editable_config_path}\n"
                    f"程序已读取文件；允许相：{', '.join(facts['allowed_phases_union'])}；"
                    f"H_generation={'已配置' if facts['H_generation'] else '未配置'}。"
                    f"尚未执行母结构枚举及导入审核。\n\n{reply}\n\n"
                    "发送“读取配置 JSON”后，程序会生成各相 H 并给出具体检查结果。")
        readiness = self.configuration_readiness(updated)
        return _format_agent_reply(reply, changes, readiness, questions)

    def _write_pending_config_patch(self, session, user_message):
        pending = session.get("pending_config_patch") or {}
        patch = pending.get("patch")
        if not isinstance(patch, dict) or not patch:
            return "当前没有待写入的参数修改；请先告诉我具体字段和值。"
        if self._editable_config_hash() != pending.get("source_hash"):
            return "配置文件已在建议后变化；请重新说明修改内容，避免覆盖你的编辑。"
        try:
            _validate_patch(patch)
            from config_layer.session.project_config_json import write_project_config_patch
            changes = write_project_config_patch(
                self.editable_config_path, patch, expected_hash=pending["source_hash"],
                baseline_config=session.get("config") or {},
            )
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            return f"配置文件未修改：{error}"
        updated = self._with_turn(session, user_message, "")
        updated.pop("pending_config_patch", None)
        _clear_config_review(updated)
        self._save(updated)
        review = self._import_and_review_config_json(updated, user_message)
        return (f"已把 {len(changes)} 项修改写入 {self.editable_config_path}。"
                "当前运行仍使用原配置，待你确认新版本。\n" + review)

    def _confirm_and_start_search(self, session, *, user_message, conversation_id):
        if not callable(self.runtime_factory):
            return (
                "Agent 与程序检查已通过，但当前没有配置搜索运行入口；"
                "配置尚未确认，请检查 runtime_factory 后重试。"
            )
        confirmation_session = (
            session if user_message is None else self._with_turn(session, user_message, "")
        )
        confirmation = self._confirm(confirmation_session)
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
                "本地 DeepSeek API 尚未启用。请打开本机设置页 http://127.0.0.1:8765/phase/setup，"
                "粘贴 API Key 并测试连接；成功后无需重启服务。密钥只保存在本机系统凭据库。")


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


def _config_json_command_mode(message):
    """Accept short commands and clear natural-language requests to re-read the file."""
    normalized = re.sub(r"[\s,，;；。.!！？?、:：]", "", str(message).strip().lower())
    continue_commands = {
        "读取配置json并继续", "读取配置json检查通过后继续", "读取配置json审核通过后继续",
        "读取配置json如果没问题就继续", "读取配置json没问题就继续",
        "读取配置json如检查通过则继续", "readconfigjsonandcontinue",
        "readconfigjsoncontinueifvalid",
    }
    if normalized in continue_commands:
        return "continue"
    review_commands = {
        "读取配置json", "检查配置json", "审查配置json",
        "读取配置", "检查配置", "审查配置", "重新读取配置",
        "readconfigjson", "reviewconfigjson",
    }
    if normalized in review_commands:
        return "review"
    if normalized in {"配置改好了", "设置改好了", "我改好了", "已改好配置", "配置已保存", "设置已保存"}:
        return "review"
    if any(word in normalized for word in ("不要读取", "暂不读取", "别读取", "先不读取")):
        return None
    if normalized.startswith(("请", "帮我", "重新", "读取", "检查", "审查", "配置已改好", "我改好了")):
        mentions_file = any(word in normalized for word in (
            "配置json", "配置文件", "设置文件", "当前配置", "search_config.project.json",
            "search_config.draft.json",
        ))
        requests_read = any(word in normalized for word in ("读取", "检查", "审查"))
        if mentions_file and requests_read:
            return "continue" if "通过后继续" in normalized or "没问题就继续" in normalized else "review"
    return None


def _is_config_json_import_command(message):
    """Only explicit local-JSON review/conditional-continue commands are accepted."""
    return _config_json_command_mode(message) is not None


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
        r"^(?:工作区根目录|工作区根路径|根路径|工作区路径|存储路径|保存路径|地址是)\s*[:：]?\s*",
        "",
        candidate,
    ).strip()
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "\"'`":
        candidate = candidate[1:-1].strip()
    if re.search(r"(?:^|\s)(?:agent|deepseek|模型)(?:\s*模型)?(?:\s*版本)?\s*[:：]", candidate, re.I):
        return None
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
    elif re.search(r"\s+(?=(?:agent|deepseek|模型)(?:\s*模型)?(?:\s*版本)?\s*[:：])", raw, re.I):
        segments = [part.strip() for part in re.split(
            r"\s+(?=(?:agent|deepseek|模型)(?:\s*模型)?(?:\s*版本)?\s*[:：])",
            raw, flags=re.I) if part.strip()]
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
    from config_layer.session.validate_workspace_root import validate_workspace_root
    candidate = validate_workspace_root(candidate)
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


def _recent_dialogue_context(session, *, limit=8, max_chars=1600):
    """Keep only dialogue since the current configuration source was imported."""
    context = []
    entries = session.get("dialogue") or []
    start = 0
    for index, entry in enumerate(entries):
        if (entry.get("type") in {"config_import", "editable_config_source_changed"}
                or entry.get("type") == "config_revision"
                and entry.get("author") == "user_json_draft"):
            start = index + 1
    for entry in entries[start:][-limit:]:
        if entry.get("type") != "message" or entry.get("role") not in {"user", "assistant"}:
            continue
        message = str(entry.get("message") or "")
        if _open_webui_metadata_reply(message) is not None:
            continue
        context.append({"role": entry["role"], "message": message[-max_chars:]})
    return context


def _open_webui_metadata_reply(message):
    """Keep Open WebUI title/tag/follow-up helpers out of the scientific dialogue."""
    text = str(message or "").lstrip()
    if not text.startswith("### Task:"):
        return None
    lowered = text.lower()
    if "suggest 3-5 relevant follow-up" in lowered or '"follow_ups"' in lowered:
        return '{"follow_ups": []}'
    if "broad tags categorizing" in lowered or '"tags"' in lowered:
        return '{"tags": ["材料相图搜索"]}'
    if "generate a concise" in lowered and "title" in lowered:
        return "相图搜索配置"
    return None


def _compact_agent_config(config):
    """Send decision-relevant settings; H matrices stay in local numeric code."""
    keys = ("frozen_parameters", "branch_partition_suggestions", "generation_actions",
            "calculation", "budgets", "mlip", "dft", "convergence", "mc_policy",
            "round_strategy", "run")
    excerpt = {key: deepcopy(config[key]) for key in keys if key in config}
    system = config.get("system") or {}
    excerpt["system"] = {key: deepcopy(system[key]) for key in (
        "system_id", "configuration_space", "phase_reference_directory", "phase_references", "H_generation",
        "constraints", "branch_schema",
    ) if key in system}
    boundary = system.get("boundary") or {}
    excerpt["system"]["boundary"] = {key: deepcopy(boundary[key])
                                        for key in ("P", "TM_ratio") if key in boundary}
    finetune = config.get("mlip_finetune") or {}
    excerpt["mlip_finetune"] = {key: deepcopy(finetune[key]) for key in
                                ("enabled", "validation", "refresh_validation") if key in finetune}
    qbc = config.get("qbc") or {}
    excerpt["qbc"] = {key: deepcopy(qbc[key]) for key in
                      ("mode", "extreme_uncertainty") if key in qbc}
    cluster = config.get("supercomputer") or {}
    excerpt["supercomputer"] = {"scheduler": deepcopy(cluster.get("scheduler") or {})}
    return excerpt


def _agent_source_contradiction(result, facts):
    """Catch only claims directly disproved by the current local file facts."""
    output = str(result.get("reply") or "") + " " + " ".join(
        map(str, result.get("questions") or [])
    )
    source = str(facts.get("source_path") or "")
    if source.endswith("search_config.project.json") and any(
        "search_config.draft.json" in line
        and any(label in line for label in ("当前草稿", "当前配置", "可编辑配置", "正在读取", "来源="))
        and not any(label in line for label in ("旧", "历史", "已停用"))
        for line in output.splitlines()
    ):
        return "引用了旧设置文件"
    if any(phrase in output for phrase in (
        "我无法读取该文件", "无法读取本地文件", "无法打开本地文件",
        "请把配置文件内容贴出", "请提供配置文件内容",
    )):
        return "程序已读取当前本地设置文件"
    if facts.get("generated_H") and all(
        (facts.get("H_by_phase") or {}).get(phase, {}).get("count", 0) > 0
        for phase in facts.get("allowed_phases_union") or []
    ) and any(phrase in output for phrase in (
        "H 尚未生成", "H尚未生成", "请提供具体矩阵", "请把矩阵填入",
        "请提供各相合法矩阵", "boundary.H 四个相键缺失",
    )):
        return "各相 H 已由程序生成并核实"
    return None


def _fill_default_mlip_versions(config):
    """Migrate legacy null version fields when the configured model is mh-1."""
    mlip = config.get("mlip") or {}
    calculation = config.get("calculation") or {}
    bohb_scope = ((config.get("bohb") or {}).get("scope") or {})
    selected_version = calculation.get("mlip_version")
    defaulted = []
    if not selected_version and mlip.get("name") == "mace-mh-1":
        selected_version = "mace-mh-1"
        calculation["mlip_version"] = selected_version
        config["calculation"] = calculation
        defaulted.append("calculation.mlip_version")
    if not bohb_scope.get("mlip_version") and selected_version:
        bohb_scope["mlip_version"] = selected_version
        config.setdefault("bohb", {})["scope"] = bohb_scope
        defaulted.append("bohb.scope.mlip_version")
    return config, defaulted


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


def _explicit_config_write_request(message):
    normalized = re.sub(r"\s+", "", str(message or "").lower())
    if any(phrase in normalized for phrase in (
        "你写", "帮我写", "替我写", "写入配置", "写进配置", "直接写",
        "直接改", "帮我改", "帮我修订配置", "修改配置文件", "保存到配置", "修改设置文件",
    )):
        return True
    if any(word in normalized for word in (
        "如果", "是否", "能否", "可否", "能不能", "可不可以", "建议", "怎么样", "合适吗", "吗?", "吗？",
    )):
        return False
    assignment = any(word in normalized for word in (
        "改为", "改成", "设置为", "设为", "调整为", "修改为", "增加到", "提高到", "降低到",
    ))
    parameter = any(word in normalized for word in (
        "上限", "预算", "配额", "branch", "初态", "步数", "patience", "ehull", "阈值",
        "相", "超胞", "det(h)", "路径", "模型版本", "mlip", "dft", "mc",
    ))
    return assignment and parameter


def _project_config_migration_note(path):
    try:
        from config_layer.session.load_editable_config_json import _strip_jsonc_comments
        document = json.loads(_strip_jsonc_comments(Path(path).read_text(encoding="utf-8")))
        migration = document.get("_migration") if isinstance(document, dict) else None
        backup = migration.get("backup_file") if isinstance(migration, dict) else None
        if backup:
            return f"已按已保存配置兼容旧模板，并保留原文件备份 {Path(path).with_name(backup)}；"
    except (OSError, ValueError, TypeError):
        pass
    return ""


def _is_pending_config_write_command(message):
    return re.sub(r"\s+", "", str(message or "").lower()) in {
        "写入", "保存", "写吧", "改吧", "确认写入", "执行写入",
        "写入配置", "确认修改", "同意修改", "写进配置", "保存到配置",
    }


def _is_config_revision_request(message):
    text = re.sub(r"\s+", "", str(message or "").lower())
    config_scope = any(word in text for word in (
        "配置", "设置文件", "参数文件", "search_config.project.json",
    ))
    edit_intent = any(word in text for word in (
        "修改", "调整", "改", "设为", "更新", "修订", "写入", "保存",
    ))
    return config_scope and edit_intent


def _clear_config_review(session):
    for key in ("agent_reviewed_revision", "agent_reviewed_config_hash",
                "agent_reviewed_config_digest", "agent_reviewed_mother_digest"):
        session.pop(key, None)


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
