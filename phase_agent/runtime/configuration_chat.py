"""Open WebUI configuration dialogue and versioned draft revisions."""

from __future__ import annotations


import hashlib

import json

import re

from copy import deepcopy

import os

from pathlib import Path


from phase_agent.configuration.schema.validate_search_config import validate_search_config

from phase_agent.configuration.session.apply_config_revision import apply_config_revision

from phase_agent.configuration.session.answer_default_parameter_prompt import (
    answer_default_parameter_prompt,
)

from phase_agent.configuration.session.confirm_config_snapshot import confirm_config_snapshot

from phase_agent.configuration.session.record_config_dialogue import record_config_dialogue

from phase_agent.configuration.session.resolve_workspace_paths import (
    default_workspace_storage,
    resolve_workspace_paths,
)

from phase_agent.configuration.session.save_config_session import save_config_session

from phase_agent.science.structures.boundary_utils import allowed_phases


CONFIG_AGENT_SYSTEM_PROMPT = """Natural-language parameter setup applies to every editable field in editable_field_catalog, not only scientific scope: sampling counts, per-branch initial structures, MC settings, convergence criteria, training/validation, DFT parameters, model metadata, local/remote environments, paths and resource settings. Understand paraphrases and multiple requirements in one message. Explicit values should populate a draft directly; advice-only questions return recommendations without writing. When the user asks you to recommend AND fill reasonable settings, propose evidence-supported values in the draft, identify them as Agent recommendations pending confirmation, and give short reasons in reasons and reply. Do not silently treat recommendations as user-specified or confirmed. Retain current values for unspecified parameters and identify retained defaults when relevant. Units must be explicit: convert meV/atom to eV/atom where required; relative_cost is not GPU-hours, wall-time, money, or job count. Distinguish candidate quota, selected branch cap, structures per branch and scheduler batch size. Only adjust linked caps when needed to implement the explicitly requested scope; never enlarge budgets to make a proposal pass. List-valued MC tier settings must preserve tier meaning and distinguish max_mc_steps from patience_steps. Do not invent remote paths, environment names, scheduler resources, model hashes or validation evidence. Unsupported parameters or units need a concise clarification, never a new undocumented field. Summarize only changed settings and necessary questions in plain Chinese rather than dumping internal paths. Saving the draft does not authorize computation or activate models.
Natural-language scope entry: infer editable configuration paths from the user's complete meaning, without requiring field names or a fixed sentence template. A hypothetical example, quotation, capability question, or discussion is not an instruction to write that example into the current project: return patch={} and write_requested=false. When the user actually defines a project, fill all unambiguous requested fields together and explain the interpreted scope in concise Chinese. "All listed phases compete throughout Na0 to Na1" means the same phase list at both endpoints and intermediate compositions; do not impose conventional phase exclusions. Use system.boundary.P as a list for that full-range choice. A composition-dependent phase request instead uses at_x/intermediate. The standard layered Na domain is [0,1]; never claim arbitrary subranges are enforced if the schema cannot express them. Single TM maps to boundary.TM_ratio with only that species at ratio 1. A global supercell multiplicity cap maps to H_generation.size_max and the matching budgets.structure_limits.max_det_H and qbc.budget_limits.structure_limits.max_det_H when these exist, retaining unrelated budgets. Do not invent a lower bound or parity rule; state retained defaults separately. If an existing explicit boundary.H would conflict with requested H_generation, clear its generated matrices in the draft so they can be recomputed at import. For added phase freedom, check whether old single-phase configuration disabled competing_phase and restore it only when the user requested phase competition. Missing reference files do not justify removing requested phases or inventing paths. Save known intent to the draft and report only the necessary missing setup. Never interpret "all phases" as exhaustive enumeration of every configuration or automatic HPC submission. Approval remains required for scientific actions.
Scientific scope is mandatory before planning. Interpret user meaning, not keyword matching. Explain and summarize phases, TM_ratio, H bounds and H/P/x/T/N roles at review. Do not treat profile examples or generic continuation as scientific choices. If TM exchange is disallowed even in a binary system, set system.configuration_space.roles.T=fixed and fixed_T_source=phase_reference together; use actual ordered reference occupancy. This does not freeze atomic coordinates during relaxation. Single TM has no exchange/ordering degree of freedom; single allowed phase has no competing-phase allocation. Do not remove physical TM interactions from the model. A fixed ratio alone does not mean fixed ordering. If the user's intended freedom is unclear, ask one concise question.

超胞限制必须有明确用户依据：未指定奇偶时采用size_step=1，selected_recommendation_indices=[]，不自动添加additional_containment_matrices。只有用户明确要求仅偶数扩胞时，才设偶数size_min和size_step=2；不得用推荐包含矩阵代替奇偶要求。用户要求其他周期包含条件时才选择相应矩阵。审查时用简短中文明确列出尺寸范围、允许奇偶、包含约束及几何筛选，区分用户指定与程序默认；模板默认不等于用户限制。
你是材料相图项目的配置助手。理解当前用户的目标，结合最近对话及程序核实的当前配置，返回配置修改或必要问题。你没有科学执行权限。

当前用户指令优先于模板默认和旧对话。draft_config与verified_config_source表示修改前的事实，不是用户希望保留的目标。不得把模板的元素、相边界、路径当科学必然条件。只修改用户本轮明确要求的内容；信息明确则完成全部修改，信息不确定才集中追问，不要求用户提供内部字段名。

editable_field_catalog是字段契约：editable=false表示程序派生事实，修改derived_from指向的源字段；replace_types表示允许整体替换的类型。相边界与元素比例属于用户选择，派生约束由程序同步。配置按中文含义和作用范围定位字段，不能为了满足数量擅自扩大预算或全局边界。候选生成量、入选量、初态量和每作业分组量是不同参数；本轮动作条件与持久配置也要区分。

首次配置重点是环境、相/元素边界、母结构、远端模型及必要计算设置。本地环境沿用已列出的默认值，超算环境由用户提供。环境名、远端路径是元数据，不在本机加载远端模型。当前目录和文件以setup_facts与verified_config_source为准，不引用其他项目的信息。

配置修改直接返回patch及write_requested=true；仅答疑或提建议时不请求写入。程序校验字段、值和文件版本后写入未确认草稿。缺母结构不阻止保存明确配置；保存不代表生成H、审核或确认。不要声称已写入或已计算，程序负责报告结果。

configuration_json_review只审查程序刚导入的事实，patch为空。ready_for_search依据readiness；列出有字段和值证据的问题，不重复索要已知事实。空atomate默认DFT参数、动态配额或未接通调度器不阻止仅生成建议。后续MC、微调、预算及验证项到使用时再处理。明确审批或条件确认由程序处理，模型不能授予执行权限。

最近对话用于理解短答，不能把一次确认泛化为所有参数。输出简短中文，未知项直接说明，不编造结构、路径、计算结果或科学阈值。

只返回JSON对象：{"reply":"中文说明","patch":{"点分配置路径":JSON值},"reasons":{"点分配置路径":"对应用户意图"},"questions":[],"write_requested":false,"ready_for_search":false}。

"""


BOOTSTRAP_HINTS = {
    "notes": [
        "母结构目录使用当前项目的配置；不要沿用其他项目的目录或相列表。",
        "远端模型路径由用户提供，仅保存为元数据。",
    ],
}


RECOMMENDED_CONFIG_CHECKS = (
    "system.boundary.P 的 Na 含量→允许相映射、H_generation 尺寸与包含条件、TM_ratio 是否准确。",
    "母结构目录是否包含与相名一致的文件（例如 O3.vasp）；文件内容及相名映射是否正确。",
    "mlip.model_path 是否是超算端实际可访问的模型路径；本地 Agent 不会加载该模型。",
    "python_environments 中本地Python/MLIP与超算Python/MLIP环境是否分别确认；超算环境不可沿用本机名称。",
    "总预算及 MC、DFT 子预算是否使用一致的项目成本单位，且子预算没有超过总预算。",
    "DFT 单点与弛豫的 user_incar_settings、赝势和计算参数来源是否符合你的既定流程。",
    "收敛阈值及单位是否符合预期（能量误差为 eV/atom；相图变化按配置口径）；覆盖率是否仅作为证据而非硬门槛。",
)


def configuration_readiness(
    session: dict, *, base_directory, phase_references_path=None, workspace_root_default=None
) -> dict:
    """Return missing/conflicting facts required before confirming first-run config."""

    config = session.get("config") or {}

    try:
        audit = validate_search_config(config, stage="startup")

    except (TypeError, ValueError, KeyError, AttributeError) as error:
        audit = {"missing": [], "conflicts": [f"配置字段格式无效：{error}"], "ambiguities": []}

    missing = list(audit.get("missing") or [])

    conflicts = list(audit.get("conflicts") or [])

    ambiguities = list(audit.get("ambiguities") or [])

    from phase_agent.configuration.schema.python_environments import python_environment

    for host in ("local", "remote"):
        for mlip in (False, True):
            try:
                python_environment(config, host, mlip=mlip)

            except ValueError as error:
                missing.append(str(error))

    system = config.get("system") or {}

    awaiting_workspace = session.get("setup_stage") in {
        "awaiting_storage_path",
        "awaiting_storage_confirmation",
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
            from phase_agent.science.structures.boundary_utils import allowed_phases_at_x

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
                        isinstance(matrix, list)
                        and len(matrix) in {2, 3}
                        and all(
                            isinstance(row, list)
                            and len(row) == len(matrix)
                            and all(
                                isinstance(value, int) and not isinstance(value, bool)
                                for value in row
                            )
                            for row in matrix
                        )
                    )

                    if not valid_matrix:
                        conflicts.append(
                            f"system.boundary.H.{phase}[{index}] 必须是 2×2 或 3×3 整数矩阵"
                        )

        tm_ratio = boundary.get("TM_ratio")

        if (
            not isinstance(tm_ratio, dict)
            or not tm_ratio
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
                for value in tm_ratio.values()
            )
        ):
            missing.append("system.boundary.TM_ratio (元素及正比例值)")

        configured_phases = [
            str(phase) for phase in (system.get("constraints") or {}).get("phases") or []
        ]

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

    remote_model = mlip.get("model_path") or (
        (config.get("supercomputer") or {}).get("paths") or {}
    ).get("mlip_model")

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

    def __init__(
        self,
        workflow_kwargs,
        *,
        config_session_path,
        base_directory,
        phase_references_path=None,
        editable_config_path=None,
        editable_config_filename="search_config.project.json",
        editable_config_directory="",
        workspace_root_default=None,
        agent_client=None,
        runtime_factory=None,
        current_deepseek_model="deepseek-v4-pro",
        deepseek_model_switcher=None,
    ):

        self.workflow_kwargs = dict(workflow_kwargs)

        self.state_path = Path(workflow_kwargs["state_path"])

        self.config_session_path = Path(config_session_path)

        self.base_directory = Path(base_directory)

        self.phase_references_path = Path(phase_references_path) if phase_references_path else None

        self.editable_config_path = Path(editable_config_path) if editable_config_path else None

        self.editable_config_filename = safe_config_filename(editable_config_filename)

        self.editable_config_directory = editable_config_directory

        session = workflow_kwargs.get("config_session") or {}

        selected_root = session.get("pending_workspace_root") or (
            (session.get("config") or {}).get("storage") or {}
        ).get("workspace_root")

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

            from phase_agent.configuration.session.project_config_json import (
                write_project_config_patch,
            )

            write_project_config_patch(
                self.editable_config_path,
                {"round_strategy.maximum_mc_budget": int(steps)},
                expected_hash=source["hash"],
                baseline_config=session.get("config") or {},
            )

        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            return f"MC 单轮预算上限未修改：{error}"

        updated = deepcopy(session)

        _clear_config_review(updated)

        self._save(updated)

        review = self._import_and_review_config_json(
            updated,
            f"将 MC 单轮预算上限设为 {steps} 步",
            conversation_id=conversation_id,
        )

        return (
            f"已只将 MC 单轮预算上限设为 {steps} 步；其他参数未改。"
            "新配置仍需你确认，旧任务不会自动执行。\n" + review
        )

    def _setup_facts(self, session, *, config=None):

        config = config if config is not None else session.get("config") or {}

        storage = config.get("storage") or {}

        calculation = config.get("calculation") or {}

        mlip = config.get("mlip") or {}

        bohb_scope = (config.get("bohb") or {}).get("scope") or {}

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
            if _is_config_revision_request(message) or _is_config_json_import_command(message):
                updated = deepcopy(session)

                updated["status"] = "draft"

                updated["setup_stage"] = "json_ready"

                updated["draft_revision"] = int(updated.get("draft_revision", 0)) + 1

                updated["previous_confirmed_snapshot"] = deepcopy(session.get("confirmed_snapshot"))

                for key in (
                    "agent_reviewed_revision",
                    "agent_reviewed_config_hash",
                    "agent_reviewed_config_digest",
                    "agent_reviewed_mother_digest",
                ):
                    updated.pop(key, None)

                updated.setdefault("dialogue", []).append(
                    {
                        "type": "configuration_reopened",
                        "reason": message,
                        "previous_config_version": (session.get("confirmed_snapshot") or {}).get(
                            "config_version"
                        ),
                    }
                )

                session = self._save(updated)

                self.delegate = None

            else:
                if self.delegate is None:
                    if not callable(self.runtime_factory):
                        return (
                            "配置快照已确认。请重启本地 Agent 进入搜索对话模式；本次没有启动计算。"
                        )

                    self.delegate = self.runtime_factory()

                return self.delegate(messages, conversation_id=conversation_id)

        setup_stage = session.get("setup_stage")

        if setup_stage in {"awaiting_storage_path", "awaiting_storage_confirmation"}:
            return self._handle_workspace_setup(session, message)

        from phase_agent.runtime.resolve_deepseek_model_request import (
            resolve_deepseek_model_request,
        )

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

            from phase_agent.configuration.session.replace_config_from_json import (
                config_digest,
                mother_structure_digest,
            )

            if (
                readiness["ready"]
                and reviewed_revision
                and reviewed_revision == session.get("draft_revision")
                and reviewed_hash
                and reviewed_hash == current_hash
                and reviewed_config_digest == config_digest(session.get("config") or {})
                and reviewed_mother_digest == mother_structure_digest(session.get("config") or {})
            ):
                return self._confirm_and_start_search(
                    session,
                    user_message=message,
                    conversation_id=conversation_id,
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
                session,
                message,
                continue_if_ready=message.strip() != "继续",
                conversation_id=conversation_id,
            )

        if _is_config_json_import_command(message):
            return self._import_and_review_config_json(
                session,
                message,
                continue_if_ready=_config_json_command_mode(message) == "continue",
                conversation_id=conversation_id,
            )

        if _is_pending_config_write_command(message):
            return self._write_pending_config_patch(session, message)

        if session.get("default_parameter_prompt", {}).get("status") == "awaiting_response":
            choice = _default_choice(message)

            if choice is not None:
                updated = answer_default_parameter_prompt(session, use_defaults=choice)

                path_note = (
                    f"完整可编辑参数 JSON：{self.editable_config_path}。"
                    "你可以直接修改并发送“读取配置 JSON”导入检查。"
                    if self.editable_config_path
                    else ""
                )

                return self._save_and_report(
                    updated, "已记录默认参数选择。配置仍是草稿，尚未确认或运行。" + path_note
                )

        if not self.agent_client:
            return self._unavailable_message()

        try:
            source_context = self._project_source_context(session)

        except (OSError, TypeError, ValueError, KeyError) as error:
            return (
                f"当前设置文件 {self.editable_config_path} 读取失败：{error}。"
                "请修正该文件后发送“读取配置 JSON”；本次没有使用旧会话配置作答。"
            )

        working_config = source_context["config"] if source_context else session.get("config") or {}

        from phase_agent.configuration.session.build_config_field_catalog import (
            build_config_field_catalog,
        )

        try:
            result = self._request_configuration_result(
                {
                    "mode": "configuration_dialogue",
                    "instruction": message,
                    "draft_config": _compact_agent_config(working_config),
                    "editable_field_catalog": build_config_field_catalog(working_config),
                    "draft_revision": session.get("draft_revision"),
                    "conversation_context": _recent_dialogue_context(session),
                    "setup_facts": self._setup_facts(session, config=working_config),
                    "bootstrap_hints": BOOTSTRAP_HINTS,
                    "editable_config_json": str(self.editable_config_path)
                    if self.editable_config_path
                    else None,
                    "verified_config_source": source_context["facts"] if source_context else None,
                    "readiness": (
                        self.configuration_readiness(session)
                        if not source_context or source_context["imported"]
                        else {"ready": False, "status": "file_not_imported"}
                    ),
                }
            )

        except Exception as error:
            diagnostic = getattr(error, "safe_message", type(error).__name__)

            self._record(session, "user", message)

            self._record(session, "assistant", f"配置助手暂时不可用（{diagnostic}）；草稿未更改。")

            return f"配置助手暂时不可用：{diagnostic}\n配置草稿未更改；请修正提示的问题后重试。"

        return self._apply_agent_response(session, message, result, source_context=source_context)

    def _call_model(self, payload):

        from phase_agent.runtime.turn_process import timed_call

        return timed_call(
            "configuration_model",
            self.agent_client,
            payload,
            category="model",
            mode=payload.get("mode"),
        )

    def _request_configuration_result(self, payload):
        """One schema-guided correction before writing; the model owns semantics."""

        from phase_agent.configuration.session.build_config_field_catalog import (
            config_patch_field_errors,
        )

        from phase_agent.runtime.turn_process import timed_call

        def request(request_payload):

            return timed_call(
                "configuration_model",
                self.agent_client,
                request_payload,
                category="model",
                mode=request_payload.get("mode"),
            )

        result = request(payload)

        errors = config_patch_field_errors(result, payload["editable_field_catalog"])

        if errors:
            result = request(
                {
                    **payload,
                    "mode": "configuration_patch_repair",
                    "invalid_response": result,
                    "validation_errors": errors,
                    "repair_instruction": "根据当前用户原始指令和字段契约修正完整JSON。派生字段不能写；重新检查源字段表达的目标是否一致。不要用旧默认代替用户意图。未写入任何字段，不要声称成功。",
                }
            )

            remaining = config_patch_field_errors(result, payload["editable_field_catalog"])

            if remaining:
                error = ValueError("配置字段修正失败：" + "; ".join(remaining))

                error.safe_message = str(error)

                raise error

        return result

    def _editable_location(self, root):

        from phase_agent.configuration.session.editable_config_location import (
            editable_config_location,
        )

        return editable_config_location(
            root, Path(self.editable_config_directory) / self.editable_config_filename
        )

    def _sync_editable_source(self, session):
        """Follow the selected workspace's configured short file after migration."""

        if not self.editable_config_filename.endswith(".project.json"):
            return session

        preferred = self._editable_location(self.workspace_root_default)

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

        updated.setdefault("dialogue", []).append(
            {
                "type": "editable_config_source_changed",
                "old_path": old_path,
                "new_path": str(preferred),
                "reason": "已按工作区运行时配置切换到现存短配置；等待重新导入。",
            }
        )

        return self._save(updated)

    def _project_source_context(self, session):

        if self.editable_config_path is None or not self.editable_config_path.name.endswith(
            ".project.json"
        ):
            return None

        from phase_agent.configuration.session.load_editable_config_json import (
            load_editable_config_json,
        )

        from phase_agent.configuration.session.replace_config_from_json import (
            config_digest,
            mother_structure_digest,
        )

        from phase_agent.configuration.session.resolve_phase_reference_directory import (
            resolve_phase_reference_directory,
        )

        from phase_agent.configuration.session.summarize_config_for_agent import (
            summarize_config_for_agent,
        )

        source_hash = self._editable_config_hash()

        if not source_hash:
            raise ValueError("文件不存在或不可读")

        imported = (
            session.get("last_imported_config_hash") == source_hash
            and session.get("last_imported_config_path") == str(self.editable_config_path)
            and session.get("last_imported_config_digest")
            == config_digest(session.get("config") or {})
            and session.get("last_imported_mother_digest")
            == mother_structure_digest(session.get("config") or {})
        )

        if imported:
            config = session.get("config") or {}

        else:
            config = load_editable_config_json(
                self.editable_config_path,
                session.get("config") or {},
                default_storage=default_workspace_storage(self.workspace_root_default),
            )

            config = resolve_phase_reference_directory(config, base_directory=self.base_directory)

            if self._editable_config_hash() != source_hash:
                raise ValueError("配置文件在读取过程中发生变化，请保存后重新读取")

        return {
            "hash": source_hash,
            "imported": imported,
            "config": config,
            "facts": summarize_config_for_agent(
                config,
                source_path=self.editable_config_path,
                source_hash=source_hash,
                generated_h=imported,
            ),
        }

    def _handle_workspace_setup(self, session, user_message):

        normalized = " ".join(str(user_message).strip().lower().split())

        from phase_agent.runtime.resolve_deepseek_model_request import (
            resolve_deepseek_model_request,
        )

        root_value, requested_model = _parse_workspace_setup_values(user_message)

        requested_model = requested_model or resolve_deepseek_model_request(user_message)

        if requested_model:
            updated = dict(session)

            updated["pending_deepseek_model"] = requested_model

            if root_value:
                return self._preview_workspace_root(
                    updated,
                    user_message,
                    path_value=root_value,
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

        if session.get("setup_stage") == "awaiting_storage_confirmation" and normalized in {
            "确认",
            "确认并开始配置",
            "确认并开始",
            "开始配置",
            "确认存储路径",
            "确认工作区",
            "确认路径",
            "confirm workspace",
        }:
            return self._confirm_workspace_root(session, user_message)

        return self._preview_workspace_root(
            session,
            user_message,
            path_value=root_value,
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

        from phase_agent.configuration.session.resolve_workspace_paths import (
            default_workspace_storage,
        )

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
            f"设置 JSON：{self._editable_location(root)}",
        ]

        lines.append("确认这两项后回复“确认”；如需修改，请重新发送正确的路径或模型版本。")

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

        from phase_agent.configuration.session.resolve_workspace_paths import (
            default_workspace_storage,
        )

        storage = default_workspace_storage(root)

        draft_path = self._editable_location(root)

        selected_model = session.get("pending_deepseek_model")

        if selected_model and selected_model != self.current_deepseek_model:
            if not callable(self.deepseek_model_switcher):
                reply = (
                    "工作区路径有效，但本地运行时未配置 DeepSeek 模型切换接口；"
                    "设置 JSON 尚未创建。请检查 settings/agent_runtime.json 后重试。"
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

        from phase_agent.configuration.session.create_editable_config_json import (
            create_editable_config_json,
        )

        from phase_agent.configuration.session.project_config_json import create_project_config_json

        try:
            creator = (
                create_project_config_json
                if draft_path.name.endswith(".project.json")
                else create_editable_config_json
            )

            created = creator(
                draft_path,
                session.get("config") or {},
                bootstrap_hints=session.get("bootstrap_hints"),
                workspace_defaults=storage,
            )

            if draft_path.name.endswith(".project.json"):
                from phase_agent.configuration.session.split_project_config import (
                    split_project_file,
                )

                split_project_file(draft_path)

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

        from phase_agent.configuration.session.split_project_config import (
            read_document,
            run_config_path,
        )

        try:
            split_document = (
                read_document(draft_path) if draft_path.name.endswith(".project.json") else {}
            )

        except (OSError, ValueError):
            split_document = {}  # Preserve unreadable pre-existing drafts for explicit import diagnostics.

        run_note = (
            f"\n运行配置：{run_config_path(draft_path, split_document)}。"
            if "run_config_file" in split_document
            else ""
        )

        if created:
            status = f"已在确认的工作区生成初始配置 JSON：{draft_path}。" + run_note

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

        from phase_agent.configuration.schema.validate_search_config import validate_search_config

        from phase_agent.configuration.session.load_editable_config_json import (
            load_editable_config_json,
        )

        from phase_agent.configuration.session.replace_config_from_json import (
            replace_config_from_json,
        )

        from phase_agent.configuration.session.resolve_phase_reference_directory import (
            resolve_phase_reference_directory,
        )

        from phase_agent.configuration.session.summarize_config_for_agent import (
            summarize_config_for_agent,
        )

        try:
            from phase_agent.configuration.session.load_editable_config_json import (
                _strip_jsonc_comments,
            )

            from phase_agent.configuration.session.project_config_json import (
                FORMAT_ID as PROJECT_FORMAT_ID,
            )

            source_hash_before = self._editable_config_hash()

            document = json.loads(
                _strip_jsonc_comments(self.editable_config_path.read_text(encoding="utf-8"))
            )

            config = load_editable_config_json(
                self.editable_config_path,
                session.get("config") or {},
                default_storage=default_workspace_storage(self.workspace_root_default),
            )

            config, _ = _fill_default_mlip_versions(config)

            config = resolve_phase_reference_directory(config, base_directory=self.base_directory)

            from phase_agent.configuration.session.replace_config_from_json import (
                mother_structure_digest,
            )

            mother_digest_before = mother_structure_digest(config)

            from phase_agent.configuration.session.materialize_layered_h import (
                materialize_layered_h,
            )

            config = materialize_layered_h(config)

            if mother_structure_digest(config) != mother_digest_before:
                raise ValueError("母结构文件在生成 H 期间发生变化，请确认文件稳定后重新读取")

            audit = validate_search_config(config, stage="startup")

            imported_config_hash = self._editable_config_hash()

            if source_hash_before != imported_config_hash:
                raise ValueError("配置文件在读取或生成 H 期间发生变化，请保存后重新读取")

        except (OSError, TypeError, ValueError, KeyError) as error:
            return f"配置 JSON 未导入：{error}\n当前草稿未更改。"

        updated = replace_config_from_json(
            session,
            config,
            source_path=str(self.editable_config_path),
            source_hash=imported_config_hash,
        )

        if document.get("_format") == PROJECT_FORMAT_ID:
            updated = dict(updated)

            updated["config_profile"] = {
                "name": document["profile"],
                "digest": document["profile_digest"],
            }

        updated = self._with_turn(updated, user_message, "")

        updated = self._save(updated)

        agent_reply = ""

        questions = []

        agent_note = ""

        agent_passed = False

        readiness = self.configuration_readiness(updated)

        verified_source = summarize_config_for_agent(
            config,
            source_path=self.editable_config_path,
            source_hash=imported_config_hash,
            generated_h=bool((config.get("system") or {}).get("H_generation", {}).get("enabled")),
        )

        if callable(self.agent_client):
            try:
                result = self._call_model(
                    {
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
                    }
                )

                if isinstance(result, dict):
                    contradiction = _agent_source_contradiction(result, verified_source)

                    agent_reply = (
                        f"Agent 本次回答与程序核实结果矛盾，已忽略：{contradiction}。"
                        if contradiction
                        else str(result.get("reply") or "配置 JSON 已检查。")
                    )

                    questions = [] if contradiction else result.get("questions") or []

                    if not isinstance(questions, list):
                        questions = []

                    if result.get("patch"):
                        agent_note = (
                            "Agent 返回了修改建议，但本次审查不会自动应用；请编辑 JSON 后再次读取。"
                        )

                    agent_passed = (
                        result.get("ready_for_search") is True
                        and not contradiction
                        and readiness["ready"]
                        and not result.get("patch")
                        and imported_config_hash is not None
                        and self._editable_config_hash() == imported_config_hash
                    )

                    if agent_passed:
                        from phase_agent.configuration.session.replace_config_from_json import (
                            config_digest,
                            mother_structure_digest,
                        )

                        updated["agent_reviewed_revision"] = updated.get("draft_revision")

                        updated["agent_reviewed_config_hash"] = imported_config_hash

                        updated["agent_reviewed_config_digest"] = config_digest(
                            updated.get("config") or {}
                        )

                        updated["agent_reviewed_mother_digest"] = mother_structure_digest(
                            updated.get("config") or {}
                        )

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

                agent_note = (
                    f"Agent 暂时无法完成审核（{diagnostic}）；已导入的草稿仍保留，不能开始。"
                )

        else:
            updated.pop("agent_reviewed_revision", None)

            updated.pop("agent_reviewed_config_hash", None)

            agent_note = (
                "DeepSeek Agent 当前不可用；文件已导入，可稍后重新发送“读取配置 JSON”进行审核。"
            )

        reply = "配置 JSON 已读取，程序检查结果已交给 Agent 审核；此次读取不会自动确认或开始搜索。"

        h_counts = (
            ", ".join(
                f"{phase}={row['count']}" for phase, row in verified_source["H_by_phase"].items()
            )
            or "无"
        )

        reply += (
            f"\n程序核实：来源={self.editable_config_path}；"
            f"允许相={','.join(verified_source['allowed_phases_union'])}；"
            f"实际 H 数量：{h_counts}。"
        )

        if document.get("_format") == PROJECT_FORMAT_ID:
            selected = [
                f"{section}.{key}"
                for section, values in document.get("config", {}).items()
                if isinstance(values, dict)
                for key in values
            ]

            overrides = list(document.get("overrides", {}))

            reply += (
                f"\n默认模板：{document['profile']} ({document['profile_digest']})。"
                f"\n项目填写项：{', '.join(selected) or '无'}。"
                f"\n高级覆盖章节：{', '.join(overrides) or '无'}。"
                f"\n生效关键值：MLIP={config['calculation']['mlip_version']}，"
                f"总预算={config['budgets']['total_relative_cost']}，"
                f"初态数={config['run']['initial_states_per_branch']}。"
            )

        if continue_if_ready:
            reply = reply.replace(
                "此次读取不会自动确认或开始搜索。", "按你的条件确认指令继续检查。"
            )

        from phase_agent.configuration.session.summarize_config_for_agent import (
            format_scientific_scope,
        )

        reply += "\n\n" + format_scientific_scope(config)

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

        formatted = _concise_import_reply(
            h_counts,
            readiness,
            questions,
            agent_passed=agent_passed,
            agent_note=agent_note,
        )

        report = formatted

        if continue_if_ready and agent_passed:
            continuation = self._confirm_and_start_search(
                updated,
                user_message=None,
                conversation_id=conversation_id,
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
                    "patch": {},
                    "questions": [],
                }

        patch = result.get("patch") or {}

        if not isinstance(patch, dict) or len(patch) > 40:
            patch = None

        updated = self._with_turn(session, user_message, "")

        changes = []

        write_authorized = (
            _explicit_config_write_request(user_message) or result.get("write_requested") is True
        )

        if patch and source_context and write_authorized:
            try:
                from phase_agent.configuration.session.validate_requested_config_patch import (
                    validate_requested_config_patch,
                )

                patch = validate_requested_config_patch(user_message, patch)

                _validate_patch(patch)

                from phase_agent.configuration.session.project_config_json import (
                    write_project_config_patch,
                )

                changes = write_project_config_patch(
                    self.editable_config_path,
                    patch,
                    expected_hash=source_context["hash"],
                    baseline_config=session.get("config") or {},
                )

                updated.pop("pending_config_patch", None)

                _clear_config_review(updated)

                fields = "、".join(change["path"] for change in changes)

                review = self._sync_written_config_draft(updated)

                result = {
                    **result,
                    "reply": (
                        f"{_project_config_migration_note(self.editable_config_path)}"
                        f"{'已写入' if changes else '原值已生效，无需重复写入'} {fields or '所请求参数'}。"
                        f"\n{review}"
                    ),
                }

                return result["reply"]

            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                patch = None

                result = {
                    **result,
                    "reply": f"{result.get('reply', '')}\n\n配置文件写入被安全检查拒绝：{error}",
                }

        elif patch and source_context:
            updated["pending_config_patch"] = {
                "patch": deepcopy(patch),
                "source_hash": source_context["hash"],
                "reasons": deepcopy(result.get("reasons") or {}),
            }

            fields = "、".join(sorted(patch))

            result = {
                **result,
                "reply": (f"已识别配置修改建议：{fields}。尚未写入文件；回复“写入”即可执行。"),
            }

        elif patch:
            try:
                _validate_patch(patch)

                updated = apply_config_revision(
                    updated,
                    patch,
                    reasons=result.get("reasons") or {},
                    author="agent_draft_suggestion",
                )

                changes = updated["dialogue"][-1].get("changes") or []

            except (TypeError, ValueError) as error:
                patch = None

                result = {
                    **result,
                    "reply": f"{result.get('reply', '')}\n\n草稿修改被安全检查拒绝：{error}",
                }

        reply = str(result.get("reply") or "我已检查当前草稿。")

        questions = result.get("questions") or []

        if not isinstance(questions, list):
            questions = []

        updated["dialogue"].append({"type": "message", "role": "assistant", "message": reply})

        updated = self._save(updated)

        if source_context and not source_context["imported"]:
            facts = source_context["facts"]

            return (
                f"当前设置文件：{self.editable_config_path}\n"
                f"程序已读取文件；允许相：{', '.join(facts['allowed_phases_union'])}；"
                f"H_generation={'已配置' if facts['H_generation'] else '未配置'}。"
                f"尚未执行母结构枚举及导入审核。\n\n{reply}\n\n"
                "发送“读取配置 JSON”后，程序会生成各相 H 并给出具体检查结果。"
            )

        readiness = self.configuration_readiness(updated)

        return _format_agent_reply(reply, changes, readiness, questions)

    def _sync_written_config_draft(self, session):
        """Reflect saved facts without enumerating structures or approving a draft."""

        from phase_agent.configuration.session.load_editable_config_json import (
            load_editable_config_json,
        )

        from phase_agent.configuration.session.resolve_phase_reference_directory import (
            resolve_phase_reference_directory,
        )

        from phase_agent.configuration.session.replace_config_from_json import config_digest

        config = load_editable_config_json(
            self.editable_config_path,
            session.get("config") or {},
            default_storage=default_workspace_storage(self.workspace_root_default),
        )

        config = resolve_phase_reference_directory(config, base_directory=self.base_directory)

        updated = deepcopy(session)

        if updated.get("config") != config:
            updated["config"] = config

            updated["draft_revision"] = int(updated.get("draft_revision", 0)) + 1

        _clear_config_review(updated)

        for key in tuple(updated):
            if key.startswith("last_imported_"):
                updated.pop(key)

        updated["dialogue"].append(
            {
                "type": "config_draft_saved",
                "source_path": str(self.editable_config_path),
                "source_hash": self._editable_config_hash(),
                "config_digest": config_digest(config),
            }
        )

        self._save(updated)

        references = (config.get("system") or {}).get("phase_references") or {}

        missing = [
            phase
            for phase, reference in references.items()
            if not Path(
                str(reference.get("path") if isinstance(reference, dict) else reference)
            ).is_file()
        ]

        note = "配置文件与会话草稿已同步；尚未审核、确认或生成 H。"

        if missing:
            note += "\n尚缺母结构：" + "、".join(sorted(missing)) + "；补齐后再审核。"

        return note

    def _write_pending_config_patch(self, session, user_message):

        pending = session.get("pending_config_patch") or {}

        patch = pending.get("patch")

        if not isinstance(patch, dict) or not patch:
            return "当前没有待写入的参数修改；请先告诉我具体字段和值。"

        if self._editable_config_hash() != pending.get("source_hash"):
            return "配置文件已在建议后变化；请重新说明修改内容，避免覆盖你的编辑。"

        try:
            _validate_patch(patch)

            from phase_agent.configuration.session.project_config_json import (
                write_project_config_patch,
            )

            changes = write_project_config_patch(
                self.editable_config_path,
                patch,
                expected_hash=pending["source_hash"],
                baseline_config=session.get("config") or {},
            )

        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            return f"配置文件未修改：{error}"

        updated = self._with_turn(session, user_message, "")

        updated.pop("pending_config_patch", None)

        _clear_config_review(updated)

        review = self._sync_written_config_draft(updated)

        return (
            f"已把 {len(changes)} 项修改写入 {self.editable_config_path}。"
            "当前运行仍使用原配置，待你确认新版本。\n" + review
        )

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

            start_analysis = getattr(self.delegate, "start_analysis", None)

            first_turn = (
                start_analysis(conversation_id=conversation_id)
                if callable(start_analysis)
                else self.delegate(
                    [
                        {
                            "role": "user",
                            "content": (
                                "配置已由用户确认。开始首轮搜索分析并给出建议；只生成待审批建议，"
                                "不得直接执行、提交或派发任何计算任务。"
                            ),
                        }
                    ],
                    conversation_id=conversation_id,
                )
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
            from phase_agent.configuration.session.split_project_config import editable_project_hash

            return editable_project_hash(self.editable_config_path)

        except (OSError, ValueError, TypeError):
            return None

    def _confirm(self, session):

        readiness = self.configuration_readiness(session)

        if not readiness["ready"]:
            return _format_agent_reply(
                "当前配置还不能开始。请先补齐缺项并解决冲突，再让 Agent 重新审核。",
                [],
                readiness,
                [],
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
                "配置检查未通过，尚未生成快照。",
                [],
                self.configuration_readiness(confirmed),
                (confirmed.get("audit") or {}).get("missing") or [],
            )

        try:
            from phase_agent.configuration.session.save_confirmed_config_file import (
                save_confirmed_config_file,
            )

            snapshot_path = save_confirmed_config_file(
                confirmed["confirmed_snapshot"], base_directory=self.base_directory
            )

        except (OSError, TypeError, ValueError, FileExistsError) as error:
            return (
                f"配置检查通过，但版本快照未写入工作区：{error}。"
                "未提交或启动任何计算；请检查 storage.workspace_root 和 storage.paths 后重试。"
            )

        self._save(confirmed)

        version = confirmed["confirmed_snapshot"]["config_version"]

        return (
            f"配置已确认并保存为版本 {version}。配置快照：{snapshot_path}\n"
            "配置现已冻结；搜索 Agent 开始提出本轮建议，具体计算仍需按审批规则确认后才会派发。"
        )

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

        return (
            "当前是配置对话模式；尚未确认配置，也不会运行计算。"
            f"本地 DeepSeek API 尚未启用。请打开本机设置页 http://127.0.0.1:{os.environ.get('PHASE_CONTROL_PORT', '8765')}/phase/setup，"
            "粘贴 API Key 并测试连接；成功后无需重启服务。密钥只保存在本机系统凭据库。"
        )


def _latest_user_message(messages):

    for item in reversed(messages or []):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue

        content = item.get("content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            return "\n".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )

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
        "读取配置json并继续",
        "读取初始配置并继续",
        "读取运行配置并继续",
        "读取配置json检查通过后继续",
        "读取配置json审核通过后继续",
        "读取配置json如果没问题就继续",
        "读取配置json没问题就继续",
        "读取配置json如检查通过则继续",
        "readconfigjsonandcontinue",
        "readconfigjsoncontinueifvalid",
    }

    if normalized in continue_commands:
        return "continue"

    review_commands = {
        "读取配置json",
        "检查配置json",
        "审查配置json",
        "读取初始配置",
        "读取运行配置",
        "读取初始配置json",
        "读取运行配置json",
        "读取配置",
        "检查配置",
        "审查配置",
        "重新读取配置",
        "readconfigjson",
        "reviewconfigjson",
    }

    if normalized in review_commands:
        return "review"

    if normalized in {
        "配置改好了",
        "设置改好了",
        "我改好了",
        "已改好配置",
        "配置已保存",
        "设置已保存",
    }:
        return "review"

    if any(word in normalized for word in ("不要读取", "暂不读取", "别读取", "先不读取")):
        return None

    if normalized.startswith(
        ("请", "帮我", "重新", "读取", "检查", "审查", "配置已改好", "我改好了")
    ):
        mentions_file = any(
            word in normalized
            for word in (
                "配置json",
                "配置文件",
                "设置文件",
                "当前配置",
                "search_config.project.json",
                "search_config.draft.json",
            )
        )

        requests_read = any(word in normalized for word in ("读取", "检查", "审查"))

        if mentions_file and requests_read:
            return (
                "continue"
                if "通过后继续" in normalized or "没问题就继续" in normalized
                else "review"
            )

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

    if re.search(
        r"(?:^|\s)(?:agent|deepseek|模型)(?:\s*模型)?(?:\s*版本)?\s*[:：]", candidate, re.I
    ):
        return None

    return candidate


def _parse_workspace_setup_values(message):
    """Parse only a path and an optional labeled/model selection from setup input."""

    from phase_agent.runtime.resolve_deepseek_model_request import resolve_deepseek_model_request

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
        segments = [
            part.strip()
            for part in re.split(
                r"\s+(?=(?:agent|deepseek|模型)(?:\s*模型)?(?:\s*版本)?\s*[:：])", raw, flags=re.I
            )
            if part.strip()
        ]

    else:
        segments = [raw]

    root_value = None

    model_value = None

    path_label = re.compile(
        r"^(?:工作区根目录|工作区根路径|根路径|工作区路径|workspace(?:\s+root)?)\s*[:：]\s*(.+)$",
        re.I,
    )

    model_label = re.compile(
        r"^(?:agent(?:\s*模型)?(?:\s*版本)?|模型(?:版本)?|deepseek(?:\s*模型)?)\s*[:：]\s*(.+)$",
        re.I,
    )

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

    from phase_agent.configuration.session.validate_workspace_root import validate_workspace_root

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

    if (
        not candidate
        or len(candidate) > 128
        or any(ord(char) < 32 or ord(char) == 127 for char in candidate)
        or "/" in candidate
        or "\\" in candidate
        or candidate in {".", ".."}
        or not candidate.lower().endswith(".json")
    ):
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
        "并保存到本地 Agent 运行时配置；从下一条消息起生效。"
        "搜索配置、API Key 和计算任务未修改。"
    )


def _recent_dialogue_context(session, *, limit=8, max_chars=1600):
    """Keep only dialogue since the current configuration source was imported."""

    context = []

    entries = session.get("dialogue") or []

    start = 0

    for index, entry in enumerate(entries):
        if (
            entry.get("type") in {"config_import", "editable_config_source_changed"}
            or entry.get("type") == "config_revision"
            and entry.get("author") == "user_json_draft"
        ):
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

    keys = (
        "frozen_parameters",
        "branch_partition_suggestions",
        "generation_actions",
        "calculation",
        "budgets",
        "mlip",
        "dft",
        "convergence",
        "mc_policy",
        "round_strategy",
        "run",
    )

    excerpt = {key: deepcopy(config[key]) for key in keys if key in config}

    system = config.get("system") or {}

    excerpt["system"] = {
        key: deepcopy(system[key])
        for key in (
            "system_id",
            "configuration_space",
            "phase_reference_directory",
            "phase_references",
            "H_generation",
            "constraints",
            "branch_schema",
        )
        if key in system
    }

    boundary = system.get("boundary") or {}

    excerpt["system"]["boundary"] = {
        key: deepcopy(boundary[key]) for key in ("P", "TM_ratio") if key in boundary
    }

    finetune = config.get("mlip_finetune") or {}

    excerpt["mlip_finetune"] = {
        key: deepcopy(finetune[key])
        for key in ("enabled", "validation", "refresh_validation")
        if key in finetune
    }

    qbc = config.get("qbc") or {}

    excerpt["qbc"] = {
        key: deepcopy(qbc[key]) for key in ("mode", "extreme_uncertainty") if key in qbc
    }

    cluster = config.get("supercomputer") or {}

    excerpt["supercomputer"] = {"scheduler": deepcopy(cluster.get("scheduler") or {})}

    return excerpt


def _agent_source_contradiction(result, facts):
    """Catch only claims directly disproved by the current local file facts."""

    output = (
        str(result.get("reply") or "") + " " + " ".join(map(str, result.get("questions") or []))
    )

    source = str(facts.get("source_path") or "")

    if source.endswith("search_config.project.json") and any(
        "search_config.draft.json" in line
        and any(
            label in line for label in ("当前草稿", "当前配置", "可编辑配置", "正在读取", "来源=")
        )
        and not any(label in line for label in ("旧", "历史", "已停用"))
        for line in output.splitlines()
    ):
        return "引用了旧设置文件"

    if any(
        phrase in output
        for phrase in (
            "我无法读取该文件",
            "无法读取本地文件",
            "无法打开本地文件",
            "请把配置文件内容贴出",
            "请提供配置文件内容",
        )
    ):
        return "程序已读取当前本地设置文件"

    if (
        facts.get("generated_H")
        and all(
            (facts.get("H_by_phase") or {}).get(phase, {}).get("count", 0) > 0
            for phase in facts.get("allowed_phases_union") or []
        )
        and any(
            phrase in output
            for phrase in (
                "H 尚未生成",
                "H尚未生成",
                "请提供具体矩阵",
                "请把矩阵填入",
                "请提供各相合法矩阵",
                "boundary.H 四个相键缺失",
            )
        )
    ):
        return "各相 H 已由程序生成并核实"

    return None


def _fill_default_mlip_versions(config):
    """Migrate legacy null version fields when the configured model is mh-1."""

    mlip = config.get("mlip") or {}

    calculation = config.get("calculation") or {}

    bohb_scope = (config.get("bohb") or {}).get("scope") or {}

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

        if any(
            any(
                word in part.lower()
                for word in ("api_key", "token", "password", "secret", "private_key")
            )
            for part in parts
        ):
            raise ValueError(f"密钥不能写入配置草稿：{path}")

        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)

        except (TypeError, ValueError) as error:
            raise ValueError(f"{path} 不是合法 JSON 值") from error


def _explicit_config_write_request(message):

    normalized = re.sub(r"\s+", "", str(message or "").lower())

    if any(
        phrase in normalized
        for phrase in (
            "你写",
            "帮我写",
            "替我写",
            "写入配置",
            "写进配置",
            "直接写",
            "直接改",
            "帮我改",
            "帮我修订配置",
            "修改配置文件",
            "保存到配置",
            "修改设置文件",
        )
    ):
        return True

    if any(
        word in normalized
        for word in (
            "如果",
            "是否",
            "能否",
            "可否",
            "能不能",
            "可不可以",
            "建议",
            "怎么样",
            "合适吗",
            "吗?",
            "吗？",
        )
    ):
        return False

    assignment = any(
        word in normalized
        for word in (
            "改为",
            "改成",
            "设置为",
            "设为",
            "调整为",
            "修改为",
            "增加到",
            "提高到",
            "降低到",
        )
    )

    parameter = any(
        word in normalized
        for word in (
            "上限",
            "预算",
            "配额",
            "branch",
            "初态",
            "步数",
            "patience",
            "ehull",
            "阈值",
            "相",
            "超胞",
            "det(h)",
            "路径",
            "模型版本",
            "mlip",
            "dft",
            "mc",
        )
    )

    return assignment and parameter


def _project_config_migration_note(path):

    try:
        from phase_agent.configuration.session.load_editable_config_json import (
            _strip_jsonc_comments,
        )

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
        "写入",
        "保存",
        "写吧",
        "改吧",
        "确认写入",
        "执行写入",
        "写入配置",
        "确认修改",
        "同意修改",
        "写进配置",
        "保存到配置",
    }


def _is_config_revision_request(message):

    text = re.sub(r"\s+", "", str(message or "").lower())

    config_scope = any(
        word in text
        for word in (
            "配置",
            "设置文件",
            "参数文件",
            "search_config.project.json",
        )
    )

    edit_intent = any(
        word in text
        for word in (
            "修改",
            "调整",
            "改",
            "设为",
            "更新",
            "修订",
            "写入",
            "保存",
        )
    )

    return config_scope and edit_intent


def _clear_config_review(session):

    for key in (
        "agent_reviewed_revision",
        "agent_reviewed_config_hash",
        "agent_reviewed_config_digest",
        "agent_reviewed_mother_digest",
    ):
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


def _concise_import_reply(h_counts, readiness, questions, *, agent_passed=False, agent_note=""):

    lines = [f"配置已读取；合法 H：{h_counts}。"]

    issues = list(
        dict.fromkeys(
            str(x)
            for key in ("missing", "conflicts", "ambiguities")
            for x in readiness.get(key) or []
        )
    )

    if issues:
        lines.append("需处理：" + "；".join(issues))

        lines.append("修改配置文件后回复“读取配置 JSON”。")

    elif agent_passed:
        lines.append("检查通过。回复“同意”保存本次配置修订，或“拒绝”保留原配置。")

    else:
        lines.extend(str(q) for q in (questions or [])[:3])

        lines.append(agent_note or "程序检查通过，Agent 审核尚未通过；当前未确认修订。")

    return "\n".join(lines)
