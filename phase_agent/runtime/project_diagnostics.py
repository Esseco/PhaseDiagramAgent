"""Read-only project diagnosis usable without a live service or valid scientific state."""

import argparse
import json
from pathlib import Path

from phase_agent.runtime.boundary_diagnostics import describe_failure


def _inspect(sections, issues, name, callback, path=None):
    try:
        data = callback()
        sections[name] = {"available": True, "data": data}
        return data
    except Exception as error:
        diagnostic = describe_failure(error, boundary="diagnostics", object_id=name)
        cause = error.__cause__ or error
        if isinstance(cause, json.JSONDecodeError):
            diagnostic.update(code="invalid_json", line=cause.lineno, column=cause.colno)
        elif isinstance(cause, FileNotFoundError):
            diagnostic["code"] = "missing_file"
        if path is not None:
            diagnostic["file"] = str(path)
        diagnostic["next_step"] = (
            "修正或恢复此文件后重新运行相同诊断命令；原文件和其他项目不变，"
            "执行状态不明时先核对原作业及回执。"
        )
        issues.append(diagnostic)
        sections[name] = {"available": False, "diagnostic": diagnostic}
        return None


def inspect_project(state_path):
    state_path = Path(state_path).resolve()
    sections, issues = {}, []

    def inspect(name, callback):
        return _inspect(sections, issues, name, callback, state_path if name == "state" else None)

    from phase_agent.persistence.result_recovery_commit import recovery_transaction_status
    from phase_agent.runtime.ui_operations import UIOperations
    import threading

    inspect("transaction", lambda: recovery_transaction_status(state_path))
    inspect("operations", lambda: UIOperations(state_path, threading.RLock()).recent())

    def state_facts():
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        if not isinstance(state, dict) or not isinstance(state.get("tasks", []), list):
            raise ValueError("invalid_state_shape")
        from phase_agent.runtime.result_recovery_status import result_recovery_facts
        from phase_agent.tools.state.execution_receipts import recovery_report

        tasks = state.get("tasks") or []
        if any(not isinstance(task, dict) for task in tasks):
            raise ValueError("invalid_state_task_shape")
        quality = []
        for task in tasks:
            outputs = task.get("outputs")
            phase = outputs.get("phase_identification") if isinstance(outputs, dict) else None
            phase_status = phase.get("status") if isinstance(phase, dict) else None
            if task.get("result_disposition") or phase_status in {"unknown", "outside_boundary"}:
                quality.append(
                    {
                        "task_id": task.get("task_id"),
                        "disposition": task.get("result_disposition"),
                        "phase_status": phase_status,
                    }
                )
        from phase_agent.persistence.ledger.result_validation import (
            validate_scientific_result,
            InvalidScientificResult,
        )

        recorded_errors = state.get("scientific_feedback_errors") or {}
        feedback_done = set(state.get("feedback_processed_task_ids") or [])
        rechecked = {"valid_now": 0, "invalid_now": 0, "already_recorded": 0}
        for task in tasks:
            if task.get("task_id") not in recorded_errors:
                continue
            try:
                validate_scientific_result(task)
            except InvalidScientificResult:
                rechecked["invalid_now"] += 1
            else:
                rechecked["valid_now"] += 1
                rechecked["already_recorded"] += task.get("task_id") in feedback_done
        rechecked["next_step"] = (
            "valid_now 仅表示历史错误未被当前格式校验复现；用原项目继续核对入账和相图。"
            "已有反馈记录的任务不要重算，也不要手工删除错误或执行凭据。"
        )

        def status_presentation():
            from phase_agent.runtime.status_presentation import format_progress

            reply = format_progress(state)
            if not isinstance(reply, str) or not reply:
                raise ValueError("empty_status_presentation")
            return {"valid": True}

        inspect("status_presentation", status_presentation)

        return {
            "model_version": state.get("active_model_version"),
            "config_version": state.get("confirmed_config_version"),
            "task_count": len(tasks),
            "result_collection": state.get("result_collection") or {},
            "result_validation_errors": state.get("result_validation_errors") or {},
            "scientific_feedback_errors": state.get("scientific_feedback_errors") or {},
            "scientific_feedback_recheck": rechecked,
            "quality_review": quality,
            "result_recoveries": result_recovery_facts(state),
            "execution_recovery": recovery_report(state_path, state),
        }

    inspect("state", state_facts)

    def incidents():
        rows = []
        for path in sorted(
            (state_path.parent / "diagnostics/incidents").glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:20]:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("invalid_incident")
                rows.append(value)
            except (ValueError, OSError) as error:
                rows.append(describe_failure(error, boundary="diagnostics", object_id=path.name))
        return rows

    inspect("incidents", incidents)
    return {
        "schema_version": 1,
        "read_only": True,
        "submitted": False,
        "automatic_actions": False,
        "sections": sections,
        "issues": issues,
        "recovery_entries": [
            {
                "kind": "inspect_receipts",
                "method": "GET",
                "url": "/phase/pending",
                "effect": "只读核对；执行不明时先查看已有收据",
            },
            {
                "kind": "resume_admission",
                "method": "POST",
                "url": "/phase/recovery/transaction/resume",
                "effect": "仅续写已批准入账，须匹配事务版本",
            },
            {
                "kind": "reconcile_execution",
                "method": "POST",
                "url": "/phase/recovery/propose",
                "effect": "依据外部作业、输出与费用核对生成处理方案，仍须人工审阅",
            },
            {
                "kind": "collect_existing",
                "command": "python -m phase_agent.tools.relax_mc_batch.cli recover --help",
                "effect": "仅回收已有结果；校验通过后使用现有导入流程",
            },
        ],
    }


def inspect_runtime(runtime_path):
    """Diagnose independent project files, including a not-yet-created scientific state."""
    from phase_agent.runtime.runtime_config_io import (
        _load_json_object,
        _resolve_path,
        _reject_secrets,
        runtime_base_directory,
    )

    path = Path(runtime_path).expanduser().resolve()
    sections, issues = {}, []
    report = {
        "schema_version": 1,
        "read_only": True,
        "submitted": False,
        "automatic_actions": False,
        "runtime_path": str(path),
        "sections": sections,
        "issues": issues,
        "status": "needs_attention",
    }

    def runtime():
        from phase_agent.runtime.local_project_launcher import _load_config

        settings = _load_config(path)
        _reject_secrets(settings)
        for key in ("state_path", "ledger_path", "config_session_path"):
            if key in settings and (
                not isinstance(settings[key], str) or not settings[key].strip()
            ):
                raise ValueError(f"invalid_path:{key}")
        return settings

    settings = _inspect(sections, issues, "runtime", runtime, path)
    if settings is None:
        return report
    # Runtime settings may contain credentials in custom annotations; expose no payload.
    sections["runtime"]["data"] = {"file": str(path)}
    from phase_agent.runtime.local_project_launcher import validate_project_paths

    _inspect(sections, issues, "paths", lambda: validate_project_paths(path), path)
    base = _inspect(
        sections, issues, "workspace", lambda: runtime_base_directory(path, settings), path
    )
    if base is None:
        return report
    sections["workspace"]["data"] = {"directory": str(base)}
    state_path = _resolve_path(settings["state_path"], base)
    ledger_path = _resolve_path(settings["ledger_path"], base)
    session_path = _resolve_path(
        settings.get("config_session_path", "local/config_session.json"), base
    )
    session = _inspect(
        sections,
        issues,
        "configuration",
        lambda: _load_json_object(session_path, "配置会话"),
        session_path,
    )
    confirmed = isinstance(session, dict) and session.get("status") == "confirmed"
    config = None
    if session is not None:

        def configuration():
            from phase_agent.configuration.schema.validate_search_config import (
                validate_search_config,
            )

            nonlocal config
            config = (
                (session.get("confirmed_snapshot") or {}).get("config")
                if confirmed
                else session.get("config")
            )
            validation = validate_search_config(config, stage="startup")
            return {"status": session.get("status"), "file": str(session_path), **validation}

        _inspect(sections, issues, "configuration", configuration, session_path)
    if isinstance(config, dict):

        def environments():
            from phase_agent.configuration.schema.python_environments import (
                merge_python_environments,
            )

            saved = config.get("python_environments")
            values = merge_python_environments(saved, settings.get("python_environments"))
            keys = ("local_python", "local_mlip", "remote_python", "remote_mlip")
            return {
                "values": values,
                "sources": {
                    key: "confirmed_config"
                    if confirmed and (saved or {}).get(key) == values[key]
                    else "draft_config"
                    if (saved or {}).get(key) == values[key]
                    else "agent_runtime.json"
                    for key in keys
                    if key in values
                },
                "missing": [
                    key
                    for key in keys
                    if not isinstance(values.get(key), str) or not values[key].strip()
                ],
                "remote_checked": False,
                "next_step": "远端环境按项目实际配置填写；旧快照缺项可在 agent_runtime.json 的 python_environments 中补充，已确认值优先。",
            }

        _inspect(sections, issues, "python_environments", environments, path)
    if confirmed and isinstance(config, dict):

        def storage():
            from copy import deepcopy
            from phase_agent.configuration.session.relocate_workspace_paths import (
                relocate_workspace_paths,
            )
            from phase_agent.configuration.session.resolve_workspace_paths import (
                resolve_workspace_paths,
            )

            effective = deepcopy(config)
            if settings.get("local_path_relocations"):
                effective = relocate_workspace_paths(
                    effective, settings["local_path_relocations"], base
                )
            if settings.get("runtime_storage_override") is not None:
                effective["storage"] = deepcopy(settings["runtime_storage_override"])
            if "storage" not in effective:
                return {"state": state_path, "ledger": ledger_path}
            selected = resolve_workspace_paths(effective, base_directory=base)
            for name, old in (("state", state_path), ("ledger", ledger_path)):
                if old.resolve() != selected[name].resolve() and old.exists():
                    raise ValueError(f"existing_{name}_path_conflicts_with_confirmed_storage")
            return {name: selected[name] for name in ("state", "ledger")}

        selected = _inspect(sections, issues, "storage", storage, session_path)
        if selected is not None:
            state_path, ledger_path = selected["state"], selected["ledger"]
            sections["storage"]["data"] = {key: str(value) for key, value in selected.items()}
    scientific = inspect_project(state_path)
    if not state_path.exists() and not confirmed:
        # A draft has no scientific state yet; absence must not look like data corruption.
        scientific["sections"]["state"] = {"available": True, "data": {"status": "not_created"}}
        scientific["issues"] = [row for row in scientific["issues"] if row["object_id"] != "state"]
    sections.update(scientific["sections"])
    issues.extend(scientific["issues"])
    report["recovery_entries"] = scientific["recovery_entries"]
    if confirmed and sections["state"]["available"]:
        actual_version = sections["state"]["data"].get("config_version")
        expected_version = (session.get("confirmed_snapshot") or {}).get("config_version")
        sections["binding"] = {
            "available": True,
            "data": {
                "valid": bool(expected_version) and actual_version == expected_version,
                "state_config_version": actual_version,
                "session_config_version": expected_version,
                "next_step": "版本不一致时使用原配置修订/确认流程，不手工修改历史任务版本。",
            },
        }

    def ledger():
        if not ledger_path.exists() and not confirmed and not state_path.exists():
            return {"status": "not_created"}
        value = _load_json_object(ledger_path, "台账")
        for key in ("branches", "structures"):
            if not isinstance(value.get(key, {}), dict):
                raise ValueError(f"ledger.{key}_must_be_object")
        return {
            "file": str(ledger_path),
            "branch_count": len(value.get("branches", {})),
            "structure_count": len(value.get("structures", {})),
        }

    _inspect(sections, issues, "ledger", ledger, ledger_path)
    if isinstance(config, dict):

        def references():
            from phase_agent.configuration.session.relocate_workspace_paths import (
                relocate_workspace_paths,
            )
            from phase_agent.configuration.session.resolve_phase_reference_directory import (
                resolve_phase_reference_directory,
            )
            from phase_agent.science.structures.boundary_utils import allowed_phases

            effective = config
            if settings.get("local_path_relocations"):
                effective = relocate_workspace_paths(
                    effective, settings["local_path_relocations"], base
                )
            effective = resolve_phase_reference_directory(effective, base_directory=base)
            system = effective.get("system") or {}
            configured = system.get("phase_references") or {}
            phases = (
                (system.get("boundary") or {}).get("P")
                or (system.get("constraints") or {}).get("phases")
                or []
            )
            missing, resolved = [], {}
            for phase in sorted(set(allowed_phases(phases)) | set(configured)):
                value = configured.get(phase)
                value = value.get("path") if isinstance(value, dict) else value
                target = _resolve_path(value, base) if isinstance(value, str) and value else None
                resolved[phase] = str(target) if target else None
                if target is None or not target.is_file():
                    missing.append(phase)
            return {
                "resolved_paths": resolved,
                "missing_phases": missing,
                "next_step": "在配置中补齐对应母结构路径，再重新读取配置。",
            }

        _inspect(sections, issues, "references", references, session_path)
    validation = (sections.get("configuration") or {}).get("data") or {}
    reference_facts = (sections.get("references") or {}).get("data") or {}
    binding = (sections.get("binding") or {}).get("data") or {}
    state_facts = (sections.get("state") or {}).get("data") or {}
    recovery = state_facts.get("execution_recovery") or {}
    needs_review = (
        binding.get("valid") is False
        or bool(recovery.get("unsettled"))
        or bool(state_facts.get("result_validation_errors"))
        or bool(state_facts.get("scientific_feedback_errors"))
    )
    report["status"] = (
        "needs_attention"
        if issues
        else "configuration_required"
        if not confirmed
        else "needs_attention"
        if needs_review or not validation.get("valid") or reference_facts.get("missing_phases")
        else "ready"
    )
    report["next_step"] = {
        "configuration_required": "补齐初始/运行配置，在原启动器打开项目并确认配置后继续。",
        "needs_attention": "按 sections/ issues 中的文件、字段和恢复入口处理，再运行同一诊断命令。",
        "ready": "本地文件检查通过；用原启动器打开项目并继续。远端作业和科学收敛未由此验证。",
    }[report["status"]]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--state")
    source.add_argument("--runtime", help="agent_runtime.json；独立检查配置、状态、台账与母结构")
    args = parser.parse_args()
    report = inspect_runtime(args.runtime) if args.runtime else inspect_project(args.state)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
