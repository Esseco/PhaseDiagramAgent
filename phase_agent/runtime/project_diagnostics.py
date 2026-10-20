"""Read-only project diagnosis usable without a live service or valid scientific state."""

import argparse
import json
from pathlib import Path

from phase_agent.runtime.boundary_diagnostics import describe_failure


def inspect_project(state_path):
    state_path = Path(state_path).resolve()
    sections, issues = {}, []

    def inspect(name, callback):
        try:
            sections[name] = {"available": True, "data": callback()}
        except Exception as error:
            diagnostic = describe_failure(error, boundary="diagnostics", object_id=name)
            issues.append(diagnostic)
            sections[name] = {"available": False, "diagnostic": diagnostic}

    from phase_agent.persistence.result_recovery_commit import recovery_transaction_status
    from phase_agent.runtime.ui_operations import UIOperations
    import threading

    inspect("transaction", lambda: recovery_transaction_status(state_path))
    inspect("operations", lambda: UIOperations(state_path, threading.RLock()).recent())

    def state_facts():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not isinstance(state.get("tasks", []), list):
            raise ValueError("invalid_state_shape")
        from phase_agent.runtime.result_recovery_status import result_recovery_facts
        from phase_agent.tools.state.execution_receipts import recovery_report

        tasks = state.get("tasks") or []
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

        return {
            "model_version": state.get("active_model_version"),
            "config_version": state.get("confirmed_config_version"),
            "task_count": len(tasks),
            "result_validation_errors": state.get("result_validation_errors") or {},
            "scientific_feedback_errors": state.get("scientific_feedback_errors") or {},
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_project(args.state), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
