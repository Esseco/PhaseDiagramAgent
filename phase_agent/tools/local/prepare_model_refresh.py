"""Prepare approved refresh tasks through the existing manual-upload protocol."""

from copy import deepcopy
import hashlib
from pathlib import Path

from phase_agent.configuration.defaults.default_slurm_cluster_config import (
    default_slurm_cluster_config,
)
from phase_agent.tools.budget.check_budget import check_budget
from phase_agent.tools.budget.reserve_budget import reserve_budget
from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
from phase_agent.tools.workflows.model_refresh_state import refresh_preview
from phase_agent.science.mlip.slurm_executor import create_mlip_task_preparer


class RefreshUploadRunner(ManualUploadBatchRunner):
    def __init__(self, *args, refresh_id, wave, **kwargs):
        super().__init__(*args, **kwargs)
        self.refresh_id, self.wave = refresh_id, wave

    def _select(self, state, tasks):
        return super()._select(
            state,
            [
                t
                for t in tasks
                if t.get("model_refresh_id") == self.refresh_id
                and t.get("model_refresh_wave") == self.wave
            ],
        )


def prepare_model_refresh(*, action, context):
    state = deepcopy(context["event_state"])
    config = deepcopy(context["effective_config"])
    refresh = state["model_refresh"]
    wave = int(refresh.get("wave", 0))
    if wave == 0:
        plan = refresh_preview(state, context["manager"], config)
        if (action.get("parameters") or {}).get("preview_checksum") != plan["checksum"]:
            return {
                "status": "rejected",
                "state": state,
                "reason": "刷新结构或设置已改变，请重新确认方案",
            }
        rows = plan["candidates"]
        requested_cost = plan["maximum_total_cost"]
        requested_count = len(rows) + plan["maximum_supplemental_count"]
    else:
        plan = refresh["plan"]
        if not refresh.get("approved_by") or not refresh.get("supplemental_reservation_key"):
            return {
                "status": "rejected",
                "state": state,
                "reason": "缺少首批刷新批准或补充预算预留，不能自动追加",
            }
        rows = refresh.get("supplemental_candidates") or []
        requested_cost = sum(r["relative_cost"] for r in rows)
        requested_count = len(rows)
        if (
            wave != 1
            or len(rows) > plan["maximum_supplemental_count"]
            or requested_cost > plan["maximum_supplemental_cost"] + 1e-9
        ):
            return {
                "status": "rejected",
                "state": state,
                "reason": "补充刷新超出首批批准上限，需要单独审批",
            }
    root = config.get("upload_batches_directory")
    command = ((config.get("supercomputer") or {}).get("worker") or {}).get("command") or []
    if not root or not command or any("YOUR_" in str(value) for value in command):
        return {
            "status": "not_configured",
            "state": state,
            "reason": "刷新上传目录或远端worker未配置",
        }
    if (state.get("dedup_gate") or {}).get("status") != "ready":
        return {"status": "not_configured", "state": state, "reason": "structure_dedup_not_ready"}
    if wave:
        hold = (state.get("budget_reservations") or {}).get(refresh["supplemental_reservation_key"])
        if hold:
            hold["status"] = "released"
            hold["released_cost"] = hold["reserved_cost"]
    checked = check_budget(
        state,
        {"stage": "relax_and_feature", "tasks": requested_count, "relative_cost": requested_cost},
        config["budgets"],
    )
    if not checked["allowed"]:
        return {
            "status": "budget_exhausted",
            "state": context["event_state"],
            "reason": "刷新全方案预算不足，未截断；需修订分层方案或预算："
            + ",".join(checked["reasons"]),
        }
    refresh_id = plan["checksum"]
    config["mlip"] = deepcopy(plan["model"])
    config["mlip"]["relax_parameters"] = plan["relax_parameters"]
    tasks = []
    for row in rows:
        path = Path(row["structure_path"])
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != row["structure_sha256"]
        ):
            raise ValueError(f"刷新输入内容已改变：{row['structure_id']}")
        key = f"model-refresh:{refresh_id}:{wave}:{row['structure_sha256']}:{row['operation']}"
        prior = next((t for t in state.get("tasks") or [] if t.get("task_key") == key), None)
        if prior:
            tasks.append(prior)
            continue
        reserved = reserve_budget(
            state,
            task_key=key,
            stage="relax_and_feature",
            amount=row["relative_cost"],
            limits=config["budgets"],
            config_version=context["config_version"],
            model_version=plan["new_model_version"],
        )
        if reserved["status"] != "reserved":
            return {
                "status": "budget_exhausted",
                "state": context["event_state"],
                "reason": reserved.get("reasons"),
            }
        state = reserved["state"]
        task = {
            "task_id": "REFRESH-" + hashlib.sha256(key.encode()).hexdigest()[:12],
            "task_key": key,
            "structure_id": row["structure_id"],
            "object_id": row["structure_id"],
            "branch_id": row["branch_id"],
            "structure_path": row["structure_path"],
            "input_structure_sha256": row["structure_sha256"],
            "stage": "relax_and_feature",
            "status": "pending",
            "model_version": plan["new_model_version"],
            "config_version": context["config_version"],
            "planned_relative_cost": row["relative_cost"],
            "model_refresh_id": refresh_id,
            "model_refresh_wave": wave,
            "model_refresh_round": 1,
            "upload_operation_id": refresh_id,
            "parent_decision_id": context.get("approval_record_id"),
            "parameters": {
                **plan["relax_parameters"],
                "model_refresh_operation": row["operation"],
                "model_refresh_wave": wave,
            },
        }
        state.setdefault("tasks", []).append(task)
        state.setdefault("pending_tasks", []).append(task)
        tasks.append(task)
    if not wave and plan["maximum_supplemental_cost"]:
        hold_key = f"model-refresh:{refresh_id}:supplement-envelope"
        reserved = reserve_budget(
            state,
            task_key=hold_key,
            stage="relax_and_feature",
            amount=plan["maximum_supplemental_cost"],
            limits=config["budgets"],
            config_version=context["config_version"],
            model_version=plan["new_model_version"],
        )
        if reserved["status"] != "reserved":
            return {
                "status": "budget_exhausted",
                "state": context["event_state"],
                "reason": reserved.get("reasons"),
            }
        state = reserved["state"]
        state["model_refresh"]["supplemental_reservation_key"] = hold_key
    base_preparer = create_mlip_task_preparer(
        context["manager"], context.get("phase_references") or {}, config
    )

    def portable_preparer(task):
        prepared = base_preparer(task)
        prepared.pop("structure_path", None)  # worker_job carries the staged remote path.
        return prepared

    runner = RefreshUploadRunner(
        root,
        refresh_id=refresh_id,
        wave=wave,
        worker_command=command,
        stage_batch_sizes=((config.get("supercomputer") or {}).get("batch_sizes") or {}),
        stage_profiles=default_slurm_cluster_config(),
        task_preparer=portable_preparer,
    )
    batches = []
    while True:
        result = runner.prepare(state)
        state = result["state"]
        if result["status"] == "no_tasks":
            break
        batches.append(result["batch"])
    prepared_ids = {t["task_id"] for t in tasks}
    if any(t["task_id"] in prepared_ids and not t.get("input_path") for t in state["tasks"]):
        raise ValueError("刷新任务未全部生成输入；请检查去重有效ID与结构登记")
    state["model_refresh"].update(
        status="waiting_results",
        plan=plan,
        wave=wave,
        task_ids=[t["task_id"] for t in tasks],
        approved_by=context.get("approval_record_id"),
    )
    return {
        "status": "prepared",
        "state": state,
        "task_count": len(tasks),
        "batch_count": len(batches),
        "submitted": False,
        "reason": f"已准备刷新第{wave + 1}批 {len(tasks)} 个输入；上传并提交各批次GPU.sh，回传results后继续。",
    }
