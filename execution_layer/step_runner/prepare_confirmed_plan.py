"""Offline preparation of all pending tasks authorized by a confirmed plan."""

from __future__ import annotations

from copy import deepcopy

from execution_layer.step_runner.check_node_role import check_node_role
from execution_layer.step_runner.file_protocol import read_json, write_json


def prepare_confirmed_plan(
    state_path, plan_path, *, batch_runner, action_executor=None,
    expected_config_version=None, node_role=None,
):
    check_node_role("compute", role=node_role)
    state = read_json(state_path, {}) or {}
    plan = read_json(plan_path)
    if not plan or plan.get("status") != "confirmed":
        raise ValueError("prepare requires an explicitly confirmed action plan")
    if expected_config_version and plan.get("config_version") != expected_config_version:
        raise ValueError("plan config_version does not match runtime configuration")
    applied = state.setdefault("applied_plan_ids", [])
    if plan["plan_id"] not in applied:
        if action_executor is not None:
            produced = action_executor(deepcopy(state), deepcopy(plan))
            state = produced.get("state", produced)
        elif plan.get("actions") and not (state.get("tasks") or state.get("pending_tasks")):
            raise RuntimeError("plan contains actions but no offline action_executor is configured")
        state.setdefault("applied_plan_ids", []).append(plan["plan_id"])
    batches = []
    while True:
        prepared = batch_runner.prepare(state)
        state = prepared["state"]
        if prepared["status"] == "no_tasks":
            break
        batches.append(prepared["batch"])
    write_json(state_path, state)
    return {
        "status": "prepared" if batches else "already_prepared_or_no_tasks",
        "state": state, "batches": batches,
    }
