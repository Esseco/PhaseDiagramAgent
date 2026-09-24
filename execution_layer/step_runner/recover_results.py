"""Offline recovery: reconcile complete task results once and publish a summary."""

from __future__ import annotations

from copy import deepcopy

from execution_layer.state.reconcile_task_results import reconcile_task_results
from execution_layer.step_runner.build_status_summary import build_status_summary
from execution_layer.step_runner.check_node_role import check_node_role
from execution_layer.step_runner.file_protocol import read_json, write_json


def recover_results(state_path, summary_path, *, result_collector, config_version=None, node_role=None):
    check_node_role("compute", role=node_role)
    state = read_json(state_path, {}) or {}
    recovered = result_collector.collect_results(state)
    reconciled = reconcile_task_results(state, recovered)
    updated = reconciled["state"]
    summary = build_status_summary(updated, config_version=config_version)
    write_json(state_path, updated)
    write_json(summary_path, summary)
    return {
        "status": "recovered", "state": updated, "summary": summary,
        "results_found": len(recovered), "reconciled": deepcopy(reconciled["reconciled"]),
    }
