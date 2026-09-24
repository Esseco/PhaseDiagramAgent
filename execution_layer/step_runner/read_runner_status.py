"""Read-only task, job, budget, and version status."""

from execution_layer.step_runner.build_status_summary import build_status_summary
from execution_layer.step_runner.check_node_role import check_node_role
from execution_layer.step_runner.file_protocol import read_json


def read_runner_status(state_path, *, node_role=None):
    check_node_role("login_or_compute", role=node_role)
    state = read_json(state_path, {}) or {}
    return build_status_summary(state)
