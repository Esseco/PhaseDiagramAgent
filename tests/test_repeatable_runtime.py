"""Small fault-injection checks: no real services, models or scientific projects."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)
from phase_agent.configuration.schema.validate_search_config import validate_search_config


@pytest.mark.parametrize(
    "field,value",
    [
        ("system", []),
        ("budgets", "bad"),
        ("system.boundary", ["O3"]),
        ("system.configuration_space.roles.T", []),
        ("calculation.mc_allocator", []),
        ("generation_actions.enabled", [{}]),
        ("frozen_parameters", [{}]),
        ("mc_policy.tiers", [None]),
        ("agent.adaptive", []),
        ("convergence.stable_model_update_epochs", "bad"),
        ("convergence.stable_model_update_epochs", 1.5),
        ("convergence.stable_model_update_epochs", float("inf")),
        ("convergence.hull_change_tolerance", "bad"),
        ("mc_policy.random_exploration_fraction", True),
        ("mlip_finetune.validation.max_energy_mae", float("nan")),
        ("mlip_finetune.validation.max_near_hull_ranking_reversals", 0.5),
    ],
)
def test_configuration_fault_returns_field_without_mutating_input(field, value):
    config = default_layered_search_config()
    config.setdefault("mlip_finetune", {})["enabled"] = True
    target = config
    parts = field.split(".")
    for key in parts[:-1]:
        target = target.setdefault(key, {})
    target[parts[-1]] = value
    before = json.dumps(config, sort_keys=True)
    result = validate_search_config(config, stage="startup")
    assert not result["valid"]
    assert any(field in error for error in result["conflicts"]), result
    assert json.dumps(config, sort_keys=True) == before


@pytest.mark.parametrize("config", [None, [], "bad", 1])
def test_non_object_configuration_is_a_validation_result(config):
    assert not validate_search_config(config)["valid"]


@pytest.mark.parametrize("field,value", [("P", 12), ("H", [1, 2])])
def test_fixed_boundary_bad_container_is_actionable(field, value):
    from phase_agent.configuration.schema.validate_configuration_space import (
        validate_configuration_space,
    )

    system = {
        "boundary": {"P": ["O3"], "H": {"O3": [[[1, 0], [0, 1]]]}},
        "configuration_space": {
            "roles": {"H": "fixed", "P": "fixed", "x": "branch", "T": "branch", "N": "internal"},
            "fixed_values": {"P": "O3", "H": [[1, 0], [0, 1]]},
        },
    }
    system["boundary"][field] = value
    result = validate_configuration_space(system)
    assert not result["valid"] and any(field in error for error in result["errors"])


def test_runtime_config_bom_and_json_location(tmp_path):
    from phase_agent.runtime.local_project_launcher import _load_config
    from phase_agent.runtime.runtime_config_io import _load_json_object

    path = tmp_path / "agent_runtime.json"
    data = {"state_path": "workflow_state/state.json", "ledger_path": "ledger.json"}
    path.write_text(json.dumps(data), encoding="utf-8-sig")
    assert _load_config(path) == data
    assert _load_json_object(path, "runtime") == data
    path.write_text('{\n"secret": "DO_NOT_PRINT",\n"state_path": }', encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        _load_json_object(path, "runtime")
    assert str(path) in str(caught.value)
    assert "3" in str(caught.value)
    assert "DO_NOT_PRINT" not in str(caught.value)


def returned_tasks(root):
    from phase_agent.tools.remote.integrity import file_checksum

    tasks = []
    for name in ("bad", "good"):
        folder = root / name
        folder.mkdir()
        task = {
            "task_id": name,
            "task_key": name,
            "stage": "deep_search",
            "status": "pending",
            "result_path": str(folder / "result.json"),
        }
        result = {**task, "status": "failed", "actual_cost": 1, "outputs": {}}
        (folder / "task.json").write_text(json.dumps(task))
        (folder / "result.json").write_text(json.dumps(result))
        marker = {
            "task_id": name,
            "task_key": name,
            "status": "failed",
            "result_checksum": file_checksum(folder / "result.json"),
        }
        (folder / "task.finished.json").write_text(json.dumps(marker))
        tasks.append(task)
    return {"tasks": tasks}


@pytest.mark.parametrize("fault", ["checksum_io", "snapshot_shape", "manifest_shape"])
def test_collection_fault_isolated_and_repaired_file_recoverable(tmp_path, monkeypatch, fault):
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner

    state = returned_tasks(tmp_path)
    before = deepcopy(state)
    real_read = Path.read_bytes
    snapshot = tmp_path / "bad/task.json"
    saved = snapshot.read_bytes()
    manifest = tmp_path / "manifest.json"
    if fault == "snapshot_shape":
        snapshot.write_text("[]")
    elif fault == "manifest_shape":
        manifest.write_text('[null, "bad"]')
        state["slurm_batches"] = [{"manifest_path": str(manifest)}]
        before = deepcopy(state)
    else:

        def read(path):
            if Path(path) == tmp_path / "bad/result.json":
                raise PermissionError("temporarily locked")
            return real_read(path)

        monkeypatch.setattr(Path, "read_bytes", read)
    runner = RemoteBatchRunner(tmp_path / "batches", worker_command=["never-run"])
    result = runner.collect_results_with_report(state)
    assert "good" in [row["task_id"] for row in result["results"]]
    if fault != "manifest_shape":
        assert [row["task_id"] for row in result["results"]] == ["good"]
        assert result["report"]["invalid_count"] == 1
        assert result["report"]["status"] == "partial"
    assert state == before
    snapshot.write_bytes(saved)
    monkeypatch.setattr(Path, "read_bytes", real_read)
    recovered = runner.collect_results_with_report(state)
    assert {row["task_id"] for row in recovered["results"]} == {"bad", "good"}
    assert state == before


def test_optional_node_trace_cannot_fail_business_or_mask_original_error(tmp_path, monkeypatch):
    from phase_agent.graphs.node_trace import append_node_trace

    def fail(*args, **kwargs):
        raise PermissionError("log is locked")

    monkeypatch.setattr(Path, "open", fail)
    frame = {"state_path": str(tmp_path / "state.json")}
    append_node_trace(frame, "collect", 0.1)
    original = ValueError("original business error")
    append_node_trace(frame, "collect", 0.1, error=original)
