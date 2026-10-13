from copy import deepcopy

import pytest

from phase_agent.tools.state.execution_receipts import (
    begin_execution,
    record_execution_return,
    recovery_report,
)
from phase_agent.tools.state.execution_reconciliation import (
    propose_reconciliation,
    apply_reconciliation,
)
from phase_agent.tools.state.execution_recovery_evidence import inspect_execution


def prepared(tmp_path):
    state_path = tmp_path / "state.json"
    identity = {
        "invocation_id": "mc",
        "config_version": "c1",
        "action_hash": "h1",
        "tool": "allocate_mc_bohb",
    }
    begin_execution(
        state_path, identity, action={"tool": "allocate_mc_bohb", "task_key": "allocation"}
    )
    record_execution_return(state_path, identity, "completed", evidence={"task_ids": ["t1"]})
    initial = tmp_path / "initial.vasp"
    initial.write_text("existing geometry")
    task_input = tmp_path / "task.json"
    task_input.write_text("existing input")
    state = {
        "tasks": [
            {
                "task_id": "t1",
                "task_key": "child",
                "status": "pending",
                "input_path": str(task_input),
                "structure_path": str(initial),
                "result_path": str(tmp_path / "results/result.json"),
            }
        ],
        "budget_reservations": {"child": {"status": "reserved", "reserved_cost": 20}},
        "reserved_relative_cost": 20,
        "pending_execution_policies": {
            "mc": {"agent_proposal": {"raw_action": {"task_key": "allocation"}}}
        },
    }
    return state_path, identity, state


def test_pending_result_does_not_block_reuse_or_mark_task_completed(tmp_path):
    path, identity, state = prepared(tmp_path)
    before = deepcopy(state)
    receipt = (tmp_path / "execution_receipts.sqlite").read_bytes()
    checks = inspect_execution(state, recovery_report(path, state)["unsettled"][0], state_path=path)
    assert {r["status"] for r in checks["artifact_checks"]} == {"file", "awaiting_result"}
    assert "不要重复" in checks["next_action"]
    plan = propose_reconciliation(
        path,
        state,
        "mc",
        "verified_registered_effects",
        {
            "evidence_note": "已核对登记任务及输入；本次动作仅准备输入，不提交超算",
            "external_jobs": "not_applicable",
            "output_inventory": "registered_outputs",
        },
    )
    result = apply_reconciliation(path, plan["state"], plan["plan_id"], "confirm_sensitive")
    assert result["state"]["tasks"] == before["tasks"]
    assert result["state"]["budget_reservations"] == before["budget_reservations"]
    assert result["state"]["reserved_relative_cost"] == 20
    assert recovery_report(path, result["state"])["status"] == "clear"
    assert (tmp_path / "execution_receipts.sqlite").read_bytes() == receipt
    assert not result["state"]["pending_execution_policies"]
    assert result["state"]["reconciled_proposals"]
    assert not begin_execution(path, identity)["allowed"]


@pytest.mark.parametrize("missing", ["input_path", "structure_path", "checkpoint"])
def test_pending_tasks_still_require_inputs_and_checkpoints(tmp_path, missing):
    path, identity, state = prepared(tmp_path)
    task = state["tasks"][0]
    if missing == "checkpoint":
        task[missing] = str(tmp_path / "missing.checkpoint.json")
    else:
        from pathlib import Path

        Path(task[missing]).unlink()
    with pytest.raises(ValueError, match="缺失文件"):
        propose_reconciliation(
            path,
            state,
            "mc",
            "verified_registered_effects",
            {
                "evidence_note": "checked",
                "external_jobs": "not_applicable",
                "output_inventory": "registered_outputs",
            },
        )


def test_completed_result_missing_cannot_be_reconciled_as_pending(tmp_path):
    path, identity, state = prepared(tmp_path)
    state["tasks"][0]["status"] = "completed"
    with pytest.raises(ValueError, match="缺失文件"):
        propose_reconciliation(
            path,
            state,
            "mc",
            "verified_registered_effects",
            {
                "evidence_note": "checked",
                "external_jobs": "not_applicable",
                "output_inventory": "registered_outputs",
            },
        )


def test_langgraph_early_barrier_defers_full_artifact_scan_until_wait(tmp_path, monkeypatch):
    from phase_agent.graphs.batch_recovery_graph import build_batch_recovery_graph
    from phase_agent.graphs.execution_recovery_graph import (
        build_execution_recovery_graph,
        execution_recovery_report,
    )
    from phase_agent.graphs.recovery_observation_graph import build_recovery_observation_graph
    from phase_agent.tools.state import execution_recovery_evidence as module

    path, identity, state = prepared(tmp_path)
    calls = []
    inspect = module.inspect_execution

    def observed(*args, **kwargs):
        calls.append(True)
        return inspect(*args, **kwargs)

    monkeypatch.setattr(module, "inspect_execution", observed)
    graph = build_recovery_observation_graph(
        build_batch_recovery_graph(), build_execution_recovery_graph()
    )
    early = graph.invoke({"business": state, "state_path": str(path)})["execution_report"]
    assert early["unsettled"] and not early["checks"] and not calls
    final = execution_recovery_report(path, state)
    assert final["unsettled"] and final["checks"] and len(calls) == 1
    assert not final["automatic_replay_allowed"]


@pytest.mark.parametrize("changed", [False, True])
def test_durable_observation_writes_only_changed_business_facts(changed):
    from types import SimpleNamespace
    from phase_agent.graphs.project.nodes import create_lifecycle_nodes
    from phase_agent.graphs.runtime_context import WorkflowRuntime
    from phase_agent.graphs.business_checkpoint import checkpoint_frame

    frame = {
        "_workflow_prepared": True,
        "loaded_state": {"progress": "old", "history": "large unchanged history" * 1000},
    }
    previous = checkpoint_frame(frame)

    def analyze(current):
        if changed:
            current["loaded_state"]["progress"] = "new"
        return current

    context = WorkflowRuntime(frame=deepcopy(frame), stages={"durable": True})
    node = create_lifecycle_nodes(analyze=analyze)["scientific_feedback"]
    update = node({"business_frame": previous}, SimpleNamespace(context=context))
    assert ("business_frame" in update) == changed
    saved = update.get("business_frame", previous)
    assert saved["loaded_state"]["progress"] == ("new" if changed else "old")
    assert update["current_node"] == "scientific_feedback"


def test_default_diagram_uses_runtime_owner_for_editable_config_comparison(tmp_path):
    from types import SimpleNamespace
    from phase_agent.configuration.defaults.default_layered_search_config import (
        default_layered_search_config,
    )
    from phase_agent.configuration.session.project_config_json import create_project_config_json
    from phase_agent.configuration.session.resolve_workspace_paths import default_workspace_storage
    from phase_agent.runtime.project_context import handler_project_context
    import json

    owner = tmp_path / "project"
    diagram = owner / "diagrams" / "diagram_default"
    source = diagram / "parameters" / "search_config.project.json"
    config = default_layered_search_config()
    config["storage"] = default_workspace_storage(owner)
    config["storage"]["workspace_root"] = str(owner)
    create_project_config_json(source, config)
    from phase_agent.configuration.session.load_editable_config_json import (
        load_editable_config_json,
    )

    config = load_editable_config_json(source, config)
    settings = {"runtime_storage_override": {**config["storage"], "workspace_root": str(diagram)}}
    owner.mkdir(parents=True, exist_ok=True)
    (owner / "agent_runtime.json").write_text(json.dumps(settings))
    (diagram / "agent_runtime.json").write_text(json.dumps(settings))
    state = {"confirmed_config": config, "confirmed_config_version": "c1"}
    session = {
        "status": "confirmed",
        "config": config,
        "confirmed_snapshot": {"config_version": "c1", "config": config},
    }
    handler = SimpleNamespace(
        state_path=diagram / "workflow_state" / "state.json",
        base_directory=owner,
        editable_config_path=source,
        workflow_kwargs={"config_session": session},
    )
    card = handler_project_context(handler, state)
    assert not card["editable_file"].get("error")
    assert card["editable_file"]["changed_paths"] == []
    assert card["draft"]["status"] == "no_changes"
    from phase_agent.configuration.session.project_config_json import write_project_config_patch

    write_project_config_patch(
        source, {"run.seed": config["run"]["seed"] + 1}, baseline_config=config
    )
    changed = handler_project_context(handler, state)
    assert "run.seed" in changed["editable_file"]["changed_paths"]
    assert changed["draft"]["status"] == "file_changes_not_effective"
