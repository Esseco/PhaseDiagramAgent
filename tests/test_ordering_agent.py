from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from pymatgen.core import Structure, Lattice

from phase_agent.tools.workflows.run_tool_step import run_tool_step
from phase_agent.tools.workflows.create_workflow_handlers import create_workflow_handlers
from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
from tests.test_event_loop import _session


def fixture(tmp_path):
    reference = Structure(
        Lattice.hexagonal(3, 10),
        ["Na", "Na", "Ni", "Ni", "O", "O", "O", "O"],
        [
            [0, 0, 0.15],
            [0.5, 0.5, 0.65],
            [0, 0, 0.4],
            [0.5, 0.5, 0.9],
            [0, 0.5, 0.3],
            [0.5, 0, 0.5],
            [0, 0.5, 0.8],
            [0.5, 0, 0],
        ],
    )
    session = _session()
    session["confirmed_snapshot"]["config"]["mlip"]["sha256"] = "0" * 64
    config = deepcopy(session["confirmed_snapshot"]["config"])
    config["upload_batches_directory"] = str(tmp_path / "jobs")
    context = {
        "phase_references": {"P2": reference},
        "effective_config": config,
        "config_version": session["confirmed_snapshot"]["config_version"],
    }
    action = {
        "operation": "prepare_ordering_search",
        "tool": "prepare_local_batch_files",
        "task_key": "ordering-test",
        "parameters": {
            "mode": "ordering_search_inputs",
            "phase": "P2",
            "tm_ratios": {"Ni": 1, "Mn": 1},
            "na_content": "1/2",
            "h_matrices": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "oxidation_states": {"Na": 1, "Ni": 3, "Mn": 4, "O": -2},
            "options": {"min_distance": 0, "initial_pool_size": 8, "max_steps": 3, "workers": 2},
        },
        "reason": "Independent TM-focused ordering search",
    }
    registry = create_tool_registry(create_workflow_handlers())
    return session, context, action, registry


def test_agent_approval_prepares_frozen_library_plan_without_computing(tmp_path):
    session, context, action, registry = fixture(tmp_path)
    with patch(
        "Process_AL_MC.MC_sample.LayeredOxide_MCOrderingClass",
        side_effect=AssertionError("must not load models"),
    ):
        first = run_tool_step(
            {},
            session,
            registry=registry,
            agent_client=lambda p: action,
            context=context,
            execution_mode="interactive",
            invocation_id="ordering",
        )
        assert first["status"] == "awaiting_approval"
        assert not (tmp_path / "jobs").exists()
        raw = first["agent_proposal"]["raw_action"]
        assert raw["ordering_search_plan"]["workload"]["matrix_count"] == 1

        def forbidden(payload):
            raise AssertionError("approval must reuse the stored plan")

        approved = run_tool_step(
            first["state"],
            session,
            registry=registry,
            agent_client=forbidden,
            context=context,
            execution_mode="interactive",
            invocation_id="ordering",
            human_feedback={"decision": "approve", "comment": "同意"},
        )
    assert approved["status"] == "prepared"
    record = next(iter(approved["state"]["ordering_searches"].values()))
    assert Path(record["directory"], "plan.json").is_file()
    assert Path(record["directory"], "runtime", "Process_AL_MC", "ordering", "api.py").is_file()
    assert not approved.get("submitted")
    from Process_AL_MC import run_ordering_search
    from phase_agent.tools.local.prepare_ordering_search import recover_ordering_results

    def evaluate(structure, ctx):
        return {
            "structure": structure.as_dict(),
            "energy_per_atom": -1,
            "phase": "P2",
            "converged": True,
        }

    run_ordering_search(record["directory"], evaluator=evaluate)
    plan_id = record["plan_id"]
    recovered = recover_ordering_results(
        action={"parameters": {"plan_id": plan_id}}, context={"event_state": approved["state"]}
    )
    assert recovered["status"] == "completed"
    assert recovered["state"]["ordering_searches"][plan_id]["result"]["representative_count"] == 2
    assert recovered["state"].get("branches") == approved["state"].get("branches")


def test_reference_change_invalidates_frozen_ordering_approval(tmp_path):
    session, context, action, registry = fixture(tmp_path)
    first = run_tool_step(
        {},
        session,
        registry=registry,
        agent_client=lambda p: action,
        context=context,
        execution_mode="interactive",
        invocation_id="ordering",
    )
    context["phase_references"]["P2"].translate_sites([0], [0.1, 0, 0])
    rejected = run_tool_step(
        first["state"],
        session,
        registry=registry,
        agent_client=lambda p: action,
        context=context,
        execution_mode="interactive",
        invocation_id="ordering",
        human_feedback={"decision": "approve", "comment": "同意"},
    )
    assert rejected["status"] in {"rejected", "proposal_context_changed", "awaiting_approval"}
    assert not (tmp_path / "jobs").exists()
