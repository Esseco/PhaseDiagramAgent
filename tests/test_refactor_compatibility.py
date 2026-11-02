"""Historical descriptors and durable graph anchors survive package reorganization."""

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from phase_agent.compatibility import module_references


DESCRIPTOR_MOVES = [
    ("execution_layer.workflows.run_event_loop", "phase_agent.tools.workflows.run_action_turn"),
    ("phase_agent.tools.workflows.run_event_loop", "phase_agent.tools.workflows.run_action_turn"),
    ("analysis_layer.state.build_state_snapshot", "phase_agent.evidence.build_state_snapshot"),
    (
        "phase_agent.analysis.state.build_state_snapshot",
        "phase_agent.evidence.build_state_snapshot",
    ),
    ("execution_layer.policy.validate_tool_action", "phase_agent.policies.validate_tool_action"),
    ("phase_agent.tools.policy.validate_tool_action", "phase_agent.policies.validate_tool_action"),
    *[
        (prefix + name, "phase_agent.models." + name)
        for prefix in ("decision_layer.agent.", "phase_agent.decisions.agent.")
        for name in (
            "create_deepseek_client",
            "prepare_llm_request",
            "transport_references",
            "transport_tables",
            "deepagents_proposal",
            "hybrid_proposal",
        )
    ],
    *[
        (prefix + name, "phase_agent.graphs.recovery." + name)
        for prefix in ("orchestration.", "phase_agent.graphs.")
        for name in (
            "batch_recovery_graph",
            "execution_recovery_graph",
            "recovery_observation_graph",
            "training_handoff_graph",
            "direction_review_graph",
        )
    ],
]


@pytest.mark.parametrize("historical,current", DESCRIPTOR_MOVES)
def test_saved_descriptor_resolves_to_current_module_identity(historical, current):
    assert module_references.import_saved_module(historical) is importlib.import_module(current)


@pytest.mark.parametrize(
    "name",
    [
        "analysis_layer_extra.state.build_state_snapshot",
        "execution_layer_extra.policy.validate_tool_action",
        "phase_agent.analysis.stateful.build_state_snapshot",
        "phase_agent.tools.policy_extra.validate_tool_action",
        "phase_agent.decisions.agentish.create_deepseek_client",
        "phase_agent.graphs.batch_recovery_graph_extra",
    ],
)
def test_descriptor_translation_observes_package_boundaries(monkeypatch, name):
    imported = []
    sentinel = object()

    def capture(module_name):
        imported.append(module_name)
        return sentinel

    monkeypatch.setattr(module_references, "import_module", capture)
    assert module_references.import_saved_module(name) is sentinel
    # The broader decisions package really moved; its suffix must remain exact.
    expected = name.replace("phase_agent.decisions.", "phase_agent.agents.", 1)
    assert imported == [expected]


def test_current_module_name_passes_through_compatibility_boundary(monkeypatch):
    imported = []
    monkeypatch.setattr(module_references, "import_module", imported.append)
    module_references.import_saved_module("phase_agent.evidence.build_state_snapshot")
    assert imported == ["phase_agent.evidence.build_state_snapshot"]


@pytest.mark.parametrize(
    "module,builder,graph_name,anchors",
    [
        (
            "phase_agent.graphs.recovery.batch_recovery_graph",
            "build_batch_recovery_graph",
            "batch_recovery",
            {
                ("join_stage_recovery", "wait_for_partial_or_remaining_results"),
                ("wait_for_partial_or_remaining_results", "reconcile_latest_task_status"),
            },
        ),
        (
            "phase_agent.graphs.recovery.training_handoff_graph",
            "build_training_handoff_graph",
            "training_lifecycle",
            {
                ("persist_training_transition", "wait_for_manifest_return"),
                ("persist_training_transition", "wait_for_validation_return"),
                ("persist_training_transition", "wait_for_activation_decision"),
                ("persist_training_transition", "wait_for_configuration_or_repair"),
                ("wait_for_manifest_return", "collect_training_results"),
                ("wait_for_validation_return", "collect_training_results"),
                ("wait_for_activation_decision", "collect_training_results"),
                ("wait_for_configuration_or_repair", "collect_training_results"),
            },
        ),
        (
            "phase_agent.graphs.project.graph",
            "build_react_lifecycle_graph",
            "project_react_workflow",
            {
                ("initialize", "observe"),
                ("observe", "wait"),
                ("wait", "wait_for_project_input"),
                ("finalize", "wait_for_project_input"),
                ("analyze", "react"),
                ("react", "finalize"),
            },
        ),
    ],
)
def test_persistent_graph_names_and_wait_anchors_remain_stable(
    module, builder, graph_name, anchors
):
    graph = getattr(importlib.import_module(module), builder)()
    assert graph.name == graph_name
    assert anchors <= {(edge.source, edge.target) for edge in graph.get_graph().edges}


def test_direction_and_recovery_child_graph_identity_remains_stable():
    from langgraph.checkpoint.memory import InMemorySaver

    from phase_agent.graphs.recovery.batch_recovery_graph import build_batch_recovery_graph
    from phase_agent.graphs.recovery.direction_review_graph import build_direction_graph
    from phase_agent.graphs.recovery.execution_recovery_graph import build_execution_recovery_graph
    from phase_agent.graphs.recovery.recovery_observation_graph import (
        build_recovery_observation_graph,
    )

    direction = build_direction_graph(InMemorySaver())
    assert direction.name == "scientific_direction_review"
    assert {
        ("__start__", "review_direction_and_plan"),
        ("review_direction_and_plan", "activate"),
        ("review_direction_and_plan", "supplement_dft"),
        ("review_direction_and_plan", "rejected"),
    } <= {(edge.source, edge.target) for edge in direction.get_graph().edges}
    execution = build_execution_recovery_graph()
    assert execution.name == "execution_recovery"
    observation = build_recovery_observation_graph(build_batch_recovery_graph(), execution)
    assert observation.name == "recovery_observation"
    assert (
        "recover_registered_tasks",
        "audit_interrupted_execution",
    ) in {(edge.source, edge.target) for edge in observation.get_graph().edges}


def test_new_responsibility_packages_import_without_old_application_packages():
    # A fresh interpreter prevents modules loaded by other tests masking stale imports.
    script = """
import importlib
import importlib.abc
import pkgutil
import sys

obsolete = (
    "analysis_layer", "execution_layer", "decision_layer", "orchestration",
    "phase_agent.analysis.state", "phase_agent.tools.policy", "phase_agent.decisions",
    "phase_agent.module_references",
    "phase_agent.graphs.batch_recovery_graph", "phase_agent.graphs.execution_recovery_graph",
    "phase_agent.graphs.recovery_observation_graph", "phase_agent.graphs.training_handoff_graph",
    "phase_agent.graphs.direction_review_graph",
)

class RejectObsoletePackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == old or fullname.startswith(old + ".") for old in obsolete):
            raise AssertionError("New source still imports an obsolete package: " + fullname)
        return None

sys.meta_path.insert(0, RejectObsoletePackages())
count = 0
for name in (
    "phase_agent.agents", "phase_agent.models", "phase_agent.policies",
    "phase_agent.evidence", "phase_agent.compatibility",
    "phase_agent.graphs.recovery", "phase_agent.graphs.decision_team",
):
    package = importlib.import_module(name)
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module.name)
        count += 1
assert count > 0
print("Imported", count, "modules without obsolete packages")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
