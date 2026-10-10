"""Measured experience and resource provenance never change budget authorization."""

from copy import deepcopy
import json
import pytest

from phase_agent.analysis.state.budget_experience import (
    budget_experiences,
    budget_experience_context,
    task_budget_observations,
)
from phase_agent.analysis.state.round_budget_evidence import round_budget_evidence
from phase_agent.analysis.state.decision_evidence_catalog import decision_evidence_catalog
from phase_agent.persistence.memory.collect_memory_candidates import collect_memory_candidates
from phase_agent.persistence.memory.verify_evidence_refs import verify_evidence_refs


def sample_state():
    return {
        "active_model_version": "m1",
        "tasks": [
            {
                "task_id": "t1",
                "stage": "deep_search",
                "status": "completed",
                "model_version": "m1",
                "config_version": "c1",
                "parent_decision_id": "a1",
                "atom_count": 40,
                "estimated_cost": 200,
                "actual_mc_steps": 8,
                "parameters": {"max_mc_steps": 100, "patience_steps": 8},
                "runtime_observation": {
                    "source": "task_start_end",
                    "elapsed_seconds": 60,
                    "backend": "mace",
                    "hardware": "gpu-partition",
                },
            }
        ],
        "action_records": [
            {
                "record_id": "a1",
                "config_version": "c1",
                "final_action": {
                    "tool": "allocate_mc_bohb",
                    "budget": 1000,
                    "parameters": {"mc_budget": 1000},
                },
                "execution_result": {"result": {"tasks": [{"task_id": "t1"}]}},
            }
        ],
        "rewards": [
            {
                "batch_id": "r1",
                "task_ids": ["t1"],
                "reward": 0.2,
                "model_version": "m1",
                "energy_method": "mlip",
                "energy_basis_id": "e1",
            }
        ],
    }


def test_nonreviewed_action_is_attributed_without_inventing_spend():
    state = sample_state()
    original = deepcopy(state)
    experience = budget_experiences(state)[0]
    assert experience["actual_relative_cost"] is None
    assert experience["proposed_budget"] == 1000
    assert experience["serial_task_seconds"] == 60
    assert experience["gpu_hours"] is None
    assert experience["observed_rewards"][0]["energy_basis_id"] == "e1"
    assert experience["parameters"] == {"mc_budget": 1000}
    state["tasks"][0].pop("parent_decision_id")
    state["action_records"][0]["execution_result"] = {}
    assert not budget_experiences(state)  # task order or batch coincidence is not attribution
    assert original["tasks"][0]["actual_mc_steps"] == 8


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "100"])
def test_invalid_cost_and_predicted_runtime_are_not_measurements(value):
    state = sample_state()
    task = state["tasks"][0]
    task["actual_cost"] = value
    task["runtime_observation"]["source"] = "historical_runtime_estimate"
    row = task_budget_observations(state)[0]
    assert row["actual_relative_cost"] is None and row["elapsed_seconds"] is None


def test_partial_failure_and_unrelated_rewards_do_not_become_savings():
    state = sample_state()
    state["tasks"].append(
        {
            **deepcopy(state["tasks"][0]),
            "task_id": "t2",
            "status": "running",
            "actual_cost": 3,
        }
    )
    state["rewards"].append({"task_ids": ["unrelated"], "reward": 100})
    experience = budget_experiences(state)[0]
    assert experience["task_count"] == 2 and experience["status"] == "partial"
    assert experience["serial_task_seconds"] is None
    assert len(experience["observed_rewards"]) == 1
    assert "causal savings" in experience["interpretation"]


def test_cohorts_keep_versions_parameters_hardware_separate_and_bound_context():
    state = sample_state()
    for index, changes in enumerate(
        [
            {"model_version": "m0"},
            {"config_version": "c2"},
            {"parameters": {"max_mc_steps": 1000}},
            {
                "runtime_observation": {
                    "source": "task_start_end",
                    "elapsed_seconds": 3600,
                    "backend": "mace",
                    "hardware": "other-gpu",
                }
            },
        ]
    ):
        state["tasks"].append(
            {**deepcopy(state["tasks"][0]), "task_id": f"t{index + 2}", **changes}
        )
    context = budget_experience_context(state)
    assert len(context["comparable_cohorts"]) == 4
    assert all(row["conditions"]["model_version"] == "m1" for row in context["comparable_cohorts"])
    assert all(row["sample_count"] == 1 for row in context["comparable_cohorts"])
    bounded = budget_experience_context(state, character_budget=500)
    assert bounded["retrieval"]["used_characters"] <= 500
    assert state["tasks"][0]["runtime_observation"]["elapsed_seconds"] == 60


def test_shared_scheduler_accounting_not_charged_once_per_structure():
    state = sample_state()
    task = state["tasks"][0]
    task.update(
        {
            "batch_id": "batch1",
            "actual_gpu_core_hours": 2,
            "job_accounting": {"source": "sacct", "elapsed_seconds": 3600, "gpu_count": 2},
        }
    )
    assert task_budget_observations(state)[0]["gpu_hours"] is None
    task["job_accounting"]["allocation_basis"] = "task_time_slice_not_whole_batch"
    assert task_budget_observations(state)[0]["gpu_hours"] == 2


def test_idempotent_candidates_and_model_context_include_citable_experience():
    state = sample_state()
    original = deepcopy(state)
    context = round_budget_evidence(state)
    assert context["budget_experience"]["coverage"]["elapsed_seconds"] == 1
    assert not context["observed_budget_outcomes"]  # legacy format no longer blocks new experience
    catalog = decision_evidence_catalog({"round_budget_evidence": context})
    assert "budget_experience:a1" in catalog["references"]
    assert any(key.startswith("budget_cohort:") for key in catalog["references"])
    collected = collect_memory_candidates(state)
    candidates = [r for r in collected["memory_candidates"] if r["kind"] == "budget_experience"]
    assert len(candidates) == 1
    assert not verify_evidence_refs(collected, candidates[0]["evidence_refs"], scope="project")
    again = collect_memory_candidates(collected)
    assert len(again["memory_candidates"]) == len(collected["memory_candidates"])
    assert not again.get("memory_review_queue")
    assert state == original
    json.dumps(again, allow_nan=False)


def test_runtime_from_explicit_cost_history_receipt_is_reused():
    state = sample_state()
    task = state["tasks"][0]
    runtime = task.pop("runtime_observation")
    state["cost_history"] = [
        {
            "task_id": "t1",
            "runtime_observation": runtime,
            "actual_cost_known": False,
            "actual_cost": 1000,
        }
    ]
    row = task_budget_observations(state)[0]
    assert row["elapsed_seconds"] == 60 and row["actual_relative_cost"] is None


def test_whole_batch_runtime_and_mixed_models_do_not_become_task_estimates():
    state = sample_state()
    state["tasks"][0]["runtime_observation"]["allocation_basis"] = (
        "whole_batch_do_not_add_to_task_slices"
    )
    state["tasks"].append({**deepcopy(state["tasks"][0]), "task_id": "t2", "model_version": "m0"})
    experience = budget_experiences(state)[0]
    assert experience["serial_task_seconds"] is None
    assert experience["model_version"] is None


def test_deepagents_transport_retains_budget_experience_without_heavy_payloads():
    from phase_agent.decisions.agent.deepagents_proposal import proposal_material

    context = round_budget_evidence(sample_state())
    context["budget_experience"]["decision_examples"][0]["structure"] = {"sites": [1, 2, 3]}
    material = proposal_material({"decision_context": {"round_budget_evidence": context}})
    experience = material["decision_context"]["round_budget_evidence"]["budget_experience"]
    assert experience["coverage"]["elapsed_seconds"] == 1
    assert experience["decision_examples"][0]["parameters"] == {"mc_budget": 1000}
    assert experience["comparable_cohorts"][0]["observations"]["elapsed_seconds"]["median"] == 60
    assert "structure" not in experience["decision_examples"][0]


def test_dialogue_model_receives_budget_experience_with_no_execution(tmp_path):
    from tests.test_unified_react_dialogue import make_handler
    from phase_agent.tools.step_runner.file_protocol import write_json

    calls = []

    def model(payload):
        calls.append(payload)
        return {"kind": "answer", "answer": "历史耗时可参考，相对成本未知"}

    handler = make_handler(tmp_path, model)
    write_json(handler.state_path, sample_state())
    reply = handler(
        [{"role": "user", "content": "解释已有预算经验"}], conversation_id="budget-memory"
    )
    assert "相对成本未知" in reply
    experience = calls[0]["decision_context"]["round_budget_evidence"]["budget_experience"]
    assert experience["coverage"]["elapsed_seconds"] == 1
    assert experience["decision_examples"][0]["actual_relative_cost"] is None
