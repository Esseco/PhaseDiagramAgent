"""Recovered DFT enters the real chat, lifecycle, proposal and approval pipeline."""

from copy import deepcopy

import pytest

from phase_agent.runtime.interaction_reports import short_reply
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from tests.test_dft_comparison_csv import add_result
from tests.test_post_dft_review import review as post_dft_review
from tests.test_round_budget_review import review as budget_review
from tests.test_three_scientific_decisions import review as scientific_review
from tests.test_unified_react_dialogue import make_handler


@pytest.mark.parametrize("tool", ["update_mlip", "pause_search", "generate_branches"])
def test_recovered_dft_uses_actual_planner_and_stores_one_approval(tmp_path, tool):
    calls, executed = [], []
    action = {
        "tool": tool,
        "task_key": "next-dft-direction",
        "parameters": {"action": "RETRAIN_MLIP", "prepare_inputs_only": True}
        if tool == "update_mlip"
        else {},
        "reason": "Synthetic evidence supports this direction; coverage is partial.",
        "expected_purpose": "Improve the model" if tool == "update_mlip" else "Stop this search",
        "post_dft_review": post_dft_review(),
        "scientific_decision": scientific_review("round_direction"),
    }
    action["post_dft_review"].update(
        choice="finetune"
        if tool == "update_mlip"
        else "search"
        if tool == "generate_branches"
        else "stop",
        finetune_recommendation="now" if tool == "update_mlip" else "defer",
        stop_status="budget_stop" if tool == "pause_search" else "continue",
    )

    def model(payload):
        calls.append(payload.get("mode"))
        if tool == "generate_branches":
            context = payload.get("decision_context") or {}
            if not context.get("branch_pool_facts"):
                return {
                    "kind": "inspect_evidence",
                    "topics": [
                        "mc",
                        "dft",
                        "budget",
                        "configuration",
                        "convergence",
                        "tasks",
                        "model",
                    ],
                    "view": "full",
                    "full_for": [tool],
                }
            refs = list(
                ((payload.get("output_contracts") or {}).get("branch_review_evidence") or {}).get(
                    "references"
                )
                or (context.get("evidence_catalog") or {}).get("references")
                or {}
            )
            chosen = refs[:1]
            action["parameters"]["regions"][0]["evidence_refs"] = chosen
            action["branch_pool_review"] = {
                "choice": "supplement",
                "sufficiency_for_next_search": "Synthetic missing coverage",
                "rationale": "Explore the uncovered region",
                "alternative_reason": "Defer this existing branch",
                "missing_evidence": [],
                "revisit_conditions": ["After new returns"],
                "evidence_refs": chosen,
                "existing_branch_groups": [
                    {
                        "decision": "defer",
                        "branch_ids": [branch_id],
                        "reason": "Synthetic coverage priority",
                        "evidence_refs": chosen,
                    }
                ],
            }
        return deepcopy(action)

    def execute(**kwargs):
        executed.append(kwargs)
        raise AssertionError("Unapproved proposal executed")

    handler = make_handler(tmp_path, model, {tool: execute})
    original_run = handler._run.__func__
    snapshot = handler.workflow_kwargs["config_session"]["confirmed_snapshot"]
    action["scientific_decision"].update(
        config_version=snapshot["config_version"], model_version="m1"
    )
    state = {
        "active_model_version": "m1",
        "confirmed_config_version": snapshot["config_version"],
        "confirmed_config": snapshot["config"],
        "scientific_decision_policy": {"schema_version": 1},
        "last_trained_dataset_version": "synthetic-previous-epoch",
        "post_recovery_next_step_pending": True,
        "recovery_acknowledgement": {"source_stages": ["dft_single_point"]},
    }
    if tool == "generate_branches":
        from pymatgen.core import Lattice, Structure
        from tests.integration.test_unified_workflow_bohb import H

        reference = tmp_path / "O3.vasp"
        Structure(
            Lattice.cubic(6),
            ["Na", "Fe", "O", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        ).to(filename=str(reference), fmt="poscar")
        handler.workflow_kwargs["phase_references"] = {"O3": str(reference)}
        manager = handler.workflow_kwargs["manager"]
        branch_id = manager.add_branch(
            P="O3", H=H, x=1, T=["Fe"], composition={"Na": 1, "Fe": 1, "O": 2}
        )
        state["branch_candidates"] = [
            {"branch_id": branch_id, "P": "O3", "H": H, "T": ["Fe"], "x": "1"}
        ]
        state["workspace_layout"] = {"workspace_root": str(tmp_path)}
        action["parameters"] = {
            "sampling_mode": "region_adaptive",
            "initial_states_per_branch": 1,
            "regions": [
                {
                    "region_id": "next-region",
                    "priority": 1,
                    "strategy": "coverage",
                    "phase": "O3",
                    "min_det_H": 1,
                    "max_det_H": 1,
                    "na_min": 0.0,
                    "na_max": 1.0,
                    "reason": "Synthetic coverage hypothesis",
                    "evidence_refs": [],
                }
            ],
        }
        action["round_budget_review"] = budget_review("new_search")
    for index in range(12):
        add_result(state, task_id=f"dft-{index}")
    for index in range(3):
        task = deepcopy(state["tasks"][0])
        task.update(task_id=f"failed-{index}", status="failed")
        state["tasks"].append(task)
    for task in state["tasks"]:
        task["config_version"] = snapshot["config_version"]
    write_json(handler.state_path, state)

    reply = handler([{"role": "user", "content": "继续"}], conversation_id="real-entry")
    visible = short_reply(reply)

    saved = read_json(handler.state_path, {})
    assert handler._run.__func__ is original_run
    assert calls and not executed
    pending = saved.get("pending_execution_policies") or {}
    assert len(pending) == 1, visible
    proposal = next(iter(pending.values()))["agent_proposal"]
    assert proposal["raw_action"]["tool"] == tool
    assert "12/15" in visible and "下一步：建议" in visible and "是否同意" in visible, {
        "reply": reply[:2500],
        "raw_keys": list(proposal["raw_action"]),
        "parameters": {
            key: value
            for key, value in proposal["raw_action"]["parameters"].items()
            if key in {"action", "prepare_inputs_only", "prepare_inputs_after_approval", "mode"}
        },
    }
    assert not saved.get("remote_finetune_jobs")
    assert [row["status"] for row in saved["tasks"]] == [row["status"] for row in state["tasks"]]

    # Recreate the actual handler over persisted state, rather than mock reload.
    from phase_agent.runtime.chat_application import RunWorkflowChatHandler
    from phase_agent.policies.file_approval import proposal_hash

    original_hash = proposal_hash(proposal)
    count_before_reload = len(calls)
    restarted = RunWorkflowChatHandler(handler.workflow_kwargs, execution_mode="interactive")
    restored = short_reply(
        restarted([{"role": "user", "content": "继续原运行"}], conversation_id="reloaded-entry")
    )
    assert "同意" in restored, restored
    restored_pending = read_json(handler.state_path, {})["pending_execution_policies"]
    assert len(restored_pending) == 1
    assert proposal_hash(next(iter(restored_pending.values()))["agent_proposal"]) == original_hash
    assert len(calls) == count_before_reload and not executed

    restarted([{"role": "user", "content": "拒绝"}], conversation_id="reloaded-entry")
    assert not read_json(handler.state_path, {}).get("pending_execution_policies")
    assert len(calls) == count_before_reload and not executed
