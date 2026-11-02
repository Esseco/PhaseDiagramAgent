"""Continuous chat regression with real persistence; external compute is synthetic."""

from copy import deepcopy

from ase import Atoms
from ase.io import write

from phase_agent.runtime.interaction_reports import short_reply
from phase_agent.runtime.chat_application import RunWorkflowChatHandler
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from tests.test_training_return_contract import training_analysis_case
from tests.test_unified_react_dialogue import make_handler
from tests.integration.test_unified_workflow_bohb import H
import pytest


@pytest.mark.parametrize("transport", ["in_process", "chat_process", "studio_process"])
@pytest.mark.parametrize(
    "history_case", [None, "legacy_ack", "legacy_pending", "lost_plan", "legacy_checkpoint_hold"]
)
def test_training_return_through_refresh_approval_with_real_chat_and_runner(
    tmp_path, monkeypatch, history_case, transport
):
    calls = []
    generation_requests = []
    returned = [False]

    def model(payload):
        calls.append(payload)
        assert returned[0], "No new scientific decision before refresh has returned"
        latest = read_json(handler.state_path, {})
        generations = latest.get("generation_history") or []
        from tests.test_round_budget_review import review as budget_review

        if generations:
            return {
                "tool": "prepare_local_batch_files",
                "task_key": "post-generation-inputs",
                "parameters": {"mode": "relax_mc_inputs"},
                "target_ids": generations[-1]["new_branch_ids"],
                "budget": 0,
                "reason": "新branch已生成，准备搜索输入",
                "round_budget_review": budget_review("new_search"),
            }
        context = payload.get("decision_context") or {}
        if not context.get("branch_pool_facts"):
            return {
                "kind": "inspect_evidence",
                "topics": ["mc", "budget", "configuration", "tasks", "model"],
                "view": "full",
                "full_for": ["generate_branches"],
            }
        refs = list(
            ((payload.get("output_contracts") or {}).get("branch_review_evidence") or {}).get(
                "references"
            )
            or (context.get("evidence_catalog") or {}).get("references")
            or {}
        )[:1]
        generation_requests.append(True)
        return {
            "tool": "generate_branches",
            "task_key": "post-refresh-generation",
            "parameters": {
                "sampling_mode": "region_adaptive",
                "initial_states_per_branch": 1,
                "regions": [
                    {
                        "region_id": "next",
                        "priority": 1,
                        "strategy": "coverage",
                        "phase": "O3",
                        "min_det_H": 1,
                        "max_det_H": 1,
                        "na_min": 0,
                        "na_max": 1,
                        "reason": "合成覆盖缺口",
                        "evidence_refs": refs,
                    }
                ],
            },
            "branch_pool_review": {
                "choice": "supplement",
                "sufficiency_for_next_search": "合成覆盖缺口",
                "rationale": "补充Na覆盖",
                "alternative_reason": "暂缓已有池",
                "missing_evidence": [],
                "revisit_conditions": ["新结果回传后"],
                "evidence_refs": refs,
                "existing_branch_groups": [
                    {
                        "decision": "defer",
                        "branch_ids": list(handler.workflow_kwargs["manager"].data["branches"]),
                        "reason": "验证新候选生成",
                        "evidence_refs": refs,
                    }
                ],
            },
            "round_budget_review": budget_review("new_search"),
            "budget": 0,
            "reason": "依据新模型相图补充候选覆盖",
            "expected_purpose": "生成新 branch",
        }

    handler = make_handler(tmp_path, model)
    snapshot = handler.workflow_kwargs["config_session"]["confirmed_snapshot"]
    config = snapshot["config"]
    config["calculation"]["mlip_version"] = "base"
    config["mlip"] = {"version": "base", "model_path": "/hpc/base.model", "sha256": "b" * 64}
    config["supercomputer"] = {
        "worker": {
            "command": [
                "python",
                "-m",
                "phase_agent.tools.remote.run_slurm_array_task",
                "--executor",
                "phase_agent.science.mlip.slurm_executor:execute_mlip_task",
            ]
        }
    }
    handler.workflow_kwargs["run_config"].update(
        upload_batches_directory=str(tmp_path / "submissions"),
        structure_directory=str(tmp_path / "structures"),
    )
    handler.workflow_kwargs["phase_references"] = {"O3": str(tmp_path / "initial.vasp")}
    state = training_analysis_case(tmp_path / "submissions" / "epoch1")
    state.update(
        confirmed_config_version=snapshot["config_version"], confirmed_config=deepcopy(config)
    )
    state["workspace_layout"] = {"workspace_root": str(tmp_path)}
    state["upload_layout"] = {"model_rounds": {"base": 2}}
    manager = handler.workflow_kwargs["manager"]
    branch = manager.add_branch(P="O3", H=H, x=1, T=["Fe"], composition={"Na": 1, "Fe": 1, "O": 2})
    path = tmp_path / "initial.vasp"
    write(
        path,
        Atoms(
            "NaFeO2",
            positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
            cell=[5, 5, 5],
            pbc=True,
        ),
    )
    sid = manager.add_structure(branch_id=branch, arrangement="synthetic", source_path=path)
    state["dedup_gate"] = {"status": "ready", "valid_structure_ids": [sid]}
    state["phase_diagrams"] = {
        "mlip": {
            "status": "completed",
            "model_version": "base",
            "entries": [
                {
                    "structure_id": sid,
                    "structure_path": str(path),
                    "ehull": 0.005,
                    "ehull_unit": "eV/atom",
                    "phase": "O3",
                }
            ],
        }
    }
    write_json(handler.state_path, state)

    process_ids = []
    turn_number = 0

    def send(text):
        nonlocal turn_number
        turn_number += 1
        if transport == "in_process":
            full = handler([{"role": "user", "content": text}], conversation_id="chain")
        else:
            import json
            import subprocess
            import sys
            from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager

            ledger = tmp_path / "restart_ledger.json"
            runtime_path = tmp_path / "restart_runtime.json"
            handler.workflow_kwargs["manager"].save(ledger)
            write_json(
                runtime_path,
                {
                    key: value
                    for key, value in handler.workflow_kwargs.items()
                    if key not in {"manager", "agent_client"}
                },
            )
            request = {
                "runtime_path": str(runtime_path),
                "ledger_path": str(ledger),
                "text": text,
                "message_id": f"message-{turn_number}",
                "conversation_id": "chain",
                "transport": transport,
            }
            if returned[0]:
                request["returned_science"] = {
                    "task": saved["tasks"][0],
                    "phase_records": saved["phase_records"],
                    "phase_diagram": saved["phase_diagrams"]["mlip"],
                }
            request_path = tmp_path / "restart_request.json"
            write_json(request_path, request)
            with subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-X",
                    "utf8",
                    "-m",
                    "tests.epoch_process_worker",
                    str(request_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            ) as process:
                import threading

                timeout = threading.Timer(60, process.kill)
                timeout.start()
                output = []
                try:
                    for line in process.stdout:
                        if line.startswith("MODEL_REQUEST "):
                            decision = model(json.loads(line[len("MODEL_REQUEST ") :]))
                            process.stdin.write(json.dumps(decision, ensure_ascii=False) + "\n")
                            process.stdin.flush()
                        else:
                            output.append(line)
                    assert process.wait(timeout=30) == 0, "".join(output)
                finally:
                    timeout.cancel()
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=10)
            result = read_json(request_path.with_suffix(".reply.json"), {})
            process_ids.append(result["pid"])
            full = result["reply"]
            handler.workflow_kwargs.update(read_json(runtime_path, {}))
            handler.workflow_kwargs["manager"].data = PhaseDataManager.load(ledger).data
            # The subprocess owns the fresh handler; these bindings are disk-backed.
        reply = short_reply(full)
        print("TURN", transport, history_case, turn_number, text, reply, flush=True)
        assert not any(
            block in reply
            for block in [
                "当前没有唯一待审批方案",
                "当前没有可批准或拒绝",
                "approval_reconciliation_required",
            ]
        ), reply
        return reply

    reply = send("已回收")
    assert "finetuning_results.md" in reply and "是否进入 epoch2" in reply
    saved = read_json(handler.state_path, {})
    assert saved["active_model_version"] == "base"
    assert saved["candidate_models"] and not calls
    assert len(saved["remote_finetune_jobs"]) == 1
    from pathlib import Path

    report = Path(saved["remote_finetune_jobs"]["j"]["training_handoff"]["training_report_path"])
    stamp = report.stat().st_mtime_ns
    assert "是否进入 epoch2" in send("已回收")
    assert report.stat().st_mtime_ns == stamp
    assert not calls

    reply = send("同意")
    saved = read_json(handler.state_path, {})
    assert saved["active_model_version"] != "base", reply
    from phase_agent.evidence.model_epoch import model_epoch

    assert model_epoch(saved, saved["active_model_version"]) == "epoch2", reply
    if transport == "studio_process":
        assert "epoch2" in reply.splitlines()[0], reply
    assert "是否批准生成刷新文件" in reply, reply
    assert "范围：" in reply and "单点 1" in reply and "成本" in reply, reply
    assert not saved.get("tasks")
    assert len(saved["pending_execution_policies"]) == 1
    assert not calls
    if history_case:
        from phase_agent.tools.workflows.recover_saved_results import recovery_candidate_fingerprint

        saved["recovery_acknowledgement"] = {
            "fingerprint": recovery_candidate_fingerprint(saved),
            "scope": "explicit_user_recovery_approval",
            "source_stages": [],
        }
        saved["post_recovery_next_step_pending"] = history_case != "legacy_ack"
        if history_case == "lost_plan":
            saved["pending_execution_policies"] = {}
        write_json(handler.state_path, saved)
    if history_case == "lost_plan":
        reply = send("同意")
        saved = read_json(handler.state_path, {})
        assert "是否批准生成刷新文件" in reply
        assert len(saved["pending_execution_policies"]) == 1
        assert not saved.get("tasks") and not calls
    elif history_case:
        original_pending = deepcopy(saved["pending_execution_policies"])
        assert "是否批准生成刷新文件" in send("继续")
        assert read_json(handler.state_path, {})["pending_execution_policies"] == original_pending
    reply = send("同意")
    saved = read_json(handler.state_path, {})
    assert saved["model_refresh"]["status"] == "waiting_results", reply
    assert "输入已生成" in reply or "刷新文件已生成" in reply, reply
    assert "sbatch submit_all.sh" in reply and "已回传" in reply, reply
    import re

    scripts = [
        value
        for value in re.findall(r"`([^`]*submit_all\.sh)`", reply)
        if Path(value).is_absolute()
    ]
    assert scripts and all(Path(script).is_file() for script in scripts), reply
    assert len(saved["tasks"]) == 1
    assert not calls
    task_ids = [t["task_id"] for t in saved["tasks"]]
    reply = send("同意")
    assert "Relax+MC" not in reply, reply
    assert [t["task_id"] for t in read_json(handler.state_path, {})["tasks"]] == task_ids
    assert not calls

    restarted = RunWorkflowChatHandler(handler.workflow_kwargs)
    waiting = short_reply(
        restarted([{"role": "user", "content": "继续"}], conversation_id="restart")
    )
    assert "刷新" in waiting and "Relax+MC" not in waiting
    assert [t["task_id"] for t in read_json(handler.state_path, {})["tasks"]] == task_ids
    assert not calls

    # A tiny synthetic returned scientific snapshot avoids any MLIP/HPC work.
    # The next chat still uses the real lifecycle, refresh evidence gate,
    # follow-up proposal validation, saved approval and generation tool.
    saved = read_json(handler.state_path, {})
    task = saved["tasks"][0]
    task.update(
        status="completed",
        converged=True,
        checks_passed=True,
        outputs={
            "single_point_completed": True,
            "forces": [[0, 0, 0]],
            "structure_path": str(path),
        },
    )
    model_version = saved["active_model_version"]
    saved["phase_records"] = [
        {
            "record_id": "r1",
            "source_task_id": task["task_id"],
            "source_version": model_version,
            "model_version": model_version,
            "energy_method": "mlip",
            "status": "completed",
            "phase_identification_status": "identified",
            "energy": -1,
            "energy_unit": "eV",
            "structure_id": sid,
            "structure_path": str(path),
            "phase": "O3",
            "composition": {"Na": 1, "Fe": 1, "O": 2},
        }
    ]
    saved["phase_diagrams"]["mlip"] = {
        "status": "completed",
        "model_version": model_version,
        "entries": [
            {
                "record_id": "r1",
                "ehull": 0.005,
                "ehull_unit": "eV/atom",
                "normalized_total_energy": -1,
                "structure_id": sid,
                "structure_path": str(path),
                "phase": "O3",
            }
        ],
    }
    write_json(handler.state_path, saved)
    returned[0] = True
    if history_case == "legacy_checkpoint_hold":
        # Real SQLite checkpoint at the prior action boundary, not a mocked runner.
        from copy import copy
        from phase_agent.graphs.scientific_graph import scientific_graph
        from phase_agent.persistence.compact_checkpoints import CompactSqliteSaver
        from phase_agent.tools.budget.settle_budget import settle_budget

        # Match production return settlement; unknown actual cost is not zero.
        settled = settle_budget(
            saved,
            task_key=saved["tasks"][0]["task_key"],
            settlement_id="synthetic-refresh-return",
            task_status="completed",
            actual_cost=None,
            estimated_cost=saved["tasks"][0]["planned_relative_cost"],
        )
        assert settled["status"] == "settled"
        saved = settled["state"]
        write_json(handler.state_path, saved)
        hold = saved["model_refresh"]["supplemental_reservation_key"]
        assert saved["budget_reservations"][hold]["status"] == "reserved"
        assert all(t["status"] == "completed" for t in saved["tasks"])
        with CompactSqliteSaver.from_conn_string(
            str(tmp_path / "langgraph_lifecycle.sqlite")
        ) as saver:
            poisoned = copy(scientific_graph())
            poisoned.checkpointer = saver
            checkpoint_config = {"configurable": {"thread_id": "project-scientific-lifecycle-v2"}}
            poisoned.update_state(
                checkpoint_config,
                {
                    "request_id": "previous-prepared-action",
                    "response": None,
                    "completed_response": None,
                    "business_frame": {},
                },
                as_node="analyze",
            )
            assert poisoned.get_state(checkpoint_config).next == ("react",)
    # Isolate the scientific hull calculation with a small validated-result
    # fixture; retain the production lifecycle and completion/approval gates.
    import phase_agent.runtime.main as main

    actual_analysis = main._analyze_workflow_results

    def analyze(frame):
        frame = actual_analysis(frame)
        current = frame["feedback"]["state"]
        current["tasks"][0].update(
            status="completed",
            converged=True,
            checks_passed=True,
            outputs={
                "single_point_completed": True,
                "forces": [[0, 0, 0]],
                "structure_path": str(path),
            },
        )
        current["phase_records"] = deepcopy(saved["phase_records"])
        current["phase_diagrams"]["mlip"] = deepcopy(saved["phase_diagrams"]["mlip"])
        return frame

    monkeypatch.setattr(main, "_analyze_workflow_results", analyze)
    reply = send("已回收")
    saved = read_json(handler.state_path, {})
    assert saved["model_refresh"]["status"] == "completed", reply
    pending = saved.get("pending_execution_policies") or {}
    assert len(pending) == 1, reply
    proposal = next(iter(pending.values()))["agent_proposal"]
    assert proposal["raw_action"]["tool"] == "generate_branches", reply
    assert "同意" in reply and "branch" in reply.lower(), reply
    from phase_agent.graphs.dialogue.support import _load_binding

    handler.conversation_id = "chain"
    binding = deepcopy(_load_binding(handler))
    before_calls = len(calls)
    before_generation_requests = len(generation_requests)
    reply = send("同意")
    saved = read_json(handler.state_path, {})
    assert saved.get("generation_history"), reply
    assert len(generation_requests) == before_generation_requests, (
        "Approval must not select another generation plan"
    )
    assert len(calls) >= before_calls
    pending = saved.get("pending_execution_policies") or {}
    assert len(pending) == 1, reply
    assert (
        next(iter(pending.values()))["agent_proposal"]["raw_action"]["tool"]
        == "prepare_local_batch_files"
    )
    assert "同意" in reply and "Relax" in reply, reply
    if history_case:
        next_pending = deepcopy(saved["pending_execution_policies"])
        call_count = len(calls)
        assert "同意" in send("继续")
        assert read_json(handler.state_path, {})["pending_execution_policies"] == next_pending
        assert len(calls) == call_count
    history_count = len(saved["generation_history"])
    replay = handler.review_pending(
        binding["plan_id"],
        "approve",
        expected_state_version=binding["state_version"],
        expected_proposal_hash=binding["proposal_hash"],
        comment="retry same approved plan",
    )
    assert replay["status"] == "already_processed"
    assert len(read_json(handler.state_path, {})["generation_history"]) == history_count
    if history_case == "legacy_checkpoint_hold":
        # Continue beyond generation: consume the displayed input approval,
        # restart again, then wait without finding the abandoned root thread.
        call_count = len(calls)
        reply = send("同意")
        saved = read_json(handler.state_path, {})
        assert not saved.get("pending_execution_policies"), reply
        assert any(t.get("status") == "pending" for t in saved["tasks"]), reply
        assert "旧动作" not in reply and "中断记录" not in reply, reply
        prepared_batches = deepcopy(saved.get("relax_mc_batches"))
        reply = send("继续")
        assert "等待" in reply and "旧动作" not in reply, reply
        assert read_json(handler.state_path, {}).get("relax_mc_batches") == prepared_batches
        assert len(calls) == call_count
    if transport != "in_process":
        assert len(process_ids) >= 7
        print("FRESH_PROCESS_TURNS", len(process_ids), process_ids, flush=True)
        assert (tmp_path / "langgraph_approvals.sqlite").is_file()


def test_refresh_blocker_is_visible_in_short_reply():
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply

    result = {
        "status": "not_configured",
        "state": {
            "model_refresh": {"status": "approval_required", "input_authorization": {"plan": {}}}
        },
        "reason": "structure_dedup_not_ready",
    }
    reply = short_reply(format_workflow_reply(result, "synthetic-state.json"))
    assert "structure_dedup_not_ready" in reply
    assert "下一步：" in reply


def test_followup_failure_does_not_offer_an_absent_approval():
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply

    result = {
        "status": "completed",
        "state": {},
        "events": [],
        "followup_error": "network unavailable",
        "completed_generation": {"registered_branches": 1, "registered_structures": 1},
    }
    reply = short_reply(format_workflow_reply(result, "synthetic-state.json"))
    assert "network unavailable" in reply and "下一步：" in reply
    assert "当前没有待批准" in reply
    assert "是否现在审查" not in reply


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_pending_feedback_with_same_request_resumes_once(tmp_path, decision):
    from phase_agent.graphs.project.runner import run_search_workflow_graph

    path = tmp_path / "state.json"
    write_json(
        path,
        {
            "tasks": [],
            "pending_execution_policies": {
                "p": {
                    "agent_proposal": {"raw_action": {"tool": "check_convergence", "task_key": "p"}}
                }
            },
        },
    )
    feedback = [None]
    reviews = []

    def initialize():
        return {
            "_workflow_prepared": True,
            "snapshot": {"config_version": "v1"},
            "loaded_state": read_json(path, {}),
        }

    def act(frame, **kwargs):
        if feedback[0] is not None:
            reviews.append(feedback[0]["decision"])
            frame["loaded_state"]["pending_execution_policies"].pop("p")
            write_json(path, frame["loaded_state"])
        return frame

    def finalize(frame):
        return {
            "status": "awaiting_approval" if feedback[0] is None else "completed",
            "state": frame["loaded_state"],
        }

    kwargs = dict(
        checkpoint_path=tmp_path / "lifecycle.sqlite",
        invocation_id="p",
        business_state_path=path,
        project_thread="project",
        initialize=initialize,
        collect=lambda f: f,
        analyze=lambda f: f,
        wait=lambda f: f,
        assess=lambda f: f,
        act=act,
        finalize=finalize,
    )
    first = run_search_workflow_graph(**kwargs)
    assert first["status"] == "awaiting_approval" and reviews == []
    assert run_search_workflow_graph(**kwargs) == first
    feedback[0] = {"decision": decision, "comment": "explicit review"}
    result = run_search_workflow_graph(**kwargs, human_feedback=feedback[0])
    assert result["status"] == "completed" and reviews == [decision]
    assert run_search_workflow_graph(**kwargs, human_feedback=feedback[0]) == result
    assert reviews == [decision]


def test_multiple_review_objects_remain_visible_in_short_reply(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from phase_agent.graphs.dialogue.support import review_presented_proposal

    rows = [
        {
            "plan_id": "refresh-plan",
            "review_card": {"title": "生成刷新文件", "scope": "首批 2 个结构"},
        },
        {
            "plan_id": "recovery-plan",
            "review_card": {"title": "核对旧执行", "scope": "核对 1 条回执"},
        },
    ]
    monkeypatch.setattr("phase_agent.graphs.dialogue.support._review_rows", lambda *a: rows)
    handler = SimpleNamespace(state_path=tmp_path / "state.json", conversation_id=None)
    reply = short_reply(review_presented_proposal(handler, {}, "同意"))
    assert "生成刷新文件" in reply and "核对旧执行" in reply
    assert reply.count("待审对象：") == 2 and "下一步：" in reply
