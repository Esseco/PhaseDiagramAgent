"""One bounded semantic interpretation of saved work; never a scientific selector."""

import json
import os


def active_task_wait_reply(facts):
    """Answer a continuation while work is still running without model routing."""
    from phase_agent.runtime.chat_state_presentation import brief_chat_state
    from phase_agent.tools.state.task_waiting import active_pending_tasks

    active = active_pending_tasks(facts.get("tasks") or [])
    if not active:
        return None
    names = {
        "dft_single_point": "DFT 单点",
        "dft_relax": "DFT 优化",
        "deep_search": "MC",
        "relax_and_feature": "Relax",
    }
    counts = {}
    for row in active:
        name = names.get(row.get("stage"), "计算")
        counts[name] = counts.get(name, 0) + 1
    current = brief_chat_state(facts)
    history = []
    if any(
        row.get("status") == "completed"
        and row.get("model_version") == facts.get("active_model_version")
        for row in (facts.get("relax_mc_batches") or {}).values()
    ):
        history.append("已完成流程：Relax+MC 结果已回收并入账。")
    dft_prepared = any(
        ((record.get("final_action") or {}).get("tool")
         or ((record.get("agent_proposal") or {}).get("raw_action") or {}).get("tool"))
        == "select_dft_candidates"
        and (
            record.get("status") in {"prepared", "completed"}
            or (record.get("execution_result") or {}).get("status") == "completed"
        )
        for record in facts.get("action_records") or []
    )
    if dft_prepared and any(
        row.get("stage") in {"dft_single_point", "dft_relax"} for row in active
    ):
        history.append("已完成流程：补充 DFT 方案已批准，输入文件已生成。")
    if any(row.get("stage") in {"dft_single_point", "dft_relax"} for row in active):
        next_step = "下一步：等待超算完成并回传 DFT 输出；回传后回复‘已回传’，本轮不重新生成方案。"
        active_ids = {row.get("task_id") for row in active}
        input_paths, submit_paths, return_paths = set(), set(), set()
        for batch in facts.get("slurm_batches") or []:
            if not active_ids.intersection(batch.get("task_ids") or []):
                continue
            handoff = batch.get("handoff") or {}
            if handoff.get("folder"):
                input_paths.add(str(handoff["folder"]))
            if handoff.get("submit_script"):
                submit_paths.add(str(handoff["submit_script"]))
            if handoff.get("return_directory"):
                return_paths.add(str(handoff["return_directory"]))
        if input_paths:
            history.append("DFT 输入地址：" + "、".join(f"`{path}`" for path in sorted(input_paths)))
        if submit_paths:
            history.append("提交脚本：" + "、".join(f"`{path}`" for path in sorted(submit_paths)))
        if return_paths:
            history.append("DFT 回传地址：" + "、".join(f"`{path}`" for path in sorted(return_paths)))
    elif any(row.get("stage") in {"relax_and_feature", "deep_search"} for row in active):
        next_step = "下一步：等待 Relax+MC 完成并回传结果；回传后回复‘已回传’，本轮不重新生成方案。"
        active_ids = {row.get("task_id") for row in active}
        active_bundle_ids = {row.get("relax_mc_bundle_id") for row in active}
        input_paths, submit_paths, return_paths = set(), set(), set()
        for batch in facts.get("slurm_batches") or []:
            if not (
                active_ids.intersection(batch.get("task_ids") or [])
                or batch.get("batch_id") in active_bundle_ids
                or batch.get("relax_mc_bundle_id") in active_bundle_ids
            ):
                continue
            handoff = batch.get("handoff") or {}
            if handoff.get("folder"):
                input_paths.add(str(handoff["folder"]))
            if handoff.get("submit_script"):
                submit_paths.add(str(handoff["submit_script"]))
            if handoff.get("return_directory"):
                return_paths.add(str(handoff["return_directory"]))
        for bundle_id in active_bundle_ids:
            batch = (facts.get("relax_mc_batches") or {}).get(bundle_id) or {}
            handoff = batch.get("handoff") or {}
            input_path = batch.get("upload_folder") or handoff.get("folder")
            submit_path = batch.get("script_path") or handoff.get("submit_script")
            return_path = batch.get("return_directory") or handoff.get("return_directory")
            if input_path:
                input_paths.add(str(input_path))
            if submit_path:
                submit_paths.add(str(submit_path))
            if return_path:
                return_paths.add(str(return_path))
        if input_paths:
            history.append("Relax+MC 输入地址：" + "、".join(f"`{path}`" for path in sorted(input_paths)))
        if submit_paths:
            history.append("Relax+MC 提交脚本：" + "、".join(f"`{path}`" for path in sorted(submit_paths)))
        if return_paths:
            history.append("Relax+MC 回收地址：" + "、".join(f"`{path}`" for path in sorted(return_paths)))
    else:
        next_step = "下一步：等待当前任务完成并回传结果；收到结果后再进行回收核对。"
    detail = "、".join(f"{name} {count} 个" for name, count in counts.items())
    return "\n".join([*history, current, f"待完成：{detail}", next_step])


def generation_blocked_review(facts, reviews):
    """A unique saved plan can need revision before it can be approved."""
    if len(reviews) != 1:
        return None
    row = reviews[0]
    record = (facts.get("pending_execution_policies") or {}).get(row["plan_id"]) or {}
    action = (record.get("agent_proposal") or {}).get("raw_action") or {}
    from phase_agent.tools.policy.branch_conditions import branch_entry_errors

    errors = branch_entry_errors(action, facts)
    if set(errors) & {
        "new_model_branch_generation_required",
        "post_refresh_branch_generation_separate_review_required",
    }:
        return row
    return None


def prepare_saved_revision(handler, context, row, message):
    """Reuse the version-bound modification path; do not approve the old plan."""

    def revise():
        result = handler.review_pending(
            row["plan_id"],
            "modify",
            expected_state_version=row["state_version"],
            expected_proposal_hash=row["proposal_hash"],
            comment=message,
        )
        return result.get("result") or result

    context.scientific_call = revise
    return {"operation": "scientific_workflow"}


def unavailable_review_reply(handler, message, facts):
    """Resolve existing explicit controls before unnecessary result collection."""
    command = message.strip().lower()
    if handler.config_delegate is not None or command not in {"同意", "approve", "拒绝", "reject"}:
        return None
    # These plans are owned by maintenance nodes, which run after the read-only fast path.
    if facts.get("pending_mc_regeneration") or (
        command in {"拒绝", "reject"}
        and any(
            facts.get(key) for key in ("pending_finetune_regeneration", "pending_dft_regeneration")
        )
    ):
        return None
    from phase_agent.runtime.review_requests import pending_reviews
    from phase_agent.tools.workflows.reuse_dft_selection import reusable_selection

    if pending_reviews(facts) or (
        command in {"同意", "approve"} and reusable_selection(facts) is not None
    ):
        return None
    if command in {"同意", "approve"}:
        from phase_agent.tools.state.execution_receipts import recovery_report

        if recovery_report(handler.state_path, facts)["unsettled"]:
            return None
    return "当前没有可批准或拒绝的待审方案；本次未执行任务，也未调用模型重新决策。"


def recovery_reply(handler):
    from phase_agent.runtime.chat_state_presentation import brief_chat_state
    from phase_agent.tools.step_runner.file_protocol import read_json

    state = read_json(handler.state_path, {}) or {}
    url = f"http://127.0.0.1:{os.environ.get('PHASE_CONTROL_PORT', '8765')}/phase/approval"
    return (
        brief_chat_state(state)
        + "\n检测到新的回传或待核对记录；本轮尚未回收、入账或分析。\n"
        "下一步：先核对回传证据；回复‘同意回收’执行回收，或回复‘拒绝回收’保持当前状态。\n"
        "原选点与审批方案保留；上次失败尚需核对实际产物，单说同意不能解除。"
        f"请打开[执行恢复与审批]({url})，在执行恢复条目填写外部作业、产物及成本核对结果，"
        "点击“生成核对处理方案”，再确认该恢复方案。恢复后继续会复用原选点；本次未执行任务。"
    )


def route_saved_control(state, runtime, reviews):
    handler = runtime.context.handler
    if handler.config_delegate is not None:
        return None
    message = state["message"]
    # Explicit controls keep their existing zero-model, identity-bound path.
    if message.strip().lower() in {
        "同意",
        "approve",
        "拒绝",
        "reject",
        "继续",
        "continue",
        "继续原运行",
        "确认敏感操作",
        "confirm sensitive action",
        "配置修订",
        "修改配置",
    }:
        return None
    from phase_agent.tools.workflows.reuse_dft_selection import reusable_selection
    from phase_agent.tools.state.execution_receipts import recovery_report

    facts = state["facts"]
    saved = reusable_selection(facts)
    unsettled = recovery_report(handler.state_path, facts)["unsettled"]
    if not (reviews or saved or unsettled):
        return None
    client = getattr(handler, "workflow_kwargs", {}).get("agent_client")
    if not callable(client):
        return {"reply": recovery_reply(handler)} if unsettled else None
    # Never discard a long instruction and then approve its truncated prefix.
    if len(message) > 2400:
        return {
            "reply": "当前指令过长，无法用快速审批入口完整核对；请缩短为针对当前方案的一条要求。本次未执行。"
        }
    from phase_agent.persistence.memory.conversation import recent_turns

    history = recent_turns(handler.state_path, runtime.context.conversation_id)[-2:]
    rows = [
        {
            "plan_id": r["plan_id"],
            "review_kind": r.get("review_kind"),
            "summary": {
                k: str((r.get("review_card") or {}).get(k) or "")[:300]
                for k in ("title", "scope", "boundary")
            },
        }
        for r in reviews[:4]
    ]
    from phase_agent.tools.policy.branch_conditions import branch_entry_errors, branch_facts

    for row in rows:
        record = (facts.get("pending_execution_policies") or {}).get(row["plan_id"]) or {}
        row["status"] = record.get("status")
        row["task_status"] = (
            (facts.get("effective_decisions") or {}).get(row["plan_id"]) or {}
        ).get("status")
        raw = (record.get("agent_proposal") or {}).get("raw_action") or {}
        row["entry_errors"] = branch_entry_errors(raw, facts) if raw else []
    workflow = branch_facts(facts)
    workflow["new_model_generation"].pop("generated_branch_ids", None)
    workflow["branch_generation_requires_existing_ids"] = False
    workflow["branch_generation_instruction"] = (
        "生成新branch是按已确认的相、组成、超胞边界提出生成区域方案，不是从已有ID选取；"
        "无需用户提供现有Branch ID。封装已有branch的Relax/MC才需要注册ID。"
        "旧方案前置条件不满足时，用户要求生成方案属于修订当前方案，不是批准旧方案。"
    )
    from phase_agent.graphs.dialogue.support import _load_binding

    binding = _load_binding(handler) or {}
    action = (saved or {}).get("raw_action") or {}
    payload = {
        "mode": "saved_work_control",
        "semantic_dialogue": True,
        "bounded_control": True,
        "unified_dialogue": True,
        "output_contracts": {
            "dialogue_outcome": {
                "kind": [
                    "answer",
                    "show_plan",
                    "advance",
                    "configure",
                    "inspect_configuration",
                    "review",
                    "revise",
                ],
                "review": {"plan_id": "actual pending id", "decision": ["approve", "reject"]},
                "answer": "brief factual answer",
                "execution_fields": "forbidden",
            }
        },
        "user_instruction": message,
        "conversation_facts": {
            "recent_turns": [
                {k: str(r.get(k) or "")[:500] for k in ("user", "assistant")} for r in history
            ],
            "pending": rows,
            "additional_plan_count": max(0, len(reviews) - 4),
            "workflow_prerequisites": workflow,
            "saved_selection_available": saved is not None,
            "saved_selection_summary": {
                "tool": action.get("tool"),
                "selected_count": sum(
                    r.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}
                    for r in (action.get("parameters") or {}).get("decisions") or []
                ),
                "reason": str(action.get("reason") or "")[:500],
            },
            "recovery_required": bool(unsettled),
            "presented_plan_binding": {
                k: binding[k] for k in ("plan_id", "state_version", "proposal_hash") if k in binding
            },
        },
        "instruction": (
            "Interpret meaning and references, not keyword matches. Return one JSON object: "
            '{"kind":"answer","answer":"brief factual answer"}, '
            '{"kind":"show_plan"}, {"kind":"advance"}, {"kind":"configure"}, '
            '{"kind":"inspect_configuration"}, '
            '{"kind":"revise","plan_id":"actual pending id or null when only a saved selection exists"}, '
            'or {"kind":"review","plan_id":"actual pending id","decision":"approve|reject"}. '
            "An explicit unambiguous approval/rejection of the displayed stored plan is review; "
            "a request to proceed with the original selection or recover is advance, never approval. "
            "An explicit scientific modification is revise; this requests a revised proposal for "
            "later approval, never executes it. This can replace a blocked proposal with a different tool; "
            "a request for a new branch-generation plan is not a request for existing Branch IDs. "
            "Use workflow_prerequisites and pending entry_errors over historical replies. "
            "A settings modification is configure; viewing current settings is inspect_configuration. "
            "Other questions remain answer. "
            "Recovery_required prevents execution; advance then opens the recovery procedure. "
            "Answer only facts provided; if details are absent state that limit. "
            "Do not invent effects or propose new candidate choices. Memory confers no authority."
        ),
    }

    # Bound input independently of accumulated histories and number of candidates.
    def payload_size():
        return len(json.dumps(payload, ensure_ascii=False, default=str))

    while payload_size() > 8000 and payload["conversation_facts"]["recent_turns"]:
        payload["conversation_facts"]["recent_turns"].pop(0)
    if payload_size() > 8000:
        return {"reply": "当前有多份较长方案，请在审批页指定具体方案后再处理。本次未执行。"}
    from phase_agent.runtime.turn_process import timed_call, process_event
    from phase_agent.graphs.dialogue.decision import _record_usage

    try:
        process_event(
            "理解已有方案的交互意图",
            {
                "最多模型请求": 1,
                "总时限秒": 12,
                "最大输出token": 384,
                "重新选择科学候选": False,
            },
        )
        action = timed_call("saved_work_semantics", client, payload, category="model")
        if not isinstance(action, dict):
            raise ValueError("semantic response must be an object")
        facts = _record_usage(handler, facts, action.get("_llm_usage"))
        if any(
            k in action for k in ("tool", "parameters", "action", "submitted", "steps_executed")
        ):
            raise ValueError("execution fields are not permitted in control intent")
        kind = action.get("kind")
        if kind == "review":
            if action.get("decision") not in {"approve", "reject"}:
                raise ValueError("review must reference an actual pending plan")
            if action.get("plan_id") not in {r["plan_id"] for r in reviews}:
                return {
                    "reply": "审批引用与当前待审方案不一致，本次未执行。请重新查看原方案，核对编号后再审批。"
                }
            return {
                "operation": "review",
                "facts": facts,
                "result": {
                    "review": {
                        "kind": "review",
                        "plan_id": action["plan_id"],
                        "decision": action["decision"],
                    }
                },
            }
        if kind == "configure":
            return {"operation": "configure", "facts": facts}
        if kind == "inspect_configuration":
            return {"operation": "inspect_configuration", "facts": facts}
        if kind == "revise":
            if unsettled:
                return {"reply": recovery_reply(handler)}
            row = next((r for r in reviews if r["plan_id"] == action.get("plan_id")), None)
            if reviews and row is None:
                raise ValueError("revision must reference an actual pending plan")
            if row is not None:
                return prepare_saved_revision(handler, runtime.context, row, message)
            elif saved is not None and action.get("plan_id") is None:
                from phase_agent.graphs.invocation_context import new_invocation

                invocation = new_invocation()

                def revise():
                    return handler._run(invocation, None, message)
            else:
                raise ValueError("no saved proposal to revise")
            runtime.context.scientific_call = revise
            return {"operation": "scientific_workflow"}
        if kind == "answer" and isinstance(action.get("answer"), str) and action["answer"].strip():
            handler.last_turn_outcome = "answered"
            return {"reply": action["answer"][:1600]}
        if kind not in {"advance", "show_plan"}:
            raise ValueError("unsupported saved-work intent")
        blocked = generation_blocked_review(facts, reviews)
        if kind == "advance" and blocked is not None and not unsettled:
            return prepare_saved_revision(handler, runtime.context, blocked, message)
        if reviews:
            from phase_agent.graphs.dialogue.support import bind_presented_proposal
            from phase_agent.runtime.plan_queries import pending_plan_reply

            bind_presented_proposal(handler, facts)
            return {"reply": pending_plan_reply(facts)}
        if unsettled:
            return {"reply": recovery_reply(handler)}
        if kind == "advance" and saved is not None:
            from phase_agent.graphs.dialogue.saved_selection_resume import prepare_resume

            process_event(
                "恢复原选点进入输入准备",
                {
                    "来源记录": saved.get("reuse_source_record_id"),
                    "重新选择科学候选": False,
                },
            )
            prepare_resume(handler, runtime.context)
            return {"operation": "scientific_workflow"}
        return {"reply": "原选点已保留，继续时会恢复为具体待审方案；本次未执行。"}
    except Exception as error:
        # Never turn a failed interpretation into a scientific retry or approval.
        _record_usage(handler, facts, getattr(error, "llm_usage", None))
        return {
            "reply": "本次未取得有效语义判断，未批准、执行或重新选点。请在审批页查看原方案，或稍后重试。"
        }
