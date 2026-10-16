"""Explicit first-model choice for new diagrams with reference DFT labels."""

from copy import deepcopy


def prepare_choice(state, config):
    from phase_agent.persistence.shared_dft import select, capture_snapshot

    rows = select(state, config, available=True)
    sources = [
        {
            "record_id": key,
            "source_diagram": p["source_diagram"],
            "source_task_id": p["source_task_id"],
            "frame_count": len(p["frames"]),
        }
        for key, p in rows
        if p["source_diagram"] != state["shared_dft"]["diagram_id"]
    ]
    return {
        "status": "pending" if sources else "no_reference_data",
        "choice": None,
        "references": sources,
        "shared_dft_snapshot": capture_snapshot(state, config),
    }


def startup_gate(state, config, *, message=None, decision=None):
    gate = state.get("reference_dft_startup")
    if not gate or gate["status"] == "completed":
        return None
    if gate["status"] == "no_reference_data" and state.get("tasks"):
        return None
    if gate["status"] == "no_reference_data" or (
        gate["status"] == "pending" and "shared_dft_snapshot" not in gate
    ):
        fresh = prepare_choice(state, config)
        gate.update(fresh)
        if gate["status"] == "no_reference_data":
            return None
    choices = {
        "先微调": "finetune",
        "先用参考DFT微调": "finetune",
        "直接用原始模型": "foundation",
        "使用原始模型": "foundation",
        "finetune": "finetune",
        "foundation": "foundation",
    }
    chosen = choices.get(str(message or "").strip()) or choices.get(str(decision or "").strip())
    if chosen and gate["status"] == "pending":
        from phase_agent.persistence.shared_dft import accept_snapshot

        accept_snapshot(state, config, gate.get("shared_dft_snapshot"))
        gate["choice"] = chosen
        gate["confirmed"] = True
        gate["confirmed_reply"] = str(message or decision)
        gate["foundation_version"] = (
            state.get("active_model_version")
            or (config.get("mlip") or {}).get("version")
            or (config.get("mlip") or {}).get("name")
        )
        if chosen == "foundation":
            gate["status"] = "completed"
            gate["result"] = "foundation_selected"
            seed_ids = {r["record_id"] for r in gate["references"]}
            state["new_dft_records"] = [
                r
                for r in state.get("new_dft_records", [])
                if r.get("shared_dft_record_id") not in seed_ids
            ]
            return None
        gate["status"] = "awaiting_finetune"
    if gate["status"] == "pending":
        sources = gate["references"]
        return {
            "status": "awaiting_reference_dft_choice",
            "state": state,
            "reference_dft_startup": deepcopy(gate),
            "reason": f"本相图有 {sum(r['frame_count'] for r in sources)} 条参考 DFT，来自 "
            + "、".join(sorted({r["source_diagram"] for r in sources}))
            + "。请回复“先微调”或“直接用原始模型”。确认前不启动搜索。",
            "steps_executed": 0,
            "submitted": False,
        }
    if state.get("active_model_version") and state["active_model_version"] != gate.get(
        "foundation_version"
    ):
        from phase_agent.tools.workflows.model_refresh_state import refresh_active

        if not refresh_active(state):
            gate.update(
                status="completed",
                result="finetuned_model_selected",
                result_model_version=state["active_model_version"],
            )
        return None
    if state.get("remote_finetune_jobs"):
        # Existing training/review/activation approvals must remain reachable.
        return None
    from phase_agent.tools.workflows.prepare_remote_finetune import prepare_remote_finetune

    prepared = prepare_remote_finetune(state, config)
    prepared["state"]["reference_dft_startup"] = deepcopy(gate)
    prepared["reference_dft_startup"] = deepcopy(gate)
    prepared["reason"] = "已确认先用参考 DFT 微调本相图独立模型。\n" + str(
        prepared.get("reason") or ""
    )
    prepared["steps_executed"] = 0
    prepared["submitted"] = False
    return prepared


def blocked_action(state, tool):
    gate = state.get("reference_dft_startup") or {}
    if gate.get("status") == "pending":
        return "请先确认参考 DFT 的启动方式：先微调或直接用原始模型。"
    if gate.get("status") == "awaiting_finetune" and tool not in {
        "update_mlip",
        "adjust_strategy",
        "pause_search",
    }:
        return "已选择先微调；请先完成本相图的训练、模型验证与启用，再开始搜索。"
    return None
