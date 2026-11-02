"""Keep the next proposal within the recovered stage's approval boundary."""


def post_recovery_tools(state):
    if not (state.get("post_recovery_next_step_pending") or state.get("_recovery_just_approved")):
        return None
    from phase_agent.tools.workflows.model_refresh_state import current_model_refresh

    refresh = current_model_refresh(state)
    if refresh and refresh.get("status") != "completed":
        # A completed model switch establishes a newer stage than old DFT/MC
        # recovery markers. Its own gate still controls exact targets/approval.
        return {"prepare_local_batch_files"}
    from phase_agent.policies.model_generation import new_model_generation_facts

    generation = new_model_generation_facts(state)
    if (
        generation["post_refresh"]
        and generation["ready"]
        and not generation["first_search_completed"]
    ):
        # Current-model branch registration supersedes older recovery markers.
        return {"prepare_local_batch_files", "generate_branches", "select_dft_candidates"}
    from phase_agent.evidence.task_failure_facts import task_failure_facts

    # A recovered run may have an empty/legacy acknowledgement scope even
    # though the current state contains actionable failed scientific tasks.
    # Keep the recovery route available in that case; otherwise the user can
    # only select new candidates and cannot complete the failed batch.
    if task_failure_facts(state).get("actionable_failures"):
        from phase_agent.agents.common.post_dft_review import REVIEW_TOOLS

        tools = set(REVIEW_TOOLS)
        tools.add("restart_failed_task")
        return tools
    stages = set((state.get("recovery_acknowledgement") or {}).get("source_stages") or [])
    dft_recovered = bool(stages & {"dft_single_point", "dft_relax"})
    if not stages:
        # Older confirmations did not record stages; use current model evidence.
        from phase_agent.evidence.post_dft_assessment import post_dft_assessment

        dft_recovered = bool(post_dft_assessment(state, state.get("confirmed_config") or {}))
    if dft_recovered:
        from phase_agent.agents.common.post_dft_review import REVIEW_TOOLS

        tools = set(REVIEW_TOOLS) - {"restart_failed_task"}
        if task_failure_facts(state).get("actionable_failures") or []:
            # Failed scientific attempts are a separate, concrete recovery
            # route.  It targets real task_ids and preserves the normal budget
            # and approval gates; it must not be emulated by selecting the
            # structure again as a new DFT candidate.
            tools.add("restart_failed_task")
        return tools
    return {"select_dft_candidates", "generate_branches"}


def legacy_post_recovery_plan_ids(state):
    """One read-only compatibility check shared by display, review and cleanup."""
    pending = state.get("pending_execution_policies") or {}
    if not state.get("recovery_acknowledgement"):
        return []
    old_dft = any(
        ((row.get("agent_proposal") or {}).get("raw_action") or {}).get("tool")
        == "select_dft_candidates"
        and not (
            ((row.get("agent_proposal") or {}).get("raw_action") or {}).get("parameters") or {}
        ).get("prepare_inputs_after_approval")
        for row in pending.values()
    )
    if state.get("post_recovery_next_step_pending") and not old_dft:
        return []
    from phase_agent.tools.workflows.model_refresh_state import current_model_refresh

    stale = []
    for key, row in pending.items():
        action = (row.get("agent_proposal") or {}).get("raw_action") or {}
        params = action.get("parameters") or {}
        preview = params.get("refresh_preview") or params.get("relax_mc_preview") or {}
        if current_model_refresh(state) and (
            preview.get("new_model_version") or preview.get("model_version")
        ) == state.get("active_model_version"):
            continue
        if (
            (
                action.get("tool")
                in {"prepare_local_batch_files", "prepare_relax_mc_inputs", "select_dft_candidates"}
                and not params.get("prepare_inputs_after_approval")
            )
            or action.get("mode") == "relax_mc_inputs"
            or params.get("mode") == "relax_mc_inputs"
        ):
            stale.append(key)
    return stale
