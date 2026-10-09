# Hybrid scientific proposal client

Set deepseek.proposal_harness=hybrid in agent_runtime.json. Routine dialogue, config editing, intent recognition and ordinary scientific selection stay on legacy. Deep Agents is lazily constructed for an explicitly marked complex proposal (analysis_harness=deepagents) or a round-budget proposal with at least two training reports. This structural policy avoids another model call merely to decide which model workflow to use; ordinary large candidate lists do not automatically trigger DA.

Default DA limits: deepagents_max_tokens=4096 per response and deepagents_recursion_limit=8 graph steps. These bound response size/steps, not total billed tokens or exact model calls. Existing _llm_usage retains reported input/output tokens and model call counts; _analysis_route adds selected harness, route reason and elapsed_seconds. Cost stays unknown unless an actual priced receipt exists. Failures propagate to existing caller handling, without automatic hybrid fallback to another paid analysis call.

DA uses invocation-local scratch and supplied reviewed evidence; command execution, delegation, activation and scientific submission are prohibited. Program validation and approval still apply. It is not the long-term memory owner or checkpoint manager.

The current NaFeMn runtime is configured for hybrid with the above bounds; restart the service to use it. A single training report remains on legacy. Tests use fake/injected models, not paid network calls. Actual quality, provider usage and spending must be observed in production before expanding the automatic DA route. Arbitrary read-only multi-report questions currently use routine conversation; this change enables complex scientific proposals, not an unrestricted file-reading chat agent.
