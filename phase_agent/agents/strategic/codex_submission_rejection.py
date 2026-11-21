"""Verified checkpoint proof for a proposal rejected before scientific execution."""

import hashlib
import json
from contextlib import closing
from pathlib import Path
import sqlite3

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from phase_agent.persistence.compact_checkpoints import CompressedSerde
from phase_agent.persistence.sqlite_paths import readonly_uri


def _checkpoint_project_bound(frame, path, action):
    saved = frame.get("runtime_state_path")
    if saved:
        return Path(saved).resolve() == path
    # Older compact terminal frames omit runtime_state_path.  Accept their
    # inline project paths only for branch validation, with both frozen
    # configuration and model identities, and retain all other proof gates.
    if action.get("tool") != "generate_branches":
        return False
    fields = (frame.get("loaded_state") or {}).get("__phase_state_fields_v1__") or {}
    def inline(name):
        value = fields.get(name)
        return value.get("inline") if isinstance(value, dict) else None
    ledger = inline("branch_energy_pool_ledger_path")
    output = inline("output_index_path")
    decision = action.get("scientific_decision") or {}
    return bool(
        ledger and output and decision.get("config_version") and decision.get("model_version")
        and Path(ledger).resolve() == (path.parent / "ledgers" / "branch_energy_pools.json").resolve()
        and Path(output).resolve() == (path.parent / "analysis" / "output_index.md").resolve()
        and inline("confirmed_config_version") == decision["config_version"]
        and inline("active_model_version") == decision["model_version"]
    )


def rejection_checkpoint(
    state_path, session, *, checkpoint_id, request_id, checkpoint_sha256, include_frame=False
):
    submission = session.get("submission") or {}
    reason = submission.get("reason") or ""
    marker = "llm_failed: ValueError: Invalid budget trade-off: "
    from phase_agent.agents.strategic.round_budget_review import round_budget_review_errors

    errors = round_budget_review_errors(
        session["draft"]["action"], {"round_budget_evidence": {"required": True}}
    )
    proven_reason = any(marker + error in reason for error in errors)
    if session["draft"]["action"].get("scientific_decision") is None:
        from phase_agent.agents.strategic.scientific_decisions import scientific_decision_errors

        schema_errors = scientific_decision_errors(session["draft"]["action"], {}, {})
        proven_reason = proven_reason or any(
            "llm_failed: ValueError: Invalid scientific decision: " + error in reason
            for error in schema_errors
        )
    # Only reconcile this exact branch-review rejection when the unknown
    # identifiers were actually cited by the frozen action. Checkpoint and
    # execution-receipt verification below remain mandatory.
    marker = "branch_review_invalid_after_one_correction: branch_pool_review.evidence_refs: unknown source "
    if marker in reason:
        import ast
        import re

        match = re.search(re.escape(marker) + r"(\[[^\]]*\])", reason)
        action = session["draft"]["action"]
        cited = (action.get("branch_pool_review") or {}).get("evidence_refs") or []
        try:
            unknown = ast.literal_eval(match.group(1)) if match else None
        except (ValueError, SyntaxError):
            unknown = None
        proven_reason = proven_reason or (
            action.get("tool") == "generate_branches"
            and isinstance(unknown, list)
            and bool(unknown)
            and all(
                isinstance(ref, str)
                and ref.startswith(("branch_pool:", "phase_diagram:mlip_active_model:", "phase_diagram:mlip:"))
                and ref in cited
                for ref in unknown
            )
        )
    # A transport-only branch-index expansion can fail the simulation intent
    # check before an approval or scientific dispatch exists. Prove the exact
    # failure and bound plan; the checkpoint and receipt checks remain mandatory.
    simulation_rejection = (
        submission.get("status") == "rejected"
        and reason
        == "Simulation planning: simulation_intent_changed; request new planning and approval"
    )
    if simulation_rejection:
        action = session["draft"]["action"]
        intent = action.get("_simulation_intent") or {}
        groups = (intent.get("branch_pool_review") or {}).get("existing_branch_groups") or []
        branch_index_case = action.get("tool") == "generate_branches" and any(
            isinstance(group, dict) and isinstance(group.get("branch_indices"), list)
            for group in groups
        )
        review = intent.get("branch_pool_review") or {}
        default_case = (
            action.get("tool") == "generate_branches"
            and review.get("default_existing_branch_decision")
            in {"defer", "continue_search", "relax_only"}
            and isinstance(review.get("default_existing_branch_reason"), str)
            and bool(review["default_existing_branch_reason"].strip())
            and isinstance(review.get("default_existing_branch_evidence_refs"), list)
            and bool(review["default_existing_branch_evidence_refs"])
            and all(
                isinstance(ref, str) and ref.strip()
                for ref in review["default_existing_branch_evidence_refs"]
            )
        )
        dft_preview_case = (
            action.get("tool") == "select_dft_candidates"
            and intent.get("tool") == "select_dft_candidates"
            and isinstance((intent.get("parameters") or {}).get("decisions"), list)
            and bool(intent["parameters"]["decisions"])
        )
        relax_preview_case = (
            action.get("tool") == "prepare_local_batch_files"
            and action.get("operation") == "prepare_relax_mc_batch"
            and intent.get("tool") == action.get("tool")
            and intent.get("operation") == action.get("operation")
            and (intent.get("parameters") or {}).get("mode") == "relax_mc_inputs"
            and bool(((intent.get("parameters") or {}).get("relax_mc_preview") or {}).get("checksum"))
            and bool(intent.get("target_ids"))
        )
        if not (branch_index_case or default_case or dft_preview_case or relax_preview_case):
            raise ValueError("submission_has_no_proven_proposal_validation_failure")
        from phase_agent.agents.simulation.agent import validate_plan

        validate_plan(action.get("_simulation_plan"), intent)
        proven_reason = True
    # The DFT admission validator rejects before creating any scientific action.
    # Check the specific validator markers; the bound terminal checkpoint and
    # absence of execution receipts remain mandatory below.
    dft_rejection = (
        submission.get("status") == "rejected"
        and session["draft"]["action"].get("tool") == "select_dft_candidates"
        and reason.startswith(("DFT 身份或D/H审阅未通过:", "DFT 身份或D/H审阅未通过："))
        and "dft_DH_group_requires_selection_or_reasoned_deferral:" in reason
        and "independent_dft_audit_required" in reason
    )
    # GS admission rejection is safe to reconcile only for the exact validator
    # error form; the checkpoint and execution receipt still prove zero execution.
    import ast

    gs_prefix = "DFT 方案未通过筛选/预算校验，需修订后再批准："
    gs_errors = None
    if reason.startswith(gs_prefix):
        try:
            gs_errors = ast.literal_eval(reason[len(gs_prefix) :])
        except (ValueError, SyntaxError):
            pass
    # The independent-audit admission failure is also pre-dispatch.  The exact
    # terminal checkpoint and absence of receipts are still checked below.
    dft_audit_rejection = (
        submission.get("status") == "rejected"
        and session["draft"]["action"].get("tool") == "select_dft_candidates"
        and gs_errors == ["independent_dft_audit_required"]
    )
    dft_gs_rejection = (
        submission.get("status") == "rejected"
        and session["draft"]["action"].get("tool") == "select_dft_candidates"
        and isinstance(gs_errors, list)
        and bool(gs_errors)
        and all(
            isinstance(item, str) and item.startswith("ground_state_selection_required:")
            for item in gs_errors
        )
    )
    # The old advisory stage counter can reject every item before dispatch.
    # Prove that the errors cover exactly the frozen DFT selection.
    action = session["draft"]["action"]
    decisions = (action.get("parameters") or {}).get("decisions") or []
    selected = {row.get("candidate_id") for row in decisions
                if row.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}}
    dft_stage_rejection = (
        submission.get("status") == "rejected"
        and action.get("tool") == "select_dft_candidates"
        and isinstance(gs_errors, list) and bool(gs_errors)
        and len(gs_errors) == len(selected)
        and all(isinstance(item, dict) and item.get("reason") == "dft_budget"
                and item.get("details") == ["stage_task_limit"] for item in gs_errors)
        and {item.get("candidate_id") for item in gs_errors} == selected
    )
    # A frozen DFT proposal can be rejected when legacy post-DFT filtering
    # hides selected or explicitly deferred frames. Only recognize the exact
    # unknown-candidate response; terminal checkpoint and receipt gates follow.
    action = session["draft"]["action"]
    decision_rows = (action.get("parameters") or {}).get("decisions") or []
    frozen_ids = {row.get("candidate_id") for row in decision_rows if isinstance(row, dict)}
    selected_ids = {row.get("candidate_id") for row in decision_rows
                    if isinstance(row, dict) and row.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}}
    identity_prefix = "DFT 身份或D/H审阅未通过："
    identity_details = reason[len(identity_prefix):].split("；", 1)[0] if reason.startswith(identity_prefix) else ""
    identity_items = identity_details.split("; ") if identity_details else []
    missing_ids = {item.removeprefix("unknown_candidate:") for item in identity_items
                   if item.startswith("unknown_candidate:")}
    dft_frozen_pool_rejection = (
        submission.get("status") == "rejected"
        and action.get("tool") == "select_dft_candidates"
        and action.get("_simulation_plan") is not None
        and identity_items and len(missing_ids) == len(identity_items)
        and missing_ids <= frozen_ids and bool(missing_ids & selected_ids)
        and "；DFT 方案未通过筛选/预算校验，需修订后再批准：" in reason
    )
    if dft_frozen_pool_rejection:
        from phase_agent.agents.simulation.agent import validate_plan

        validate_plan(action["_simulation_plan"], action.get("_simulation_intent") or action)
    proven_reason = (proven_reason or dft_rejection or dft_audit_rejection or dft_gs_rejection
                     or dft_stage_rejection or dft_frozen_pool_rejection)
    # A structured DIRECT request could be stopped at the refresh gate when a
    # client adapter dropped its explicit scope. The same terminal checkpoint,
    # exact response, invocation binding and no-receipt checks below still apply.
    action = session["draft"]["action"]
    direct_refresh_gate = (
        submission.get("status") == "not_configured"
        and action.get("operation") == "sample_direct_extrapolation"
        and action.get("tool") == "prepare_local_batch_files"
        and (action.get("parameters") or {}).get("mode") == "direct_extrapolation"
        and (action.get("parameters") or {}).get("scope_id") == "epoch_refresh"
        and reason.startswith("刷新第")
        and "refresh_relaxation_not_converged" in reason
    )
    direct_policy_rejection = (
        submission.get("status") == "rejected"
        and action.get("operation") == "sample_direct_extrapolation"
        and action.get("tool") == "prepare_local_batch_files"
        and (action.get("parameters") or {}).get("mode") == "direct_extrapolation"
        and (action.get("parameters") or {}).get("scope_id") == "epoch_refresh"
        and reason == "方案未通过执行约束：new_model_structure_refresh_required"
    )
    # Legacy recovery gating rejected this registered-input action before dispatch.
    # The terminal checkpoint, project identity and absent receipt remain required.
    program = action.get("_simulation_program") or {}
    targets = action.get("target_ids") or []
    dft_input_recovery_rejection = (
        submission.get("status") == "rejected"
        and reason == "方案未通过执行约束：post_recovery_only_exact_candidate_relax_allowed"
        and action.get("tool") == "prepare_local_batch_files"
        and action.get("operation") == "prepare_dft_inputs"
        and (action.get("parameters") or {}) == {"mode": "dft_inputs"}
        and isinstance(targets, list) and bool(targets)
        and all(isinstance(item, str) and item for item in targets)
        and len(set(targets)) == len(targets)
        and program.get("protocol") == "registered_dft_inputs/v1"
        and program.get("status") == "passed"
        and program.get("target_ids") == targets
    )
    expected_tasks = (action.get("parameters") or {}).get("expected_task_ids")
    scoped_mc_preparation = (
        action.get("tool") == "prepare_local_batch_files"
        and action.get("operation") == "prepare_mlip_mc"
        and (action.get("parameters") or {}).get("mode") == "mc_inputs"
        and isinstance(expected_tasks, list)
        and bool(expected_tasks)
        and all(isinstance(value, str) and value for value in expected_tasks)
        and len(set(expected_tasks)) == len(expected_tasks)
    )
    mc_task_limit_rejection = (
        submission.get("status") == "rejected"
        and (action.get("tool") == "allocate_mc_bohb" or scoped_mc_preparation)
        and reason == "方案未通过执行约束：stage_task_limit"
    )
    # Older proposal adapters rejected all multi-target retries before dispatch.
    # The exact terminal checkpoint and missing execution receipt are still required.
    retry_targets = action.get("target_ids") or []
    mc_retry_guard_rejection = (
        submission.get("status") == "not_configured"
        and action.get("tool") == "restart_failed_task"
        and (action.get("parameters") or {}).get("repair_profile") == "mc_threshold_recovery"
        and len(retry_targets) > 1
        and all(isinstance(target, str) and target for target in retry_targets)
        and len(set(retry_targets)) == len(retry_targets)
        and reason.endswith(
            "llm_failed: ValueError: restart_failed_task requires a retryable scientific task_id"
        )
    )
    proven_reason = (
        proven_reason
        or direct_refresh_gate
        or direct_policy_rejection
        or dft_input_recovery_rejection
        or mc_task_limit_rejection
        or mc_retry_guard_rejection
    )
    if not proven_reason or submission.get("status") not in (
        {"not_configured", "rejected"}
        if simulation_rejection
        or dft_rejection
        or dft_gs_rejection
        or dft_audit_rejection
        or dft_stage_rejection
        or dft_frozen_pool_rejection
        or direct_policy_rejection
        or dft_input_recovery_rejection
        or mc_task_limit_rejection
        else {"not_configured"}
    ):
        raise ValueError("submission_has_no_proven_proposal_validation_failure")
    expected_status = (
        "rejected"
        if simulation_rejection
        or dft_rejection
        or dft_gs_rejection
        or dft_audit_rejection
        or dft_stage_rejection
        or dft_frozen_pool_rejection
        or direct_policy_rejection
        or dft_input_recovery_rejection
        or mc_task_limit_rejection
        else "not_configured"
    )
    path = Path(state_path).resolve()
    with closing(
        sqlite3.connect(readonly_uri(path.parent / "langgraph_lifecycle.sqlite"), uri=True)
    ) as conn:
        row = conn.execute(
            "SELECT thread_id,checkpoint_id,type,checkpoint FROM checkpoints "
            "WHERE checkpoint_ns='' ORDER BY checkpoint_id DESC LIMIT 1"
        ).fetchone()
        if (
            not row
            or row[1] != checkpoint_id
            or hashlib.sha256(row[3]).hexdigest() != checkpoint_sha256
        ):
            raise ValueError("submission_checkpoint_changed")
        payload = CompressedSerde(JsonPlusSerializer()).loads_typed(
            (row[2].removeprefix("phase-cas-v1:"), row[3])
        )
    values = payload.get("channel_values") or {}
    response = values.get("completed_response") or {}
    frame = values.get("business_frame") or {}
    if (
        values.get("request_id") != request_id
        or (session.get("dispatch_request_id") not in {None, request_id})
        or values.get("phase") != "finalized"
        or values.get("current_node") != "prepare_inputs_and_finalize"
        or values.get("node_status") != "completed"
        or not _checkpoint_project_bound(frame, path, session["draft"]["action"])
        or response.get("status") != expected_status
        or response.get("submitted") is not False
        or response.get("reason") != reason
        or any(
            response.get(k)
            for k in ("record_id", "agent_proposal", "execution", "execution_result")
        )
    ):
        raise ValueError("submission_checkpoint_does_not_prove_zero_execution")
    events = response.get("events")
    if (
        not isinstance(events, list)
        or len(events) != 1
        or events[0].get("status") != expected_status
        or events[0].get("reason") != reason
    ):
        raise ValueError("submission_checkpoint_has_other_events")
    receipt_path = path.parent / "execution_receipts.sqlite"
    if receipt_path.exists():
        with closing(sqlite3.connect(readonly_uri(receipt_path), uri=True)) as conn:
            if any(
                json.loads(identity).get("invocation_id") == request_id
                for (identity,) in conn.execute("SELECT identity FROM receipts")
            ):
                raise ValueError("submission_has_execution_receipt")
    proof = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "request_id": request_id,
        "thread_id": row[0],
        "classification": ("dft_input_recovery_gate_rejected" if dft_input_recovery_rejection
                           else "dft_legacy_stage_limit_rejected" if dft_stage_rejection
                           else "proposal_validation_rejected"),
        "executed": False,
    }
    return (proof, frame) if include_frame else proof
