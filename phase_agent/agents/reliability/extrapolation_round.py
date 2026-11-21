"""Reliability assessment of round-level extrapolation evidence."""

from __future__ import annotations

import hashlib
import json
import math

from phase_agent.evidence.relaxation_extrapolation_round import assess_round


def _monitor_observation(task, *, round_kind=None):
    """Only a finite final feature check establishes a monitored relaxation."""
    output = task.get("outputs") or {}
    monitor = output.get("extrapolation_monitor") or {}
    score = monitor.get("final_feature_difference")
    failures = monitor.get("failures")
    flag = monitor.get("extrapolation_failure")
    threshold = monitor.get("feature_difference_threshold", 1.0)
    valid = (
        not isinstance(threshold, bool)
        and threshold in (1.0, 3.0)
        and (round_kind != "epoch_refresh" or threshold == 1.0)
        and (not (task.get("parameters") or {}).get("model_refresh_operation") or threshold == 1.0)
        and
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(score)
        and score >= 0
        and isinstance(failures, list)
        and type(flag) is bool
        and flag == bool(failures)
        and (score <= threshold or flag)
    )
    identity = output.get("relaxation_observation") or {}
    if not identity:
        from phase_agent.agents.reliability.legacy_refresh_identity import (
            verified_legacy_refresh_observation,
        )

        identity = verified_legacy_refresh_observation(task) or {}
    if identity:
        valid = (
            valid
            and identity.get("identity_scheme") == "periodic_input_v1"
            and identity.get("monitor_complete") is True
            and identity.get("extrapolation_failure") is flag
        )
    return {
        "structure_id": identity.get("structure_id") or task.get("structure_id"),
        "identity_scheme": identity.get("identity_scheme"),
        "monitor_complete": valid,
        "extrapolation_failure": flag if valid else None,
    }


def assess_recovered_round(tasks, *, model_version, scope_complete, round_kind):
    """Assess only the supplied bound round; unknown work cannot clear the gate."""
    if round_kind not in {"epoch_refresh", "relax_mc"} or type(scope_complete) is not bool:
        raise ValueError("invalid_extrapolation_round_scope")
    rows = []
    complete = scope_complete
    ids = set()
    for task in tasks:
        tid = task.get("task_id")
        if not isinstance(tid, str) or not tid or tid in ids:
            raise ValueError("invalid_or_duplicate_round_task")
        ids.add(tid)
        if task.get("model_version") != model_version:
            raise ValueError("round_model_mismatch")
        stage = task.get("stage")
        output = task.get("outputs") or {}
        if stage == "deep_search":
            if task.get("status") != "completed":
                complete = False
            observations = output.get("relaxation_observations")
            steps = output.get("actual_mc_steps")
            if (
                output.get("relaxation_observations_complete") is not True
                or not isinstance(observations, list)
                or type(steps) is not int
                or steps < 0
                or len(observations) != steps + 1
            ):
                complete = False
                continue
            rows.extend(observations)
        elif stage == "relax_and_feature":
            if task.get("status") not in {"completed", "failed"}:
                complete = False
            followup = output.get("refresh_force_followup") or {}
            if output.get("single_point_completed") is True and followup.get("relaxed") is False:
                continue
            rows.append(_monitor_observation(task, round_kind=round_kind))
    if any(row.get("identity_scheme") != "periodic_input_v1" for row in rows):
        complete = False
    result = assess_round(rows, scope_complete=complete)
    return {
        **result,
        "round_kind": round_kind,
        "task_count": len(ids),
        "scope_complete": complete,
        "threshold": 0.1,
        "reason": "完整逐次弛豫证据"
        if result["status"] == "complete"
        else "实际弛豫范围或监测证据不完整",
    }


def _round_digest(scope_id, model, tasks):
    """Bind a DIRECT decision to the exact recovered monitoring evidence."""
    rows = []
    for task in sorted(tasks, key=lambda item: str(item.get("task_id"))):
        output = task.get("outputs") or {}
        rows.append(
            {
                "task_id": task.get("task_id"),
                "task_key": task.get("task_key"),
                "structure_id": task.get("structure_id"),
                "status": task.get("status"),
                "model_version": task.get("model_version"),
                "stage": task.get("stage"),
                "monitor": output.get("extrapolation_monitor"),
                "relaxation_observation": output.get("relaxation_observation"),
                "observations": output.get("relaxation_observations"),
                "observations_complete": output.get("relaxation_observations_complete"),
                "actual_mc_steps": output.get("actual_mc_steps"),
                "single_point_completed": output.get("single_point_completed"),
                "force_followup": output.get("refresh_force_followup"),
            }
        )
    payload = {"scope_id": scope_id, "model_version": model, "tasks": rows}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def assess_current_rounds(state):
    """Provide bounded reliability facts for the active refresh and returned MC rounds."""
    model = state.get("active_model_version")
    if not model:
        return {"model_version": None, "rounds": []}
    tasks = state.get("tasks") or []
    by_id = {}
    duplicates = set()
    for task in tasks:
        tid = task.get("task_id")
        if tid in by_id:
            duplicates.add(tid)
        by_id[tid] = task
    rounds = []
    refresh = state.get("model_refresh") or {}
    refresh_ids = refresh.get("task_ids") or []
    if refresh.get("new_model_version") == model and refresh_ids:
        complete = (
            len(set(refresh_ids)) == len(refresh_ids)
            and not duplicates.intersection(refresh_ids)
            and all(tid in by_id for tid in refresh_ids)
        )
        bound = [by_id[tid] for tid in refresh_ids if tid in by_id]
        try:
            result = assess_recovered_round(
                bound, model_version=model, scope_complete=complete, round_kind="epoch_refresh"
            )
        except ValueError:
            result = {
                "round_kind": "epoch_refresh",
                "status": "unknown",
                "direct_sampling_required": None,
                "failure_fraction": None,
                "reason": "刷新任务身份或模型版本不一致",
            }
        # Historical endpoint classification is weaker than step monitoring, but
        # an independently verified lower bound can still prove the 10% trigger.
        if result.get("direct_sampling_required") is None and complete:
            from phase_agent.evidence.extrapolation_review import extrapolation_review
            endpoint = extrapolation_review(state)
            if (endpoint.get("status") == "endpoint_classified_incomplete"
                    and endpoint.get("coverage_complete") is True
                    and endpoint.get("task_count") == len(bound)
                    and state.get("endpoint_supplement")):
                identities = {t.get("structure_id") for t in bound}
                import csv
                from pathlib import Path
                with (Path(state["endpoint_supplement"]["directory"]) / "evidence.csv").open(
                        encoding="utf-8-sig", newline="") as stream:
                    supplement_failures = {row["task_id"] for row in csv.DictReader(stream)
                                           if row["classification"] == "extrapolation_failure"}
                failed = {t.get("structure_id") for t in bound
                          if (t.get("extrapolation_review") or {}).get("status") == "extrapolation_failure"
                          or t.get("task_id") in supplement_failures}
                if (None not in identities and all(isinstance(x, str) and x for x in identities)
                        and len(failed) > 0 and len(failed) * 10 >= len(identities)):
                    result = {**result, "status": "endpoint_failure_lower_bound",
                              "failure_count": None, "failure_fraction": None,
                              "direct_sampling_required": True,
                              "endpoint_failure_structure_count": len(failed),
                              "endpoint_structure_count": len(identities),
                              "force_jump_evidence_available": False,
                              "reason": "已核验初末态外推失败数量达到阈值；逐步监测仍不完整"}
        rounds.append(
            {
                "scope_id": "epoch_refresh",
                "evidence_digest": _round_digest("epoch_refresh", model, bound),
                **result,
            }
        )
    for bundle_id, batch in sorted((state.get("relax_mc_batches") or {}).items()):
        if batch.get("model_version") != model or batch.get("status") == "cancelled":
            continue
        bound = [t for t in tasks if t.get("relax_mc_bundle_id") == bundle_id]
        summary = batch.get("result_summary") or {}
        expected = summary.get("task_count")
        complete = (
            batch.get("formal_recovery_status") == "accepted"
            and batch.get("status") == "completed"
            and bool(batch.get("import_checksum"))
            and type(expected) is int
            and expected > 0
            and len(bound) == expected
            and not duplicates.intersection(t.get("task_id") for t in bound)
        )
        try:
            result = assess_recovered_round(
                bound, model_version=model, scope_complete=complete, round_kind="relax_mc"
            )
        except ValueError:
            result = {
                "round_kind": "relax_mc",
                "status": "unknown",
                "direct_sampling_required": None,
                "failure_fraction": None,
                "reason": "Relax–MC 任务身份或模型版本不一致",
            }
        # Relax–MC monitoring remains available to ordinary DFT selection,
        # but only an epoch refresh can require extrapolation recovery.
        result["failure_threshold_reached"] = result.get("direct_sampling_required")
        if result.get("direct_sampling_required") is not None:
            result["direct_sampling_required"] = False
        rounds.append(
            {
                "scope_id": bundle_id,
                "evidence_digest": _round_digest(bundle_id, model, bound),
                **result,
            }
        )
    return {
        "model_version": model,
        "rounds": rounds,
        "instruction": "按实际弛豫结构汇总外推证据；仅模型轮次刷新达到10%时要求恢复采样。Relax–MC的外推值供普通DFT采点参考；未知不等于零。",
    }


def recovery_dft_failure_settled(state, cycle):
    """Permit a new recovery route only after every previously selected DFT task is terminal."""
    from pathlib import Path
    from phase_agent.evidence.extrapolation_recovery import recovery_label_facts
    if recovery_label_facts(state).get("status") == "complete":
        return False
    selected = {r.get("structure_sha256") for r in cycle.get("selected") or []}
    if not selected or None in selected:
        return False
    found = {}
    bindings = {}
    if cycle.get("policy_version") == "epoch_refresh_v1":
        from phase_agent.evidence.extrapolation_recovery import _matches_recovery_frame
        for item in cycle.get("selected") or []:
            bindings.setdefault(item.get("structure_sha256"), set()).add(item.get("candidate_id"))
    for task in state.get("tasks") or []:
        if task.get("stage") not in {"dft_single_point", "dft_relax"}:
            continue
        if bindings and not _matches_recovery_frame(task, bindings, cycle.get("invalid_model_version")):
            continue
        try:
            path = Path(task["structure_path"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except (KeyError, TypeError, OSError):
            continue
        if digest in selected:
            found.setdefault(digest, []).append(task.get("status"))
    terminal = {"completed", "failed", "timeout"}
    if cycle.get("policy_version") != "epoch_refresh_v1":
        terminal.add("cancelled")
    finished = (set(found) == selected
                and all(status in terminal for statuses in found.values() for status in statuses))
    if cycle.get("policy_version") == "epoch_refresh_v1":
        if not finished:
            return False
        records = {row.get("task_id"): row for row in state.get("dft_dataset_records") or []}
        for task in state.get("tasks") or []:
            if (task.get("stage") not in {"dft_single_point", "dft_relax"}
                or task.get("status") != "completed"
                or not _matches_recovery_frame(task, bindings, cycle.get("invalid_model_version"))):
                continue
            try:
                digest = hashlib.sha256(Path(task["structure_path"]).read_bytes()).hexdigest()
            except (KeyError, TypeError, OSError):
                continue
            if digest in selected:
                record = records.get(task.get("task_id")) or {}
                if record.get("status") != "completed" or type(record.get("training_ready")) is not bool:
                    return False
        return True
    return (finished and any(status in {"failed", "timeout"}
                             for statuses in found.values() for status in statuses))


def required_direct_round(state):
    """Return a proven mandatory round unless matching sampling was recorded."""
    facts = assess_current_rounds(state)
    completed = state.get("direct_sampling_results") or {}
    active_cycle = state.get("extrapolation_recovery_cycle") or {}
    if (active_cycle.get("policy_version") == "epoch_refresh_v1"
        and active_cycle.get("invalid_model_version") == facts.get("model_version")
        and active_cycle.get("status") == "awaiting_reviewed_dft_labels"):
        bound = next((r for r in facts["rounds"] if r.get("scope_id") == active_cycle.get("scope_id")), None)
        if not bound:
            return {"scope_id": active_cycle.get("scope_id"), "evidence_digest": None,
                    "failure_fraction": None, "status": "unknown"}
        if bound.get("evidence_digest") != active_cycle.get("round_evidence_digest"):
            return bound
    for row in facts["rounds"]:
        if row.get("direct_sampling_required") is not True:
            continue
        done = completed.get(row["scope_id"]) or {}
        from phase_agent.tools.workflows.direct_extrapolation_sampling import valid_direct_receipt
        cycle = state.get("extrapolation_recovery_cycle") or {}
        if (cycle.get("policy_version") == "epoch_refresh_v1"
            and cycle.get("scope_id") == row["scope_id"]
            and cycle.get("invalid_model_version") == facts["model_version"]):
            from phase_agent.evidence.extrapolation_recovery import recovery_label_facts
            progress = recovery_label_facts(state)
            if progress.get("status") == "complete":
                continue
            if progress.get("status") == "waiting" and recovery_dft_failure_settled(state, cycle):
                return row
        batch = state.get("perturbation_recovery_batch") or {}
        switching = (cycle.get("status") == "awaiting_reviewed_dft_labels"
                     and cycle.get("scope_id") == row["scope_id"]
                     and cycle.get("direct_source_digest") == done.get("source_digest")
                     and valid_direct_receipt(done) and recovery_dft_failure_settled(state, cycle))
        perturbation_pending = (batch.get("scope_id") == row["scope_id"]
                                and batch.get("round_evidence_digest") == row.get("evidence_digest")
                                and batch.get("status") in {"waiting_results", "results_received"}
                                and done.get("recovery_route") != "perturbation")
        if switching or perturbation_pending:
            return row

        if (
            done.get("model_version") != facts["model_version"]
            or done.get("round_evidence_digest") != row.get("evidence_digest")
            or not valid_direct_receipt(done)
            or ((done.get("feature_difference_filter") or {}).get("maximum") != 1.0
                and not ((state.get("extrapolation_recovery_cycle") or {}).get("scope_id") == row["scope_id"]
                         and (state.get("extrapolation_recovery_cycle") or {}).get("direct_source_digest") == done.get("source_digest")))
        ):
            return row
    return None


def unquarantined_direct_round(state):
    """Do not advance a verified DIRECT recovery before its hull is quarantined."""
    from phase_agent.tools.workflows.direct_extrapolation_sampling import valid_direct_receipt

    model = state.get("active_model_version")
    cycle = state.get("extrapolation_recovery_cycle") or {}
    for scope, receipt in (state.get("direct_sampling_results") or {}).items():
        if receipt.get("model_version") != model or not valid_direct_receipt(receipt):
            continue
        if (cycle.get("scope_id") != scope or cycle.get("invalid_model_version") != model
            or cycle.get("direct_source_digest") != receipt.get("source_digest")):
            archived = any(row.get("scope_id") == scope and row.get("invalid_model_version") == model
                           and row.get("direct_source_digest") == receipt.get("source_digest")
                           for row in state.get("extrapolation_recovery_history") or [])
            if not archived:
                return scope
    return None
