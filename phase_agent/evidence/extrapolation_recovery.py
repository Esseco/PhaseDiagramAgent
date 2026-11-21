"""Bind a reviewed DIRECT recovery to the affected refresh and previous hull."""

from copy import deepcopy
import hashlib
from pathlib import Path


def quarantine_refresh(state, scope_id):
    """Return a new state; callers publish and save only through the Agent lock."""
    from phase_agent.tools.workflows.direct_extrapolation_sampling import valid_direct_receipt

    receipt = (state.get("direct_sampling_results") or {}).get(scope_id) or {}
    refresh = state.get("model_refresh") or {}
    model = state.get("active_model_version")
    if not valid_direct_receipt(receipt) or receipt.get("model_version") != model:
        raise ValueError("recovery_direct_receipt_invalid")
    existing = state.get("extrapolation_recovery_cycle")
    lineage_previous = existing
    if (existing and existing.get("status") == "awaiting_new_model_refresh"
        and existing.get("replacement_model_version") == model
        and existing.get("invalid_model_version") != model):
        state = deepcopy(state)
        archived = state.pop("extrapolation_recovery_cycle")
        archived["status"] = "replacement_refresh_failed"
        state.setdefault("extrapolation_recovery_history", []).append(archived)
        existing = None
    if existing:
        if existing.get("status") == "completed" and existing.get("invalid_model_version") != model:
            state = deepcopy(state)
            state.setdefault("extrapolation_recovery_history", []).append(
                state.pop("extrapolation_recovery_cycle")
            )
            existing = None
    if (existing and existing.get("status") == "awaiting_reviewed_dft_labels"
        and existing.get("scope_id") == scope_id
        and existing.get("direct_source_digest") != receipt.get("source_digest")
        and (receipt.get("recovery_route") == "perturbation"
             or existing.get("policy_version") == "epoch_refresh_v1")):
        from phase_agent.agents.reliability.extrapolation_round import recovery_dft_failure_settled
        if not recovery_dft_failure_settled(state, existing):
            raise ValueError("recovery_prior_dft_not_settled")
        state = deepcopy(state)
        archived = state.pop("extrapolation_recovery_cycle")
        archived["status"] = ("superseded_by_next_attempt"
                              if archived.get("policy_version") == "epoch_refresh_v1"
                              else "superseded_by_perturbation")
        state.setdefault("extrapolation_recovery_history", []).append(archived)
        existing = None
    if existing:
        if (
            existing.get("scope_id") == scope_id
            and existing.get("invalid_model_version") == model
            and existing.get("direct_source_digest") == receipt.get("source_digest")
        ):
            result = deepcopy(state)
            if model not in (existing.get("invalidated_model_versions") or []):
                result["extrapolation_recovery_cycle"]["invalidated_model_versions"] = [model]
                return result, {
                    **deepcopy(result["extrapolation_recovery_cycle"]),
                    "status": "quarantine_upgraded",
                }
            baseline = existing.get("baseline_diagram") or {}
            if (baseline.get("status") != "completed"
                or baseline.get("model_version") != existing.get("baseline_model_version")
                or baseline.get("version") != existing.get("baseline_hull_version")):
                raise ValueError("recovery_previous_hull_invalid")
            if (result.get("phase_diagrams") or {}).get("mlip") != baseline:
                result.setdefault("phase_diagrams", {})["mlip"] = deepcopy(baseline)
                return result, {**deepcopy(existing), "status": "quarantine_restored"}
            return result, {**deepcopy(existing), "status": "already_quarantined"}
        raise ValueError("recovery_cycle_already_exists")
    if scope_id == "epoch_refresh":
        scope_tasks = set(refresh.get("task_ids") or [])
    else:
        batch = (state.get("relax_mc_batches") or {}).get(scope_id) or {}
        if batch.get("formal_recovery_status") != "accepted" or batch.get("model_version") != model:
            raise ValueError("recovery_mc_batch_not_verified")
        scope_tasks = {task.get("task_id") for task in state.get("tasks") or []
                       if task.get("relax_mc_bundle_id") == scope_id}
    if refresh.get("new_model_version") != model or not scope_tasks:
        raise ValueError("recovery_refresh_scope_invalid")
    previous = refresh.get("old_diagram") or {}
    current = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if (previous.get("status") != "completed" or not previous.get("version")
        or previous.get("model_version") != refresh.get("old_model_version")):
        raise ValueError("recovery_previous_hull_invalid")
    invalid_hull = current.get("version") if current.get("model_version") == model else None
    if (invalid_hull is None and current.get("model_version") == model
        and current.get("status") == "invalidated"
        and current.get("reason") == "large_extrapolation_failure"
        and scope_id in (current.get("invalidated_rounds") or [])):
        # The invalidation sentinel deliberately has no hull version; bind it
        # to the independently verified published bundle for this same model.
        from phase_agent.analysis.feedback.versioned_products import validate_bundle

        directory = state.get("analysis_output_directory")
        try:
            manifest = validate_bundle(Path(directory) / "latest" / "phase_diagram")
            identity = manifest.get("identity") or {}
            if identity.get("model_version") == model:
                invalid_hull = identity.get("hull_version")
        except (TypeError, OSError, ValueError, KeyError):
            pass
    if invalid_hull is None and (current.get("model_version") != previous.get("model_version")
                                 or current.get("version") != previous.get("version")):
        raise ValueError("recovery_current_hull_unbound")
    selected = receipt.get("selected") or []
    if (
        not selected
        or len(selected) != receipt.get("selected_count")
        or len({r.get("candidate_id") for r in selected}) != len(selected)
        or any(not r.get("structure_sha256") for r in selected)
    ):
        raise ValueError("recovery_selection_invalid")
    tasks = {row.get("task_id"): row for row in state.get("tasks") or []}
    perturbation = receipt.get("recovery_route") == "perturbation"
    if perturbation:
        from phase_agent.tools.workflows.perturbation_direct import perturbation_candidates
        verified, _ = perturbation_candidates(state, state.get("perturbation_recovery_batch") or {},
            scope_id=scope_id, round_digest=receipt.get("round_evidence_digest"))
        verified_frames = {(r["source_task_id"], r["structure_sha256"]) for r in verified}
    for row in selected:
        tid = row.get("source_task_id")
        task = tasks.get(tid) or {}
        try:
            actual = hashlib.sha256(Path(row["structure_path"]).read_bytes()).hexdigest()
        except (KeyError, OSError, TypeError) as exc:
            raise ValueError("recovery_selected_frame_invalid") from exc
        if (
            actual != row["structure_sha256"]
            or (perturbation and (tid, row["structure_sha256"]) not in verified_frames)
            or (not perturbation and (tid not in scope_tasks or task.get("model_version") != model))
        ):
            raise ValueError("recovery_selected_frame_invalid")
    result = deepcopy(state)
    cycle = {
        "scope_id": scope_id,
        "recovery_route": receipt.get("recovery_route", "extrapolation"),
        "recovery_approval_record_id": receipt.get("recovery_approval_record_id"),
        "status": "awaiting_reviewed_dft_labels",
        "invalid_model_version": model,
        "invalid_hull_version": invalid_hull,
        "baseline_model_version": previous["model_version"],
        "baseline_hull_version": previous["version"],
        "baseline_diagram": deepcopy(previous),
        "invalidated_task_ids": sorted(scope_tasks),
        "invalidated_model_versions": [model],
        "direct_source_digest": receipt["source_digest"],
        "selected": [
            {"candidate_id": row["candidate_id"], "structure_sha256": row["structure_sha256"]}
            for row in selected
        ],
    }
    if scope_id == "epoch_refresh" and receipt.get("policy_version") == "epoch_refresh_v1":
        from phase_agent.evidence.extrapolation_policy import recovery_lineage
        from phase_agent.evidence.model_epoch import model_epoch
        cycle["policy_version"] = "epoch_refresh_v1"
        cycle["round_evidence_digest"] = receipt.get("round_evidence_digest")
        cycle.update(recovery_lineage(lineage_previous, model, cycle["round_evidence_digest"], model_epoch(state, model)))
    result["extrapolation_recovery_cycle"] = cycle
    result["model_refresh"]["status"] = "abandoned_extrapolation"
    result["phase_diagrams"]["mlip"] = deepcopy(previous)
    return result, {**cycle, "status": "quarantined"}


def _matches_recovery_frame(task, bindings, model):
    """Match a registered task to its approved candidate, model and exact input."""
    digest = task.get("input_structure_sha256")
    if (task.get("model_version") != model or not digest
        or task.get("object_id") not in bindings.get(digest, set())):
        return False
    try:
        return hashlib.sha256(Path(task["structure_path"]).read_bytes()).hexdigest() == digest
    except (KeyError, OSError, TypeError):
        return False


def _epoch_recovery_label_facts(state, cycle):
    """Read verified labels for distinct frames from this refresh's attempts."""
    history = state.get("extrapolation_recovery_history") or []
    attempts = [row for row in [*history, cycle]
                if row.get("policy_version") == "epoch_refresh_v1"
                and row.get("scope_id") == cycle.get("scope_id")
                and row.get("invalid_model_version") == cycle.get("invalid_model_version")]
    identities = [row.get("direct_source_digest") for row in attempts]
    lineage = cycle.get("recovery_lineage_id")
    all_attempts = ([row for row in [*history, cycle] if row.get("recovery_lineage_id") == lineage]
                    if lineage else attempts)
    all_ids = [row.get("direct_source_digest") for row in all_attempts]
    if (not attempts or not cycle.get("round_evidence_digest")
        or any(not x for x in all_ids) or len(set(all_ids)) != len(all_ids)
        or any(row.get("round_evidence_digest") != cycle.get("round_evidence_digest") for row in attempts)
        or any(not x for x in identities) or len(set(identities)) != len(identities)):
        return {"status": "unknown", "selected_count": 0, "labelled_count": 0, "attempt_count": 0}
    expected = set()
    bindings = {}
    for row in attempts:
        if row.get("recovery_route") == "parent_first_perturbation":
            from phase_agent.evidence.parent_first_recovery import valid_parent_first_cycle
            if not valid_parent_first_cycle({**state, "extrapolation_recovery_cycle": row}):
                return {"status": "unknown", "selected_count": 0, "labelled_count": 0,
                        "attempt_count": len(attempts)}
        selected = row.get("selected") or []
        hashes = [r.get("structure_sha256") for r in selected]
        if (not hashes or any(not h for h in hashes) or len(hashes) != len(set(hashes))
            or any(not isinstance(r.get("candidate_id"), str) or not r["candidate_id"] for r in selected)):
            return {"status": "unknown", "selected_count": 0, "labelled_count": 0,
                    "attempt_count": len(attempts)}
        expected.update(hashes)
        for item in selected:
            bindings.setdefault(item["structure_sha256"], set()).add(item["candidate_id"])
    from phase_agent.science.dft.spin_acceptance import spin_standard_passed
    records = {r.get("task_id"): r for r in state.get("dft_dataset_records") or []}
    labelled = {}
    returned = {}
    failure_reasons = {}
    failed_tasks = set()
    for task in state.get("tasks") or []:
        if (task.get("stage") in {"dft_single_point", "dft_relax"}
            and _matches_recovery_frame(task, bindings, cycle.get("invalid_model_version"))):
            status = task.get("status")
            record = records.get(task.get("task_id")) or {}
            settled = (status in {"failed", "timeout"} or
                       (status == "completed" and record.get("status") == "completed"
                        and type(record.get("training_ready")) is bool))
            returned.setdefault(task["input_structure_sha256"], []).append(settled)
        if (task.get("stage") in {"dft_single_point", "dft_relax"}
            and task.get("status") in {"failed", "timeout"}
            and _matches_recovery_frame(task, bindings, cycle.get("invalid_model_version"))):
            try:
                failed_digest = hashlib.sha256(Path(task["structure_path"]).read_bytes()).hexdigest()
            except (KeyError, OSError, TypeError):
                failed_digest = None
            task_id = task.get("task_id")
            if failed_digest in expected and isinstance(task_id, str) and task_id not in failed_tasks:
                failed_tasks.add(task_id)
                reason = str(task.get("error") or task.get("failure_reason") or task.get("status"))[:200]
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        if (task.get("stage") not in {"dft_single_point", "dft_relax"}
            or task.get("status") != "completed"
            or not _matches_recovery_frame(task, bindings, cycle.get("invalid_model_version"))):
            continue
        record = records.get(task.get("task_id")) or {}
        if not (record.get("status") == "completed" and record.get("converged") is True
                and record.get("checks_passed") is True and record.get("final_frame_valid") is True
                and record.get("training_ready") is True and spin_standard_passed(record)):
            continue
        try:
            digest = hashlib.sha256(Path(task["structure_path"]).read_bytes()).hexdigest()
        except (KeyError, OSError, TypeError):
            continue
        if digest in expected and isinstance(task.get("task_id"), str) and task["task_id"]:
            labelled[digest] = task["task_id"]
    threshold_met = len(labelled) >= 50
    all_returned = set(returned) == expected and all(all(rows) for rows in returned.values())
    return {"status": "complete" if threshold_met and all_returned else "waiting",
            "threshold_met": threshold_met, "all_returned": all_returned,
            "selected_count": len(expected), "labelled_count": len(labelled),
            "attempt_count": len(all_attempts), "round_attempt_count": len(attempts),
            "label_task_ids": sorted(labelled.values()),
            "failed_count": len(failed_tasks), "failure_groups": [
                {"reason": reason, "count": count} for reason, count in sorted(failure_reasons.items())]}


def recovery_label_facts(state):
    """Require completed, trainable DFT labels bound to every selected input frame."""
    cycle = state.get("extrapolation_recovery_cycle") or {}
    if cycle.get("policy_version") == "epoch_refresh_v1":
        return _epoch_recovery_label_facts(state, cycle)
    selected = cycle.get("selected") or []
    expected = {row.get("structure_sha256") for row in selected}
    if not selected or None in expected or len(expected) != len(selected):
        return {"status": "unknown", "selected_count": len(selected), "labelled_count": 0}
    selected_count = len(expected)
    excluded_failed = set()
    if cycle.get('recovery_route') == 'parent_first_perturbation':
        from phase_agent.evidence.parent_first_recovery import valid_parent_first_cycle
        if not valid_parent_first_cycle(state):
            return {'status':'unknown', 'selected_count':selected_count, 'labelled_count':0}
        batch = state['parent_first_recovery_batches'][cycle['batch_id']]
        task_map = {t.get('task_id'):t for t in state.get('tasks') or []}
        group = [task_map[i] for i in batch['dft_task_ids']]
        if all(t.get('status') in {'completed','failed','timeout'} for t in group):
            failures = [t for t in group if t.get('status') in {'failed','timeout'}]
            if len(failures) * 5 < selected_count:
                excluded_failed = {t['input_structure_sha256'] for t in failures}
                expected -= excluded_failed
    records = {row.get("task_id"): row for row in state.get("dft_dataset_records") or []}
    labelled = set()
    label_tasks = {}
    from phase_agent.science.dft.spin_acceptance import spin_standard_passed

    for task in state.get("tasks") or []:
        if (
            task.get("stage") not in {"dft_single_point", "dft_relax"}
            or task.get("status") != "completed"
        ):
            continue
        record = records.get(task.get("task_id")) or {}
        if not (
            record.get("status") == "completed"
            and record.get("converged") is True
            and record.get("checks_passed") is True
            and record.get("final_frame_valid") is True
            and record.get("training_ready") is True
            and spin_standard_passed(record)
        ):
            continue
        try:
            digest = hashlib.sha256(Path(task["structure_path"]).read_bytes()).hexdigest()
        except (KeyError, OSError, TypeError):
            continue
        if digest in expected and isinstance(task.get("task_id"), str) and task["task_id"]:
            labelled.add(digest)
            label_tasks[digest] = task.get("task_id")
    return {
        "status": "complete" if labelled == expected else "waiting",
        "selected_count": selected_count,
        "labelled_count": len(labelled),
        "label_task_ids": sorted(label_tasks.values()),
        **({'failed_excluded_count':len(excluded_failed)} if cycle.get('recovery_route') == 'parent_first_perturbation' else {}),
    }


def epoch_recovery_policy(state, round_facts, *, legacy=False):
    """Apply the pure policy to bound refresh facts and read-only label progress."""
    from phase_agent.agents.reliability.extrapolation_round import recovery_dft_failure_settled
    from phase_agent.evidence.extrapolation_policy import assess_policy, model_update_index
    model = state.get("active_model_version")
    update_index = model_update_index(state, model)
    cycle = state.get("extrapolation_recovery_cycle") or {}
    progress = {}
    switch = False
    if (cycle.get("policy_version") == "epoch_refresh_v1"
        and cycle.get("scope_id") == round_facts.get("scope_id")
        and cycle.get("invalid_model_version") == model):
        if cycle.get("round_evidence_digest") != round_facts.get("evidence_digest"):
            return {"status": "unknown", "route": None}
        progress = recovery_label_facts(state)
        if progress.get("status") == "unknown":
            return {"status": "unknown", "route": None}
        settled = recovery_dft_failure_settled(state, cycle)
        if progress.get("status") != "complete" and not settled:
            return {"status": "waiting_results", "route": None}
        # Once a completed attempt is insufficient, continue with perturbations.
        switch = settled
    elif (cycle.get("policy_version") == "epoch_refresh_v1"
          and cycle.get("status") == "awaiting_new_model_refresh"
          and cycle.get("replacement_model_version") == model):
        previous_progress = recovery_label_facts(state)
        if previous_progress.get("status") != "complete":
            return {"status": "unknown", "route": None}
        progress = {"attempt_count": previous_progress.get("attempt_count", 0), "labelled_count": 0}
    fraction = round_facts.get("failure_fraction")
    if legacy and fraction is None and round_facts.get("status") == "endpoint_failure_lower_bound":
        n = round_facts.get("endpoint_structure_count")
        failed = round_facts.get("endpoint_failure_structure_count")
        if type(n) is int and n > 0 and type(failed) is int and 0 <= failed <= n:
            lower = failed / n
            if (update_index is not None and update_index < 2) or lower >= 0.2:
                fraction = lower
    return assess_policy(update_index, fraction,
                         progress.get("attempt_count", 0), progress.get("labelled_count", 0), switch, legacy=legacy)


def recovery_model_matches_labels(state, candidate_version):
    """Require the candidate's recorded training scope to include each recovered DFT label."""
    cycle = state.get("extrapolation_recovery_cycle") or {}
    labels = recovery_label_facts(state)
    if labels.get("status") != "complete" or not candidate_version:
        return False
    required = set(labels.get("label_task_ids") or [])
    for job in (state.get("remote_finetune_jobs") or {}).values():
        handoff = job.get("training_handoff") or {}
        if (job.get("original_model_version") != cycle.get("invalid_model_version")
            or handoff.get("candidate_model_version") != candidate_version
            or job.get("status") != "results_received"):
            continue
        ids = ((job.get("report") or {}).get("dataset_scope") or {}).get("data_ids") or []
        if all(any(str(value).startswith(task_id + ":frame:") for value in ids) for task_id in required):
            return True
    return False


def recovery_context(state):
    cycle = state.get("extrapolation_recovery_cycle") or {}
    if not cycle:
        return None
    return {"status": cycle.get("status"), "scope_id": cycle.get("scope_id"),
            "invalid_model_version": cycle.get("invalid_model_version"),
            "invalid_hull_version": cycle.get("invalid_hull_version"),
            "baseline_model_version": cycle.get("baseline_model_version"),
            "baseline_hull_version": cycle.get("baseline_hull_version"),
            "replacement_model_version": cycle.get("replacement_model_version"),
            "origin_epoch": cycle.get("origin_epoch"),
            "extrapolation_round": cycle.get("extrapolation_round"),
            "selected_candidate_ids": [row.get("candidate_id") for row in cycle.get("selected") or []],
            "labels": recovery_label_facts(state),
            "instruction": ("本 epoch 最多补充5次；累计达到50个合格去重结构且所有DFT任务已回收后进入微调。"
                            "失败任务只汇总、不自动重试；被作废轮次不能进入相图。"
                            if cycle.get("policy_version") == "epoch_refresh_v1" else
                            "DIRECT 候选需审核并取得全部合格 DFT 能量和受力标签后微调；被作废轮次不能进入相图，替代模型从上一有效相图刷新。") }
