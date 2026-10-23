"""Audited H/D thresholds; explicit manual values take precedence over agent revisions."""

from copy import deepcopy
from datetime import datetime, timezone

FIELDS = {
    "h_cutoff": "hull_shift_mev_per_O2",
    "d_cutoff": "outlier_gap_mev_per_O2",
}


def effective_policy(state, overrides=None):
    from phase_agent.analysis.phase.phase_quality import screening_policy

    saved = (state or {}).get("phase_quality_policy") or {}
    return screening_policy({**(saved.get("overrides") or {}), **(overrides or {})})


def update_quality_policy(state, *, cutoffs, source, reason, expected_revision, release_manual=()):
    from phase_agent.analysis.phase.phase_quality import screening_policy

    if source not in {"human", "agent"} or not isinstance(reason, str) or not reason.strip():
        raise ValueError("阈值调整需要来源和明确原因")
    saved = deepcopy(state.get("phase_quality_policy") or {})
    revision = saved.get("revision", 0)
    if isinstance(expected_revision, bool) or expected_revision != revision:
        raise ValueError("阈值版本已改变，请重新读取")
    if (
        set(cutoffs) - FIELDS.keys()
        or set(release_manual) - FIELDS.keys()
        or not (cutoffs or release_manual)
    ):
        raise ValueError("需要有效的H/D阈值或解除人工固定项")
    manual = set(saved.get("manual_fields") or [])
    if source == "agent" and (release_manual or manual.intersection(cutoffs)):
        raise ValueError("人工固定阈值优先，agent不能覆盖或解除")
    before = effective_policy(state)
    changed = dict(saved.get("overrides") or {})
    changed.update({FIELDS[k]: v for k, v in cutoffs.items()})
    after = screening_policy(changed)
    manual.difference_update(release_manual)
    if source == "human":
        manual.update(cutoffs)
    event = {
        "revision": revision + 1,
        "source": source,
        "reason": reason.strip(),
        "before": before,
        "after": after,
        "manual_fields": sorted(manual),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    saved.update(overrides=changed, revision=revision + 1, manual_fields=sorted(manual))
    saved.setdefault("history", []).append(event)
    current = deepcopy(state)
    current["phase_quality_policy"] = saved
    current.pop("agent_state_summary", None)
    return {
        "status": "policy_updated",
        "state": current,
        "policy": after,
        "submitted": False,
        "scientific_records_modified": False,
    }


def commit_quality_policy(state_path, state, manager, *, ledger_path, **change):
    from phase_agent.analysis.phase.phase_quality_persistence import commit_quality_change

    return commit_quality_change(
        state_path,
        state,
        manager,
        lambda saved, fresh: update_quality_policy(saved, **change),
        ledger_path=ledger_path,
        evidence={"operation": "phase_quality_policy", **change},
    )
