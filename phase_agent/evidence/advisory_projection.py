"""Small, version-bound advisory inventories with exact on-demand evidence."""

from collections import Counter
from copy import deepcopy
import hashlib
import json

PAGE_THRESHOLD = 128  # Transport only; never limits scientific sampling.
SIZE_THRESHOLD = 256 * 1024  # Transport characters, not a scientific limit.


def _needs_paging(value):
    if not isinstance(value, (list, dict)):
        return False
    if len(value) > PAGE_THRESHOLD:
        return True
    size = 0
    for piece in json.JSONEncoder(ensure_ascii=False, default=str).iterencode(value):
        size += len(piece)
        if size > SIZE_THRESHOLD:
            return True
    return False


def _deferred(value, source_ref):
    descriptor = {
        "format": "deferred_evidence_v1",
        "source_ref": source_ref,
        "count": len(value),
        "retrieval": "phase_team_evidence",
    }
    if source_ref.endswith(".tasks") and isinstance(value, list):
        counts = Counter(
            (str(t.get("stage") or "unknown"), str(t.get("status") or "unknown"))
            for t in value
            if isinstance(t, dict)
        )
        descriptor["stage_status_counts"] = [
            {"stage": stage, "status": status, "count": count}
            for (stage, status), count in sorted(counts.items())
        ]
    return descriptor


def project_advisory_payload(payload, source_identity=None):
    """Replace large collections only in advisory transport; originals stay authoritative."""
    out = dict(payload)
    deferred = []
    for section in ("state", "decision_context"):
        original = payload.get(section)
        if not isinstance(original, dict):
            continue
        value = dict(original)
        out[section] = value
        for field in ("tasks", "budget_reservations", "qbc_candidates", "relax_mc_batches"):
            raw = value.get(field)
            if _needs_paging(raw):
                ref = section + "." + field
                value[field] = _deferred(raw, ref)
                deferred.append(ref)
        refresh = value.get("model_refresh")
        if isinstance(refresh, dict) and isinstance(refresh.get("plan"), dict):
            candidates = refresh["plan"].get("candidates")
            if isinstance(candidates, list) and _needs_paging(candidates):
                refresh = dict(refresh)
                plan = dict(refresh["plan"])
                ref = section + ".model_refresh.plan.candidates"
                plan["candidates"] = _deferred(candidates, ref)
                refresh["plan"] = plan
                value["model_refresh"] = refresh
                deferred.append(ref)
    if deferred:
        if source_identity is None:
            # Synthetic callers without an indexed source still bind every omitted byte.
            digest = hashlib.sha256()
            for ref in sorted(deferred):
                value = payload
                for key in ref.split("."):
                    value = value[key]
                digest.update(
                    json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
                )
            source_identity = {"detached_digest": digest.hexdigest()}
        out["state"] = {
            **out.get("state", {}),
            "advisory_source_identity": deepcopy(source_identity),
        }
        previously_deferred = (out.get("decision_context", {}).get("deferred_evidence") or {}).get(
            "fields"
        ) or []
        out["decision_context"] = {
            **out.get("decision_context", {}),
            "deferred_evidence": {
                "fields": sorted(set(deferred) | set(previously_deferred)),
                "retrieval": "phase_team_evidence",
                "instruction": "Complete original records remain available by page; missing detail is unknown.",
            },
        }
    return out
