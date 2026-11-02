"""Invocation-local, content-bound evidence shared by all scientific roles."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from phase_agent.evidence.decision_evidence import transport_copy


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Non-serializable dependency in scientific evidence: " + type(value).__name__)


def evidence_snapshot(payload):
    material = transport_copy(
        {
            k: v
            for k, v in payload.items()
            if k
            in {
                "state",
                "decision_context",
                "allowed_tools",
                "user_instruction",
                "user_message",
                "conversation_facts",
                "current_project",
                "instruction",
                "scientific_operations",
                "module_operations",
            }
        }
    )
    state = material.get("state") or {}
    context = material.get("decision_context") or {}
    brief = context.get("scientific_decision_brief") or {}
    project = (
        state.get("current_project")
        or context.get("current_project")
        or material.get("current_project")
        or {}
    )
    identity = {
        "project_id": project.get("project_path")
        or project.get("project")
        or state.get("project_id"),
        "config_version": state.get("confirmed_config_version")
        or state.get("config_version")
        or brief.get("config_version"),
        "model_version": state.get("active_model_version") or brief.get("model_version"),
    }
    refs = set()
    for source in (state, context):
        catalog = source.get("evidence_catalog") or {}
        if isinstance(catalog, dict):
            references = catalog.get("references") or {}
            if isinstance(references, dict):
                refs.update(references)
    encoded = json.dumps(
        {"identity": identity, "facts": material},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_value,
    )
    return {
        **identity,
        "snapshot_id": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "allowed_tools": list(payload.get("allowed_tools") or []),
        "available_evidence_refs": sorted(refs),
        "facts": json.loads(encoded)["facts"],
    }


def role_context(snapshot, role):
    # Preserve shared constraints and cross-stage facts. Existing transport removes
    # atom-level arrays; role-specific focus never hides authoritative restrictions.
    from phase_agent.agents.permissions import role_permissions

    facts = snapshot.get("facts") or {}
    operations = [
        row["operation"]
        for row in [
            *(facts.get("scientific_operations") or []),
            *(facts.get("module_operations") or []),
        ]
    ]
    return {
        "role": role,
        "evidence_snapshot": deepcopy(snapshot),
        "permissions": role_permissions(role, available_operations=operations),
    }
