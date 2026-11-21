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


class _LegacySnapshotKeys(Exception):
    pass


def _snapshot_copy(value):
    """Detach ordinary JSON trees once; normalize Paths/tuples as JSON would."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_snapshot_copy(item) for item in value]
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise _LegacySnapshotKeys
        return {key: _snapshot_copy(item) for key, item in value.items()}
    return json.loads(json.dumps(value, allow_nan=False, default=_json_value))


def _encode(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_value,
    )


def evidence_snapshot(payload):
    source = {
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
    legacy_keys = False
    try:
        material = transport_copy(source, copy_value=_snapshot_copy)
    except _LegacySnapshotKeys:
        # Keep the original ordering/hash semantics for non-string historical keys.
        material = transport_copy(source)
        legacy_keys = True
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
    digest = hashlib.sha256()
    if legacy_keys:
        encoded = _encode({"identity": identity, "facts": material})
        digest.update(encoded.encode("utf-8"))
        material = json.loads(encoded)["facts"]
    else:
        # Canonical bytes are identical to the old whole-object encoding.
        # Use the C encoder one fact field at a time, not a Python float iterator.
        digest.update(b'{"facts":{')
        for index, key in enumerate(sorted(material)):
            if index:
                digest.update(b",")
            digest.update(_encode(key).encode("utf-8"))
            digest.update(b":")
            digest.update(_encode(material[key]).encode("utf-8"))
        digest.update(b'},"identity":')
        digest.update(_encode(identity).encode("utf-8"))
        digest.update(b"}")
    return {
        **identity,
        "snapshot_id": digest.hexdigest(),
        "allowed_tools": list(payload.get("allowed_tools") or []),
        "available_evidence_refs": sorted(refs),
        "facts": material,
    }


# Only role-specific protocols are omitted. Measured pools, limitations, unknown
# fields and cross-stage risks remain shared; no row/character budget truncates them.
ROLE_PROTOCOLS = {
    "exploration": {"generation_region_contract", "branch_pool_review_contract"},
    "reliability": {"dft_selection_policy.reference_framework"},
}


def _same_fact(left, right):
    # Preserve JSON types and signed zero without encoding entire evidence twice.
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_fact(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_fact(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return left.hex() == right.hex()
    return left == right


def role_context(snapshot, role):
    from phase_agent.agents.permissions import role_permissions

    projected = dict(snapshot)
    facts = dict(snapshot.get("facts") or {})
    state = dict(facts.get("state") or {})
    context = dict(facts.get("decision_context") or {})
    projected["facts"] = facts
    if "state" in facts:
        facts["state"] = state
    if "decision_context" in facts:
        facts["decision_context"] = context
    omitted = []
    duplicates = {}
    # Keep differing versions visible; equality is the only deduplication rule.
    nested = state.get("decision_context")
    if isinstance(nested, dict):
        nested = dict(nested)
        state["decision_context"] = nested
        for key in list(nested):
            if key in context and _same_fact(nested[key], context[key]):
                duplicates["state.decision_context." + key] = "decision_context." + key
                del nested[key]
        if not nested:
            del state["decision_context"]
    for key in list(state):
        if key in context and _same_fact(state[key], context[key]):
            duplicates["state." + key] = "decision_context." + key
            del state[key]
    # Existing summary aliases and reviewed memory share one transmitted value.
    for section_name, section in (("state", state), ("decision_context", context)):
        aliases = {
            "uncertainty": "qbc_uncertainty",
            "search_history": "recent_action_history",
        }
        for alias, canonical in aliases.items():
            if (
                alias in section
                and canonical in section
                and _same_fact(section[alias], section[canonical])
            ):
                duplicates[section_name + "." + alias] = section_name + "." + canonical
                del section[alias]
        memory = section.get("memory_context") or {}
        for alias, canonical in (
            ("relevant_approved_knowledge", "approved_knowledge"),
            ("short_term_memory", "short_term"),
        ):
            if (
                alias in section
                and canonical in memory
                and _same_fact(section[alias], memory[canonical])
            ):
                duplicates[section_name + "." + alias] = (
                    section_name + ".memory_context." + canonical
                )
                del section[alias]
    other_protocols = (
        set().union(*(fields for owner, fields in ROLE_PROTOCOLS.items() if owner != role))
        if role in ROLE_PROTOCOLS
        else set()
    )
    for section_name, section in (("state", state), ("decision_context", context)):
        for field in sorted(other_protocols):
            container = section
            parts = field.split(".")
            for part in parts[:-1]:
                child = container.get(part) if isinstance(container, dict) else None
                if isinstance(child, dict):
                    child = dict(child)
                    container[part] = child
                container = child
            if isinstance(container, dict) and parts[-1] in container:
                omitted.append(section_name + "." + field)
                del container[parts[-1]]
    omitted_operations = []
    if role in ROLE_PROTOCOLS:
        from phase_agent.agents.permissions import ADVISORY_OPERATIONS

        for field in ("scientific_operations", "module_operations"):
            if field in facts:
                retained = []
                for row in facts[field]:
                    if (
                        row["operation"] in ADVISORY_OPERATIONS[role]
                        or row.get("tool") == "pause_search"
                        or (role == "exploration" and row.get("tool") == "adjust_strategy")
                    ):
                        retained.append(row)
                    else:
                        omitted_operations.append(row["operation"])
                facts[field] = retained
    operations = [
        row["operation"]
        for row in [
            *(facts.get("scientific_operations") or []),
            *(facts.get("module_operations") or []),
        ]
    ]
    from phase_agent.evidence.requests import TOPICS

    focus_topics = {"mc", "convergence"} if role == "exploration" else {"dft", "model"}
    focus = set().union(*(TOPICS[topic] for topic in focus_topics))
    projected["context_projection"] = {
        "role_evidence_fields": [
            section_name + "." + key
            for section_name, section in (("state", state), ("decision_context", context))
            for key in sorted(section.keys() & focus)
        ]
        if role in ROLE_PROTOCOLS
        else [],
        "role": role,
        "source_snapshot_id": snapshot["snapshot_id"],
        "duplicate_fields": duplicates,
        "omitted_protocol_fields": omitted,
        "omitted_operation_contracts": sorted(set(omitted_operations)),
        "rule": "Identity, constraints, budgets, approvals, shared reviewed memory and scientific evidence remain authoritative. Only listed duplicate values, other-role protocols and unavailable specialist operation contracts are omitted. Missing evidence is unknown, never success or permission.",
    }
    from phase_agent.tools.dispatch.scientific_operations import operation_readiness

    if role in {"exploration", "reliability"} and isinstance(
        state.get("remote_dft_sampling"), dict
    ):
        state["remote_dft_sampling"] = {
            bid: {k: v for k, v in row.items() if k != "selected"}
            for bid, row in state["remote_dft_sampling"].items()
            if isinstance(row, dict)
        }

    return {
        "role": role,
        "operation_readiness": operation_readiness(facts),
        "evidence_snapshot": _snapshot_copy(projected),
        "permissions": role_permissions(role, available_operations=operations),
    }


def simulation_facts(action, *, state, config=None, context=None, snapshot=None):
    """The same self-contained preflight facts for ordinary and native planners."""
    from phase_agent.evidence.build_decision_context import build_decision_context
    from phase_agent.evidence.project_authority import current_project, project_identity

    context = context or {}
    config = config if config is not None else state.get("confirmed_config") or {}
    from phase_agent.evidence.requests import TOPICS, required_fields
    from phase_agent.agents.strategic.human_review import authorization_settings

    decision = deepcopy(context) if context else build_decision_context(state)
    known_fields = set().union(*TOPICS.values())
    required = required_fields(action, decision) | set().union(
        *(TOPICS[name] for name in ("tasks", "budget", "configuration", "plans", "model"))
    )
    omitted = sorted(known_fields.intersection(decision) - required)
    decision = {key: value for key, value in decision.items() if key not in omitted}
    decision["budget_limits"] = deepcopy(
        config.get("budgets", decision.get("budget_limits", state.get("budget_limits")))
    )
    decision.setdefault(
        "execution_authorization_policy",
        {
            "source": "confirmed_config" if state.get("confirmed_config_version") else "none",
            **authorization_settings(config),
        },
    )
    decision["human_guidance"] = deepcopy(
        (action.get("_decision_team") or {}).get("guidance")
        or (context.get("codex_consultation") or {}).get("guidance")
        or []
    )
    # Do not send the whole persistent history/checkpoint tree to a planner.
    # Current task rows retain all readiness, checksums, status and diagnostics.
    facts_state = {
        key: state[key]
        for key in (
            "tasks",
            "current_status",
            "human_constraints",
            "budget_remaining",
            "budget_usage",
            "budget_reservations",
            "pending_execution_policies",
            "active_model_version",
            "confirmed_config_version",
            "active_model",
            "relax_mc_batches",
            "remote_finetune_jobs",
            "batch_return_facts",
        )
        if key in state
    }
    for key in required_fields(action, state):
        if key != "decision_context":
            facts_state.setdefault(key, state[key])
    facts_state["current_project"] = deepcopy(
        ((snapshot or {}).get("facts", {}).get("state") or {}).get("current_project")
        or current_project(state)
    )
    local_payload = {
        "state": facts_state,
        "decision_context": decision,
        "allowed_tools": [],
        "user_instruction": "Plan only the supplied frozen action; no approval or retry authority.",
    }
    source_identity = (((snapshot or {}).get("facts") or {}).get("state") or {}).get(
        "advisory_source_identity"
    )
    if source_identity is not None:
        # The native team coordinator can retrieve these exact records from the
        # same consultation; the scientific action still uses the full state.
        from phase_agent.evidence.advisory_projection import project_advisory_payload

        local_payload = project_advisory_payload(local_payload, source_identity=source_identity)
    local = evidence_snapshot(local_payload)
    if snapshot is not None:
        # The native task stays bound to the full consultation, not a new approval.
        local.update(
            {
                key: deepcopy(snapshot[key])
                for key in (
                    "snapshot_id",
                    "project_id",
                    "config_version",
                    "model_version",
                    "available_evidence_refs",
                )
            }
        )
    projected = role_context(local, "simulation")["evidence_snapshot"]
    projected["context_projection"]["omitted_unrelated_fields"] = omitted
    projected["action_scope"] = {
        "tool": action.get("tool"),
        "parameters": deepcopy(action.get("parameters") or {}),
    }
    if snapshot is None:
        projected.update(project_identity(state))
    return projected
