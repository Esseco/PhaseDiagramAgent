"""Reversible references for equal evidence repeated within one model request."""

from copy import deepcopy
import hashlib
import json

FORMAT = "evidence_ref_v1"
ROOTS = {
    "state",
    "decision_context",
    "conversation_facts",
    "current_project",
    "session",
    "config_session",
    "current_config",
    "draft_config",
    "configuration",
}


def compact_transport_references(payload, *, minimum_chars=512):
    seen = {}
    protected_keys = {
        "human_constraints",
        "human_system_knowledge",
        "long_term_advice",
        "user_instruction",
        "human_comment",
        "instruction",
        "system_prompt",
        "repair_instruction",
        "output_contracts",
        "editable_field_catalog",
    }
    protected_cache = {}

    def protected(value):
        if not isinstance(value, (dict, list)):
            return False
        identity = id(value)
        if identity not in protected_cache:
            protected_cache[identity] = (
                bool(set(value) & protected_keys) or any(protected(v) for v in value.values())
                if isinstance(value, dict)
                else any(protected(v) for v in value)
            )
        return protected_cache[identity]

    def visit(value, path):
        if path.count("/") > 1 and isinstance(value, (dict, list)) and not protected(value):
            encoded = json.dumps(
                value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")
            )
            if len(encoded) >= minimum_chars:
                digest = hashlib.sha256(encoded.encode()).digest()
                prior = seen.get(digest)
                if prior is not None and type(prior[1]) is type(value) and prior[1] == value:
                    reference = {"format": FORMAT, "source_ref": prior[0]}
                    if isinstance(value, list):
                        reference["count"] = len(value)
                    if len(json.dumps(reference)) < len(encoded):
                        return reference
                seen.setdefault(digest, (path, value))
        if isinstance(value, dict):
            return {
                k: deepcopy(v)
                if k in protected_keys
                else visit(v, path + "/" + str(k).replace("~", "~0").replace("/", "~1"))
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [visit(v, path + "/" + str(i)) for i, v in enumerate(value)]
        return deepcopy(value)

    return {
        key: visit(value, "/" + key) if key in ROOTS else deepcopy(value)
        for key, value in payload.items()
    }


def expand_transport_references(payload):
    def lookup(pointer):
        if not isinstance(pointer, str) or not pointer.startswith("/"):
            raise ValueError("Invalid evidence reference")
        value = payload
        try:
            for token in pointer[1:].split("/"):
                key = token.replace("~1", "/").replace("~0", "~")
                value = value[int(key)] if isinstance(value, list) else value[key]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("Unresolved evidence reference") from error
        return value

    def expand(value, active):
        if isinstance(value, dict):
            if (
                set(value) in ({"format", "source_ref"}, {"format", "source_ref", "count"})
                and value["format"] == FORMAT
            ):
                pointer = value["source_ref"]
                if pointer in active:
                    raise ValueError("Cyclic evidence reference")
                return expand(lookup(pointer), active | {pointer})
            return {key: expand(item, active) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item, active) for item in value]
        return deepcopy(value)

    return expand(payload, set())
