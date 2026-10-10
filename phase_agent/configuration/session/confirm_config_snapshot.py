"""Create an immutable versioned snapshot only after explicit confirmation."""

import hashlib
import json
from copy import deepcopy

from phase_agent.configuration.schema.validate_search_config import validate_search_config


def confirm_config_snapshot(session: dict, *, user_confirmed: bool) -> dict:
    if not user_confirmed:
        return {**deepcopy(session), "confirmation_error": "explicit_user_confirmation_required"}
    audit = validate_search_config(session["config"], stage="startup")
    if not audit["valid"]:
        return {
            **deepcopy(session),
            "confirmation_error": "configuration_not_valid",
            "audit": audit,
        }
    payload = deepcopy(session["config"])
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]
    previous = session.get("confirmed_snapshot") or {}
    same_config = (
        previous.get("config_hash") == digest
        and previous.get("config") == payload
        and isinstance(previous.get("config_version"), str)
    )
    updated = deepcopy(session)
    updated["status"] = "confirmed"
    updated["confirmed_snapshot"] = (
        deepcopy(previous)
        if same_config
        else {
            "config_version": f"config-{updated['draft_revision']:04d}-{digest}",
            "config_hash": digest,
            "config": payload,
            "audit": audit,
        }
    )
    updated["confirmation_error"] = None
    if session.get("config_profile"):
        updated["confirmed_snapshot"]["profile"] = deepcopy(session["config_profile"])
    updated["dialogue"].append(
        {
            "type": "confirmation",
            "config_version": updated["confirmed_snapshot"]["config_version"],
            "explicit": True,
        }
    )
    return updated
