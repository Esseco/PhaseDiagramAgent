"""Apply an explicit path patch and record every change and reason."""

from copy import deepcopy
from config_layer.session.classify_config_permission import classify_config_permission


def apply_config_revision(session: dict, patch: dict, *, reasons=None, author="user") -> dict:
    if session.get("status") != "draft":
        raise ValueError("confirmed session cannot be edited; create a new draft version")
    updated = deepcopy(session); changes = []
    for path, new_value in sorted(patch.items()):
        parts = path.split("."); target = updated["config"]
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        old = deepcopy(target.get(parts[-1])); target[parts[-1]] = deepcopy(new_value)
        permission = classify_config_permission(path, updated["config"])
        changes.append({"path": path, "old": old, "new": deepcopy(new_value), "reason": (reasons or {}).get(path), "constraint_type": "hard" if permission == "hard_constraint" else "adjustable" if permission == "adjustable_policy" else "unclassified"})
    updated["draft_revision"] += 1
    updated["dialogue"].append({"type": "config_revision", "author": author, "revision": updated["draft_revision"], "changes": changes})
    return updated
