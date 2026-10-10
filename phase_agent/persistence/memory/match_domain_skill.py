"""Match a published domain skill to a new project's confirmed signature."""

from copy import deepcopy


def match_domain_skill(skill_profile, project_signature):
    expected = skill_profile.get("signature") or {}
    actual = project_signature or {}
    matched, mismatched, unconfirmed = [], [], []
    for key, required in expected.items():
        if key not in actual or actual[key] is None:
            unconfirmed.append(key)
        elif isinstance(required, list):
            if set(map(str, required)).issubset(
                set(map(str, actual[key] if isinstance(actual[key], list) else [actual[key]]))
            ):
                matched.append(key)
            else:
                mismatched.append(key)
        elif actual[key] == required:
            matched.append(key)
        else:
            mismatched.append(key)
    status = (
        "match"
        if not mismatched and not unconfirmed
        else "partial"
        if matched
        else "not_applicable"
    )
    return {
        "status": status,
        "matched": matched,
        "mismatched": mismatched,
        "unconfirmed": unconfirmed,
        "recommendations": deepcopy(skill_profile.get("recommendations") or []),
        "requires_project_confirmation": True,
    }
