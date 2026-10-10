"""Validate fixed structural constraints while retaining actual re-identification."""


def validate_structure_transition(expected: dict, identified: dict, *, policy=None) -> dict:
    settings = {"constraint_fields": ["x", "T"], "reidentify_fields": ["P", "H"]}
    settings.update(policy or {})
    violations = [
        field
        for field in settings["constraint_fields"]
        if field in expected and expected.get(field) != identified.get(field)
    ]
    changes = [
        field
        for field in settings["reidentify_fields"]
        if field in expected and expected.get(field) != identified.get(field)
    ]
    return {
        "valid": not violations,
        "constraint_violations": violations,
        "reidentified_fields": changes,
        "structure_family_changed": bool(changes),
        "original_assignment": {
            key: expected.get(key)
            for key in set(settings["constraint_fields"] + settings["reidentify_fields"])
        },
        "actual_identification": {
            key: identified.get(key)
            for key in set(settings["constraint_fields"] + settings["reidentify_fields"])
        },
    }
