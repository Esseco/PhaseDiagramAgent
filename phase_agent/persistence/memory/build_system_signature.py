"""Extract a portable system signature from confirmed configuration."""


def build_system_signature(system):
    system = system or {}
    system_id = system.get("system_id")
    if not system_id:
        raise ValueError("confirmed system_id is required")
    species = system.get("species") or {}
    roles = (system.get("configuration_space") or {}).get("roles") or {}
    phases = (system.get("constraints") or {}).get("phases") or []
    return {
        "system_id": system_id,
        "mobile_ion": sorted(species.get("mobile") or []),
        "structure_family": "layered_oxide" if "layered" in system_id else system_id,
        "allowed_phases": sorted(phases),
        "branch_variables": sorted(key for key, role in roles.items() if role == "branch"),
        "internal_variables": sorted(key for key, role in roles.items() if role == "internal"),
    }
