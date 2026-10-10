"""Read published skills and return reviewable matches for a new project."""

import json
from pathlib import Path

from phase_agent.persistence.memory.match_domain_skill import match_domain_skill


def load_matching_domain_skills(library_root, project_signature):
    root = Path(library_root).resolve()
    registry_path = root / "registry.json"
    if not registry_path.is_file():
        return []
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    matches = []
    for entry in registry.get("skills") or []:
        profile_path = (root / entry["profile_path"]).resolve()
        if root not in profile_path.parents:
            raise ValueError("skill registry path escapes knowledge library")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if profile.get("status") != "published":
            continue
        result = match_domain_skill(profile, project_signature)
        matches.append(
            {
                "skill_id": profile["skill_id"],
                "version": profile["version"],
                "source_project": profile.get("source_project"),
                "profile_path": str(profile_path),
                **result,
            }
        )
    return matches
