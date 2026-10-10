"""Publish one explicitly approved domain skill version to a shared registry."""

import json
from pathlib import Path
import re
import shutil


def publish_domain_skill(draft_directory, library_root, *, approved, version="1.0.0"):
    if approved is not True:
        raise ValueError("explicit user approval is required")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("skill version must be semantic version x.y.z")
    draft = Path(draft_directory).resolve()
    profile = json.loads((draft / "profile.json").read_text(encoding="utf-8"))
    if profile.get("status") != "pending_review":
        raise ValueError("only reviewed drafts can be published")
    if not profile.get("recommendations"):
        raise ValueError("no user-approved recommendations to publish")
    skill_id = profile["skill_id"]
    root = Path(library_root).resolve()
    target = root / "domain_skills" / skill_id / version
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True, exist_ok=False)
    try:
        for name in ("SKILL.md", "profile.json", "evidence.json", "CHANGELOG.md"):
            shutil.copy2(draft / name, target / name)
        profile.update({"status": "published", "version": version})
        (target / "profile.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        registry_path = root / "registry.json"
        registry = (
            json.loads(registry_path.read_text(encoding="utf-8"))
            if registry_path.is_file()
            else {"skills": []}
        )
        registry.setdefault("skills", []).append(
            {
                "skill_id": skill_id,
                "version": version,
                "profile_path": str((target / "profile.json").relative_to(root)).replace("\\", "/"),
                "source_project": profile.get("source_project"),
            }
        )
        temporary = registry_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(registry_path)
    except Exception:
        shutil.rmtree(target)
        raise
    return {
        "status": "published",
        "skill_id": skill_id,
        "version": version,
        "directory": str(target),
        "registry_path": str(registry_path),
    }
