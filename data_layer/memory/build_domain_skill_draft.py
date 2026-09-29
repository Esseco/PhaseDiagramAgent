"""Write a reviewable project-local knowledge draft after accepted convergence."""

from copy import deepcopy
import json
from pathlib import Path
import re


def build_domain_skill_draft(state, *, convergence, system_signature, workspace_root,
                             skill_id, source_project, convergence_rules=None):
    if not (convergence.get("status") == "finished" and convergence.get("converged") is True
            and state.get("user_accepted_convergence") is True):
        raise ValueError("confirmed convergence is required before a formal skill draft")
    if not state.get("confirmed_config_version") or not state.get("active_model_version"):
        raise ValueError("config and model versions are required")
    skill_id = _safe_id(skill_id)
    approved = [_sanitize(deepcopy(row)) for row in ((state.get("decision_memory") or {}).get("records") or [])
                if row.get("approved_by_user") is True and row.get("status") == "active"]
    candidates = [_sanitize(deepcopy(row)) for row in state.get("memory_candidates") or []
                  if row.get("status") not in {"rejected", "stale"}]
    if not approved and not candidates:
        raise ValueError("no reviewed knowledge or factual candidate evidence to draft")
    root = Path(workspace_root).resolve()
    folder = root / "knowledge_export" / f"{skill_id}-draft"
    if (folder / "profile.json").is_file():
        existing = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
        return {"status": "already_drafted", "skill_id": skill_id,
                "draft_directory": str(folder),
                "approved_knowledge_count": len(existing.get("recommendations") or []),
                "candidate_count": len(existing.get("candidate_ids_for_review") or [])}
    folder.mkdir(parents=True, exist_ok=True)
    profile = {"skill_id": skill_id, "version": "0.1.0-draft", "status": "pending_review",
               "signature": deepcopy(system_signature), "source_project": source_project,
               "config_version": state["confirmed_config_version"],
               "model_version": state["active_model_version"],
               "convergence_rules": deepcopy(convergence_rules or {}),
               "convergence": {key: deepcopy(convergence.get(key)) for key in
                               ("status", "checks", "model_update_epochs_counted",
                                "recent_final_frame_mae_ev_per_atom", "coverage_risk")},
               "recommendations": approved,
               "candidate_ids_for_review": [row["candidate_id"] for row in candidates]}
    evidence = {"skill_id": skill_id, "approved_evidence_refs": sorted({
        ref for row in approved for ref in row.get("evidence_refs") or []}),
        "candidate_observations": candidates}
    _write_json(folder / "profile.json", profile)
    _write_json(folder / "evidence.json", evidence)
    lines = [f"# {skill_id}", "", "Draft for human review. Applies only where profile.json matches.", "",
             "## Reviewed knowledge", ""]
    lines.extend(f"- {row['statement']} ({row['knowledge_id']}; {row['maturity']})"
                 for row in approved)
    if not approved:
        lines.append("- No approved reusable recommendations yet.")
    lines.extend(["", "## Evidence pending review", "",
                  f"{len(candidates)} factual observations are indexed in evidence.json.", "",
                  "This file contains no executable code or hard-constraint overrides.", ""])
    (folder / "SKILL.md").write_text("\n".join(lines), encoding="utf-8")
    (folder / "CHANGELOG.md").write_text("# Changelog\n\n- 0.1.0-draft: generated for review.\n", encoding="utf-8")
    return {"status": "pending_review", "skill_id": skill_id, "draft_directory": str(folder),
            "approved_knowledge_count": len(approved), "candidate_count": len(candidates)}


def _safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,63}", value):
        raise ValueError("skill_id must be lowercase letters, digits and hyphens")
    return value


def _write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sanitize(value):
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"\b[A-Za-z]:[\\/][^\s,;]+", "${WORKSPACE_ROOT}", value)
        value = re.sub(r"(?<!\w)/(?:data|home|fsb)/[^\s,;]+", "${REMOTE_PATH}", value)
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{16,}", "${API_KEY}", value)
    return value
