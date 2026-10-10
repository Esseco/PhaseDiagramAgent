"""Draft reviewed domain knowledge when confirmed numerical search finishes."""

from phase_agent.persistence.memory.build_domain_skill_draft import build_domain_skill_draft
from phase_agent.persistence.memory.build_system_signature import build_system_signature
from pathlib import Path


def maybe_draft_converged_skill(state, config, convergence):
    if convergence.get("status") != "finished" or convergence.get("converged") is not True:
        return {"status": "not_converged"}
    system = config.get("system_config") or config.get("system") or {}
    storage = config.get("storage") or {}
    root = storage.get("workspace_root") or config.get("workspace_root")
    if not root or not system.get("system_id"):
        return {"status": "not_configured", "reason": "workspace_or_system_id_missing"}
    signature = build_system_signature(system)
    skill_id = system["system_id"].lower().replace("_", "-")
    try:
        return build_domain_skill_draft(
            state,
            convergence=convergence,
            system_signature=signature,
            workspace_root=root,
            skill_id=skill_id,
            source_project=Path(root).name,
            convergence_rules=config.get("convergence") or {},
        )
    except ValueError as error:
        return {"status": "insufficient_knowledge", "reason": str(error)}
