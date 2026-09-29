"""Queue published system advice for per-item review in another project."""

from copy import deepcopy

from data_layer.memory.load_matching_domain_skills import load_matching_domain_skills
from data_layer.memory.knowledge_record import knowledge_record


def propose_domain_skill_import(state, library_root, project_signature):
    """Import neither constraints nor advice automatically; only create review proposals."""
    updated = deepcopy(state)
    proposals = []
    for match in load_matching_domain_skills(library_root, project_signature):
        if match["status"] != "match":
            continue
        for source in match["recommendations"]:
            record = knowledge_record({**source, "maturity": "heuristic",
                                       "status": "needs_revalidation",
                                       "source_skill": f"{match['skill_id']}@{match['version']}",
                                       "source_evidence_refs": source.get("evidence_refs") or [],
                                       "evidence_refs": []})
            proposal_id = f"import-{match['skill_id']}-{record['knowledge_id']}"
            queue = updated.setdefault("memory_review_queue", [])
            if any(row.get("proposal_id") == proposal_id for row in queue):
                continue
            row = {"proposal_id": proposal_id, "record": record,
                   "source": "published_domain_skill", "status": "pending_review",
                   "requires_project_validation": True,
                   "source_skill": record["source_skill"]}
            queue.append(row)
            proposals.append(row)
    return {"state": updated, "proposals": proposals}
