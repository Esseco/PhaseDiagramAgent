"""Memory facts, human review, convergence and cross-project skill reuse."""

import json
from pathlib import Path

import pytest

from analysis_layer.state.build_decision_context import build_decision_context
from data_layer.memory.build_domain_skill_draft import build_domain_skill_draft
from data_layer.memory.collect_memory_candidates import collect_memory_candidates
from data_layer.memory.load_matching_domain_skills import load_matching_domain_skills
from data_layer.memory.publish_domain_skill import publish_domain_skill
from data_layer.memory.propose_domain_skill_import import propose_domain_skill_import
from data_layer.memory.retrieve_relevant_knowledge import retrieve_relevant_knowledge
from data_layer.memory.review_queue import propose_knowledge_record, review_memory_update


def _state():
    return {"confirmed_config_version": "cfg-1", "active_model_version": "mace-1",
            "user_accepted_convergence": True,
            "convergence": {"status": "finished", "converged": True},
            "slurm_batches": [{"batch_id": "B1", "model_version": "mace-1"}],
            "tasks": [{"task_id": "T1", "batch_id": "B1", "stage": "relax_and_feature",
                       "status": "completed"}]}


def test_candidate_backfill_is_idempotent_and_not_agent_advice():
    state = collect_memory_candidates(_state())
    assert len(state["memory_candidates"]) == 1
    assert state["memory_candidates"][0]["facts"]["completed_count"] == 1
    assert len(collect_memory_candidates(state)["memory_candidates"]) == 1
    assert build_decision_context(state)["relevant_approved_knowledge"] == []


def test_structured_knowledge_requires_evidence_and_review():
    state = collect_memory_candidates(_state())
    record = {"scope": "system", "category": "search_rule", "statement": "Run modest MC first",
              "status": "active", "maturity": "project_observed", "system_id": "na-layered",
              "source_project": "FeMn", "applicability": {"phase": "O3"},
              "evidence_refs": ["completed_batch:B1"]}
    queued = propose_knowledge_record(state, record)
    assert retrieve_relevant_knowledge(queued["state"], system_id="na-layered", phase="O3") == []
    approved = review_memory_update(queued["state"], queued["proposal"]["proposal_id"], approved=True)
    matched = retrieve_relevant_knowledge(approved["state"], system_id="na-layered", phase="O3")
    assert len(matched) == 1 and matched[0]["approved_by_user"] is True
    assert retrieve_relevant_knowledge(approved["state"], system_id="na-layered", phase="P3") == []
    with pytest.raises(ValueError, match="references are missing"):
        propose_knowledge_record(state, {**record, "evidence_refs": ["task:missing"]})


def test_skill_draft_gate_publication_and_matching(tmp_path):
    state = collect_memory_candidates(_state())
    record = {"scope": "system", "category": "search_rule", "statement": "Check O3 coverage",
              "status": "active", "maturity": "project_observed", "system_id": "na-layered",
              "source_project": "FeMn", "applicability": {"phase": "O3"},
              "evidence_refs": ["completed_batch:B1"]}
    proposal = propose_knowledge_record(state, record)
    state = review_memory_update(proposal["state"], proposal["proposal"]["proposal_id"],
                                 approved=True)["state"]
    signature = {"mobile_ion": ["Na"], "structure_family": "layered_oxide"}
    with pytest.raises(ValueError, match="confirmed convergence"):
        build_domain_skill_draft(state, convergence={"status": "budget_exhausted"},
            system_signature=signature, workspace_root=tmp_path, skill_id="na-o3-layered-oxide",
            source_project="FeMn")
    drafted = build_domain_skill_draft(state, convergence=state["convergence"],
        system_signature=signature, workspace_root=tmp_path, skill_id="na-o3-layered-oxide",
        source_project="FeMn")
    assert drafted["status"] == "pending_review"
    assert drafted["candidate_count"] == 1
    assert build_domain_skill_draft(state, convergence=state["convergence"],
        system_signature=signature, workspace_root=tmp_path, skill_id="na-o3-layered-oxide",
        source_project="FeMn")["status"] == "already_drafted"
    with pytest.raises(ValueError, match="approval"):
        publish_domain_skill(drafted["draft_directory"], tmp_path / "knowledge", approved=False)
    published = publish_domain_skill(drafted["draft_directory"], tmp_path / "knowledge", approved=True)
    profile = json.loads(Path(published["directory"], "profile.json").read_text(encoding="utf-8"))
    assert profile["status"] == "published"
    matches = load_matching_domain_skills(tmp_path / "knowledge", signature)
    assert matches[0]["status"] == "match" and matches[0]["requires_project_confirmation"]
    assert load_matching_domain_skills(tmp_path / "knowledge", {"mobile_ion": ["Li"]})[0]["status"] == "not_applicable"
    imported = propose_domain_skill_import({}, tmp_path / "knowledge", signature)
    assert len(imported["proposals"]) == 1
    assert imported["proposals"][0]["record"]["status"] == "needs_revalidation"
    assert propose_domain_skill_import(imported["state"], tmp_path / "knowledge", signature)["proposals"] == []
    approved_import = review_memory_update(imported["state"],
        imported["proposals"][0]["proposal_id"], approved=True)["state"]
    assert retrieve_relevant_knowledge(approved_import, system_id="na-layered", phase="O3")


def test_legacy_strings_remain_readable_as_scoped_records():
    state = {"decision_memory": {"long_term_advice": ["Reduce DFT relax"], "version": 1}}
    rows = retrieve_relevant_knowledge(state, system_id="na-layered")
    assert rows[0]["statement"] == "Reduce DFT relax"
    assert rows[0]["knowledge_id"].startswith("LEGACY-")
