import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

from ase import Atoms
from ase.io import write
import pytest

from phase_agent.agents.simulation.program_checks import parent_first_input_record
from phase_agent.runtime.codex.mcp import (
    compact_context_response,
    compact_operation_status_response,
    compact_pending_response,
)
from phase_agent.runtime.codex.team import call_team_tool
from phase_agent.tools.workflows.initial_tool_proposal import _deterministic_cold_start_action
from tests.test_codex_native_team import advice, make_control, team


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_cold_start_project(tmp_path: Path):
    project_root = tmp_path / "O3_NaCoO2"
    (project_root / "structures" / "branch_pool").mkdir(parents=True, exist_ok=True)
    (project_root / "workflow_state").mkdir(parents=True, exist_ok=True)
    (project_root / "models").mkdir(parents=True, exist_ok=True)

    atoms = Atoms("NaCoO2", positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0]])
    atoms.set_cell([4.0, 4.0, 4.0])
    atoms.set_pbc(True)
    branch_path = project_root / "structures" / "branch_pool" / "branch_0001.cif"
    write(branch_path, atoms)
    branch_sha = _sha256(branch_path)

    feature_model = project_root / "models" / "mace-mh-1.model"
    feature_model.write_bytes(b"fake-mace-mh-1-weights")
    feature_sha = _sha256(feature_model)

    state = {
        "schema_version": "clean_v2",
        "project_name": "O3_NaCoO2",
        "epoch": 0,
        "status": "initialized",
        "confirmed_config_version": "c1",
        "active_model_version": "mace-mh-1",
        "active_model": {
            "name": "mace-mh-1",
            "model_sha256": feature_sha,
        },
        "branches": [
            {
                "branch_id": "branch_0001",
                "record_id": "branch:branch_0001",
                "structure_path": str(branch_path),
                "structure_sha256": branch_sha,
                "composition": "NaCoO2",
                "status": "generated",
            }
        ],
    }
    config = {
        "project": {"name": "O3_NaCoO2", "root": str(project_root)},
        "mlip": {"name": "mace-mh-1", "model_sha256": feature_sha},
        "direct": {
            "feature_model_path": str(feature_model),
            "feature_model_sha256": feature_sha,
        },
        "dft_parameters": {"encut": 520, "kspacing": 0.25},
    }
    return project_root, branch_path, branch_sha, feature_model, feature_sha, state, config


def test_cold_start_clean_v2_inherits_defaults_and_prepares_archive(tmp_path):
    project_root, branch_path, branch_sha, feature_model, feature_sha, state, config = (
        _make_cold_start_project(tmp_path)
    )
    manager = SimpleNamespace(
        data={
            "structures": {
                "parent_0001": {
                    "record_id": "parent_0001",
                    "structure_id": "parent_0001",
                    "kind": "initial_parent",
                    "natoms": 4,
                    "det_H": 1,
                    "structure_path": str(branch_path),
                    "structure_sha256": branch_sha,
                    "source": "mp",
                    "composition": "NaCoO2",
                }
            },
        }
    )
    config["dft"] = {"parameters": config["dft_parameters"]}
    action = _deterministic_cold_start_action(
        state,
        config,
        ["prepare_local_batch_files", "prepare_parent_first_recovery"],
        context={"manager": manager},
    )
    assert action is not None
    assert action["parameters"]["model_sha256"] == feature_sha
    assert action["parameters"]["feature_model_path"] == str(feature_model)
    assert action["parameters"]["feature_model_sha256"] == feature_sha

    program = parent_first_input_record(action, state, config=config, manager=manager)
    assert program is not None
    assert program["status"] == "passed", program["unmet_conditions"]


def test_preflight_blocks_missing_cold_start_fields_in_draft_without_receipts(tmp_path):
    control, context = make_control(tmp_path)
    project_root, branch_path, branch_sha, feature_model, feature_sha, state, config = (
        _make_cold_start_project(tmp_path)
    )
    context["allowed_tools"] = [*context["allowed_tools"], "prepare_local_batch_files"]
    context["team_state"] = state
    t = team(control)
    orig_t_ctx = t._context
    t._context = lambda **kw: {
        **orig_t_ctx(**kw),
        "allowed_tools": context["allowed_tools"],
        "team_state": state,
    }
    view = call_team_tool(
        control,
        "phase_team_start",
        {
            "question": "Prepare initial_direct local bundle",
            "human_directed": True,
            "consult": [],
        },
    )
    bad_action = {
        "tool": "prepare_local_batch_files",
        "parameters": {
            "bundle_type": "parent_first_recovery_inputs",
            "sampling_method": "direct",
            "confirmed": True,
            "round_evidence_digest": "a" * 64,
            "initial_parents": [{"record_id": "missing"}],
        },
    }
    with pytest.raises(ValueError, match="simulation_program_blocked") as exc_info:
        t.draft(
            view["team_id"],
            1,
            bad_action,
            rationale="Attempt incomplete initial_direct draft",
        )
    msg = str(exc_info.value)
    assert "missing_or_invalid_feature_model_path" in msg
    assert "missing_or_invalid_feature_model_sha256" in msg
    assert "missing_or_invalid_model_sha256" in msg


def test_human_directed_other_rounds_local_prep_skips_all_subagents(tmp_path, monkeypatch):
    control, context = make_control(tmp_path)
    _, _, _, _, _, state, _ = _make_cold_start_project(tmp_path)
    context["allowed_tools"] = [*context["allowed_tools"], "prepare_local_batch_files"]
    from phase_agent.agents.strategic.codex_team import CodexTeam

    orig_ctx = CodexTeam._context
    monkeypatch.setattr(
        CodexTeam,
        "_context",
        lambda self, **kw: {
            **orig_ctx(self, **kw),
            "allowed_tools": context["allowed_tools"],
            "team_state": state,
        },
    )
    start_resp = call_team_tool(
        control,
        "phase_team_start",
        {"question": "Pack relax_mc inputs", "human_directed": True, "consult": []},
    )
    assert start_resp["requests"] == {}
    draft_resp = call_team_tool(
        control,
        "phase_team_draft",
        {
            "team_id": start_resp["team_id"],
            "revision": 1,
            "action": {
                "tool": "prepare_local_batch_files",
                "parameters": {"mode": "training_handoff_inputs", "confirmed": True},
            },
            "rationale": "Deterministic training_handoff local packaging",
        },
    )
    assert draft_resp["status"] == "awaiting_human_input"
    assert draft_resp["draft"]["simulation_program_status"] == "passed"
    assert "requests" not in draft_resp

    heavy_ctx = {
        "context_version": "cv-1",
        "context_digest": "d" * 64,
        "allowed_tools": ["prepare_local_batch_files"],
        "effective_config": {"blob": "x" * 20000},
        "scientific_operations": [{"op": i} for i in range(100)],
        "calculation_cost_reference": {"cost": "y" * 10000},
        "state": {"structures": [f"s{i}" for i in range(500)]},
        "team_state": {"structures": [f"s{i}" for i in range(500)]},
    }
    compacted_ctx = compact_context_response(heavy_ctx)
    assert "effective_config" not in compacted_ctx
    assert "scientific_operations" not in compacted_ctx
    assert "calculation_cost_reference" not in compacted_ctx
    assert "state" not in compacted_ctx
    assert "team_state" not in compacted_ctx
    assert len(json.dumps(compacted_ctx, ensure_ascii=False).encode("utf-8")) < 4096
    assert "initial_parents" in msg or "parent_selection" in msg
    saved_state = json.loads(control.state_path.read_text(encoding="utf-8"))
    assert not saved_state.get("execution_receipts")


def test_deterministic_prep_revise_and_reconciled_no_effect_do_not_deadlock(tmp_path):
    control, context = make_control(tmp_path)
    _, branch_path, branch_sha, feature_model, feature_sha, state, _ = _make_cold_start_project(
        tmp_path
    )
    context["allowed_tools"] = [*context["allowed_tools"], "prepare_local_batch_files"]
    context["team_state"] = state
    t = team(control)
    orig_t_ctx = t._context
    t._context = lambda **kw: {
        **orig_t_ctx(**kw),
        "allowed_tools": context["allowed_tools"],
        "team_state": state,
    }

    start_resp = t.start(
        question="Prepare initial_direct bundle",
        human_directed=True,
        consult=["exploration", "reliability"],
        compact=True,
    )
    tid = start_resp["team_id"]
    t.advise(tid, 1, advice(start_resp, "exploration"), compact=True)
    t.advise(tid, 1, advice(start_resp, "reliability"), compact=True)

    good_action = {
        "tool": "prepare_local_batch_files",
        "parameters": {
            "bundle_type": "parent_first_recovery_inputs",
            "sampling_method": "direct",
            "confirmed": True,
            "round_evidence_digest": "initial_direct_v1",
            "model_sha256": feature_sha,
            "feature_model_path": str(feature_model),
            "feature_model_sha256": feature_sha,
            "dft_parameters": {},
            "initial_parents": [
                {
                    "record_id": "branch:branch_0001",
                    "structure_path": str(branch_path),
                    "structure_sha256": branch_sha,
                }
            ],
        },
    }
    draft_resp = t.draft(
        tid,
        1,
        good_action,
        rationale="Pack initial_direct inputs",
        compact=True,
    )
    assert draft_resp["status"] == "awaiting_human_input"
    assert draft_resp["draft"]["simulation_program_status"] == "passed"

    fb_revise = t.feedback(
        tid,
        1,
        "revise",
        "Adjust technical parameter without re-waking subagents",
        human_directed=True,
        compact=True,
    )
    assert fb_revise["revision"] == 2
    assert fb_revise["remaining_roles"] == []
    assert fb_revise["registered_roles"] == ["exploration", "reliability"]

    draft_resp_2 = t.draft(
        tid,
        2,
        good_action,
        rationale="Updated technical parameter",
        compact=True,
    )
    assert draft_resp_2["status"] == "awaiting_human_input"
    fb_accept = t.feedback(tid, 2, "accept", "Approve local packaging", compact=True)
    assert fb_accept["status"] == "ready_to_propose"

    # Simulate pre-dispatch execution_reconciliation_required that has now been reconciled (no unsettled)
    session = t._load(tid)
    session["status"] = "submission_uncertain"
    session["submission"] = {"status": "execution_reconciliation_required"}
    t._save(session)

    control.propose_codex = lambda *a, **kw: {
        "reply": {"status": "awaiting_approval", "plan_id": "plan-1", "proposal_hash": "h" * 64}
    }
    control.pending = lambda: {
        "count": 1,
        "state_version": "sv-1",
        "pending": [
            {
                "plan_id": "plan-1",
                "proposal_hash": "h" * 64,
                "recommended_action": "prepare_local_batch_files",
                "target_ids": [f"s-{i}" for i in range(50)],
                "effective_config": {"huge": "x" * 10000},
                "scientific_operations": [{"op": "y"} for _ in range(50)],
                "calculation_cost_reference": {"cost": "z" * 5000},
            }
        ],
    }

    propose_resp = call_team_tool(control, "phase_team_propose", {"team_id": tid, "revision": 2})
    assert propose_resp["status"] == "awaiting_approval"
    assert propose_resp["plan_id"] == "plan-1"
    assert propose_resp["proposal_hash"] == "h" * 64
    assert len(json.dumps(propose_resp, ensure_ascii=False).encode("utf-8")) < 4096


def test_compact_mcp_payload_size_limits():
    heavy_pending = {
        "count": 1,
        "state_version": "sv-1",
        "effective_config": {"blob": "x" * 20000},
        "scientific_operations": [{"name": f"op_{i}"} for i in range(100)],
        "calculation_cost_reference": {"notes": "y" * 10000},
        "state": {"structures": [f"st_{i}" for i in range(500)]},
        "pending": [
            {
                "plan_id": "plan-99",
                "proposal_hash": "a" * 64,
                "recommended_action": "prepare_local_batch_files",
                "target_ids": [f"id_{i}" for i in range(200)],
                "effective_config": {"blob": "z" * 10000},
                "parameters": {"target_ids": [f"id_{i}" for i in range(200)]},
            }
        ],
    }
    compacted_pending = compact_pending_response(heavy_pending)
    assert compacted_pending["plan_id"] == "plan-99"
    assert compacted_pending["proposal_hash"] == "a" * 64
    assert "effective_config" not in compacted_pending
    assert "scientific_operations" not in compacted_pending
    assert "calculation_cost_reference" not in compacted_pending
    assert "state" not in compacted_pending
    assert len(json.dumps(compacted_pending, ensure_ascii=False).encode("utf-8")) < 4096

    heavy_op = {
        "operation_id": "op-1",
        "status": "completed",
        "effective_config": {"blob": "x" * 20000},
        "scientific_operations": [{"name": f"op_{i}"} for i in range(100)],
        "calculation_cost_reference": {"notes": "y" * 10000},
        "result": {
            "plan_id": "plan-99",
            "proposal_hash": "a" * 64,
            "state_version": "sv-1",
            "status": "completed",
            "state": {"huge": "w" * 20000},
            "effective_config": {"huge": "v" * 20000},
        },
    }
    compacted_op = compact_operation_status_response(heavy_op)
    assert compacted_op["plan_id"] == "plan-99"
    assert compacted_op["proposal_hash"] == "a" * 64
    assert compacted_op["state_version"] == "sv-1"
    assert len(json.dumps(compacted_op, ensure_ascii=False).encode("utf-8")) < 4096


def test_extrapolation_round_parent_first_preflight_blocks_missing_fields(tmp_path):
    control, context = make_control(tmp_path)
    _, _, _, _, _, state, _ = _make_cold_start_project(tmp_path)
    state["dft_dataset_records"] = [{"record_id": "dft-1"}]
    context["allowed_tools"] = [*context["allowed_tools"], "prepare_local_batch_files"]
    t = team(control)
    orig_t_ctx = t._context
    t._context = lambda **kw: {
        **orig_t_ctx(**kw),
        "allowed_tools": context["allowed_tools"],
        "team_state": state,
    }
    view = call_team_tool(
        control,
        "phase_team_start",
        {"question": "Prepare epoch_refresh_v1 bundle", "human_directed": True, "consult": []},
    )
    bad_extrapolation = {
        "tool": "prepare_local_batch_files",
        "parameters": {
            "bundle_type": "parent_first_recovery_inputs",
            "sampling_method": "direct",
            "extrapolation_policy_version": "epoch_refresh_v1",
            "confirmed": True,
        },
    }
    with pytest.raises(ValueError, match="simulation_program_blocked") as exc_info:
        t.draft(view["team_id"], 1, bad_extrapolation, rationale="Incomplete extrapolation draft")
    msg = str(exc_info.value)
    assert "parent_record_ids_required" in msg
    assert "missing_or_invalid_feature_model_path" in msg
    assert "missing_or_invalid_feature_model_sha256" in msg
    assert "missing_or_invalid_model_sha256" in msg


def test_human_directed_other_rounds_local_prep_skips_all_subagents(tmp_path, monkeypatch):
    control, context = make_control(tmp_path)
    _, _, _, _, _, state, _ = _make_cold_start_project(tmp_path)
    context["allowed_tools"] = [*context["allowed_tools"], "prepare_local_batch_files"]
    from phase_agent.agents.strategic.codex_team import CodexTeam

    orig_ctx = CodexTeam._context
    monkeypatch.setattr(
        CodexTeam,
        "_context",
        lambda self, **kw: {
            **orig_ctx(self, **kw),
            "allowed_tools": context["allowed_tools"],
            "team_state": state,
        },
    )
    start_resp = call_team_tool(
        control,
        "phase_team_start",
        {"question": "Pack relax_mc inputs", "human_directed": True, "consult": []},
    )
    assert start_resp["requests"] == {}
    draft_resp = call_team_tool(
        control,
        "phase_team_draft",
        {
            "team_id": start_resp["team_id"],
            "revision": 1,
            "action": {
                "tool": "prepare_local_batch_files",
                "parameters": {"mode": "training_handoff_inputs", "confirmed": True},
            },
            "rationale": "Deterministic training_handoff local packaging",
        },
    )
    assert draft_resp["status"] == "awaiting_human_input"
    assert draft_resp["draft"]["simulation_program_status"] == "passed"
    assert "requests" not in draft_resp

    heavy_ctx = {
        "context_version": "cv-1",
        "context_digest": "d" * 64,
        "allowed_tools": ["prepare_local_batch_files"],
        "effective_config": {"blob": "x" * 20000},
        "scientific_operations": [{"op": i} for i in range(100)],
        "calculation_cost_reference": {"cost": "y" * 10000},
        "state": {"structures": [f"s{i}" for i in range(500)]},
        "team_state": {"structures": [f"s{i}" for i in range(500)]},
    }
    compacted_ctx = compact_context_response(heavy_ctx)
    assert "effective_config" not in compacted_ctx
    assert "scientific_operations" not in compacted_ctx
    assert "calculation_cost_reference" not in compacted_ctx
    assert "state" not in compacted_ctx
    assert "team_state" not in compacted_ctx
    assert len(json.dumps(compacted_ctx, ensure_ascii=False).encode("utf-8")) < 4096
