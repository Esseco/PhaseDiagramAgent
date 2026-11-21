"""Tests for hypothesis ledger, bounded read-only drilldown, and conditional rebuttal."""

from __future__ import annotations

from phase_agent.evidence.decision_feedback import recent_decision_feedback
from phase_agent.evidence.drilldown import run_drilldown_queries
from phase_agent.graphs.decision_team.graph import request_with_langgraph


def test_hypothesis_ledger_in_recent_decision_feedback():
    state = {
        "active_model_version": "m1",
        "confirmed_config_version": "c1",
        "action_records": [
            {
                "record_id": "rec-1",
                "status": "executed",
                "final_action": {"tool": "generate_branches", "task_key": "tk-1"},
                "agent_proposal": {
                    "decision_summary": {
                        "rationale": "Expand P2 Na range",
                        "expected_outcome": "Register 2 new branches",
                    }
                },
                "execution_result": {"status": "completed", "result": {"bundle_id": "b1"}},
            }
        ],
        "tasks": [
            {"task_id": "t1", "task_key": "tk-1", "status": "completed"},
        ],
        "remote_dft_sampling": {
            "b1": {
                "status": "accepted",
                "gs_count": 3,
                "kmeans_count": 97,
                "excluded_hull_outlier_count": 1,
            }
        },
    }
    rows = recent_decision_feedback(state)
    assert len(rows) == 1
    assert rows[0]["outcome"]["hypothesis_status"] == "supported"
    assert rows[0]["outcome"]["scientific_outcome"]["gs_count"] == 3


def test_drilldown_queries_return_compact_summaries():
    snapshot = {
        "facts": {
            "state": {
                "tasks": [
                    {
                        "task_id": "t1",
                        "stage": "deep_search",
                        "actual_phase": "P2",
                        "outputs": {"extrapolation_monitor": {"extrapolation_failure": True}},
                    }
                ],
                "dft_dataset_records": [
                    {
                        "task_id": "t1",
                        "status": "completed",
                        "atom_count": 4,
                        "energy": -20.0,
                        "mlip_prediction": {"status": "completed", "energy": -20.08},
                    }
                ],
                "remote_dft_sampling": {
                    "b1": {
                        "status": "accepted",
                        "gs_count": 1,
                        "kmeans_count": 2,
                        "excluded_hull_outlier_count": 0,
                        "selected": [
                            {"branch_id": "B1", "weight": 0.2},
                            {"branch_id": "B1", "weight": 0.8},
                        ],
                    }
                },
            }
        }
    }
    res = run_drilldown_queries(
        snapshot,
        ["phase_error_breakdown", "sampling_weight_stats", "extrapolation_hotspots"],
    )
    # Capped at 2 queries per call
    assert list(res.keys()) == ["phase_error_breakdown", "sampling_weight_stats"]
    assert res["phase_error_breakdown"][0]["phase"] == "P2"
    assert res["sampling_weight_stats"]["weight_max"] == 0.8

    res2 = run_drilldown_queries(snapshot, ["extrapolation_hotspots"])
    assert res2["extrapolation_hotspots"][0]["failure_count"] == 1


class _ConflictAndDrilldownClient:
    supports_expert_advice = True

    def __init__(self, *, conflict: bool, with_drilldown: bool):
        self.conflict = conflict
        self.with_drilldown = with_drilldown
        self.calls = []
        self._drilled = set()

    def __call__(self, request):
        self.calls.append(request)
        mode = request.get("mode")
        role = request.get("agent_role")
        sid = (request.get("evidence_snapshot") or {}).get("snapshot_id") or request.get("snapshot_id")
        if mode == "team_consultation":
            return {
                "consult": ["exploration", "reliability"],
                "question": "Should we explore or finetune?",
                "snapshot_id": sid,
                "_llm_usage": {"calls": 1, "input_tokens": 1, "output_tokens": 1, "cost": None},
            }
        if mode == "expert_assessment":
            if self.with_drilldown and role == "reliability" and role not in self._drilled:
                self._drilled.add(role)
                return {
                    "role": role,
                    "snapshot_id": sid,
                    "assessment": "Need phase error breakdown.",
                    "recommendation": "finetune" if self.conflict else "defer",
                    "options": [],
                    "evidence_refs": [],
                    "expected_outcome": "Identify error hotspot.",
                    "limitations": "Summary only.",
                    "query_requests": ["phase_error_breakdown"],
                    "_llm_usage": {"calls": 1, "input_tokens": 1, "output_tokens": 1, "cost": None},
                }
            rec = (
                ("explore" if role == "exploration" else "finetune")
                if self.conflict
                else "defer"
            )
            return {
                "role": role,
                "snapshot_id": sid,
                "assessment": f"{role} assessment.",
                "recommendation": rec,
                "options": [],
                "evidence_refs": [],
                "expected_outcome": "Outcome.",
                "limitations": "None.",
                "_llm_usage": {"calls": 1, "input_tokens": 1, "output_tokens": 1, "cost": None},
            }
        if mode == "expert_drilldown":
            assert "drilldown_results" in request
            assert "evidence_snapshot" not in request
            return {
                "role": role,
                "snapshot_id": sid,
                "assessment": "Refined after drilldown.",
                "recommendation": "finetune" if self.conflict else "defer",
                "options": [],
                "evidence_refs": [],
                "expected_outcome": "Refined outcome.",
                "limitations": "Bounded.",
                "query_requests": [],
                "_llm_usage": {"calls": 1, "input_tokens": 1, "output_tokens": 1, "cost": None},
            }
        if mode == "expert_rebuttal":
            assert "own_summary" in request and "peer_summary" in request
            assert "evidence_snapshot" not in request
            return {
                "role": role,
                "snapshot_id": sid,
                "recommendation": "explore" if role == "exploration" else "finetune",
                "rebuttal": f"{role} maintains position with evidence.",
                "evidence_refs": [],
                "_llm_usage": {"calls": 1, "input_tokens": 1, "output_tokens": 1, "cost": None},
            }
        return {
            "tool": "pause_search",
            "parameters": {},
            "reason": "Coordinated decision.",
            "_llm_usage": {"calls": 1, "input_tokens": 1, "output_tokens": 1, "cost": None},
        }


def test_conditional_rebuttal_and_drilldown_only_fire_when_needed():
    payload = {
        "mode": "autonomous_search",
        "allowed_tools": ["pause_search"],
        "state": {
            "confirmed_config_version": "c1",
            "active_model_version": "m1",
            "current_project": {"project_path": "synthetic"},
            "tasks": [],
        },
        "decision_context": {},
        "instruction": "Decide next step.",
    }

    # No conflict, no drilldown -> 0 extra calls
    quiet = _ConflictAndDrilldownClient(conflict=False, with_drilldown=False)
    res_quiet = request_with_langgraph(quiet, payload)
    assert [c["mode"] for c in quiet.calls] == [
        "team_consultation",
        "expert_assessment",
        "expert_assessment",
        "autonomous_search",
    ]
    assert "rebuttals" not in res_quiet["_decision_team"]

    # Conflict + 1 drilldown -> exactly 1 drilldown + 1 rebuttal round
    active = _ConflictAndDrilldownClient(conflict=True, with_drilldown=True)
    res_active = request_with_langgraph(active, payload)
    modes = [c["mode"] for c in active.calls]
    assert modes.count("expert_drilldown") == 1
    assert modes.count("expert_rebuttal") == 2
    assert len(res_active["_decision_team"]["rebuttals"]) == 2


def test_joint_convergence_checklist_computes_margin_and_blind_spots():
    from phase_agent.analysis.convergence.joint_convergence_evidence import (
        joint_convergence_checklist,
    )

    state = {
        "phase_diagrams": {
            "mlip": {
                "status": "completed",
                "entries": [
                    {"is_stable": True, "ehull": 0.0},
                    {"is_stable": False, "ehull": 0.015},
                    {"is_stable": False, "ehull": 0.050},
                ],
            }
        },
        "dft_dataset_records": [
            {
                "status": "completed",
                "atom_count": 4,
                "energy": -20.0,
                "mlip_prediction": {"status": "completed", "energy": -20.12},
                "qbc": {"energy_per_atom_std": 0.002},
            }
        ],
        "tasks": [
            {"outputs": {"extrapolation_monitor": {"extrapolation_failure": True}}},
        ],
    }
    post_dft = {
        "gs_metrics": {"energy_per_atom": {"mae": 0.010}},
        "sample_metrics": {"energy_per_atom": {"mae": 0.030}},
    }
    res = joint_convergence_checklist(state, post_dft=post_dft)
    assert res["min_competitor_margin_eV_per_atom"] == 0.015
    assert res["ordering_margin_safe"] is True
    assert res["near_hull_count"] == 2
    assert res["qbc_paired_count"] == 1
    assert res["qbc_blind_spot_count"] == 1
    assert res["qbc_blind_spot_fraction"] == 1.0
    assert res["extrapolation_failure_count"] == 1
    assert res["extrapolation_hotspots"][0]["count"] == 1
    assert res["marginal_gain_saturated"] is None


def test_coverage_and_marginal_matrix_labels_unexplored_high_gain_and_saturated():
    from phase_agent.agents.exploration.branch_pool_review import coverage_and_marginal_matrix

    state = {
        "confirmed_config": {"boundary": {"P": ["O3", "P2", "O1"]}},
    }
    existing = [
        {
            "branch_id": "B-O3",
            "phase": "O3",
            "x": "0.5",
            "mc_task_states": {"current_model:completed": 1},
            "current_model_ehull_ev_per_atom": 0.005,
            "is_hull_ground_state": True,
        },
        {
            "branch_id": "B-P2",
            "phase": "P2",
            "x": "0.5",
            "mc_task_states": {"current_model:completed": 2},
            "current_model_ehull_ev_per_atom": 0.050,
            "is_hull_ground_state": False,
        },
    ]
    matrix = coverage_and_marginal_matrix(state, existing, latest_branch_ids={"B-O3"})
    by_phase = {row["phase"]: row for row in matrix}
    assert by_phase["O3"]["exploration_status"] == "high_gain"
    assert by_phase["O3"]["marginal_gain_last_round"] == 1
    assert by_phase["P2"]["exploration_status"] == "saturated"
    assert by_phase["O1"]["exploration_status"] == "unexplored"


def test_choose_generation_strategy_adapts_to_matrix_and_avoids_periodic_bloat():
    from phase_agent.agents.strategic.strategy.choose_generation_strategy import (
        choose_generation_strategy,
    )

    unexplored_res = choose_generation_strategy(
        {
            "coverage_and_marginal_matrix": [
                {"phase": "O1", "exploration_status": "unexplored"},
                {"phase": "P2", "exploration_status": "unexplored"},
            ]
        },
        total_quota=10,
    )["quotas"]
    assert unexplored_res["coverage"] > unexplored_res["periodic_extension"]

    high_gain_res = choose_generation_strategy(
        {
            "coverage_and_marginal_matrix": [
                {"phase": "O3", "exploration_status": "high_gain"},
                {"phase": "P2", "exploration_status": "high_gain"},
            ]
        },
        total_quota=10,
    )["quotas"]
    assert high_gain_res["composition"] > high_gain_res["coverage"]
    assert high_gain_res["tm_ordering"] > high_gain_res["periodic_extension"]


def test_deterministic_cold_start_and_first_finetune_skip_llm():
    from types import SimpleNamespace
    from phase_agent.tools.workflows.initial_tool_proposal import _deterministic_cold_start_action

    ctx = {
        "manager": SimpleNamespace(
            data={"branches": {}, "structures": {"s1": {"det_H": 2}, "s2": {"det_H": 12}}}
        )
    }
    cold = _deterministic_cold_start_action(
        {}, {}, ["prepare_local_batch_files", "update_mlip"], context=ctx
    )
    assert cold is not None
    assert cold["operation"] == "prepare_parent_first_recovery"
    assert cold["parameters"]["scope_id"] == "initial_direct"
    assert cold["parameters"]["confirmed"] is True
    assert cold["parameters"]["initial_parents"] == ["s1"]
    assert cold["parameters"]["small_supercell_only"] is True
    assert cold["parameters"]["initial_setup"] is True
    assert cold["parameters"]["sampling_method"] == "direct"
    assert cold["parameters"]["eval_mlip_on_selected_frames"] is True
    assert cold["parameters"]["include_baseline_hull_and_errors"] is True

    disabled = _deterministic_cold_start_action(
        {}, {"workflow": {"direct_cold_start_without_llm": False}}, ["prepare_local_batch_files"]
    )
    assert disabled is None

    after_direct = {
        "initial_direct_sampling": {"status": "completed"},
        "dft_dataset_records": [{"structure_id": f"s-{i}", "status": "completed"} for i in range(50)],
    }
    finetune = _deterministic_cold_start_action(after_direct, {}, ["update_mlip"])
    assert finetune is not None
    assert finetune["tool"] == "update_mlip"
    assert finetune["operation"] == "prepare_mlip_training"
    assert finetune["parameters"]["action"] == "RETRAIN_MLIP"


def test_bundle_post_mc_dft_requires_small_supercells(tmp_path):
    from types import SimpleNamespace
    from ase import Atoms
    from ase.io import write as ase_write
    from phase_agent.tools.relax_mc_batch.plan import diagnose

    poscar = tmp_path / "POSCAR"
    ase_write(poscar, Atoms("CoO2", cell=[3, 3, 3], pbc=True), format="vasp")
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"fake-model")

    def _manager(det_h):
        return SimpleNamespace(
            boundary={"P": ["O3"]},
            data={
                "branches": {
                    "b1": {
                        "branch_id": "b1",
                        "P": "O3",
                        "x": "0",
                        "det_H": det_h,
                        "structure_ids": ["s1"],
                    }
                },
                "structures": {
                    "s1": {
                        "structure_id": "s1",
                        "source_path": str(poscar),
                        "composition": {"Co": 1, "O": 2},
                    }
                },
                "system_config": {},
            },
        )

    from phase_agent.configuration.defaults.default_budget_rules import default_budget_rules

    config = {
        "mlip": {"version": "mace-v1", "model_path": str(model_file)},
        "bohb": {"relax_structures_per_branch": 1},
        "budgets": default_budget_rules(),
        "run": {"seed": 1},
    }
    state = {"active_model_version": "mace-v1", "dedup_gate": {"status": "ready"}}

    plan = diagnose(
        ["b1"],
        state,
        _manager(4),
        config,
        "cfg-1",
        phase_references={"O3": poscar},
        bundle_post_mc_dft=True,
        max_dft_frames=40,
    )
    assert plan["prepare_dft_on_remote"] is True
    assert plan["bundled_dft_policy"]["max_dft_frames"] == 40
    assert plan["diagnosis"]["small_supercell_only"] is True

    import pytest

    with pytest.raises(ValueError, match="bundle_post_mc_dft_requires_small_supercells"):
        diagnose(
            ["b1"],
            state,
            _manager(12),
            config,
            "cfg-1",
            phase_references={"O3": poscar},
            bundle_post_mc_dft=True,
        )


def test_deterministic_generation_and_skip_reliability_before_first_mc():
    from types import SimpleNamespace
    from phase_agent.tools.workflows.initial_tool_proposal import (
        _deterministic_generation_action,
        _deterministic_post_generation_relax_mc_action,
    )
    from phase_agent.graphs.decision_team.nodes import consultation_plan

    action = _deterministic_generation_action(
        {
            "decision_context": {
                "branch_pool_facts": {
                    "coverage_and_marginal_matrix": [
                        {"phase": "O1", "exploration_status": "unexplored"},
                    ]
                }
            }
        },
        {"exploration": {"branches_per_round": 8}},
        current={},
    )
    assert action["tool"] == "generate_branches"
    assert action["operation"] == "generate_branches"
    assert sum(action["parameters"]["quotas"].values()) == 8
    assert action["parameters"]["quotas"]["coverage"] > action["parameters"]["quotas"]["periodic_extension"]

    relax_mc = _deterministic_post_generation_relax_mc_action(
        {"active_model_version": "m1", "dedup_gate": {"status": "ready"}},
        {"decision_context": {}},
        {},
        ["prepare_local_batch_files"],
        context={
            "manager": SimpleNamespace(
                data={"branches": {"b1": {"det_H": 4, "structure_ids": ["s1"]}}}
            )
        },
    )
    assert relax_mc is not None
    assert relax_mc["operation"] == "prepare_relax_mc_batch"
    assert relax_mc["target_ids"] == ["b1"]
    assert relax_mc["parameters"]["bundle_post_mc_dft"] is True

    runtime = SimpleNamespace(
        context=SimpleNamespace(
            expert_call=lambda request: {
                "snapshot_id": "snap-1",
                "consult": ["exploration", "reliability"],
                "question": "next?",
            }
        )
    )
    out = consultation_plan(
        {
            "snapshot": {
                "snapshot_id": "snap-1",
                "facts": {"decision_context": {"skip_reliability_before_first_mc": True}},
            }
        },
        runtime,
    )
    assert out["consultation"]["consult"] == ["exploration"]


def test_weighted_kmeans_pp_uses_deterministic_fps_by_default():
    import numpy as np
    from phase_agent.science.qbc.remote_dft_sampling import select_weighted_kmeans_pp

    refs = np.array([[0.0, 0.0]])
    cands = np.array([[0.1, 0.0], [5.0, 0.0], [10.0, 10.0]])
    weights = np.array([1.0, 1.0, 1.0])
    chosen = select_weighted_kmeans_pp(cands, weights, reference_features=refs, n_select=2, seed=0)
    assert chosen == [2, 1]


def test_post_mc_hull_branching_unlocks_exploration_without_stalling():
    from types import SimpleNamespace
    from phase_agent.tools.workflows.initial_tool_proposal import _post_mc_allowed_tools
    from phase_agent.graphs.decision_team.nodes import consultation_plan

    tools = ["prepare_local_batch_files", "generate_branches", "select_dft_candidates"]
    unlocked = _post_mc_allowed_tools(
        tools, {"post_refresh": True, "first_search_completed": True}, post_mc=True, allow_continue_exploration=True
    )
    assert "generate_branches" in unlocked
    assert "prepare_local_batch_files" in unlocked
    assert "select_dft_candidates" in unlocked

    runtime = SimpleNamespace(
        context=SimpleNamespace(
            expert_call=lambda request: {
                "snapshot_id": "snap-2",
                "consult": ["exploration", "reliability"],
                "question": "continue or dft?",
            }
        )
    )
    out = consultation_plan(
        {
            "snapshot": {
                "snapshot_id": "snap-2",
                "facts": {
                    "state": {"remote_dft_sampling": {"b1": {"status": "accepted"}}},
                    "decision_context": {"post_mc_hull_decision_facts": {"unexplored_count": 2}},
                },
            }
        },
        runtime,
    )
    assert out["consultation"]["consult"] == ["exploration", "reliability"]


def test_has_unsearched_latest_generation_prevents_post_mc_loop():
    from types import SimpleNamespace
    from phase_agent.tools.workflows.initial_tool_proposal import _has_unsearched_latest_generation

    ctx = {
        "manager": SimpleNamespace(
            data={
                "branches": {
                    "b1": {"structure_ids": ["s1"]},
                    "b2": {"structure_ids": ["s2"]},
                }
            }
        )
    }
    state = {
        "generation_history": [{"model_version": "m1", "registered_ids": ["s2"]}],
        "tasks": [{"task_id": "t1", "stage": "deep_search", "branch_id": "b1", "model_version": "m1"}],
    }
    assert _has_unsearched_latest_generation(state, "m1", ctx) is True
    state["tasks"].append(
        {"task_id": "t2", "stage": "deep_search", "branch_id": "b2", "model_version": "m1"}
    )
    assert _has_unsearched_latest_generation(state, "m1", ctx) is False


def test_post_mc_orthogonal_arbitration_skips_rebuttal():
    from types import SimpleNamespace
    from phase_agent.evidence.snapshot import evidence_snapshot
    from phase_agent.graphs.decision_team.nodes import coordinated_proposal

    payload = {
        "state": {"active_model_version": "m1"},
        "decision_context": {"post_mc_hull_decision_facts": {"unexplored_count": 0}},
    }
    snap = evidence_snapshot(payload)
    calls = []

    def _expert_call(req):
        calls.append("rebuttal")
        return {}

    def _proposal_call(req):
        calls.append("proposal")
        return {
            "tool": "select_dft_candidates",
            "action_type": "select_dft_candidates",
            "parameters": {
                "decisions": [
                    {"candidate_id": "c1", "action": "DFT_SINGLE_POINT", "reason": "urgent"}
                ]
            },
        }

    runtime = SimpleNamespace(
        context=SimpleNamespace(
            payload=payload,
            expert_call=_expert_call,
            proposal_call=_proposal_call,
        )
    )
    out = coordinated_proposal(
        {
            "snapshot": snap,
            "consultation": {"consult": ["exploration", "reliability"]},
            "advice": [
                {"role": "exploration", "recommendation": "explore"},
                {"role": "reliability", "recommendation": "dft"},
            ],
        },
        runtime,
    )
    assert calls == ["proposal"]
    assert "rebuttals" not in out["proposal"]["_decision_team"]


def test_post_dft_assessment_decision_signals_and_sampling_buffer():
    from phase_agent.evidence.post_dft_assessment import post_dft_assessment
    from tests.test_post_dft_assessment import add_result

    def _state(n_records, gs_err, sample_err):
        state = {}
        for i in range(n_records):
            is_gs = i == 0
            err = gs_err if is_gs else sample_err
            add_result(state, task_id=f"DFT-{i}", energy_error=err, force_error=0.01)
            state["tasks"][-1]["is_ground_state"] = is_gs
        return state

    cfg = {
        "mlip": {"version": "m1"},
        "workflow": {"finetune_min_new_dft": 50},
        "convergence": {"gs_only_convergence_mode": False},
    }
    low_data = post_dft_assessment(_state(10, 0.001, 0.02), cfg)
    assert low_data["untrained_valid_dft_count"] == 10
    assert low_data["post_dft_decision_signals"]["recommended_route_hint"] == "accumulate_dft_or_explore"

    enough_data = post_dft_assessment(_state(50, 0.001, 0.02), cfg)
    assert enough_data["post_dft_decision_signals"]["data_sufficient_for_finetune"] is True
    assert enough_data["post_dft_decision_signals"]["recommended_route_hint"] == "finetune"

    gs_only_cfg = {
        "mlip": {"version": "m1"},
        "workflow": {"finetune_min_new_dft": 50},
        "convergence": {"gs_only_convergence_mode": True},
    }
    gs_ok = post_dft_assessment(_state(50, 0.001, 0.02), gs_only_cfg)
    assert gs_ok["post_dft_decision_signals"]["gs_converged"] is True
    assert gs_ok["post_dft_decision_signals"]["recommended_route_hint"] == "explore_or_converge"

    accurate = post_dft_assessment(_state(50, 0.001, 0.002), cfg)
    assert accurate["post_dft_decision_signals"]["recommended_route_hint"] == "explore_or_converge"


def test_clean_project_settings_and_sampling_features_preserved():
    from phase_agent.configuration.runtime.project_template import CLEAN_PROJECT_SETTINGS

    assert CLEAN_PROJECT_SETTINGS["editable_config_draft_path"] == "inputs/search_config.json"
    assert CLEAN_PROJECT_SETTINGS["phase_diagram_directory"] == "outputs"
    assert CLEAN_PROJECT_SETTINGS["new_runs_directory"] == "runs"
    assert CLEAN_PROJECT_SETTINGS["state_path"].startswith(".phase/")
    prefs = CLEAN_PROJECT_SETTINGS["output_preferences"]
    assert "png" in prefs["formats"]
    assert "pdf" in prefs["exclude_formats"]
    assert prefs["concise_summary"] is True
    assert prefs["save_sampling_features"] is True


def test_compact_project_storage_prunes_settled_duplicates_safely(tmp_path):
    import json
    from phase_agent.tools.local.compact_project_storage import compact_project_storage

    settled_upload = tmp_path / "runs" / "b1" / "upload_tmp"
    settled_upload.mkdir(parents=True)
    (settled_upload / "dup.vasp").write_text("dummy" * 100, encoding="utf-8")

    active_upload = tmp_path / "runs" / "b2" / "upload_tmp"
    active_upload.mkdir(parents=True)
    (active_upload / "active.vasp").write_text("keep", encoding="utf-8")

    ref_dir = tmp_path / "runs" / "b3" / "upload_tmp"
    ref_dir.mkdir(parents=True)
    ref_file = ref_dir / "keep_ref.vasp"
    ref_file.write_text("referenced", encoding="utf-8")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "tasks": [{"task_id": "t1", "structure_path": str(ref_file)}],
                "relax_mc_batches": {
                    "b1": {
                        "status": "results_received",
                        "import_checksum": "sha256:abc",
                        "upload_folder": str(settled_upload),
                    },
                    "b2": {"status": "prepared", "upload_folder": str(active_upload)},
                    "b3": {
                        "status": "results_received",
                        "import_checksum": "sha256:def",
                        "upload_folder": str(ref_dir),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    dry = compact_project_storage(state_file, dry_run=True, archive_state=False)
    assert dry["status"] == "dry_run"
    assert len(dry["pruned_folders"]) == 1
    assert settled_upload.is_dir()

    compacted = compact_project_storage(state_file, dry_run=False, archive_state=False)
    assert compacted["status"] == "compacted"
    assert not settled_upload.exists()
    assert active_upload.is_dir()
    assert ref_file.is_file()


def test_mp_competing_phase_repo_and_post_dft_global_ehull(tmp_path):
    from phase_agent.analysis.phase.mp_global_ehull import (
        CompetingPhaseRepo,
        compute_gs_global_dft_ehull,
    )
    from phase_agent.evidence.post_dft_assessment import post_dft_assessment
    from tests.test_post_dft_assessment import add_result

    repo = CompetingPhaseRepo(tmp_path / "mp_repo.json")
    params_pbe_u = {"functional": "PBE+U", "U": {"Co": 3.32}, "ENCUT": 520}
    params_other = {"functional": "r2SCAN", "ENCUT": 680}
    competing = [
        {
            "mp_id": "mp-1",
            "formula": "Na",
            "composition": {"Na": 1},
            "dft_energy_eV": -1.3,
            "geometry_sha256": "g-na",
        },
        {
            "mp_id": "mp-2",
            "formula": "Co",
            "composition": {"Co": 1},
            "dft_energy_eV": -7.0,
            "geometry_sha256": "g-co",
        },
        {
            "mp_id": "mp-3",
            "formula": "O2",
            "composition": {"O": 2},
            "dft_energy_eV": -9.8,
            "geometry_sha256": "g-o2",
        },
    ]
    assert repo.register(competing, params_pbe_u) == 3
    assert len(repo.lookup(["Na", "Co", "O"], params_pbe_u)) == 3
    assert repo.lookup(["Na", "Co", "O"], params_other) == []

    # Pure DFT global Ehull check: NaCoO2 at -20.0 eV is stable vs (-1.3 - 7.0 - 9.8 = -18.1 eV)
    stable_eval = compute_gs_global_dft_ehull(
        [{"task_id": "t1", "structure_id": "s1", "composition": {"Na": 1, "Co": 1, "O": 2}, "energy": -20.0}],
        repo.lookup(["Na", "Co", "O"], params_pbe_u),
    )
    assert stable_eval["status"] == "evaluated"
    assert stable_eval["unstable_gs_count"] == 0
    assert stable_eval["entries"][0]["is_globally_stable"] is True

    # Integration inside post_dft_assessment
    state = {}
    add_result(state, task_id="DFT-GS-1", energy_error=0.001, force_error=0.01)
    state["tasks"][-1]["is_ground_state"] = True
    state["dft_dataset_records"][-1]["composition"] = {"Na": 1, "Co": 1, "O": 2}
    state["dft_dataset_records"][-1]["energy"] = -20.0
    cfg = {
        "mlip": {"version": "m1"},
        "dft": {"parameters": params_pbe_u},
        "mp_competing_phases": {"enabled": True, "shared_repo_path": str(tmp_path / "mp_repo.json")},
    }
    assessment = post_dft_assessment(state, cfg)
    assert assessment["gs_global_dft_ehull"]["status"] == "evaluated"
    assert assessment["gs_global_dft_ehull"]["competing_phase_count"] == 3
    assert assessment["gs_global_dft_ehull"]["unstable_gs_count"] == 0


def test_fetch_mp_stable_binaries_uses_chemsys_subsystems():
    from types import SimpleNamespace
    from pymatgen.core import Lattice, Structure
    from phase_agent.analysis.phase.mp_global_ehull import fetch_mp_stable_binaries

    captured = {}

    class FakeRester:
        def __init__(self, key):
            captured["key"] = key

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @property
        def materials(self):
            class _Summary:
                @staticmethod
                def search(**kwargs):
                    captured["search"] = kwargs
                    s = Structure(Lattice.cubic(3.0), ["Na"], [[0, 0, 0]])
                    return [
                        SimpleNamespace(
                            material_id="mp-10",
                            formula_pretty="Na",
                            structure=s,
                            energy_above_hull=0.0,
                        )
                    ]

            return SimpleNamespace(summary=_Summary())

    rows = fetch_mp_stable_binaries(["Na", "O"], api_key="k", max_arity=2, rester_cls=FakeRester)
    assert captured["key"] == "k"
    assert set(captured["search"]["chemsys"]) == {"Na", "O", "Na-O"}
    assert captured["search"]["is_stable"] is True
    assert len(rows) == 1 and rows[0]["composition"] == {"Na": 1.0} and "structure" in rows[0]


def test_export_gs_global_dft_ehull_csv(tmp_path):
    import csv
    from phase_agent.analysis.phase.mp_global_ehull import export_gs_global_dft_ehull_csv

    out_csv = export_gs_global_dft_ehull_csv(
        {
            "status": "evaluated",
            "entries": [
                {
                    "task_id": "t1",
                    "structure_id": "s1",
                    "composition": {"Na": 1, "Co": 1, "O": 2},
                    "dft_energy_eV": -20.0,
                    "dft_global_ehull_eV_per_atom": 0.0,
                    "is_globally_stable": True,
                    "decomposition_products": {"s1": 1.0},
                }
            ],
        },
        tmp_path / "gs_global_dft_ehull.csv",
    )
    assert out_csv.is_file()
    with out_csv.open("r", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["task_id"] == "t1"
    assert rows[0]["dft_global_ehull_eV_per_atom"] == "0.0"
    assert rows[0]["is_globally_stable"] == "True"


def test_prepare_mp_competing_phases_merges_local_and_missing_mp(tmp_path):
    from types import SimpleNamespace
    import hashlib
    import json
    from pymatgen.core import Lattice, Structure
    from phase_agent.analysis.phase.mp_global_ehull import prepare_mp_competing_phases

    s_na = Structure(Lattice.cubic(3.0), ["Na"], [[0, 0, 0]])
    s_o2 = Structure(Lattice.cubic(4.0), ["O", "O"], [[0, 0, 0], [0.3, 0, 0]])
    h_na = hashlib.sha256(
        json.dumps(s_na.as_dict(), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    class FakeRester:
        def __init__(self, key):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @property
        def materials(self):
            class _Summary:
                @staticmethod
                def search(**kwargs):
                    return [
                        SimpleNamespace(structure=s_na),
                        SimpleNamespace(structure=s_o2),
                    ]

            return SimpleNamespace(summary=_Summary())

    res = prepare_mp_competing_phases(
        ["Na", "O"],
        tmp_path / "repo.json",
        dft_parameters={"potcar_functional": "PBE_54"},
        local_entries=[
            {
                "composition": {"Na": 1.0},
                "structure": s_na.as_dict(),
                "geometry_sha256": h_na,
                "dft_energy_eV": -1.3,
            }
        ],
        api_key="k",
        rester_cls=FakeRester,
    )
    assert res["status"] == "dft_required"
    assert len(res["cached_entries"]) == 1
    assert len(res["pending_dft_tasks"]) == 1
    assert res["pending_dft_tasks"][0]["composition"] == {"O": 2.0}
    assert res["pending_dft_tasks"][0]["parameter_source"] == "mp_defaults"
