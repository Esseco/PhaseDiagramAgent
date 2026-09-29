import unittest

from analysis_layer.convergence.check_global_convergence import check_global_convergence
from analysis_layer.feedback.validate_branch_static_energies import validate_branch_static_energies
from execution_layer.state.reconcile_task_results import reconcile_task_results
from scientific_layer.mc.schedule_tiered_mc import schedule_tiered_mc
from config_layer.runtime.validate_energy_conventions import validate_energy_conventions
from decision_layer.scoring.rank_branch_relax_prescreen import rank_branch_relax_prescreen


POLICY = {
    "tiers": [
        {"name": "small", "max_mc_steps": 10, "patience_steps": 3, "min_improvement": .01},
        {"name": "large", "max_mc_steps": 30, "patience_steps": 6, "min_improvement": .01},
    ],
    "max_segments_per_branch": 2, "max_cumulative_cost_per_branch": 50,
    "random_exploration_fraction": 0, "high_ehull_defer_threshold": .3,
    "near_hull_retry_threshold": .1,
}


class TieredClosedLoopTest(unittest.TestCase):
    def test_relax_prescreen_filters_illegal_and_prefers_coverage_per_cost(self):
        result = rank_branch_relax_prescreen([
            {"branch_id": "bad", "legal": False},
            {"branch_id": "cheap-gap", "P": "O3", "x": .5,
             "phase_soc_coverage_gap": 1, "estimated_relax_cost": 1},
            {"branch_id": "costly", "P": "P3", "x": .5,
             "phase_soc_coverage_gap": 1, "estimated_relax_cost": 10}],
            count=1, seed=1, random_fraction=0)
        self.assertEqual(result["selected"][0]["branch_id"], "cheap-gap")
        self.assertEqual(result["selected"][0]["phase_soc_region"], "O3:x=0.5")

    def test_small_first_and_patience_near_hull_retries_new_segment(self):
        candidate = {"branch_id": "B1", "relaxed_ehull": .02,
                     "structure_path": "initial.vasp", "structure_id": "S1"}
        first = schedule_tiered_mc([candidate], {}, policy=POLICY, total_budget=100,
                                   seed=1, model_version="m1", hull_reference_version="h1")
        self.assertEqual(first["actions"][0]["tier"], "small")
        state = first["state"]
        state["segments"][0].update(status="completed", result_path="lowest.vasp",
                                    stop_reason="patience",
                                    actual_mc_steps=4, actual_gpu_core_hours=1)
        second = schedule_tiered_mc([candidate], state, policy=POLICY, total_budget=100,
                                    seed=1, model_version="m1", hull_reference_version="h1")
        self.assertEqual(second["actions"][0]["segment_reason"], "near_hull_patience_retry_new_seed")
        self.assertEqual(second["actions"][0]["restart_mode"], "new_segment_from_structure")

    def test_append_limit_and_high_hull_defer(self):
        high = {"branch_id": "B2", "relaxed_ehull": .8}
        result = schedule_tiered_mc([high], {}, policy=POLICY, total_budget=100,
                                    seed=1, model_version="m1", hull_reference_version="h1")
        self.assertEqual(result["actions"], [])

    def test_na0_endpoint_is_excluded_from_na_v_mc(self):
        candidates = [
            {"branch_id": "Na0", "x": "0/1", "relaxed_ehull": .01},
            {"branch_id": "NaHalf", "x": "1/2", "relaxed_ehull": .02},
        ]
        result = schedule_tiered_mc(candidates, {}, policy=POLICY, total_budget=100,
                                    seed=1, model_version="m1", hull_reference_version="h1")
        self.assertEqual([row["branch_id"] for row in result["actions"]], ["NaHalf"])
        self.assertEqual(result["excluded_candidates"], [
            {"branch_id": "Na0", "reason": "no_mobile_ion_sites"}
        ])

    def test_equal_hull_mc_allocation_rotates_phase_and_composition(self):
        policy = {**POLICY, "max_cumulative_cost_per_branch": 200,
                  "initial_ehull_bands": [
                      {"max_ehull_ev_per_atom": .01, "max_mc_steps": 100, "patience_steps": 8},
                      {"max_ehull_ev_per_atom": .02, "max_mc_steps": 60, "patience_steps": 6},
                      {"max_ehull_ev_per_atom": .04, "max_mc_steps": 30, "patience_steps": 4},
                  ]}
        candidates = [
            {"branch_id": "OP2-x1", "P": "OP2", "x": "1/4", "relaxed_ehull": .005},
            {"branch_id": "OP2-x2", "P": "OP2", "x": "1/2", "relaxed_ehull": .005},
            {"branch_id": "OP2-x3", "P": "OP2", "x": "3/4", "relaxed_ehull": .005},
            {"branch_id": "O3-x1", "P": "O3", "x": "1/4", "relaxed_ehull": .005},
            {"branch_id": "P3-x1", "P": "P3", "x": "1/4", "relaxed_ehull": .005},
        ]
        result = schedule_tiered_mc(candidates, {}, policy=policy, total_budget=300,
                                    seed=4, model_version="m1", hull_reference_version="h1")
        self.assertEqual(len(result["actions"]), 3)
        phase_by_branch = {row["branch_id"]: row["P"] for row in candidates}
        self.assertEqual({phase_by_branch[row["branch_id"]] for row in result["actions"]},
                         {"OP2", "O3", "P3"})

    def test_relax_energy_sigma_prioritizes_candidates_within_same_phase_and_x(self):
        policy = {**POLICY, "max_cumulative_cost_per_branch": 200,
                  "initial_ehull_bands": [
                      {"max_ehull_ev_per_atom": .01, "max_mc_steps": 10, "patience_steps": 3},
                      {"max_ehull_ev_per_atom": .02, "max_mc_steps": 20, "patience_steps": 4},
                  ]}
        candidates = [
            {"branch_id": "low-sigma", "P": "O3", "x": "1/2", "relaxed_ehull": .005,
             "branch_energy_std_per_atom": .001, "allocation_score": .00475},
            {"branch_id": "high-sigma", "P": "O3", "x": "1/2", "relaxed_ehull": .005,
             "branch_energy_std_per_atom": .01, "allocation_score": .0025},
        ]
        result = schedule_tiered_mc(candidates, {}, policy=policy, total_budget=10,
                                    seed=4, model_version="m1", hull_reference_version="h1")
        self.assertEqual(result["actions"][0]["branch_id"], "high-sigma")

    def test_budget_exhaustion_model_switch_and_resume_are_distinct(self):
        candidate = {"branch_id": "B1", "relaxed_ehull": .02}
        exhausted = schedule_tiered_mc([candidate], {}, policy=POLICY, total_budget=0,
                                       seed=2, model_version="m1", hull_reference_version="h1")
        self.assertEqual(exhausted["status"], "budget_exhausted")
        first = schedule_tiered_mc([candidate], {}, policy=POLICY, total_budget=20,
                                   seed=2, model_version="m1", hull_reference_version="h1")
        restored = schedule_tiered_mc([candidate], first["state"], policy=POLICY, total_budget=20,
                                      seed=2, model_version="m1", hull_reference_version="h1")
        self.assertEqual(restored["status"], "tasks_in_progress")
        switched = schedule_tiered_mc([candidate], {}, policy=POLICY, total_budget=20,
                                      seed=2, model_version="m2", hull_reference_version="h2")
        self.assertNotEqual(first["actions"][0]["task_key"], switched["actions"][0]["task_key"])

    def test_partial_and_duplicate_recovery_charge_once_and_release(self):
        state = {"tasks": [{"task_id": "T1", "task_key": "K1", "status": "pending"}],
                 "budget_reservations": {"K1": {"status": "reserved", "reserved_cost": 10,
                                                  "stage": "deep_search"}},
                 "reserved_relative_cost": 10}
        partial = reconcile_task_results(state, [{"task_id": "T1", "task_key": "K1", "status": "running",
                                                  "actual_mc_steps": 3}])
        done = reconcile_task_results(partial["state"], [{"task_id": "T1", "task_key": "K1",
            "status": "completed", "actual_cost": 4, "stop_reason": "patience"}])
        repeated = reconcile_task_results(done["state"], [{"task_id": "T1", "task_key": "K1",
            "status": "completed", "actual_cost": 4}])
        self.assertEqual(done["state"]["budget_usage"]["total_relative_cost"], 4)
        self.assertEqual(done["state"]["budget_reservations"]["K1"]["released_cost"], 6)
        self.assertEqual(repeated["state"]["budget_usage"]["total_relative_cost"], 4)

    def test_early_stop_releases_estimated_reservation_without_claiming_gpu_hours(self):
        state = {"tasks": [{"task_id": "T2", "task_key": "K2", "status": "pending"}],
                 "budget_reservations": {"K2": {"status": "reserved", "reserved_cost": 10,
                                                  "stage": "deep_search"}},
                 "reserved_relative_cost": 10}
        result = reconcile_task_results(state, [{"task_id": "T2", "task_key": "K2", "status": "completed",
            "requested_max_mc_steps": 10, "actual_mc_steps": 4, "stop_reason": "patience"}])
        task = result["state"]["tasks"][0]
        self.assertEqual(result["state"]["budget_usage"]["total_relative_cost"], 4)
        self.assertEqual(result["state"]["budget_usage"]["estimated_total_relative_cost"], 4)
        self.assertEqual(result["state"]["budget_reservations"]["K2"]["released_cost"], 6)
        self.assertIsNone(task["actual_gpu_core_hours"])
        self.assertIsNone(task["actual_cost"])
        self.assertEqual(task["estimated_cost"], 4)
        self.assertIn("estimated", task["estimated_cost_basis"])

    def test_two_model_epochs_require_user_acceptance_and_coverage_is_evidence(self):
        state = {"model_update_epochs": [
                    {"model_version": "m1", "hull_change": .001, "ground_state_unchanged": True},
                    {"model_version": "m2", "hull_change": .001, "ground_state_unchanged": True}],
                 "final_frame_dft_errors": [{"error_ev_per_atom": .001}],
                 "coverage": {"fraction": .1}, "budget_remaining": 10}
        rules = {"hull_change_tolerance": .003, "stable_model_update_epochs": 2,
                 "final_energy_mae_tolerance": .003, "minimum_recent_dft_checks": 1}
        pending = check_global_convergence(state, rules=rules)
        self.assertEqual(pending["status"], "numerical_criteria_satisfied")
        self.assertFalse(pending["converged"])
        state["user_accepted_convergence"] = True
        self.assertEqual(check_global_convergence(state, rules=rules)["status"], "finished")

    def test_branch_grouped_mae_rejects_mixed_model_and_geometry(self):
        result = validate_branch_static_energies([
            {"record_id": "1", "branch_id": "B1", "geometry_id": "G1", "model_version": "m2",
             "mlip_energy": -1.0, "dft_energy": -.9, "energy_unit": "eV/atom"},
            {"record_id": "2", "branch_id": "B1", "geometry_id": "G2", "dft_geometry_id": "G3",
             "model_version": "m2", "mlip_energy": -1, "dft_energy": -1, "energy_unit": "eV/atom"},
            {"record_id": "3", "branch_id": "B2", "geometry_id": "G4", "model_version": "m1",
             "mlip_energy": -1, "dft_energy": -1, "energy_unit": "eV/atom"}], model_version="m2")
        self.assertAlmostEqual(result["by_branch"]["B1"]["mae_ev_per_atom"], .1)
        self.assertEqual(len(result["rejected"]), 2)

    def test_energy_reference_records_source_without_claiming_review(self):
        result = validate_energy_conventions({"oxygen_reference_unit": "eV/O2",
            "model_error_unit": "eV/atom", "voltage_references": {
                "Na": {"value": -1.2, "source": "user supplied"}}})
        self.assertTrue(result["valid"])
        self.assertFalse(result["voltage_references"]["Na"]["reliability_reviewed_by_program"])


if __name__ == "__main__":
    unittest.main()
