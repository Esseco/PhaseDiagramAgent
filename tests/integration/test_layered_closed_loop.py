import tempfile
import unittest

from config_layer.session.apply_config_revision import apply_config_revision
from config_layer.session.confirm_config_snapshot import confirm_config_snapshot
from config_layer.session.create_config_draft import create_config_draft
from config_layer.defaults.default_layered_search_config import default_layered_search_config
from config_layer.defaults.default_budget_rules import default_budget_rules
from analysis_layer.convergence.check_global_convergence import check_global_convergence
from data_layer.ledger.collect_calculation_results import collect_calculation_results
from data_layer.ledger.load_legacy_ledger import load_legacy_ledger
from data_layer.ledger.phase_data_manager import PhaseDataManager
from scientific_layer.structures.create_generation_registry import create_generation_registry
from execution_layer.state.normalize_task import normalize_task
from execution_layer.state.normalize_task_result import normalize_task_result
from execution_layer.budget.reserve_budget import reserve_budget
from execution_layer.budget.settle_budget import settle_budget


class LayeredClosedLoopTest(unittest.TestCase):
    def test_config_generation_mock_collection_feedback_budget_and_resume(self):
        boundary = {"P": ["O3"], "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]}, "TM_ratio": {"Fe": 1, "Mn": 1}}
        session = create_config_draft(default_layered_search_config(boundary=boundary))
        session = apply_config_revision(session, {"calculation.mlip_version": "m1", "dft.parameters": {"encut": 520}, "convergence.hull_tolerance": .01, "convergence.minimum_dft_validations": 1, "convergence.coverage_threshold": .9})
        session = confirm_config_snapshot(session, user_confirmed=True)
        self.assertEqual(session["status"], "confirmed")
        self.assertIn("coverage", create_generation_registry().names())

        manager = PhaseDataManager(boundary)
        branch = manager.add_branch(P="O3", H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], x=.5, T=["Fe", "Mn"], composition={"Na": 1, "Fe": 1, "Mn": 1, "O": 4})
        structure = manager.add_structure(branch_id=branch, arrangement={"V": [1, 0]})
        task = normalize_task({"task_id": "T1", "object_id": structure, "structure_id": structure, "stage": "simple_check", "config_version": session["confirmed_snapshot"]["config_version"], "model_version": "m1"})
        limits = default_budget_rules(); reserved = reserve_budget({}, task_key="T1", stage="simple_check", amount=.05, limits=limits)
        result = normalize_task_result(task, {"status": "completed", "converged": True, "outputs": {"checks_passed": True}, "actual_cost": .03})
        collected = collect_calculation_results(manager, structure, result)
        self.assertIsNotNone(collected["result_id"])
        settled = settle_budget(reserved["state"], task_key="T1", settlement_id="R1", task_status="completed", actual_cost=.03)
        self.assertEqual(settled["state"]["budget_usage"]["total_relative_cost"], .03)
        convergence = check_global_convergence({"tasks": [], "model_update_epochs": [{"model_version": "m1", "hull_change": .002, "ground_state_unchanged": True}, {"model_version": "m2", "hull_change": .001, "ground_state_unchanged": True}], "final_frame_dft_errors": [{"error_ev_per_atom": .002}], "budget_remaining": 1}, rules={"hull_change_tolerance": .003, "stable_model_update_epochs": 2, "final_energy_mae_tolerance": .003, "minimum_recent_dft_checks": 1})
        self.assertEqual(convergence["status"], "numerical_criteria_satisfied")
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = f"{directory}/phase.json"; manager.save(path); loaded = load_legacy_ledger(path)
        self.assertEqual(set(loaded.data["branches"]), {branch})


if __name__ == "__main__": unittest.main()
