import unittest

from phase_agent.science.bohb.collect_bohb_results import collect_bohb_results
from experiments.bohb.compare_bohb_methods import compare_bohb_methods
from phase_agent.science.bohb.default_bohb_config import default_bohb_config
from experiments.bohb.evaluate_budget_correlation import evaluate_budget_correlation
from phase_agent.science.bohb.run_bohb_iteration import run_bohb_iteration


def case():
    config = default_bohb_config()
    config["scope"] = {"mlip_version": "m1", "hull_reference_version": "h1", "candidate_set_version": "c1"}
    config["new_candidates_per_iteration"] = 6
    candidates = [{"branch_id": f"B{i}", "P": "O3", "x": str(i % 2), "T": ["Fe"], "composition_group": f"x={i % 2}", "bohb_features": [i % 2, i / 10], "important": i == 0} for i in range(12)]
    return config, candidates


def simulator(*, branch, action):
    value = int(branch["branch_id"][1:]) / 10 + 1 / action["budget"]
    return {"status": "completed", "minimum_energy_per_atom": value, "composition_group": branch["composition_group"], "group_reference_energy_per_atom": 0.0, "group_energy_scale": 1.0, "actual_cost": action["incremental_budget"], "checkpoint": f"cp-{branch['branch_id']}-{action['budget']}"}


class BohbTest(unittest.TestCase):
    def test_pending_recovery_is_idempotent(self):
        config, candidates = case()
        first = run_bohb_iteration(candidates, None, config=config, total_mc_budget=100, seed=1)
        self.assertEqual(first["actions"][0]["incremental_budget"], 10)
        self.assertEqual(first["actions"][0]["planned_relative_cost"], 1)
        task = first["state"]["pending_tasks"][0]
        result = {**simulator(branch=candidates[int(task["branch_id"][1:])], action=task), "task_key": task["task_key"]}
        state = collect_bohb_results(first["state"], candidates, [result], config=config)
        repeated = collect_bohb_results(state, candidates, [result], config=config)
        self.assertEqual(state["consumed_mc_budget"], repeated["consumed_mc_budget"])

    def test_promotion_and_scope_isolation(self):
        config, candidates = case()
        first = run_bohb_iteration(candidates, None, config=config, total_mc_budget=300, seed=2, evaluator=simulator)
        second = run_bohb_iteration(candidates, first["state"], config=config, total_mc_budget=300, seed=3, evaluator=simulator)
        self.assertTrue(any(item["budget"] == 30 for item in second["state"]["observations"]))
        changed = {**config, "scope": {**config["scope"], "mlip_version": "m2"}}
        isolated = run_bohb_iteration(candidates, second["state"], config=changed, total_mc_budget=300, seed=4)
        self.assertEqual(isolated["status"], "scope_changed")

    def test_correlation_and_equal_budget_comparison(self):
        config, candidates = case()
        comparison = compare_bohb_methods(candidates, config=config, total_mc_budget=180, seed=5, simulator=simulator)
        self.assertEqual(comparison["status"], "completed")
        self.assertTrue(all(item["actual_total_cost"] <= 180 for item in comparison["results"].values()))
        rows = [{"branch_id": f"B{i}", "budget": budget, "loss": float(i), "status": "completed"} for budget in (10, 30) for i in range(5)]
        correlation = evaluate_budget_correlation(rows, low_budget=10, high_budget=30)
        self.assertAlmostEqual(correlation["spearman"], 1.0)


if __name__ == "__main__":
    unittest.main()
