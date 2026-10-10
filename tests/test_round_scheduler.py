import tempfile
import unittest

from phase_agent.decisions.strategy.default_round_strategy_config import default_round_strategy_config
from phase_agent.persistence.state.load_round_scheduler_state import load_round_scheduler_state
from phase_agent.persistence.state.pause_strategy_round import pause_strategy_round
from phase_agent.tools.workflows.run_round_scheduler import run_round_scheduler
from phase_agent.persistence.state.save_round_scheduler_state import save_round_scheduler_state
from phase_agent.science.bohb.default_bohb_config import default_bohb_config


class RoundSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.candidates = [{"branch_id": f"B{i}", "P": "O3", "x": i / 10, "T": ["Fe", "Mn"], "region_id": "r1" if i < 3 else "r2", "composition_group": str(i % 2), "bohb_features": [i]} for i in range(6)]
        self.bohb = default_bohb_config(); self.bohb["scope"] = {"mlip_version": "m1", "hull_reference_version": "h1", "candidate_set_version": "c1"}; self.bohb["new_candidates_per_iteration"] = 2
        self.strategy = default_round_strategy_config(); self.strategy["rule_default"].update({"mc_budget": 100, "dft_budget": 5})
        self.summary = {"known_region_ids": ["r1"], "seed": 3}

    def test_normal_fallback_resume_and_duplicate_invocation(self):
        llm = lambda _: {"focus_regions": ["r1"], "generation_quotas": {"coverage": 3}, "mc_budget": 100, "dft_budget": 5, "exploration_fraction": .2, "reason": "focus gap"}
        first = run_round_scheduler(None, self.candidates, summary=self.summary, bohb_config=self.bohb, strategy_config=self.strategy, invocation_id="call-1", agent_client=llm)
        self.assertTrue(first["actions"]); self.assertEqual(first["decision_source"], "llm_agent")
        self.assertEqual(first["round"]["branch_selection_owner"], "agent")
        self.assertEqual(first["round"]["mc_fidelity_owner"], "hyperband")
        self.assertEqual(
            {item["selection_source"] for item in first["actions"]},
            {"agent_recommended_scope", "global_exploration"},
        )
        by_id = {item["branch_id"]: item for item in self.candidates}
        recommended = [item for item in first["actions"] if item["selection_source"] == "agent_recommended_scope"]
        explored = [item for item in first["actions"] if item["selection_source"] == "global_exploration"]
        self.assertTrue(all(by_id[item["branch_id"]]["region_id"] == "r1" for item in recommended))
        self.assertTrue(all(by_id[item["branch_id"]]["region_id"] != "r1" for item in explored))
        repeated = run_round_scheduler(first["state"], self.candidates, summary=self.summary, bohb_config=self.bohb, strategy_config=self.strategy, invocation_id="call-1", agent_client=lambda _: (_ for _ in ()).throw(RuntimeError()))
        self.assertTrue(repeated["idempotent_replay"]); self.assertEqual(repeated["actions"], first["actions"])
        with tempfile.TemporaryDirectory(dir=".") as directory:
            path = f"{directory}/state.json"; save_round_scheduler_state(first["state"], path); restored = load_round_scheduler_state(path)
        resumed = run_round_scheduler(restored, self.candidates, summary=self.summary, bohb_config=self.bohb, strategy_config=self.strategy, invocation_id="call-2")
        self.assertFalse({a["task_key"] for a in first["actions"]} & {a["task_key"] for a in resumed["actions"]})

    def test_invalid_llm_falls_back_and_scope_change_is_blocked(self):
        invalid = lambda _: {"branch_ids": ["B1"], "mc_budget": 99999}
        result = run_round_scheduler(None, self.candidates, summary=self.summary, bohb_config=self.bohb, strategy_config=self.strategy, invocation_id="bad", agent_client=invalid)
        self.assertTrue(result["round"]["fallback_used"]); self.assertEqual(result["decision_source"], "rule")
        changed = dict(self.bohb); changed["scope"] = dict(self.bohb["scope"]); changed["scope"]["mlip_version"] = "m2"
        blocked = run_round_scheduler(result["state"], self.candidates, summary=self.summary, bohb_config=changed, strategy_config=self.strategy, invocation_id="changed")
        self.assertEqual(blocked["status"], "scope_changed")
        paused = pause_strategy_round(result["state"]["active_round"], reason="model update")
        self.assertEqual(paused["status"], "paused")

    def test_agent_cannot_assign_branch_or_branch_budget(self):
        invalid = lambda _: {
            "focus_regions": ["r1"], "generation_quotas": {},
            "mc_budget": 100, "dft_budget": 5, "exploration_fraction": .2,
            "branch_budgets": {"B1": 90}, "reason": "manual allocation",
        }
        result = run_round_scheduler(None, self.candidates, summary=self.summary, bohb_config=self.bohb, strategy_config=self.strategy, invocation_id="branch-budget", agent_client=invalid)
        self.assertTrue(result["round"]["fallback_used"])
        self.assertEqual(result["decision_source"], "rule")


if __name__ == "__main__":
    unittest.main()
