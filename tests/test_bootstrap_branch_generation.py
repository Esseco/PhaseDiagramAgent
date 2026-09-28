import unittest
from types import SimpleNamespace
from unittest.mock import patch

from execution_layer.workflows.create_active_learning_handlers import _generate_branches
from scientific_layer.structures.generate_coverage_branches import generate_coverage_branches
from scientific_layer.structures.select_candidates import select_candidates
from run.open_webui_api import format_workflow_reply


class BootstrapBranchGenerationTest(unittest.TestCase):
    def test_empty_ledger_assigns_generation_to_coverage(self):
        manager = SimpleNamespace(data={"branches": {}}, boundary={"P": ["O1", "O3"]})
        result = {"registered": [], "coverage": [], "summary": {"registered_branches": 0}}
        context = {"manager": manager, "phase_references": {}, "event_state": {},
                   "effective_config": {"total_quota": 20, "batch_size": 8,
                                        "initial_states_per_branch": 3, "seed": 42,
                                        "structure_directory": "unused"}}
        with patch("execution_layer.workflows.create_active_learning_handlers.run_branch_generation",
                   return_value=result) as generator:
            _generate_branches(action={"task_key": "first", "parameters": {}}, context=context)
        self.assertEqual(generator.call_args.kwargs["quotas"], {"coverage": 20})

    def test_missing_phases_use_coverage_even_when_old_branches_exist(self):
        manager = SimpleNamespace(data={"branches": {"B-old": {"P": "O1"}}},
                                  boundary={"P": ["O1", "O3", "P3", "OP2"]})
        context = {"manager": manager, "phase_references": {}, "event_state": {},
                   "effective_config": {"total_quota": 20, "batch_size": 8,
                                        "initial_states_per_branch": 3, "seed": 42,
                                        "structure_directory": "unused"}}
        with patch("execution_layer.workflows.create_active_learning_handlers.run_branch_generation",
                   return_value={"registered": [], "coverage": [], "summary": {}}) as generator:
            _generate_branches(action={"task_key": "next", "parameters": {}}, context=context)
        self.assertEqual(generator.call_args.kwargs["quotas"], {"coverage": 20})

    def test_coverage_round_robins_phases_and_prefers_small_cells(self):
        phases = ["O1", "O3", "OP2", "P3"]
        frameworks = []
        for phase in phases:
            for size in (12, 4):
                frameworks.append({"P": phase,
                                   "H": [[size, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   "det_H": size, "atom_count_full": size * 4,
                                   "allowed_x": ["0" if phase == "O1" else "1/2"]})
        manager = SimpleNamespace(data={"branches": {}}, boundary={})

        def identify(_structure, _boundary, *, phase_hint, **_kwargs):
            framework = _structure
            return {"P": phase_hint, "H": framework["H"],
                    "det_H": framework["det_H"], "x": framework["allowed_x"][0],
                    "T": ["Fe", "Mn"], "composition": {}}

        with patch("scientific_layer.structures.generate_coverage_branches.generate_branch_structure",
                   side_effect=lambda _boundary, *, H, phase, **_kwargs: next(
                       item for item in frameworks if item["H"] == H and item["P"] == phase)), patch(
                   "scientific_layer.structures.generate_coverage_branches.identify_branch_parameters",
                   side_effect=identify):
            rows = generate_coverage_branches(manager, frameworks, {}, quota=8, seed=1)
        self.assertEqual([row["P"] for row in rows], phases * 2)
        self.assertEqual([row["det_H"] for row in rows], [4] * 4 + [12] * 4)

    def test_selection_covers_phases_before_repeating_one(self):
        candidates = [
            {"candidate_id": phase, "P": phase, "x": "1/2", "T": ["Fe", "Mn"],
             "H": [[size, 0, 0], [0, 1, 0], [0, 0, 1]],
             "estimated_cost": {"value": size}}
            for phase, size in [("O1", 4), ("O1", 12), ("O3", 4),
                                ("O3", 12), ("P3", 4), ("OP2", 4)]
        ]
        result = select_candidates(candidates, batch_size=4, random_fraction=0, seed=1)
        selected = result["selected_candidates"]
        self.assertEqual({row["P"] for row in selected}, {"O1", "O3", "P3", "OP2"})
        self.assertTrue(all(row["estimated_cost"]["value"] == 4 for row in selected))

    def test_chat_reports_actual_generated_counts(self):
        reply = format_workflow_reply({"status": "completed", "events": [{
            "final_action": {"tool": "generate_branches"},
            "execution": {"result": {"summary": {
                "registered_branches": 8, "registered_structures": 20,
                "selected_branches": 8, "proposed_branches": 20,
                "phase_counts": {"O1": 2, "O3": 2, "OP2": 2, "P3": 2},
                "det_H_range": [4, 8],
            }}},
        }]}, "state.json")
        self.assertIn("8 个 branch、20 个初始结构", reply)
        self.assertIn("det(H) 4–8", reply)


if __name__ == "__main__":
    unittest.main()
