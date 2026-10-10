"""不启动训练或 DFT 的模拟验证。"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

from phase_agent.science.mlip.build_mlip_committee import build_mlip_committee
from phase_agent.science.qbc.evaluate_qbc import evaluate_qbc
from phase_agent.science.training.update_mlip import update_mlip
from phase_agent.science.training.validate_mlip import validate_mlip


class ActiveLearningTest(unittest.TestCase):
    def setUp(self):
        self.structure = Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]])

    def test_duplicate_model_content_is_rejected(self):
        first, second = Path(__file__), Path(__file__).parent / "__init__.py"
        committee = build_mlip_committee(
            [
                {"model_path": first, "model_id": "a"},
                {"model_path": first, "model_id": "duplicate"},
                {"model_path": second, "model_id": "b"},
            ],
            loader=lambda **kwargs: kwargs["model_path"].name,
        )
        self.assertEqual(committee["status"], "ready")
        self.assertEqual(committee["rejected"][0]["reason"], "duplicate_model_content")

    def test_qbc_is_disagreement(self):
        committee = {
            "loaded_members": [
                {"model_id": "a", "model": 0.0},
                {"model_id": "b", "model": 1.0},
            ]
        }
        result = evaluate_qbc(
            self.structure,
            committee,
            predictor=lambda structure, model, member: {
                "energy": model,
                "forces": np.full((len(structure), 3), model),
            },
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            result["interpretation"], "committee_disagreement_not_true_error"
        )

    def test_training_requires_adapter_and_validation_does_not_activate(self):
        update = update_mlip(
            [
                {
                    "data_id": "d1",
                    "status": "completed",
                    "converged": True,
                    "energy": -1.0,
                }
            ],
            dataset_version="D1",
            candidate_model_version="M2",
        )
        self.assertEqual(update["status"], "not_configured")
        result = validate_mlip(
            {"version": "M1"},
            {"version": "M2"},
            [{}],
            evaluator=lambda model, data: {
                "energy_mae": 0.1 if model["version"] == "M1" else 0.08,
                "force_rmse": 0.2 if model["version"] == "M1" else 0.18,
                "critical_failure_fraction": 0,
                "near_hull_ranking_reversals": 0,
            },
            criteria={"max_energy_mae": .2, "max_critical_failure_fraction": .5,
                      "max_near_hull_ranking_reversals": 0},
            validation_data_version="V1",
        )
        self.assertTrue(result["passed"])
        self.assertFalse(result["activate_new_model"])


if __name__ == "__main__":
    unittest.main()
