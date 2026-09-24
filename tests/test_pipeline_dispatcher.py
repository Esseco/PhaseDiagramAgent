from __future__ import annotations

import importlib
from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run.run_active_learning_cycle import run_active_learning_cycle
from run.run_confirmed_active_learning_cycle import run_confirmed_active_learning_cycle
from run import run_workflow
from execution_layer.dispatch.calculation_stage_registry import CalculationStageRegistry
from execution_layer.dispatch.create_pipeline_dispatcher import create_pipeline_dispatcher


class _Manager:
    def __init__(self):
        self.boundary = {"P": ["O3"], "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]}, "TM_ratio": {"Fe": 1}}
        self.data = {
            "branches": {"B1": {"branch_id": "B1", "P": "O3", "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "x": "1", "T": ["Fe"]}},
            "structures": {"S1": {"structure_id": "S1", "branch_id": "B1", "source_path": "S1.vasp", "metadata": {}, "stage_history": {"dft_single_point": [], "dft_relax": []}}},
        }
        self.saved = []

    def save(self, path):
        self.saved.append(str(path))
        return Path(path)


class PipelineDispatcherTest(unittest.TestCase):
    def test_public_entry_rejects_unconfirmed_configuration(self):
        result = run_workflow(_Manager(), {}, {}, {"status": "draft"})
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["submitted"])

    def test_dispatcher_resolves_ledger_context(self):
        manager = _Manager()
        registry = CalculationStageRegistry()
        registry.register(
            "probe",
            handler=lambda task, context: {
                "status": "completed",
                "outputs": {
                    "branch_id": context["branch"]["branch_id"],
                    "structure": context["structure"],
                },
            },
        )
        dispatcher = create_pipeline_dispatcher(
            manager,
            {"O3": "reference.vasp"},
            {"mlip": {"name": "m1", "model_path": "model.pt"}},
            registry=registry,
        )
        result = dispatcher({"task_id": "T1", "structure_id": "S1", "stage": "probe"})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["outputs"]["branch_id"], "B1")
        self.assertEqual(result["outputs"]["structure"], "S1.vasp")

    def test_active_learning_cycle_reuses_dispatcher_for_dft(self):
        manager = _Manager()
        module = importlib.import_module("run.run_active_learning_cycle")
        search = {"status": "completed", "state": {"iteration": 1, "pending_tasks": [], "processed_task_ids": []}}
        dispatcher = lambda task: {**task, "status": "pending"}
        config = {
            "qbc": {"selection_mode": "qbc_fixed"},
            "dft": {"parameters": {"encut": 520}},
            "budgets": {"stage_limits": {"dft_single_point": {"max_cost": 100}, "dft_relax": {"max_cost": 0}}},
        }
        candidates = [{"candidate_id": "S1", "branch_id": "B1", "qbc": {"status": "completed", "f_std_max": 1.0}, "predicted_Ehull": 0.01, "estimated_cost": 1.0}]
        with tempfile.TemporaryDirectory() as directory:
            config.update({"ledger_path": f"{directory}/ledger.json", "state_path": f"{directory}/state.json"})
            with patch.object(module, "run_pipeline", return_value=search):
                result = run_active_learning_cycle(
                    manager,
                    {},
                    config,
                    config_version="config-1",
                    dispatcher=dispatcher,
                    candidates=candidates,
                    dft_budget=100,
                )
        self.assertEqual(result["acquisition"]["submissions"][0]["status"], "pending")
        self.assertEqual(result["state"]["pending_tasks"][0]["stage"], "dft_single_point")

    def test_active_cycle_disables_pipeline_dft_without_mutating_config(self):
        manager = _Manager()
        module = importlib.import_module("run.run_active_learning_cycle")
        config = {
            "system_config": {"calculation_workflow": {"stages": [
                {"name": "deep_search", "enabled": True},
                {"name": "dft_single_point", "enabled": True},
                {"name": "dft_relax", "enabled": True},
            ]}},
            "qbc": {"selection_mode": "qbc_fixed"},
            "budgets": {"stage_limits": {}},
        }
        original = deepcopy(config)
        captured = {}

        def fake_pipeline(_manager, _references, effective, **_kwargs):
            captured["config"] = effective
            return {"status": "completed", "state": {"iteration": 1}}

        with patch.object(module, "run_pipeline", side_effect=fake_pipeline):
            run_active_learning_cycle(
                manager, {}, config, config_version="config-1", candidates=[], dft_budget=0
            )
        stages = {item["name"]: item["enabled"] for item in captured["config"]["system_config"]["calculation_workflow"]["stages"]}
        self.assertTrue(stages["deep_search"])
        self.assertFalse(stages["dft_single_point"])
        self.assertFalse(stages["dft_relax"])
        self.assertEqual(config, original)

    def test_confirmed_active_cycle_rejects_mismatched_version(self):
        result = run_confirmed_active_learning_cycle(
            _Manager(), {}, {},
            {"status": "confirmed", "confirmed_snapshot": {"config_version": "config-2"}},
            config_version="config-1",
        )
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "config_version_mismatch")

    def test_default_qbc_pool_excludes_duplicate_pending_and_completed_dft(self):
        manager = _Manager()
        manager.data["structures"].update({
            "S2": {"structure_id": "S2", "branch_id": "B1", "metadata": {"duplicate_of": "S1"}, "stage_history": {}},
            "S3": {"structure_id": "S3", "branch_id": "B1", "metadata": {}, "stage_history": {}},
            "S4": {"structure_id": "S4", "branch_id": "B1", "metadata": {}, "stage_history": {"dft_single_point": [{"result_id": "R1"}]}},
        })
        module = importlib.import_module("run.run_active_learning_cycle")
        search = {"status": "completed", "state": {
            "iteration": 1,
            "pending_tasks": [{"structure_id": "S3", "stage": "dft_relax", "status": "pending"}],
        }}
        captured = {}

        def fake_acquisition(candidates, state, **_kwargs):
            captured["ids"] = [item["candidate_id"] for item in candidates]
            return {"status": "completed", "state": state, "submissions": [], "retrain": None}

        with patch.object(module, "run_pipeline", return_value=search), patch.object(
            module, "run_qbc_dft_decision_flow", side_effect=fake_acquisition
        ):
            result = run_active_learning_cycle(
                manager, {}, {"qbc": {"selection_mode": "qbc_fixed"}, "budgets": {"stage_limits": {}}},
                config_version="config-1", dft_budget=0,
            )
        self.assertEqual(captured["ids"], ["S1"])
        reasons = {item["structure_id"]: item["reason"] for item in result["state"]["qbc_candidate_filter"]["excluded"]}
        self.assertEqual(reasons, {"S2": "duplicate_structure", "S3": "dft_already_pending", "S4": "dft_already_completed"})


if __name__ == "__main__":
    unittest.main()
