from __future__ import annotations

import importlib
from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phase_agent.runtime import run_workflow
from phase_agent.tools.dispatch.calculation_stage_registry import CalculationStageRegistry
from phase_agent.tools.dispatch.create_pipeline_dispatcher import create_pipeline_dispatcher


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






if __name__ == "__main__":
    unittest.main()
