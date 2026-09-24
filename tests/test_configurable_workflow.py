import unittest

from scientific_layer.structures.create_generation_registry import create_generation_registry
from execution_layer.dispatch.calculation_stage_registry import CalculationStageRegistry
from decision_layer.calculation.decide_next_calculation import decide_next_calculation
from execution_layer.dispatch.dispatch_calculation_stage import execute_registered_stage
from data_layer.ledger.phase_data_manager import PhaseDataManager
from config_layer.defaults.layered_oxide_system_config import layered_oxide_system_config
from config_layer.schema.validate_system_config import validate_system_config


class ConfigurableWorkflowTest(unittest.TestCase):
    def test_default_system_and_registries(self):
        system = validate_system_config(layered_oxide_system_config())
        self.assertEqual(system["branch_schema"]["fields"], ["P", "H", "x", "T"])
        self.assertIn("coverage", create_generation_registry().names())

    def test_custom_branch_schema(self):
        system = layered_oxide_system_config()
        system["branch_schema"] = {"fields": ["phase", "content"], "internal_search_variables": ["ordering"]}
        manager = PhaseDataManager({"P": ["O3"], "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]}, "TM_ratio": {"Fe": 1}}, system_config=system)
        branch_id = manager.add_branch_record({"phase": "alpha", "content": "1/2"})
        self.assertEqual(manager.data["branches"][branch_id]["phase"], "alpha")

    def test_custom_stage_and_explicit_status(self):
        registry = CalculationStageRegistry()
        registry.register("cheap", handler=lambda task, context: {"status": "completed", "converged": True, "outputs": {"value": 1}})
        task = {"task_id": "T1", "object_id": "S1", "stage": "cheap"}
        result = execute_registered_stage(task, {}, registry)
        self.assertEqual(result["status"], "completed")
        record = {"metadata": {"calculation_attempts": []}, "stage_history": {"cheap": []}}
        decision = decide_next_calculation(record, rules={"stages": [{"name": "cheap", "enabled": True, "max_retries": 0}]})
        self.assertEqual(decision["stage"], "cheap")
        self.assertEqual(decision["run_status"], "ready")


if __name__ == "__main__":
    unittest.main()
