"""Public source layout and saved descriptors must survive removal of old packages."""
import ast
import json
from pathlib import Path
from phase_agent.module_references import import_saved_module
from phase_agent.runtime.project_service import prepare_studio_config

ROOT = Path(__file__).resolve().parents[1]
OLD = {"run", "orchestration", "config_layer", "data_layer", "decision_layer",
       "analysis_layer", "scientific_layer", "execution_layer"}


def test_application_has_one_package_and_no_old_imports():
    assert all(not (ROOT / name).exists() for name in OLD)
    for path in (ROOT / "phase_agent").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in OLD, path
            elif isinstance(node, ast.Import):
                assert not any(alias.name.split(".")[0] in OLD for alias in node.names), path


def test_saved_adapter_reference_resolves_without_old_package():
    new = import_saved_module("phase_agent.runtime.default_run_config")
    assert import_saved_module("run.default_run_config") is new
    assert import_saved_module("scientific_layer.dft.parse_vasp_result") is import_saved_module("phase_agent.science.dft.parse_vasp_result")


def test_generated_studio_config_uses_actual_code_root(tmp_path):
    runtime = tmp_path / "agent_runtime.json"
    runtime.write_text("{}", encoding="utf-8")
    config = json.loads(prepare_studio_config(runtime).read_text(encoding="utf-8"))
    assert Path(config["dependencies"][0]) == ROOT
    for reference in [*config["graphs"].values(), config["http"]["app"]]:
        module, attribute = reference.split(":")
        assert module.startswith("phase_agent.")
        assert hasattr(import_saved_module(module), attribute)


def test_obsolete_controllers_and_rule_scores_are_absent():
    paths = ['phase_agent/decisions/scoring/allocate_generation_quotas.py', 'phase_agent/decisions/scoring/decide_next_stage.py', 'phase_agent/decisions/scoring/estimate_calculation_cost.py', 'phase_agent/decisions/scoring/score_action_reward.py', 'phase_agent/decisions/scoring/score_additional_mc_value.py', 'phase_agent/decisions/scoring/score_branch_potential.py', 'phase_agent/decisions/scoring/score_coverage_gap.py', 'phase_agent/decisions/scoring/score_dft_value.py', 'phase_agent/decisions/scoring/score_generation_strategy.py', 'phase_agent/decisions/scoring/score_parent_branch.py', 'phase_agent/decisions/scoring/score_validation_improvement.py', 'phase_agent/decisions/scoring/update_strategy_weights.py', 'phase_agent/runtime/web_chat_gate.py', 'phase_agent/decisions/agent/langgraph_decision.py', 'phase_agent/decisions/strategy/update_search_policy.py', 'phase_agent/analysis/convergence/check_branch_stop.py', 'phase_agent/analysis/feedback/reevaluate_candidates.py', 'phase_agent/analysis/phase/identify_mc_result_phase.py', 'phase_agent/configuration/runtime/estimate_draft_cost.py', 'phase_agent/configuration/session/pause_for_config_change.py', 'phase_agent/tools/workflows/create_qbc_dft_tool_handler.py']
    assert all(not (ROOT / path).exists() for path in paths)
