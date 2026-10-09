from copy import deepcopy
from pathlib import Path
import yaml

from execution_layer.workflows.create_workflow_handlers import create_workflow_handlers
from execution_layer.workflows.request_strategy_revision import request_strategy_revision
from config_layer.runtime.authorize_budget_extension import authorize_budget_extension
from execution_layer.workflows.run_tool_step import _apply_execution_result
from tests.test_dft_comparison_csv import add_result


def test_config_revision_is_draft_not_enablement(monkeypatch):
    config = {"mlip_finetune": {"enabled": False}}
    state = {"confirmed_config_version": "old", "confirmed_config": deepcopy(config)}
    monkeypatch.setattr("execution_layer.workflows.create_workflow_handlers._update_mlip",
        lambda **kwargs: {"status": "awaiting_remote_training", "state": kwargs["context"]["event_state"]})
    result = request_strategy_revision(action={"parameters": {
        "request_configuration_revision": True, "patch": {"mlip_finetune.enabled": True}}},
        context={"event_state": state, "effective_config": config})
    assert result["status"] == "awaiting_remote_training"
    assert config["mlip_finetune"]["enabled"] is False
    assert result["state"]["confirmed_config"]["mlip_finetune"]["enabled"] is False
    updated = deepcopy(config)
    updated["mlip_finetune"]["enabled"] = True
    result["state"]["tasks"] = [{"task_id": "old-dft", "stage": "dft_single_point", "status": "pending", "recovery_wait_waived": True}]
    snapshot = {"config_version": "new", "config": updated}
    result["state"]["requested_config_revision"] = {"patch": {"mlip_finetune.enabled": True}, "source_config_version": "old"}
    assert authorize_budget_extension(result["state"], snapshot)["status"] == "approval_required"
    assert authorize_budget_extension(result["state"], snapshot, user_approved=True)["status"] == "extended"
    changed = deepcopy(snapshot)
    changed["config"]["mlip_finetune"]["training"] = {"patience": 20}
    assert authorize_budget_extension(result["state"], changed, user_approved=True)["status"].startswith("rejected")


def test_successful_supplement_consumes_only_source_round():
    updated, status = _apply_execution_result({}, {"tool": "select_dft_candidates", "_post_dft_scope_key": "source-round"},
        {"status": "completed", "result": {"status": "prepared", "state": {}, "tasks": [{"task_id": "new"}]}},
        record_id="r", formal=False)
    assert updated["post_dft_decided_rounds"] == ["source-round"]
    failed, _ = _apply_execution_result({}, {"tool": "select_dft_candidates", "_post_dft_scope_key": "source-round"},
        {"status": "completed", "result": {"status": "rejected", "state": {}}}, record_id="r", formal=False)
    assert not failed.get("post_dft_decided_rounds")


def test_default_training_prepares_portable_inputs_not_local_training(tmp_path):
    from config_layer.defaults.default_mace_committee_config import default_mace_committee_config
    state = {}
    for i in range(10):
        add_result(state, task_id=f"T{i}")
    settings = default_mace_committee_config()
    settings["enabled"] = True
    settings["labels"]["include_stress"] = False
    config = {"mlip_finetune": settings, "upload_batches_directory": str(tmp_path),
              "mlip": {"version": "m1", "model_path": "/remote/models/m1.model", "mace_head": "omat_pbe"},
              "python_environments": {"remote_mlip": "mace"}}
    result = create_workflow_handlers()["update_mlip"](action={}, context={"event_state": state, "effective_config": config})
    assert result["status"] == "awaiting_remote_training"
    job = next(iter(result["state"]["remote_finetune_jobs"].values()))
    assert job["submitted"] is False and job["activated"] is False
    directory = Path(job["directory"]) / "inputs"
    params = yaml.safe_load((directory / "com_1/mace_train.yaml").read_text())
    assert params["foundation_model"] == "/remote/models/m1.model"
    assert not Path(params["valid_file"]).is_absolute()
    assert "minimum_new_dft_records" not in params
    assert "-n mace" in (directory / "run_training.sh").read_text()
    main = job["report"]["main_model"]
    assert main["folds"] == 5
    seen = []
    for fold in main["fold_jobs"]:
        assert not set(fold["train_groups"]) & set(fold["valid_groups"])
        seen.extend(fold["valid_groups"])
    assert len(seen) == len(set(seen)) == main["groups"]
    assert not (directory / "main_final").exists()
    assert main["final_model"] == "com_1"
    final = params
    assert final["train_file"] == final["valid_file"] == "../_shared_data/train.xyz"
    assert final["valid_fraction"] == 0.0
    assert final["patience"] == 20
    assert final["foundation_head"] == "omat_pbe"
    assert final["lr"] == settings["committee"][0]["lr"]
    assert params["E0s"] == "estimated"
    assert params["amsgrad"] is True
    assert params["scaling"] == "rms_forces_scaling"
    assert final["E0s"] == "estimated"
    assert final["amsgrad"] is True
    assert final["scaling"] == "rms_forces_scaling"
    gpu = (directory / "GPU.sh").read_text()
    assert "bash run_training.sh" in gpu and "run_mlip_task.py" not in gpu
    assert "__JOB_NAME__" not in gpu
    repeated = create_workflow_handlers()["update_mlip"](action={}, context={
        "event_state": result["state"], "effective_config": config})
    assert repeated["status"] == "awaiting_remote_training"
    blocked = create_workflow_handlers()["update_mlip"](action={}, context={
        "event_state": state, "effective_config": config})
    assert blocked["status"] == "not_configured"
    assert "未覆盖" in blocked["reason"]
    assert params["train_file"] == params["valid_file"] == "../_shared_data/train.xyz"
    assert params["patience"] == 20
    assert "collect_training_results.py" in (directory / "run_training.sh").read_text()


def test_legacy_training_environment_uses_remote_template_not_local():
    from execution_layer.remote.write_training_submission import training_environment
    assert training_environment({"python_environments": {"local_mlip": "py-mace"}}) == ("mace", "remote_gpu_template")
    assert training_environment({"python_environments": {"remote_mlip": "my-mace"}}) == ("my-mace", "confirmed_config")


def test_input_only_update_bypasses_enablement_not_training_safety(monkeypatch):
    from execution_layer.workflows.create_workflow_handlers import _update_mlip
    from run.chat_approval_rules import is_sensitive_proposal
    from run.workflow_reply_presentation import format_workflow_reply
    action = {"tool": "update_mlip", "parameters": {"prepare_inputs_only": True}}
    monkeypatch.setattr("execution_layer.workflows.prepare_remote_finetune.prepare_remote_finetune",
        lambda state, config: {"status": "awaiting_remote_training", "state": state})
    def forbidden(**kwargs):
        raise AssertionError("input preparation must not invoke trainer")
    config = {"mlip_finetune": {"enabled": False, "training": {"minimum_new_dft_records": 0}}}
    result = _update_mlip(action=action, context={"effective_config": config, "event_state": {},
        "mlip_trainer": forbidden, "model_update_handler": forbidden})
    assert result["status"] == "awaiting_remote_training"
    proposal = {"raw_action": action, "recommended_action": "update_mlip", "estimated_cost": {"estimated_total_cost": 0}}
    assert not is_sensitive_proposal(proposal)
    assert is_sensitive_proposal({"raw_action": {"tool": "update_mlip"}})
    text = format_workflow_reply({"status": "awaiting_approval", "agent_proposal": proposal, "state": {}}, "state.json")
    assert "生成超算训练提交文件" in text
    assert "启用微调" not in text
    assert "预计相对成本：0" not in text


def test_hpc_kfold_finalizer_emits_metrics_and_component_rows(tmp_path, monkeypatch):
    import json
    import sys
    import types
    import numpy as np
    from ase import Atoms
    from ase.io import write
    from ase.calculators.calculator import Calculator, all_changes
    from execution_layer.remote import collect_training_results as collector

    class DummyMACE(Calculator):
        implemented_properties = ["energy", "forces"]
        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": 2.0, "forces": np.zeros((len(atoms), 3))}

    monkeypatch.setitem(sys.modules, "mace.calculators", types.SimpleNamespace(MACECalculator=DummyMACE))
    monkeypatch.setattr(collector, "__file__", str(tmp_path / "collect_training_results.py"))
    plan = {"config": {"labels": {}}, "main_model": {"final_model": "com_1", "fold_jobs": [{"model_id": "main_cv_1"}]},
            "committees": [{"model_id": "com_1"}]}
    (tmp_path / "training_plan.json").write_text(json.dumps(plan))
    for name in ("main_cv_1", "com_1"):
        (tmp_path / name).mkdir()
        (tmp_path / name / (name + ".model")).touch()
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms.info["REF_energy"] = 1.0
    atoms.arrays["REF_forces"] = np.ones((2, 3))
    write(tmp_path / "main_cv_1/valid.xyz", atoms, format="extxyz")
    collector.main()
    import csv
    with (tmp_path / "results/kfold_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[-1]["energy_MAE_meV_per_atom"]) == 500.0
    assert float(rows[-1]["force_RMSE_meV_per_A"]) == 1000.0
    with (tmp_path / "results/force_comparison.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 6
    manifest = json.loads((tmp_path / "results/models.json").read_text())
    assert len(manifest) == 1
    assert manifest[0]["is_main_model"] and manifest[0]["is_committee_member"]
    assert manifest[0]["storage"] == "remote"
    assert manifest[0]["remote_model_path"] == str((tmp_path / "com_1/com_1.model").resolve())
    assert not (tmp_path / "results/models").exists()
