"""Saved configuration facts never trigger scientific setup implicitly."""
import json
from unittest.mock import patch
from phase_agent.runtime.local_project_launcher import create_project
from phase_agent.runtime.agent_api import create_agent_runtime
from phase_agent.configuration.session.build_config_field_catalog import build_config_field_catalog


def test_config_write_syncs_unreviewed_draft_without_enumeration(tmp_path):
    runtime = create_project(tmp_path / "project", registry=tmp_path / "registry.json")
    with patch("phase_agent.runtime.deepseek_credentials.load_deepseek_api_key", return_value=None):
        handler = create_agent_runtime(runtime)
    handler.agent_client = lambda payload: {
        "reply": "写入指定配置", "write_requested": True,
        "patch": {"system.boundary.P": ["O3"], "system.boundary.TM_ratio": {"Cr": 1.0},
                  "system.H_generation.size_max": 12, "python_environments.remote_mlip": "mace",
                  "python_environments.remote_python": "python"}}
    with patch("phase_agent.configuration.session.materialize_layered_h.materialize_layered_h",
               side_effect=AssertionError("Saving must not enumerate H")):
        reply = handler([{"role": "user", "content": "直接写入草稿，计算前我会审核"}], conversation_id="a")
    session = json.loads((runtime.parent / "parameters/config_session.json").read_text(encoding="utf-8"))
    config = session["config"]
    assert config["system"]["constraints"]["phases"] == ["O3"]
    assert config["system"]["constraints"]["TM_ratio"] == {"Cr": 1.0}
    assert config["system"]["species"]["substitutional"] == ["Cr"]
    assert config["system"]["H_generation"]["size_max"] == 12
    assert config["python_environments"]["remote_mlip"] == "mace"
    assert config["python_environments"]["remote_python"] == "python"
    assert session["status"] == "draft"
    assert not session.get("confirmed_snapshot")
    assert not session.get("last_imported_config_hash")
    assert not session.get("agent_reviewed_revision")
    assert not (runtime.parent / "workflow_state/state.json").exists()
    assert "尚缺母结构：O3" in reply
    assert "尚未审核、确认或生成 H" in reply
    assert "草稿未更改" not in reply


def test_catalog_exposes_source_fields_and_derived_fields():
    catalog = build_config_field_catalog({"system": {
        "boundary": {"P": {"at_x": {"0": ["O1"]}}, "TM_ratio": {}},
        "constraints": {"phases": ["O1"], "TM_ratio": {"Fe": 1}}}})
    assert catalog["system.boundary.P"]["replace_types"] == ["list", "dict"]
    assert catalog["system.boundary.TM_ratio"]["editable"]
    assert not catalog["system.constraints.TM_ratio"]["editable"]
    assert catalog["system.constraints.TM_ratio"]["derived_from"] == "system.boundary.TM_ratio"
    assert "system.constraints.TM_ratio.Fe" not in catalog


def test_model_repairs_derived_field_before_any_write(tmp_path):
    runtime = create_project(tmp_path / "project", registry=tmp_path / "registry.json")
    with patch("phase_agent.runtime.deepseek_credentials.load_deepseek_api_key", return_value=None):
        handler = create_agent_runtime(runtime)
    calls = []
    def model(payload):
        calls.append(payload)
        if len(calls) == 1:
            return {"reply": "改相", "patch": {"system.constraints.phases": ["O3"]}, "write_requested": True}
        assert "read-only" in payload["validation_errors"][0]
        assert payload["instruction"] == "只研究O3，直接保存"
        return {"reply": "改相", "patch": {"system.boundary.P": ["O3"]}, "write_requested": True}
    handler.agent_client = model
    reply = handler([{"role": "user", "content": "只研究O3，直接保存"}], conversation_id="a")
    assert len(calls) == 2
    assert "尚缺母结构：O3" in reply
    assert handler.workflow_kwargs["config_session"]["config"]["system"]["constraints"]["phases"] == ["O3"]


def test_repeated_invalid_field_does_not_write(tmp_path):
    runtime = create_project(tmp_path / "project", registry=tmp_path / "registry.json")
    with patch("phase_agent.runtime.deepseek_credentials.load_deepseek_api_key", return_value=None):
        handler = create_agent_runtime(runtime)
    source = runtime.parent / "parameters/search_config.project.json"
    original = source.read_bytes()
    calls = []
    handler.agent_client = lambda payload: calls.append(payload) or {
        "reply": "改相", "patch": {"system.constraints.phases": ["O3"]}, "write_requested": True}
    reply = handler([{"role": "user", "content": "只研究O3，直接保存"}], conversation_id="a")
    assert len(calls) == 2
    assert source.read_bytes() == original
    assert "草稿未更改" in reply


def test_parent_object_cannot_bypass_derived_field_check():
    from phase_agent.configuration.session.build_config_field_catalog import config_patch_field_errors
    catalog = build_config_field_catalog({"system": {"constraints": {"phases": ["O3"]}}})
    errors = config_patch_field_errors({"patch": {"system": {"constraints": {"phases": ["P3"]}}}}, catalog)
    assert errors and "system.boundary.P" in errors[0]
