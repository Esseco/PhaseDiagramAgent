def test_stale_approved_input_with_invalid_layers_returns_rejected(tmp_path, monkeypatch):
    from phase_agent.tools.relax_mc_batch import prepare as module

    state = {"tasks": []}
    context = {
        "effective_config": {"upload_batches_directory": str(tmp_path / "uploads")},
        "manager": object(),
        "event_state": state,
        "config_version": "v1",
    }
    action = {
        "target_ids": ["b"],
        "parameters": {"relax_mc_preview": {"bundle_id": "b", "checksum": "frozen"}},
    }
    monkeypatch.setattr(module, "context_config", lambda config, context: config)

    def bad(*a, **k):
        raise ValueError("na_layer_occupancy_invalid:S-bad")

    monkeypatch.setattr(module, "diagnose", bad)
    result = module.prepare_batch(action=action, context=context)
    assert result["status"] == "rejected"
    assert result["reason"] == "na_layer_occupancy_invalid:S-bad"
    assert result["submitted"] is False
    assert not (tmp_path / "uploads").exists() and state["tasks"] == []


def test_unrelated_preflight_error_is_not_silenced(tmp_path, monkeypatch):
    import pytest
    from phase_agent.tools.relax_mc_batch import prepare as module

    context = {
        "effective_config": {},
        "manager": object(),
        "event_state": {},
        "config_version": "v1",
    }
    action = {
        "target_ids": ["b"],
        "parameters": {"relax_mc_preview": {"bundle_id": "b", "checksum": "frozen"}},
    }
    monkeypatch.setattr(module, "context_config", lambda config, context: config)

    def bad(*a, **k):
        raise ValueError("different_error")

    monkeypatch.setattr(module, "diagnose", bad)
    with pytest.raises(ValueError, match="different_error"):
        module.prepare_batch(action=action, context=context)
