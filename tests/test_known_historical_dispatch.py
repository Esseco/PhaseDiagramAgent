from phase_agent.tools.remote.transfer_bundle import has_registered_return


def test_only_matching_historical_failure_is_ignored(tmp_path, monkeypatch):
    from phase_agent.tools.remote import transfer_bundle as module
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(module, "registered_return_locations", lambda state: {str(root)})
    monkeypatch.setattr(module, "registered_result_directories", lambda state: set())
    monkeypatch.setattr(module, "dispatch_failure", lambda source: "remote_dispatch_failed:old")
    state = {"result_collection":{"historical_transfer_errors":[{"archive":str(root/"diagnostics.zip"), "reason":"remote_dispatch_failed:old"}]}}
    assert has_registered_return(state) is True
    assert has_registered_return(state, ignore_known_diagnostics=True) is False
    state["result_collection"]["historical_transfer_errors"][0]["reason"] = "remote_dispatch_failed:different"
    assert has_registered_return(state, ignore_known_diagnostics=True) is True


def test_new_result_archive_still_blocks(tmp_path, monkeypatch):
    from phase_agent.tools.remote import transfer_bundle as module
    root = tmp_path / "output"
    root.mkdir()
    (root/"new.results.zip").write_bytes(b"new")
    monkeypatch.setattr(module, "registered_return_locations", lambda state: {str(root)})
    monkeypatch.setattr(module, "registered_result_directories", lambda state: set())
    monkeypatch.setattr(module, "dispatch_failure", lambda source: "remote_dispatch_failed:old")
    state = {"result_collection":{"historical_transfer_errors":[{"archive":str(root/"diagnostics.zip"), "reason":"remote_dispatch_failed:old"}]}}
    assert has_registered_return(state, ignore_known_diagnostics=True) is True
