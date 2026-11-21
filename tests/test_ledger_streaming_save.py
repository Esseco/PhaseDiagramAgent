import pytest

import phase_agent.persistence.ledger.phase_data_manager as ledger


def make_manager():
    return ledger.PhaseDataManager({"P": ["O3"], "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]}, "TM_ratio": {"Ni": 1}})


def test_save_streams_without_materializing_complete_json(tmp_path, monkeypatch):
    manager = make_manager()
    path = tmp_path / "ledger.json"
    def forbid_dumps(*args, **kwargs):
        raise AssertionError("whole ledger must not be encoded at once")
    with monkeypatch.context() as patch:
        patch.setattr(ledger.json, "dumps", forbid_dumps)
        manager.save(path)
    assert ledger.PhaseDataManager.load(path).data == manager.data


def test_interrupted_save_preserves_previous_ledger(tmp_path, monkeypatch):
    manager = make_manager()
    path = tmp_path / "ledger.json"
    manager.save(path)
    before = path.read_bytes()
    manager.data["changed"] = 1
    def fail(data, stream, **kwargs):
        stream.write("partial")
        raise MemoryError()
    monkeypatch.setattr(ledger.json, "dump", fail)
    with pytest.raises(MemoryError):
        manager.save(path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
