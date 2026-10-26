"""Checkpoint reads do not persist defaults and stores release file handles."""

from collections import defaultdict

from phase_agent.persistence.studio_checkpoint_store import StudioCheckpointDict


def test_missing_default_read_is_side_effect_free(tmp_path):
    store = StudioCheckpointDict(dict, filename=tmp_path / "store.pckl")
    before = store.backend.conn.total_changes
    view = store["missing"]
    assert isinstance(view, StudioCheckpointDict)
    assert len(store) == 0
    assert store.backend.conn.total_changes == before
    store.close()


def test_nested_lazy_default_materializes_only_on_mutation(tmp_path):
    filename = tmp_path / "store.pckl"
    store = StudioCheckpointDict(lambda: defaultdict(dict), filename=filename)
    store["thread"]["namespace"]["checkpoint"] = {"value": 1}
    assert len(store) == 1
    store.close()
    reopened = StudioCheckpointDict(lambda: defaultdict(dict), filename=filename)
    assert reopened["thread"]["namespace"]["checkpoint"]["value"] == 1
    reopened.close()


def test_shared_store_connection_closes_after_last_owner(tmp_path):
    filename = tmp_path / "store.pckl"
    first = StudioCheckpointDict(dict, filename=filename)
    second = StudioCheckpointDict(dict, filename=filename)
    first.close()
    second["value"] = 1
    second.close()
    sqlite_path = filename.with_name(filename.name + ".sqlite")
    sqlite_path.unlink()
    assert not sqlite_path.exists()
