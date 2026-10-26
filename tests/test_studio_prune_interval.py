from types import SimpleNamespace
from unittest.mock import Mock

from phase_agent.persistence.studio_storage import StudioStorage


class Store(dict):
    incremental = False

    def sync(self):
        self.synced = getattr(self, "synced", 0) + 1


def test_studio_storage_delays_non_incremental_pruning_until_bounded_interval(monkeypatch):
    memory = SimpleNamespace(storage=Store(), writes=Store(), blobs=Store())
    manager = StudioStorage(memory)
    prune = Mock()
    monkeypatch.setattr(manager, "prune", prune)
    for _ in range(manager.PRUNE_INTERVAL - 1):
        manager.changed()
        memory.storage.sync()
    assert prune.call_count == 0
    manager.changed()
    memory.storage.sync()
    assert prune.call_count == 1
    assert manager.pruned == manager.revision


def test_incremental_store_uses_a_longer_prune_interval(monkeypatch):
    from types import SimpleNamespace

    class IncrementalStore(Store):
        incremental = True

    memory = SimpleNamespace(
        storage=IncrementalStore(), writes=IncrementalStore(), blobs=IncrementalStore()
    )
    manager = StudioStorage(memory)
    prune = Mock()
    monkeypatch.setattr(manager, "prune", prune)
    for _ in range(manager.INCREMENTAL_PRUNE_INTERVAL - 1):
        manager.changed()
        memory.storage.sync()
    assert prune.call_count == 0
    manager.changed()
    memory.storage.sync()
    assert prune.call_count == 1
