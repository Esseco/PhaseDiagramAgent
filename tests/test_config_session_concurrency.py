"""Configuration session writes stay atomic under concurrent callers."""

import json
from threading import Barrier, Thread

from phase_agent.configuration.session.save_config_session import save_config_session


def test_concurrent_config_session_saves_are_atomic(tmp_path):
    target = tmp_path / "config_session.json"
    barrier = Barrier(8)
    errors = []

    def save(index):
        try:
            barrier.wait()
            save_config_session({"turn": index, "dialogue": [str(index)]}, target)
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    workers = [Thread(target=save, args=(index,)) for index in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert errors == []
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["turn"] in range(8)
    assert not list(tmp_path.glob("*.tmp"))
