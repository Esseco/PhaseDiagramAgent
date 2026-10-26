from copy import deepcopy
import hashlib
import json
from unittest.mock import patch
from phase_agent.decisions.agent.transport_tables import (
    compact_evidence_tables,
    expand_evidence_tables,
    evidence_row_count,
)
from phase_agent.decisions.agent.summary_request import _evidence_revision
from phase_agent.runtime.turn_process import record_turn, _current
from phase_agent.tools.step_runner.file_protocol import read_json


def test_mixed_scientific_rows_roundtrip_without_nulls_or_missing_candidates():
    rows = []
    for i in range(200):
        row = {
            "task_id": str(i),
            "status": "completed",
            "model_version": "m",
            "stage": "relax" if i % 2 else "mc",
            "outputs": {"energy": i / 1000},
        }
        if i % 2:
            row["converged"] = True
        else:
            row["segment_index"] = i % 3
        rows.append(row)
    payload = {
        "tasks": rows,
        "qbc_candidates": [
            {"id": str(i), "ehull": i / 1000, "qbc": None} for i in range(200)
        ],
    }
    before = deepcopy(payload)
    compact = compact_evidence_tables(payload)
    assert compact["tasks"]["format"] == "schema_group_table_v1"
    assert expand_evidence_tables(compact) == payload == before
    assert evidence_row_count(compact["tasks"]) == 200
    assert len(json.dumps(compact)) < len(json.dumps(payload)) * 0.75


def test_streaming_revision_is_identical_and_tracks_hidden_changes():
    sources = {
        "a": [{"候选": i, "value": None} for i in range(30)],
        "hidden": {"budget": 5},
    }
    old = hashlib.sha256(
        json.dumps(sources, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert _evidence_revision(sources) == old
    sources["hidden"]["budget"] = 6
    assert _evidence_revision(sources) != old


def test_initial_state_parsed_once_and_not_shared_across_later_reads(tmp_path):
    (tmp_path / "workflow_state").mkdir()
    path = tmp_path / "workflow_state/state.json"
    path.write_text('{"tasks":[],"value":1}')
    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    with record_turn(config, "status", "thread"):
        first = read_json(path)
        first["value"] = 2
        assert read_json(path)["value"] == 1
        assert "_initial_state" not in _current.get()
    record = json.loads(
        next((tmp_path / "workflow_state/dialogue_process").glob("*.json")).read_text(
            encoding="utf-8"
        )
    )
    assert "_initial_state" not in record


def test_initial_state_rejected_when_file_changes(tmp_path):
    (tmp_path / "workflow_state").mkdir()
    path = tmp_path / "workflow_state/state.json"
    path.write_text('{"tasks":[],"value":1}')
    config = tmp_path / "agent_runtime.json"
    config.write_text("{}")
    with record_turn(config, "status", "thread"):
        path.write_text('{"tasks":[],"value":999}')
        assert read_json(path)["value"] == 999


def test_lossless_business_checkpoint_migration_keeps_every_record(tmp_path):
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.base import empty_checkpoint
    from phase_agent.persistence.checkpoint_maintenance import compact_sqlite

    source = tmp_path / "old.sqlite"
    target = tmp_path / "converted.sqlite"
    cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    with SqliteSaver.from_conn_string(str(source)) as saver:
        for i in range(12):
            cp = empty_checkpoint()
            cp["id"] = str(i).zfill(4)
            cp["channel_values"] = {
                "business_frame": {"loaded_state": {"tasks": ["fact" * 20000]}}
            }
            cfg = saver.put(cfg, cp, {"step": i}, {})
        saver.put_writes(cfg, [("__interrupt__", {"approval": "keep"})], "approval")
    result = compact_sqlite(source, target, preserve_all=True)
    assert result["verified"] and result["old_checkpoints_archived"] == 0
    with sqlite3.connect(target) as conn:
        assert conn.execute("select count(*) from checkpoints").fetchone()[0] == 12
    with SqliteSaver.from_conn_string(str(source)) as old:
        assert old.get_tuple(cfg).pending_writes[0][2] == {"approval": "keep"}


def test_checkpoint_aliases_serialize_once_but_mutation_is_never_stale(tmp_path):
    import sqlite3
    import threading
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from phase_agent.persistence.checkpoint_blobs import CheckpointBlobSerde

    delegate = JsonPlusSerializer()
    conn = sqlite3.connect(tmp_path / "cas.sqlite")
    serde = CheckpointBlobSerde(delegate, conn, threading.RLock(), threshold=32)
    state = {"large": ["fact" * 1000]}
    frame = {
        "loaded_state": state,
        "pre_reconciled": {"state": state},
        "feedback": {"state": state},
    }
    calls = []
    original = delegate.dumps_typed
    with patch.object(
        delegate,
        "dumps_typed",
        side_effect=lambda value: (calls.append(value), original(value))[1],
    ):
        raw = serde.dumps_typed(frame)
    assert sum(value is state["large"] for value in calls) == 1
    assert serde.loads_typed(raw) == frame
    state["large"].append("new evidence")
    assert serde.loads_typed(serde.dumps_typed(frame)) == frame


def test_server_bootstrap_installs_store_before_starting_cli(monkeypatch):
    from langgraph_runtime_inmem import checkpoint
    from phase_agent.runtime.studio_server import main
    from phase_agent.persistence.studio_checkpoint_store import checkpoint_container

    previous = checkpoint.PersistentDict
    calls = []
    monkeypatch.setattr(checkpoint, "InMemorySaver", checkpoint.InMemorySaver)
    monkeypatch.setattr(checkpoint, "PersistentDict", previous)
    monkeypatch.setattr(
        "runpy.run_module",
        lambda name, **kw: calls.append((name, checkpoint.PersistentDict)),
    )
    main()
    assert calls == [("langgraph_cli", checkpoint_container)]


def test_server_filters_only_benign_windows_transport_disconnects():
    from types import SimpleNamespace
    from phase_agent.runtime.studio_server import _is_benign_transport_disconnect

    callback = SimpleNamespace(__qualname__="_ProactorBasePipeTransport._call_connection_lost")
    handle = SimpleNamespace(_callback=callback)

    assert _is_benign_transport_disconnect(
        {"exception": ConnectionResetError(10054), "handle": handle}
    )
    assert not _is_benign_transport_disconnect(
        {"exception": ValueError("workflow failed"), "handle": handle}
    )
