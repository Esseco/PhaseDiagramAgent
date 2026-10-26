import json
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from phase_agent.persistence.checkpoint_maintenance import main


def _checkpoint(index):
    return {
        "v": 4,
        "id": f"{index:04}",
        "ts": "2026-10-15T00:00:00+00:00",
        "channel_values": {"facts": {"index": index}},
        "channel_versions": {"facts": index},
        "versions_seen": {},
        "pending_sends": [],
    }


def test_checkpoint_maintenance_cli_compacts_and_reports_verified_sqlite(tmp_path, capsys):
    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"
    cfg = {"configurable": {"thread_id": "t", "checkpoint_ns": ""}}
    with SqliteSaver.from_conn_string(str(source)) as saver:
        for i in range(3):
            cfg = saver.put(cfg, _checkpoint(i), {}, {})
    assert main(
        ["--sqlite-source", str(source), "--destination", str(target), "--preserve-all"]
    ) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["verified"] is True
    assert receipt["latest_recovery_points_verified"] == 1
    assert target.is_file()


def test_checkpoint_maintenance_cli_rejects_in_place_destination(tmp_path):
    source = tmp_path / "source.sqlite"
    source.write_bytes(b"not a database")
    try:
        main(["--sqlite-source", str(source), "--destination", str(source)])
    except ValueError as error:
        assert "different" in str(error)
    else:
        raise AssertionError("in-place compaction must be rejected")
