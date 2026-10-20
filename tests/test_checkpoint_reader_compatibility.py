"""Persisted graph recovery must accept all existing checkpoint encodings."""

import hashlib
import sqlite3
import threading

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from phase_agent.graphs.approval_graph import durable_approval
from phase_agent.graphs.batch_recovery_graph import recover_batch_stages
from phase_agent.graphs.direction_review_graph import direction_checkpoint
from phase_agent.persistence.checkpoint_blobs import CheckpointBlobSerde
from phase_agent.persistence.compact_checkpoints import CompactSqliteSaver, CompressedSerde


def encode_existing(path, encoding):
    """Offline encoding change on a test database only; leave IDs and parents intact."""
    with sqlite3.connect(path) as conn:
        serde = (
            CompressedSerde(JsonPlusSerializer())
            if encoding == "zlib"
            else CheckpointBlobSerde(CompressedSerde(JsonPlusSerializer()), conn, threading.RLock())
        )
        old = CheckpointBlobSerde(CompressedSerde(JsonPlusSerializer()), conn, threading.RLock())
        for table, column in (("checkpoints", "checkpoint"), ("writes", "value")):
            for rowid, kind, data in list(conn.execute(f"SELECT rowid,type,{column} FROM {table}")):
                encoded = serde.dumps_typed(old.loads_typed((kind, data)))
                conn.execute(
                    f"UPDATE {table} SET type=?,{column}=? WHERE rowid=?", (*encoded, rowid)
                )


@pytest.mark.parametrize("encoding", ["zlib", "cas"])
def test_batch_recovery_resumes_encoded_interrupt_without_losing_history(tmp_path, encoding):
    path = tmp_path / "state.json"
    state = {
        "tasks": [
            {"task_id": f"t{i}", "stage": "dft_single_point", "status": "submitted"}
            for i in range(80)
        ]
    }
    assert len(recover_batch_stages(state, path)["dft_single_point"]["waiting_ids"]) == 80
    db = tmp_path / "langgraph_batches.sqlite"
    encode_existing(db, encoding)
    with sqlite3.connect(db) as conn:
        old_ids = set(
            conn.execute(
                "SELECT thread_id,checkpoint_ns,checkpoint_id,parent_checkpoint_id FROM checkpoints"
            )
        )
    for task in state["tasks"]:
        task["status"] = "completed"
    result = recover_batch_stages(state, path)
    assert result["dft_single_point"]["completed_ids"] == [t["task_id"] for t in state["tasks"]]
    assert result["dft_single_point"]["waiting_ids"] == []
    with sqlite3.connect(db) as conn:
        assert old_ids <= set(
            conn.execute(
                "SELECT thread_id,checkpoint_ns,checkpoint_id,parent_checkpoint_id FROM checkpoints"
            )
        )
        assert {r[0] for r in conn.execute("SELECT thread_id FROM checkpoints")} == {
            hashlib.sha256(str(path.resolve()).encode()).hexdigest()
        }


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_encoded_approval_preserves_binding_and_delivery_once(tmp_path, decision):
    path = tmp_path / "state.json"
    envelope = {
        "invocation_id": "original",
        "proposal_hash": "h",
        "config_version": "c",
        "revision": 0,
    }
    assert durable_approval(path, envelope)["status"] == "awaiting_approval"
    encode_existing(tmp_path / "langgraph_approvals.sqlite", "cas")
    assert durable_approval(path, envelope, decision)["status"] == "decision_delivered"
    result = durable_approval(path, envelope, decision)
    assert result["status"] == "approval_reconciliation_required"
    assert result["decision"] == decision
    assert durable_approval(path, {**envelope, "revision": 1})["status"] == "awaiting_approval"


@pytest.mark.parametrize("decision,status", [("approve", "approved"), ("reject", "rejected")])
def test_encoded_direction_resumes_original_proposal(tmp_path, decision, status):
    path = tmp_path / "state.json"
    proposal = {"direction": "activate", "candidate_model_version": "new"}
    direction_checkpoint(path, proposal)
    encode_existing(tmp_path / "langgraph_directions.sqlite", "cas")
    result = direction_checkpoint(path, proposal, decision)
    assert result["status"] == status and result["proposal"] == proposal
    assert direction_checkpoint(path, proposal, decision) == result
    with pytest.raises(ValueError, match="already_recorded"):
        direction_checkpoint(path, proposal, "reject" if decision == "approve" else "approve")


def test_compatibility_context_does_not_prune_existing_history(tmp_path):
    from tests.test_compact_checkpoints import checkpoint

    path = str(tmp_path / "history.sqlite")
    config = {"configurable": {"thread_id": "old", "checkpoint_ns": ""}}
    with CompactSqliteSaver.from_conn_string(path, maintain_history=False) as saver:
        for i in range(18):
            config = saver.put(config, checkpoint(i), {"source": "loop", "step": i}, {})
            saver.put_writes(config, [("result", i)], "task")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM checkpoints").fetchone()[0] == 18
        assert conn.execute("SELECT count(*) FROM writes").fetchone()[0] == 18


@pytest.mark.parametrize("corruption", ["missing", "checksum"])
def test_corrupt_payload_blocks_recovery_instead_of_resetting_thread(tmp_path, corruption):
    path = str(tmp_path / "corrupt.sqlite")
    config = {"configurable": {"thread_id": "original", "checkpoint_ns": ""}}
    from tests.test_compact_checkpoints import checkpoint

    value = checkpoint(0)
    value["channel_values"]["business_frame"] = {"loaded_state": {"tasks": ["payload" * 12000]}}
    with CompactSqliteSaver.from_conn_string(path) as saver:
        config = saver.put(config, value, {"source": "input", "step": 0}, {})
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM checkpoint_payloads").fetchone()[0] > 0
        if corruption == "missing":
            conn.execute("DELETE FROM checkpoint_payloads")
        else:
            conn.execute("UPDATE checkpoint_payloads SET kind='msgpack',data=?", (b"wrong",))
    with CompactSqliteSaver.from_conn_string(path, maintain_history=False) as saver:
        with pytest.raises(ValueError, match="payload missing|checksum mismatch"):
            saver.get_tuple(config)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM checkpoints").fetchone()[0] == 1


def test_encoded_native_interrupt_rejects_wrong_approval_identity(tmp_path):
    from phase_agent.graphs.approval_graph import build_approval_graph
    from langgraph.types import Command

    path = tmp_path / "identity.sqlite"
    config = {"configurable": {"thread_id": "original"}}
    envelope = {"invocation_id": "a", "config_version": "c", "proposal_hash": "h", "revision": 0}
    with CompactSqliteSaver.from_conn_string(str(path), maintain_history=False) as saver:
        build_approval_graph(saver).invoke({"envelope": envelope}, config)
    with CompactSqliteSaver.from_conn_string(str(path), maintain_history=False) as saver:
        graph = build_approval_graph(saver)
        with pytest.raises(ValueError, match="identity mismatch"):
            graph.invoke(
                Command(
                    resume={
                        "decision": "approve",
                        "envelope": {**envelope, "proposal_hash": "wrong"},
                    }
                ),
                config,
            )
        assert not graph.get_state(config).values.get("delivered")


def test_loading_diagnostics_cover_checkpoint_entrypoints_and_codecs():
    from phase_agent.runtime.loaded_code import WATCHED_MODULES

    assert {
        "phase_agent.graphs.batch_recovery_graph",
        "phase_agent.graphs.approval_graph",
        "phase_agent.graphs.direction_review_graph",
        "phase_agent.persistence.compact_checkpoints",
        "phase_agent.persistence.checkpoint_blobs",
    } <= set(WATCHED_MODULES)
