import pytest

from phase_agent.graphs.business_checkpoint import restore_frame
from phase_agent.tools.state.workflow_control import FIELD


def test_shared_nodes_restore_independent_mutable_values():
    shared = {"rows": [1]}
    saved = {"loaded_state": {}, "first": shared, "second": shared}
    restored = restore_frame({"loaded_state": {}}, saved)
    restored["first"]["rows"].append(2)
    assert restored["second"]["rows"] == [1]
    assert shared["rows"] == [1]


def test_legacy_tuple_keeps_json_roundtrip_behavior():
    restored = restore_frame({"loaded_state": {}}, {"loaded_state": {}, "value": (1, 2)})
    assert restored["value"] == [1, 2]


def test_changed_control_revision_still_blocks_restore():
    with pytest.raises(ValueError, match="stale_lifecycle_checkpoint"):
        restore_frame({"loaded_state": {FIELD: "new"}}, {"loaded_state": {FIELD: "old"}})


def test_snapshot_list_detaches_shared_values():
    from phase_agent.graphs.business_checkpoint import checkpoint_frame
    shared = {"rows": [1]}
    result = checkpoint_frame({"loaded_state": {"items": [shared, shared]}})
    result["loaded_state"]["items"][0]["rows"].append(2)
    assert result["loaded_state"]["items"][1]["rows"] == [1]
    assert shared["rows"] == [1]


def test_snapshot_inplace_mutation_is_preserved():
    from phase_agent.graphs.business_checkpoint import checkpoint_frame
    frame = {"loaded_state": {"items": [1]}}
    old = checkpoint_frame(frame)
    frame["loaded_state"]["items"].append(2)
    new = checkpoint_frame(frame, old)
    assert old["loaded_state"]["items"] == [1]
    assert new["loaded_state"]["items"] == [1, 2]


def test_snapshot_legacy_path_and_tuple():
    from pathlib import Path
    from phase_agent.graphs.business_checkpoint import checkpoint_frame
    result = checkpoint_frame({"loaded_state": {"items": [Path("file"), (1, 2)]}})
    assert result["loaded_state"]["items"] == ["file", [1, 2]]
