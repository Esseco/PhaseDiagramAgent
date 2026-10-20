"""Ownership and incremental-cache guards; no timing assertions or real work."""

from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace
import json

import pytest


class NoCopy(list):
    def __deepcopy__(self, memo):
        raise AssertionError("A private historical payload was copied again")


@pytest.mark.parametrize(
    "module,name,kwargs,index",
    [
        ("tools.state.restore_generation_gate", "restore_generation_gate", {"manager": None}, None),
        ("tools.local.recover_remote_training", "recover_remote_training", {}, 0),
        ("tools.relax_mc_batch.results", "recover_available", {"context": {}}, None),
        (
            "analysis.phase.update_local_mlip_hull_pool",
            "update_local_mlip_hull_pool",
            {"config": {}, "path": None},
            None,
        ),
        (
            "analysis.phase.ensure_phase_identification",
            "ensure_phase_identification",
            {"manager": SimpleNamespace(data={})},
            0,
        ),
        (
            "tools.workflows.apply_scientific_feedback",
            "apply_scientific_feedback",
            {"recovered_results": [], "manager": None},
            "state",
        ),
        ("graphs.training_handoff_graph", "run_training_handoffs", {"config": {}}, 0),
    ],
)
def test_owned_internal_state_does_not_copy_history(module, name, kwargs, index):
    state = {"tasks": [], "history_payload": NoCopy([{"proof": [1]}])}
    result = getattr(import_module("phase_agent." + module), name)(state, _owned=True, **kwargs)
    current = result if index is None else result[index]
    assert current is state


def test_public_helpers_keep_defensive_copy():
    from phase_agent.tools.state.restore_generation_gate import restore_generation_gate

    state = {"tasks": [], "history_payload": [{"proof": [1]}]}
    saved = deepcopy(state)
    returned = restore_generation_gate(state, None)
    returned["history_payload"][0]["proof"].append(2)
    assert state == saved


def test_owned_compaction_retains_history_without_copy():
    from phase_agent.tools.workflows.compact_action_history import compact_state_history

    record = {"record_id": "r", "result": {"state": {"redundant": True}, "proof": NoCopy([1])}}
    state = {"action_records": [record], "tasks": []}
    assert compact_state_history(state, _owned=True) is state
    assert state["action_records"][0]["record_id"] == "r"
    assert "state" not in record["result"] and record["result"]["proof"] == [1]


def test_incremental_phase_cache_compacts_once(tmp_path, monkeypatch):
    module = import_module("phase_agent.analysis.phase.ensure_phase_identification")
    writes = []
    persist = module._persist_cache_file

    def record(path, cache):
        writes.append(len(cache))
        return persist(path, cache)

    monkeypatch.setattr(module, "_persist_cache_file", record)

    def identify(task, manager, **kwargs):
        output = {
            **task.get("outputs", {}),
            "phase_identification": {
                "status": "identified",
                "phase": "O3",
                "classification_scope": "scope",
                "structure_sha256": task["task_id"],
            },
        }
        return {**task, "outputs": output, "actual_phase": "O3"}

    monkeypatch.setattr(module, "identify_result_phase", identify)
    state = {
        "tasks": [
            {"task_id": str(i), "stage": "deep_search", "status": "completed"} for i in range(30)
        ]
    }
    path = tmp_path / "cache.json"
    _, report = module.ensure_phase_identification(state, SimpleNamespace(data={}), cache_path=path)
    assert report["newly_cached_results"] == 30
    assert len(writes) == 1, "Do not serialize the entire growing cache for every result"
    assert len(json.loads(path.read_text())) == 30


def test_phase_cache_recovers_committed_entries_after_interruption(tmp_path, monkeypatch):
    module = import_module("phase_agent.analysis.phase.ensure_phase_identification")

    def identify(task, manager, **kwargs):
        if task["task_id"] == "stop":
            raise RuntimeError("simulated interruption")
        return {
            **task,
            "outputs": {
                "phase_identification": {
                    "status": "identified",
                    "phase": "O3",
                    "classification_scope": "scope",
                    "structure_sha256": task["task_id"],
                }
            },
        }

    monkeypatch.setattr(module, "identify_result_phase", identify)
    state = {
        "tasks": [
            {"task_id": i, "stage": "deep_search", "status": "completed"} for i in ["done", "stop"]
        ]
    }
    path = tmp_path / "cache.json"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        module.ensure_phase_identification(state, SimpleNamespace(data={}), cache_path=path)
    loaded = {}
    assert module._merge_cache_file(loaded, path) is None
    assert loaded["scope:done"]["phase"] == "O3"


def test_phase_journal_handles_torn_tail_and_concurrent_writers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    store = import_module("phase_agent.analysis.phase.phase_identification_cache")
    path = tmp_path / "cache.json"
    row = {"status": "identified", "phase": "O3"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: store.append_entry(path, str(i), row), range(12)))
    journal = store.journal_path(path)
    with journal.open("ab") as stream:
        stream.write(b'{"torn":')
    store.append_entry(path, "after", row)
    loaded = {}
    assert store.load_cache(loaded, path)
    assert set(loaded) == {str(i) for i in range(12)} | {"after"}
    store.compact_cache(path, loaded)
    assert json.loads(path.read_text()) == loaded
    assert journal.read_bytes() == b""


def test_late_input_timings_survive_checkpoint_volume():
    from phase_agent.runtime.turn_process import _current, record_timing

    record = {"timings": []}
    token = _current.set(record)
    try:
        for i in range(150):
            record_timing("checkpoint-" + str(i), 0.01, category="checkpoint")
        for i in range(10):
            record_timing("input-" + str(i), 0.02, category="training_inputs")
        record_timing("input-failure", 0.1, category="training_inputs", status="failed")
        assert len(record["timings"]) <= 80
        assert len([r for r in record["timings"] if r["category"] == "training_inputs"]) == 11
        assert record["timings"][-1]["status"] == "failed"
    finally:
        _current.reset(token)


def test_unchanged_final_geometry_does_not_reparse_and_invalidates_content(tmp_path, monkeypatch):
    from pymatgen.core import Structure, Lattice

    module = import_module("phase_agent.analysis.phase.identify_result_phase")
    geometry = tmp_path / "final.vasp"
    structure = Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    structure.to(filename=geometry, fmt="poscar")
    manager = SimpleNamespace(data={"branches": {}, "structures": {}})
    result = {
        "stage": "deep_search",
        "status": "completed",
        "outputs": {"structure_path": str(geometry)},
    }
    monkeypatch.setattr(
        "phase_agent.science.structures.identify_branch.identify_phase",
        lambda *a, **k: {"phase": "P3", "method": "test"},
    )
    first = module.identify_result_phase(result, manager)
    assert first["actual_phase"] == "P3"
    parse = Structure.from_file
    calls = []

    def counted(*a, **k):
        calls.append(1)
        return parse(*a, **k)

    monkeypatch.setattr(Structure, "from_file", counted)
    assert module.identify_result_phase(first, manager)["actual_phase"] == "P3"
    assert calls == []
    geometry.write_bytes(geometry.read_bytes() + b"\n")
    assert module.identify_result_phase(first, manager)["actual_phase"] == "P3"
    assert len(calls) == 1
    geometry.unlink()
    missing = module.identify_result_phase(first, manager)
    assert missing["phase_identification_status"] == "unknown"


@pytest.mark.parametrize("change", ["boundary", "reference", "frame", "inline"])
def test_geometry_reuse_keeps_scope_and_final_frame_checks(tmp_path, monkeypatch, change):
    from pymatgen.core import Structure, Lattice

    module = import_module("phase_agent.analysis.phase.identify_result_phase")
    path = tmp_path / "final.vasp"
    reference = tmp_path / "reference.vasp"
    structure = Structure(
        Lattice.cubic(4),
        ["Na", "Fe", "O", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]],
    )
    structure.to(filename=path, fmt="poscar")
    reference.write_bytes(path.read_bytes())
    manager = SimpleNamespace(boundary={"P": ["P3"]}, data={"branches": {}, "structures": {}})
    result = {
        "stage": "deep_search",
        "status": "completed",
        "outputs": {"structure_path": str(path)},
    }
    refs = {"P3": str(reference)}
    first = module.identify_result_phase(result, manager, phase_references=refs)
    assert first["outputs"]["phase_identification"]["method"] == "unique_configured_phase"
    outputs = first["outputs"]
    if change == "boundary":
        manager.boundary = {"P": ["O3"]}
    elif change == "reference":
        reference.write_bytes(reference.read_bytes() + b"\n")
    elif change == "frame":
        outputs["final_frame_valid"] = False
    elif change == "inline":
        outputs["structure"] = structure.as_dict()
    assert module._cached_file_identity(outputs, manager, refs) is None
    second = module.identify_result_phase(first, manager, phase_references=refs)
    if change == "boundary":
        assert second["actual_phase"] == "O3"
    elif change == "frame":
        assert second["phase_identification_status"] == "unknown"
    else:
        assert second["actual_phase"] == "P3"


def test_history_directory_index_skips_settled_mc_but_keeps_dft(tmp_path):
    from phase_agent.tools.remote.batch_runner import _task_directories_by_id

    old = tmp_path / "old"
    old.mkdir()
    (old / "task.json").write_text("{broken")
    dft = tmp_path / "dft"
    dft.mkdir()
    (dft / "task.json").write_text(json.dumps({"task_id": "dft", "batch_id": "D", "task_key": "k"}))
    state = {
        "slurm_batches": [
            {"batch_id": "M", "task_ids": ["mc"], "task_directories": [str(old)]},
            {"batch_id": "D", "task_ids": ["dft"], "task_directories": [str(dft)]},
        ]
    }
    assert set(_task_directories_by_id(state, task_ids={"dft"})) == {("D", "dft")}
    assert _task_directories_by_id(state, task_ids=set()) == {}
    assert ("D", "dft") in _task_directories_by_id(state)


def test_owned_already_imported_mc_avoids_copy_but_rechecks_binding(tmp_path):
    from phase_agent.tools.relax_mc_batch.results import import_results
    from phase_agent.tools.remote.transfer_bundle import checksum

    archive = tmp_path / "results.zip"
    archive.write_bytes(b"existing archive identity")
    state = {
        "active_model_version": "m1",
        "history": NoCopy([1]),
        "relax_mc_batches": {
            "b": {
                "model_version": "m1",
                "config_version": "c1",
                "expected_results_archive": str(archive),
                "import_checksum": checksum(archive),
            }
        },
    }
    context = {"event_state": state, "effective_config": {}, "config_version": "c1"}
    action = {"parameters": {"bundle_id": "b"}}
    assert import_results(action=action, context=context, _owned=True)["state"] is state
    with pytest.raises(ValueError, match="batch_model_or_config_version_mismatch"):
        import_results(action=action, context={**context, "config_version": "c2"}, _owned=True)
