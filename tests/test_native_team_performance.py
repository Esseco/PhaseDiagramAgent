"""Regression boundaries for bounded native-team context work."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from phase_agent.evidence.snapshot import evidence_snapshot, role_context
from phase_agent.evidence.decision_evidence import transport_copy


@pytest.mark.parametrize(
    "extra",
    [
        {"path": Path("synthetic/input"), "values": (1, -0.0, True)},
        {"legacy_keys": {10: "ten", 2: "two"}},
    ],
)
def test_snapshot_keeps_original_canonical_hash_and_json_normalization(extra):
    payload = {
        "state": {"current_project": {"project_path": "synthetic"}, **extra},
        "decision_context": {
            "qbc_candidates": [{"candidate_id": "c", "structure": [1, 2], "H": 0.2}]
        },
        "allowed_tools": ["pause_search"],
    }
    before = deepcopy(payload)
    material = transport_copy(payload)
    identity = {"project_id": "synthetic", "config_version": None, "model_version": None}
    encoded = json.dumps(
        {"identity": identity, "facts": material},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda value: str(value) if isinstance(value, Path) else None,
    )
    snapshot = evidence_snapshot(payload)
    assert snapshot["snapshot_id"] == hashlib.sha256(encoded.encode()).hexdigest()
    assert snapshot["facts"] == json.loads(encoded)["facts"]
    snapshot["facts"]["decision_context"]["qbc_candidates"][0]["H"] = 5
    assert payload == before


def test_table_candidates_do_not_deepcopy_nested_scientific_payload(monkeypatch):
    from phase_agent.models import transport_tables as tables

    rows = [
        {"id": i, "observations": [j / 7 for j in range(64)], "verified": True} for i in range(8)
    ]
    source = deepcopy(rows)
    with monkeypatch.context() as patch:
        patch.setattr(tables, "deepcopy", lambda value: pytest.fail("whole table copied"))
        result = tables.table_rows(rows)
    assert tables.expand_evidence_tables(result) == source
    assert rows == source


def test_native_indexed_read_skips_unrelated_history_and_keeps_full_fallback(tmp_path, monkeypatch):
    from phase_agent.tools.step_runner.file_protocol import write_json
    from phase_agent.agents.strategic import codex_team
    from tests.test_codex_native_team import make_control, team

    control, _ = make_control(tmp_path)
    write_json(
        control.state_path,
        {
            "qbc_candidates": [{"candidate_id": "c", "ehull": 0.1}],
            "event_history": [{"audit": "retained"}],
        },
    )
    reads = []
    original = codex_team.read_json

    def read(path, *args, **kwargs):
        if Path(path) == control.state_path:
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(codex_team, "read_json", read)
    t = team(control)
    context = t._context()
    assert context["team_decision_context"]["qbc_candidates"][0]["ehull"] == 0.1
    assert reads == []
    control.state_path.with_name("state.json.index.json").unlink()
    assert t._context()["team_decision_context"] == context["team_decision_context"]
    assert len(reads) == 0  # Strongly bound derived cache does not need the disposable index.
    codex_team._READ_CACHE.clear()
    assert t._context()["team_decision_context"] == context["team_decision_context"]
    assert len(reads) == 1


def test_registering_advice_reuses_one_snapshot_and_can_ack_without_task_rebuild(
    tmp_path, monkeypatch
):
    from phase_agent.agents.strategic import codex_team
    from tests.test_codex_native_team import make_control, team, advice

    control, _ = make_control(tmp_path)
    t = team(control)
    view = t.start(question="review")
    snapshots = []
    original = codex_team.evidence_snapshot

    def counted(*args, **kwargs):
        snapshots.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(codex_team, "evidence_snapshot", counted)
    registered = t.advise(view["team_id"], 1, advice(view))
    assert len(snapshots) == 1
    assert "reliability" in registered["requests"]
    snapshots.clear()
    ack = t.advise(view["team_id"], 1, advice(view), compact=True)
    assert len(snapshots) == 1
    assert ack["snapshot_id"] == view["snapshot"]["snapshot_id"]
    assert ack["remaining_roles"] == ["reliability"]
    assert ack["execution_approved"] is False
    assert "requests" not in ack and "snapshot" not in ack
    assert t.read(request_role="reliability")["requests"]["reliability"]


def test_mcp_advice_ack_is_default_and_full_response_remains_available(tmp_path):
    from phase_agent.runtime.codex.team import call_team_tool
    from tests.test_codex_native_team import make_control, team, advice

    control, _ = make_control(tmp_path)
    t = team(control)
    view = t.start(question="review")
    arguments = {"team_id": view["team_id"], "revision": 1, "advice": advice(view)}
    ack = call_team_tool(control, "phase_team_advise", arguments)
    assert ack["remaining_roles"] == ["reliability"]
    assert "requests" not in ack
    full = call_team_tool(control, "phase_team_advise", {**arguments, "compact": False})
    assert full["requests"]["reliability"]["evidence_snapshot"]["snapshot_id"] == ack["snapshot_id"]
    assert len(t._load()["advice"]) == 1


def test_index_unsupported_access_falls_back_without_changing_evidence(tmp_path, monkeypatch):
    from phase_agent.persistence.state.field_index import IndexUnavailable
    from phase_agent.tools.step_runner.file_protocol import write_json
    from phase_agent.evidence import build_decision_context as builder
    from tests.test_codex_native_team import make_control, team

    control, _ = make_control(tmp_path)
    source = {"qbc_candidates": [{"candidate_id": "c", "H": 0.3}]}
    write_json(control.state_path, source)
    original = builder.build_decision_context

    def needs_dict(state):
        if type(state) is not dict:
            raise IndexUnavailable("synthetic unsupported mapping consumer")
        return original(state)

    monkeypatch.setattr(builder, "build_decision_context", needs_dict)
    assert (
        team(control)._context()["team_decision_context"]["qbc_candidates"]
        == source["qbc_candidates"]
    )


def test_indexed_team_read_rejects_concurrent_unread_field_change(tmp_path, monkeypatch):
    import os
    from phase_agent.tools.step_runner.file_protocol import write_json
    from phase_agent.evidence import build_decision_context as builder
    from tests.test_codex_native_team import make_control, team

    control, _ = make_control(tmp_path)
    write_json(
        control.state_path, {"qbc_candidates": [], "event_history": [{"note": "old-marker"}]}
    )
    before = control.state_path.stat()
    original = builder.build_decision_context
    changed = False

    def update_unread(state):
        nonlocal changed
        result = original(state)
        if not changed:
            raw = control.state_path.read_bytes()
            assert b"old-marker" in raw
            control.state_path.write_bytes(raw.replace(b"old-marker", b"new-marker"))
            os.utime(control.state_path, ns=(before.st_atime_ns, before.st_mtime_ns))
            changed = True
        return result

    monkeypatch.setattr(builder, "build_decision_context", update_unread)
    with pytest.raises(ValueError, match="sources_changed"):
        team(control)._context()
    assert b"new-marker" in control.state_path.read_bytes()


@pytest.mark.parametrize("left,right", [(-0.0, 0.0), (True, 1), (1, 1.0)])
def test_projection_does_not_merge_distinct_json_numbers(left, right):
    payload = {"state": {"budget": left}, "decision_context": {"budget": right}}
    projected = role_context(evidence_snapshot(payload), "exploration")["evidence_snapshot"]
    assert "budget" in projected["facts"]["state"]
    assert "budget" in projected["facts"]["decision_context"]


def test_snapshot_candidate_tuple_and_repeated_rows_are_detached():
    row = {"candidate_id": "c", "H": 0.2}
    payload = {"decision_context": {"qbc_candidates": (row, row)}}
    snapshot = evidence_snapshot(payload)
    candidates = snapshot["facts"]["decision_context"]["qbc_candidates"]
    assert type(candidates) is list
    candidates[0]["H"] = 9
    assert candidates[1]["H"] == row["H"] == 0.2


def test_snapshot_rejects_nonfinite_values_and_runtime_dependencies():
    for value in [float("nan"), float("inf"), object()]:
        with pytest.raises((TypeError, ValueError)):
            evidence_snapshot({"state": {"untrusted": value}})


@pytest.mark.parametrize("inline", [False, True])
def test_native_planning_reuses_snapshot_without_weakening_context_binding(
    tmp_path, monkeypatch, inline
):
    from phase_agent.agents.strategic import codex_team
    from phase_agent.agents.simulation.agent import planning_request
    from tests.test_simulation_agent import SimulationModel
    from tests.test_codex_native_team import make_control, team

    control, context = make_control(tmp_path)
    context["allowed_tools"] = ["prepare_local_batch_files"]
    t = team(control)
    view = t.start(question="prepare", consult=['exploration'])
    from tests.test_codex_native_team import register_opinions
    register_opinions(t, view, ['exploration'])
    view = t.read(request_role="strategic")
    action = {"tool": "prepare_local_batch_files", "parameters": {"mode": "relax_inputs"}}
    plan = SimulationModel()(planning_request(action, facts=view["snapshot"])) if inline else None
    calls = []
    original = codex_team.evidence_snapshot

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(codex_team, "evidence_snapshot", counted)
    draft = t.draft(view["team_id"], 1, action, rationale="prepare", simulation_plan=plan)
    assert len(calls) == 1
    if not inline:
        calls.clear()
        t.simulate(view["team_id"], 1, SimulationModel()(draft["requests"]["simulation"]))
        assert len(calls) == 1


def test_advice_resolves_paged_candidate_ids_against_bound_source(tmp_path):
    from phase_agent.tools.step_runner.file_protocol import write_json
    from tests.test_codex_native_team import make_control, team, advice

    control, context = make_control(tmp_path)
    rows = [{"candidate_id": f"c{i}", "structure_id": f"s{i}"} for i in range(130)]
    write_json(control.state_path, {"qbc_candidates": rows})
    context["allowed_tools"] = ["select_dft_candidates"]
    from phase_agent.tools.dispatch.scientific_operations import operation_catalog
    context["scientific_operations"] = operation_catalog()
    t = team(control)
    view = t.start(question="review", consult=["reliability"])
    opinion = advice(view, "reliability")
    opinion["recommendation"] = "dft"
    opinion["options"] = [{"tool": "select_dft_candidates", "reason": "review", "target_ids": ["c129"], "parameters": {}}]
    opinion["risk_structure_ids"] = ["s129"]
    ack = t.advise(view["team_id"], 1, opinion, compact=True)
    assert ack["registered_role"] == "reliability"


def test_advice_rejects_fabricated_paged_candidate_id(tmp_path):
    from phase_agent.tools.step_runner.file_protocol import write_json
    from tests.test_codex_native_team import make_control, team, advice

    control, context = make_control(tmp_path)
    write_json(control.state_path, {"qbc_candidates": [{"candidate_id": f"c{i}"} for i in range(130)]})
    context["allowed_tools"] = ["select_dft_candidates"]
    from phase_agent.tools.dispatch.scientific_operations import operation_catalog
    context["scientific_operations"] = operation_catalog()
    t = team(control)
    view = t.start(question="review", consult=["reliability"])
    opinion = advice(view, "reliability")
    opinion["recommendation"] = "dft"
    opinion["options"] = [{"tool": "select_dft_candidates", "reason": "review", "target_ids": ["invented"], "parameters": {}}]
    with pytest.raises(ValueError, match="unavailable"):
        t.advise(view["team_id"], 1, opinion, compact=True)
