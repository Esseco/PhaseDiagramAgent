from copy import deepcopy

import pytest

from phase_agent.analysis.phase.phase_dft_sampling import (
    dft_sampling_reference, fixed_snapshot_sampling_evidence,
)
from phase_agent.analysis.phase.phase_low_energy_D import evaluate_low_energy_prefixes
from phase_agent.analysis.phase.phase_quality import diagnose_phase_diagram
from phase_agent.analysis.phase.phase_quality_reports import candidate_quality_evidence
from phase_agent.science.qbc.build_candidate_metrics import build_qbc_candidate_metrics
from tests.test_phase_quality import sample, add
from tests.test_phase_energy_islands import masked_endpoint


def test_fixed_D_prefixes_unmask_multiple_low_clusters_and_keep_missingness():
    def points(values):
        return [dict(key=str(i), phase="O3", composition_key=(("O", 2),), x=0., energy_mev_per_O2=float(e)) for i, e in enumerate(values)]
    clusters, candidates, _ = evaluate_low_energy_prefixes(points([0, 0, 37, 37, 196, 196, 197, 198, 199, 200]))
    assert clusters[0]["D"] == clusters[1]["D"] == pytest.approx(15.9)
    assert candidates[1]["candidate_size"] == 4
    assert candidates[1]["denominator_meV_per_O2"] == 10
    assert evaluate_low_energy_prefixes(points([0, 9, 18, 27, 36, 45, 200, 201]))[0][0]["D"] is None
    assert evaluate_low_energy_prefixes(points([0, 100]))[0][0]["D"] is None


def test_static_D_is_fixed_and_union_H_changes_with_saved_recursion():
    state = masked_endpoint()
    report = diagnose_phase_diagram(state)
    before = deepcopy(state)
    data = fixed_snapshot_sampling_evidence(state["phase_diagrams"]["mlip"], report)
    assert state == before
    assert all(v["H_layers"] == [r["analysis_layer"] for r in report["analysis_layers"]] for v in data.values())
    assert any(v["H_D_candidate_union_sequence"][0] == pytest.approx(196) for v in data.values() if v["D_score"] is not None)
    assert any(None in v["H_single_sequence"] for v in data.values())
    assert all(v["D_unit"] == "dimensionless" and v["H_unit"] == "meV/O2" for v in data.values())
    bad = deepcopy(report)
    bad["snapshot_evidence_digest"] = "stale"
    with pytest.raises(ValueError, match="不匹配"):
        fixed_snapshot_sampling_evidence(state["phase_diagrams"]["mlip"], bad)


def test_adapter_preserves_order_count_QBC_Ehull_and_propagates_metrics():
    state = sample()
    candidate = dict(candidate_id="c", structure_sha256="g1", model_version="m1", predicted_Ehull=.02, ehull_unit="eV/atom", phase_diagram_version="h1", qbc={"status": "not_available_single_model"})
    candidates = [candidate, {**candidate, "candidate_id": "stale", "model_version": "m0"}]
    before = deepcopy((state, candidates))
    result = candidate_quality_evidence(state, candidates)
    assert (state, candidates) == before
    assert [r["candidate_id"] for r in result] == ["c", "stale"]
    assert result[0]["predicted_Ehull"] == .02 and result[0]["qbc"] == candidate["qbc"]
    assert result[0]["dft_sampling_evidence"]["status"] == "available"
    assert result[1]["dft_sampling_evidence"]["status"] == "unavailable"
    metrics = build_qbc_candidate_metrics(result)["metrics"][0]
    assert metrics["dft_sampling_evidence"] == result[0]["dft_sampling_evidence"]
    assert metrics["energy_per_atom_std"] is None


def test_far_D_is_visible_for_audit_but_not_a_sampling_priority():
    state = sample()
    add(state, "far", 10, x=.25)
    for i in range(6):
        add(state, f"reference{i}", 11 + .001*i, x=.25)
    report = diagnose_phase_diagram(state)
    row = next(r for r in report["structures"] if "far" in r["record_ids"])
    candidate = dict(candidate_id="far", structure_sha256=row["structure_sha256"][0])
    enriched = candidate_quality_evidence(state, [candidate], report=report)[0]
    evidence = enriched["dft_sampling_evidence"]
    assert evidence["D_score"] > 90 and evidence["quality_priority_eligible"] is False
    assert enriched["phase_quality_signal"]["dft_evidence_priority_mev_per_O2"] == 0


def test_stale_candidate_hull_does_not_receive_current_D_H():
    state = sample()
    row = dict(candidate_id="c", structure_sha256="g1", phase_diagram_version="old")
    result = candidate_quality_evidence(state, [row])[0]
    assert result["dft_sampling_evidence"]["status"] == "unavailable"


def test_reference_has_no_weights_quotas_or_D_cutoff():
    rules = dft_sampling_reference()
    assert rules["role"] == "agent_reference_not_automatic_ranking"
    assert not {"weights", "quotas", "D_cutoff", "score_formula"}.intersection(rules)
    text = str(rules)
    assert "不能替换predicted_Ehull" in text and "不填零" in text and "不能要求同时高" in text


def test_current_read_context_refreshes_evidence_without_changing_saved_candidates():
    from phase_agent.analysis.state.build_decision_context import build_decision_context
    s = sample()
    s["qbc_candidates"] = [dict(candidate_id="c", structure_sha256="g1", model_version="m1")]
    before = deepcopy(s)
    context = build_decision_context(s)
    assert s == before
    assert context["qbc_candidates"][0]["dft_sampling_evidence"]["status"] == "available"


def test_saved_selection_details_preserve_fixed_D_and_conditional_H():
    from phase_agent.tools.workflows.dft_selection_artifacts import selection_record
    from phase_agent.tools.workflows.dft_review_view import markdown_view
    s = sample()
    s["qbc_candidates"] = candidate_quality_evidence(s, [dict(candidate_id="c", structure_sha256="g1", model_version="m1")])
    action = {"tool": "select_dft_candidates", "parameters": {"decisions": [{"candidate_id": "c", "action": "DFT_SINGLE_POINT", "reason": "核查低能离群"}]}}
    record = selection_record({"raw_action": action}, s)
    assert record["decisions"][0]["dft_sampling_evidence"]["geometry_key"]
    text = markdown_view(record)
    assert "固定D_无量纲" in text and "H候选联合序列" in text


def test_semantic_proposal_receives_reference_without_new_model_call(monkeypatch):
    import phase_agent.decisions.agent.propose_tool_action as module
    from phase_agent.analysis.state.build_decision_context import build_decision_context
    # Intercept the existing validated client boundary, not the scientific executor.
    captured = []
    def validated(client, payload):
        captured.append(payload)
        return {"tool": "select_dft_candidates", "task_key": "one", "parameters": {"decisions": []}, "reason": "参考当前证据"}
    monkeypatch.setattr(module, "request_validated_action", validated)
    s = sample()
    s["qbc_candidates"] = [dict(candidate_id="c", structure_sha256="g1", model_version="m1")]
    view = {"decision_context": build_decision_context(s)}
    module.propose_agent_tool_action(view, agent_client=lambda p: None, allowed_tools=["select_dft_candidates"], config={})
    assert len(captured) == 1
    reference = captured[0]["decision_context"]["dft_selection_policy"]["reference_framework"]
    assert reference == dft_sampling_reference()
    assert reference["D_unit"] == "dimensionless"
    assert len(reference["evidence_patterns"]) == 6
    assert "默认先选一个代表" in reference["representative_selection"]
    assert "post_dft_assessment" in str(reference["adaptation"])
    assert captured[0]["decision_context"]["qbc_candidates"][0]["dft_sampling_evidence"]["status"] == "available"


def test_missing_structure_file_is_explicitly_unknown(tmp_path):
    state = sample()
    candidate = dict(candidate_id="missing", structure_sha256="g1", structure_path=str(tmp_path / "missing.vasp"))
    signal = candidate_quality_evidence(state, [candidate])[0]["dft_sampling_evidence"]
    assert signal["status"] == "unavailable" and signal["structure_file_status"] == "unavailable"
    assert signal["reason"].startswith("structure_file_unavailable:")
    assert "D_score" not in signal


def test_review_plot_preserves_points_and_makes_sparse_phases_visible(tmp_path, monkeypatch):
    import Process_Phasediagram
    from matplotlib.figure import Figure
    from phase_agent.analysis.phase.phase_quality_comparison import publish_quality_comparison
    state = masked_endpoint()
    report = diagnose_phase_diagram(state)
    before = deepcopy(state)
    received, legends = [], []
    original_plot = Process_Phasediagram.plot_phase_diagram
    original_save = Figure.savefig
    def plot(*args, **kwargs):
        received.append(kwargs)
        return original_plot(*args, **kwargs)
    def save(figure, *args, **kwargs):
        legends.extend(t.get_text() for legend in figure.legends for t in legend.get_texts())
        return original_save(figure, *args, **kwargs)
    monkeypatch.setattr(Process_Phasediagram, "plot_phase_diagram", plot)
    monkeypatch.setattr(Figure, "savefig", save)
    result = publish_quality_comparison(state, report, directory=tmp_path)
    assert result["status"] == "completed" and state == before
    assert len(received) == 2 and all(r["unstable_alpha"] >= .6 for r in received)
    assert any("(n=" in text for text in legends)
    assert len(result["plot_paths"]) == 1 and result["plot_paths"][0].endswith(".png")



def test_reference_patterns_use_existing_purposes_and_preserve_scientific_authority():
    from typing import get_args
    from phase_agent.decisions.agent.scientific_decisions import SelectionPurpose

    rules = dft_sampling_reference()
    allowed = set(get_args(SelectionPurpose.model_fields["purpose"].annotation))
    assert {row["purpose"] for row in rules["evidence_patterns"]} <= allowed
    assert any("QBC低" in row["when"] and row["purpose"] == "bias_check" for row in rules["evidence_patterns"])
    assert any("递归后" in row["when"] and "前置" in row["consider"] for row in rules["evidence_patterns"])
    assert "同组分同相" in rules["representative_selection"]
    assert "已完成且质量合格" in rules["representative_selection"]
    assert "不能编造" in rules["representative_selection"]
    assert "不自动改cutoff或预算" in str(rules["adaptation"])
    assert "也不归入三者均低" in str(rules["cautions"])
