from phase_agent.evidence.model_diagram_scope import active_model_state, selection_diagram
from phase_agent.analysis.phase import phase_dft_candidates as candidates


def test_selection_uses_active_diagram_and_preserves_input():
    old = {"model_version": "new", "version": "mixed", "entries": [{"source_version": "old"}]}
    current = {"model_version": "new", "version": "current", "entries": [{"source_version": "new"}]}
    state = {"active_model_version": "new", "phase_diagrams": {"mlip": old}, "phase_diagrams_by_model": {"new": current}}
    assert selection_diagram(state) is current
    assert active_model_state(state)["phase_diagrams"]["mlip"] is current
    assert state["phase_diagrams"]["mlip"] is old


def test_selection_preserves_explicit_recovery_baseline():
    baseline = {"model_version": "old", "version": "base"}
    state = {"active_model_version": "new", "phase_diagrams": {"mlip": baseline},
             "phase_diagrams_by_model": {"new": {"model_version": "new", "version": "current"}},
             "extrapolation_recovery_cycle": {"status": "awaiting_reviewed_dft_labels", "invalid_model_version": "new", "baseline_model_version": "old", "baseline_hull_version": "base"}}
    assert selection_diagram(state) is baseline


def test_ordinary_candidates_do_not_automatically_add_extrapolation_failures(monkeypatch):
    def unexpected(*args):
        raise AssertionError("ordinary DFT selection must not auto-add failure frames")

    monkeypatch.setattr(candidates, "extrapolation_dft_candidates", unexpected)
    state = {"model_refresh": {"task_ids": ["refresh"]}, "tasks": []}
    rows = [{"candidate_id": "normal"}, {"candidate_id": "failed", "extrapolation_failure": True}]
    assert candidates.refresh_failure_candidates(state, rows) == [{"candidate_id": "normal"}]


def test_mc_candidate_binds_current_model(tmp_path, monkeypatch):
    from phase_agent.science.qbc import post_mc_candidates as module
    source = tmp_path / 'structure.vasp'
    source.write_text('frame', encoding='utf8')
    entry = {'structure_path': str(source), 'phase_identification_status': 'identified',
             'ehull_unit': 'eV/atom', 'ehull': 0.0, 'source_version': 'new'}
    diagram = {'status': 'completed', 'model_version': 'new', 'version': 'v', 'entries': [entry]}
    state = {'active_model_version': 'new', 'phase_diagrams': {'mlip': diagram}, 'tasks': [
        {'stage': 'deep_search', 'status': 'completed', 'model_version': 'new',
         'branch_id': 'b', 'structure_id': 's', 'outputs': {'structure_path': str(source)}}]}
    monkeypatch.setattr(module, '_geometry_evidence', lambda *_: None)
    monkeypatch.setattr(candidates, 'phase_review_candidates', lambda state, rows: rows)
    result = module.post_mc_candidates(state, 'new')
    assert result[0]['model_version'] == 'new'
    assert result[0]['phase_diagram_version'] == 'v'


def test_ground_state_binding_receives_active_model_diagram(monkeypatch):
    from phase_agent.tools.workflows.validate_dft_agent_decisions import _ground_state_selection_errors
    from phase_agent.analysis.feedback import dft_comparison_tables
    old = {'status': 'completed', 'model_version': 'new', 'version': 'mixed', 'entries': [{'source_version': 'old'}]}
    current = {'status': 'completed', 'model_version': 'new', 'version': 'current',
               'entries': [{'source_version': 'new', 'is_stable': True}]}
    state = {'active_model_version': 'new', 'phase_diagrams': {'mlip': old},
             'phase_diagrams_by_model': {'new': current}, 'qbc_candidates': []}
    seen = []
    monkeypatch.setattr(candidates, 'bind_ground_state_frames', lambda state, rows: seen.append(state['phase_diagrams']['mlip']) or [])
    monkeypatch.setattr(dft_comparison_tables, 'build_comparison_tables', lambda *a, **k: ([], []))
    assert _ground_state_selection_errors(state, set()) == []
    assert seen == [current]
    assert state['phase_diagrams']['mlip'] is old


def test_total_and_independent_diagrams_have_distinct_evidence_references():
    from phase_agent.evidence.decision_evidence_catalog import decision_evidence_catalog
    catalog = decision_evidence_catalog({'current_phase_diagram': {
        'mlip': {'version': 'mixed', 'status': 'completed'},
        'mlip_active_model': {'version': 'current', 'status': 'completed'}}})['references']
    assert 'phase_diagram:mlip:mixed' in catalog
    assert 'phase_diagram:mlip_active_model:current' in catalog


def test_completed_latest_hull_is_used_for_selection_and_mc_candidates(tmp_path, monkeypatch):
    from phase_agent.science.qbc import post_mc_candidates as module
    source = tmp_path / "frame.vasp"
    source.write_text("geometry", encoding="utf8")
    entry = {"structure_path": str(source), "phase_identification_status": "identified",
             "ehull_unit": "eV/atom", "ehull": 0.12, "source_version": "old"}
    total = {"status": "completed", "model_version": "new", "version": "total",
             "energy_policy": "latest_valid_per_structure_v1", "mixed_model_versions": True, "entries": [entry]}
    independent = {"status": "completed", "model_version": "new", "version": "independent", "entries": []}
    state = {"active_model_version": "new", "phase_diagrams": {"mlip": total},
             "phase_diagrams_by_model": {"new": independent}, "tasks": [
                 {"task_id": "t", "stage": "deep_search", "status": "completed", "model_version": "new",
                  "branch_id": "b", "structure_id": "s", "outputs": {"structure_path": str(source)}}]}
    monkeypatch.setattr(module, "_geometry_evidence", lambda *_: None)
    monkeypatch.setattr(candidates, "phase_review_candidates", lambda state, rows: rows)
    assert selection_diagram(state) is total
    row = module.post_mc_candidates(state, "new")[0]
    assert row["phase_diagram_version"] == "total"
    assert row["predicted_Ehull"] == 0.12
    assert row["energy_source_version"] == "old"
    assert state["phase_diagrams"]["mlip"] is total


def test_pending_refresh_does_not_select_retained_total_hull():
    from phase_agent.evidence.model_diagram_scope import decision_diagram
    total = {"status": "completed", "model_version": "new", "version": "total",
             "energy_policy": "latest_valid_per_structure_v1", "entries": []}
    current = {"status": "completed", "model_version": "new", "version": "current", "entries": []}
    state = {"active_model_version": "new", "phase_diagrams": {"mlip": total},
             "phase_diagrams_by_model": {"new": current},
             "model_refresh": {"new_model_version": "new", "status": "running"}}
    assert selection_diagram(state) is decision_diagram(state) is current


def test_mc_candidate_rejects_changed_structure_contents(tmp_path, monkeypatch):
    from phase_agent.science.qbc import post_mc_candidates as module
    source = tmp_path / "frame.vasp"
    source.write_text("modified", encoding="utf8")
    entry = {"structure_path": str(source), "phase_identification_status": "identified",
             "ehull_unit": "eV/atom", "ehull": 0.0, "structure_sha256": "0" * 64}
    diagram = {"status": "completed", "model_version": "m", "version": "v", "entries": [entry]}
    state = {"active_model_version": "m", "phase_diagrams": {"mlip": diagram}, "tasks": [
        {"task_id": "t", "stage": "deep_search", "status": "completed", "model_version": "m",
         "branch_id": "b", "structure_id": "s", "outputs": {"structure_path": str(source)}}]}
    monkeypatch.setattr(candidates, "phase_review_candidates", lambda state, rows: rows)
    assert module.post_mc_candidates(state, "m") == []
