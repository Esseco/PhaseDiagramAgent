from phase_agent.persistence.models.require_structure_refresh import require_structure_refresh
from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram


def test_new_epoch_preserves_latest_valid_results_and_excludes_failures():
    def row(rid, sid, version, energy, na, status='completed'):
        return {'record_id': rid, 'structure_id': sid, 'structure_sha256': sid,
                'model_version': version, 'source_version': version, 'energy_method': 'mlip',
                'energy_unit': 'eV', 'energy': energy, 'status': status,
                'composition': {'Na': na, 'Co': 1, 'O': 2}, 'phase': 'O3'}
    state = {'tasks': [], 'upload_layout': {'model_rounds': {'old': 1}},
             'phase_diagrams': {'mlip': {'model_version': 'old'}}}
    require_structure_refresh(state, 'old', 'new')
    assert state['mlip_energy_policy'] == 'latest_valid_per_structure_v1'
    records = [row('a0', 'a', 'old', -5, 0), row('a1', 'a', 'new', -6, 0),
               row('b0', 'b', 'old', -7, 1),
               row('b1', 'b', 'new', -8, 1, 'excluded_extrapolation_failure')]
    diagram = update_phase_diagram(records, active_model_version='new', output_state=state)['diagrams']['mlip']
    assert diagram['status'] == 'completed'
    assert {r['record_id'] for r in diagram['entries']} == {'a1', 'b0'}
    assert {r['source_version'] for r in diagram['entries']} == {'old', 'new'}
