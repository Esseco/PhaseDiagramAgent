from phase_agent.analysis.phase.phase_dft_candidates import (
    bind_ground_state_frames,
    phase_review_candidates,
)
from tests.test_phase_dft_candidates import files
from tests.test_phase_quality import sample


def test_mixed_hull_candidates_keep_frame_energy_source(tmp_path):
    state = sample()
    files(state, tmp_path)
    snapshot = state['phase_diagrams']['mlip']
    snapshot['mixed_model_versions'] = True
    snapshot['energy_policy'] = 'latest_valid_per_structure_v1'
    snapshot['source_model_versions'] = ['m1', 'old-model']
    for row in snapshot['entries']:
        row['source_version'] = state['active_model_version']
    old = snapshot['entries'][1]
    old['source_version'] = 'old-model'
    old['is_stable'] = False
    result = phase_review_candidates(state, [])
    candidate = next(row for row in result if row['candidate_id'] == old['structure_id'])
    assert candidate['energy_source_version'] == 'old-model'
    assert candidate['model_version'] == 'old-model'
    assert candidate['phase_diagram_version'] == snapshot['version']


def test_mixed_ground_state_does_not_invent_missing_source(tmp_path):
    state = sample()
    files(state, tmp_path)
    snapshot = state['phase_diagrams']['mlip']
    snapshot['mixed_model_versions'] = True
    entry = snapshot['entries'][1]
    entry['is_stable'] = True
    entry['x_Na_per_O2'] = 2 * entry['composition']['Na'] / entry['composition']['O']
    entry.pop('source_version', None)
    candidate = next(row for row in bind_ground_state_frames(state, []) if row['structure_id'] == entry['structure_id'])
    assert candidate['energy_source_version'] is None
    assert candidate['model_version'] is None
