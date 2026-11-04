from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from phase_agent.analysis.phase.phase_dft_candidates import extrapolation_dft_candidates
from phase_agent.analysis.phase.phase_quality_disposition import filter_excluded_candidates


def test_failure_is_a_separate_bound_dft_candidate_but_not_a_hull_record():
    with TemporaryDirectory(dir=Path.cwd()) as folder:
        path = Path(folder) / 'frame.vasp'
        path.write_text('structure', encoding='utf8')
        digest = sha256(path.read_bytes()).hexdigest()
        state = {'active_model_version': 'm', 'tasks': [{'task_id': 't', 'structure_id': 's',
            'model_version': 'm', 'branch_id': 'b', 'extrapolation_review': {'status': 'extrapolation_failure', 'score': 2},
            'outputs': {'structure_path': str(path), 'structure_checksum': digest,
                        'composition': {'Na': 1, 'O': 2}, 'optimizer_converged': False}}],
            'phase_records': [{'record_id': 'r', 'source_task_id': 't', 'model_version': 'm',
                               'structure_path': str(path), 'status': 'excluded_extrapolation_failure'}]}
        current = [{'candidate_id': 's', 'structure_id': 's', 'model_version': 'm'}]
        rows = extrapolation_dft_candidates(state, current)
        assert len(rows) == 2
        failure = rows[1]
        assert failure['candidate_id'] == 'EXTRAP-t' and failure['structure_id'] == 's'
        assert failure['structure_sha256'] == digest and failure['predicted_Ehull'] is None
        assert failure['optimizer_converged'] is False
        assert filter_excluded_candidates(state, rows) == rows
        path.write_text('modified', encoding='utf8')
        assert extrapolation_dft_candidates(state, current) == current


def test_other_quality_exclusions_still_filter_candidates():
    state = {'active_model_version': 'm', 'phase_records': [{'record_id': 'r',
             'model_version': 'm', 'structure_path': 'bad', 'status': 'excluded_invalid_result'}]}
    assert filter_excluded_candidates(state, [{'record_id': 'r', 'structure_path': 'bad'}]) == []
