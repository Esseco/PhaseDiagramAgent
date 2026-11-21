from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from phase_agent.tools.workflows.dft_selection_plot import plot_dft_selection


def test_selected_exact_frame_is_marked_and_extrapolation_is_not_plotted():
    with TemporaryDirectory() as folder:
        path = Path(folder) / 'structure.vasp'
        path.write_text('bound frame', encoding='utf8')
        digest = sha256(path.read_bytes()).hexdigest()
        entries = [
            {'composition': {'O': 2}, 'eform_per_O2': 0, 'phase': 'O1', 'is_stable': True},
            {'composition': {'Na': 0.5, 'O': 2}, 'eform_per_O2': -0.1, 'phase': 'P3', 'is_stable': True, 'structure_sha256': digest},
            {'composition': {'Na': 1, 'O': 2}, 'eform_per_O2': 0, 'phase': 'O3', 'is_stable': True},
        ]
        state = {'active_model_version': 'm', 'phase_diagrams': {'mlip': {'status': 'completed', 'model_version': 'm', 'version': 'v', 'entries': entries}},
                 'qbc_candidates': [{'candidate_id': 'good', 'structure_path': str(path), 'structure_sha256': digest},
                                    {'candidate_id': 'bad', 'structure_path': str(path), 'structure_sha256': digest, 'extrapolation_failure': True}]}
        record = {'model_version': 'm', 'hull_version': 'v', 'decisions': [{'candidate_id': 'good', 'action': 'DFT_SINGLE_POINT'}, {'candidate_id': 'bad', 'action': 'DFT_SINGLE_POINT'}]}
        result = plot_dft_selection(state, record, Path(folder) / 'plot.png')
        import json
        info = json.loads(Path(result['metadata']).read_text(encoding='utf8'))
        assert info['marked_candidates'] == ['good']
        assert info['unplotted_selected'][0]['candidate_id'] == 'bad'
        assert Path(result['figure']).stat().st_size > 1000


def test_required_plot_blocks_selected_frames_not_on_hull():
    import pytest
    from phase_agent.tools.workflows.dft_selection_plot import require_dft_selection_plot
    with TemporaryDirectory() as folder:
        path = Path(folder) / "frame.vasp"
        path.write_text("frame", encoding="utf8")
        digest = sha256(path.read_bytes()).hexdigest()
        entries = [{"composition": {"O": 2}, "eform_per_O2": 0, "phase": "O3", "is_stable": True}, {"composition": {"Na": 1, "O": 2}, "eform_per_O2": 0, "phase": "O3", "is_stable": True}]
        state = {"active_model_version": "m", "phase_diagrams": {"mlip": {"status": "completed", "model_version": "m", "version": "v", "entries": entries}}, "qbc_candidates": [{"candidate_id": "missing", "structure_path": str(path), "structure_sha256": digest}]}
        with pytest.raises(ValueError, match="dft_selection_plot_unplotted"):
            require_dft_selection_plot(state, [{"candidate_id": "missing", "action": "DFT_SINGLE_POINT"}])

def test_required_plot_blocks_without_current_hull():
    import pytest
    from phase_agent.tools.workflows.dft_selection_plot import require_dft_selection_plot
    with pytest.raises(ValueError, match="current MLIP hull is unavailable"):
        require_dft_selection_plot({}, [{"candidate_id": "x", "action": "DFT_SINGLE_POINT"}])


def test_required_plot_allows_verified_extrapolation_frame_without_plotting_it():
    from phase_agent.tools.workflows.dft_selection_plot import require_dft_selection_plot
    with TemporaryDirectory() as folder:
        path=Path(folder)/'frame.vasp';path.write_text('frame',encoding='utf8');digest=sha256(path.read_bytes()).hexdigest()
        entries=[{'composition':{'O':2},'eform_per_O2':0,'phase':'O1','is_stable':True},{'composition':{'Na':1,'O':2},'eform_per_O2':0,'phase':'O3','is_stable':True}]
        state={'active_model_version':'m','phase_diagrams':{'mlip':{'status':'completed','model_version':'m','version':'v','entries':entries}},'qbc_candidates':[{'candidate_id':'bad','structure_path':str(path),'structure_sha256':digest,'extrapolation_failure':True}]}
        require_dft_selection_plot(state,[{'candidate_id':'bad','action':'DFT_SINGLE_POINT'}])


def test_recovery_uses_verified_baseline_and_excludes_invalid_epoch_frames():
    import pytest
    from phase_agent.tools.workflows.dft_selection_plot import require_dft_selection_plot
    with TemporaryDirectory() as folder:
        path = Path(folder) / "frame.vasp"
        path.write_text("recovery frame", encoding="utf8")
        digest = sha256(path.read_bytes()).hexdigest()
        entries = [{"composition": {"O": 2}, "eform_per_O2": 0, "phase": "O1", "is_stable": True}, {"composition": {"Na": 1, "O": 2}, "eform_per_O2": 0, "phase": "O3", "is_stable": True}]
        state = {"active_model_version": "bad", "phase_diagrams": {"mlip": {"status": "completed", "model_version": "old", "version": "v", "entries": entries}}, "qbc_candidates": [{"candidate_id": "r", "structure_path": str(path), "structure_sha256": digest}], "extrapolation_recovery_cycle": {"status": "awaiting_reviewed_dft_labels", "invalid_model_version": "bad", "baseline_model_version": "old", "baseline_hull_version": "v", "selected": [{"candidate_id": "r", "structure_sha256": digest}]}}
        decisions = [{"candidate_id": "r", "action": "DFT_SINGLE_POINT"}]
        require_dft_selection_plot(state, decisions)
        state["extrapolation_recovery_cycle"]["selected"][0]["structure_sha256"] = "wrong"
        with pytest.raises(ValueError, match="recovery_frame_hash_mismatch"):
            require_dft_selection_plot(state, decisions)
        state["extrapolation_recovery_cycle"]["baseline_hull_version"] = "wrong"
        with pytest.raises(ValueError, match="current MLIP hull is unavailable"):
            require_dft_selection_plot(state, decisions)
