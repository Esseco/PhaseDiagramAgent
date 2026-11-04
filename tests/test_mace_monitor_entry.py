from pathlib import Path
from tempfile import TemporaryDirectory

from ase import Atoms
from ase.io import write
from phase_agent.science.mlip import mace_worker


def test_standalone_worker_installs_monitor_and_preserves_evidence(monkeypatch):
    from Process_AL_MC import relax as module
    with TemporaryDirectory(dir=Path.cwd()) as folder:
        root = Path(folder)
        source = root / 'input.vasp'
        atoms = Atoms('NaCoO2', positions=[[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0]], cell=[5, 5, 5], pbc=True)
        write(source, atoms, format='vasp')
        sentinel = object()
        monkeypatch.setattr(mace_worker, '_extrapolation_monitor_factory', lambda: lambda initial: sentinel)
        def fake_relax(structure, model_path, **kwargs):
            assert kwargs['step_callback'] is sentinel
            write(kwargs['output_path'], atoms, format='vasp')
            return {'optimizer_converged': False, 'extrapolation_monitor': {'extrapolation_failure': True, 'final_feature_difference': 2.0}}
        monkeypatch.setattr(module, 'relax_structure_mace', fake_relax)
        result = mace_worker.run_mace_worker({'operation': 'relax', 'structure_path': str(source),
            'output_directory': str(root), 'model_path': '/fake', 'model_version': 'm', 'parameters': {}})
        assert result['extrapolation_monitor']['extrapolation_failure'] is True
        assert result['optimizer_converged'] is False and Path(result['structure_path']).is_file()


def test_mc_initial_failure_preserves_failed_frame(monkeypatch):
    import Process_AL_MC
    from Process_AL_MC.MC_sample import InitialExtrapolationFailure
    with TemporaryDirectory(dir=Path.cwd()) as folder:
        root = Path(folder)
        frame = root / 'extrapolation_000000.vasp'
        frame.write_text('frame', encoding='utf8')
        evidence = {'structure_path': str(frame), 'structure_checksum': 'checksum',
                    'extrapolation_monitor': {'extrapolation_failure': True}}
        class Runner:
            def __init__(self, **kwargs):
                assert kwargs['relaxation_monitor_factory'] is not None
            def run(self, **kwargs):
                raise InitialExtrapolationFailure(evidence)
        monkeypatch.setattr(Process_AL_MC, 'LayeredOxide_MCOrderingClass', Runner)
        monkeypatch.setattr(mace_worker, '_extrapolation_monitor_factory', lambda: object())
        result = mace_worker.run_mace_worker({'operation': 'mc', 'structure_path': 'input.vasp',
            'output_directory': str(root), 'model_path': '/fake', 'model_version': 'm', 'parameters': {}})
        assert result['status'] == 'failed' and result['extrapolation_failures'] == [evidence]
        assert result['structure_path'] == str(frame)
