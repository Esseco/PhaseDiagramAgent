from pathlib import Path
from execution_layer.remote.run_mlip_batch import main
if __name__ == '__main__':
    manifest = str(Path(__file__).resolve().with_name('manifest.json'))
    raise SystemExit(main(['--manifest', manifest, '--executor', 'scientific_layer.mlip.slurm_executor:execute_mlip_task']))
