"""Fast example with a mock relax backend; no MACE calculation."""

from pathlib import Path
from tempfile import TemporaryDirectory

from pymatgen.core import Lattice, Structure

from scientific_layer.surrogate_models.default_surrogate_config import default_surrogate_config
from scientific_layer.surrogate_models.run_surrogate_experiment import run_surrogate_experiment


def run_example(root="."):
    with TemporaryDirectory(dir=root) as directory:
        path = Path(directory) / "initial.vasp"
        Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [.5, .5, .5]]).to(filename=path)
        dataset = {"task": "mlip_mc_fixed_budget_hull_distance", "rows": [{"branch_id": "B1", "branch": {"P": "O3"}, "split": "train", "split_group": "G1", "target": {"distance_to_fixed_hull": -.1}, "initial_structures": [{"structure_id": "S1", "source_path": str(path)}]}]}
        config = default_surrogate_config(); config["relax"].update({"model_path": "mock.model", "model_version": "mock-v1"})
        def backend(**kwargs):
            return {"status": "completed", "converged": True, "structure": kwargs["structure"], "cost": 2.0, "force_evaluation_count": 5, "cost_unit": "relative"}
        first = run_surrogate_experiment(dataset, config=config, cache_directory=Path(directory) / "cache", relax_backend=backend)
        second = run_surrogate_experiment(dataset, config=config, cache_directory=Path(directory) / "cache", relax_backend=backend)
        second["first_run"] = first
        return second


if __name__ == "__main__":
    print(run_example())
