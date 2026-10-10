"""No-calculation example using a tiny in-memory historical ledger."""

from tempfile import TemporaryDirectory

from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from experiments.branch_surrogate.default_branch_surrogate_config import default_branch_surrogate_config
from experiments.branch_surrogate.run_branch_dataset_preparation import run_branch_dataset_preparation


def run_example(output_directory=None):
    boundary = {"P": ["O3"], "H": {"O3": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]}, "TM_ratio": {"Fe": 1, "Mn": 1}}
    manager = PhaseDataManager(boundary)
    branch_id = manager.add_branch(P="O3", H=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], x=.5, T=["Fe", "Mn"], composition={"Na": 1, "Fe": 1, "Mn": 1, "O": 4})
    structure_id = manager.add_structure(branch_id=branch_id, arrangement={"V": [1, 0]}, metadata={"calculation_attempts": [{"task_id": "mc-1", "stage": "deep_search", "status": "completed", "model_version": "mh-1", "budget": 30, "actual_cost": 30}]})
    manager.record_result(structure_id=structure_id, stage="deep_search", converged=True, mlip_version="mh-1", mlip_energy=-10.0, energy_unit="eV", metadata={"task_id": "mc-1", "mc_budget": 30, "atom_count": 4, "actual_cost": 30})
    config = default_branch_surrogate_config()
    config.update({"target_mc_budget": 30, "mlip_version": "mh-1", "hull_reference_version": "hull-v1"})
    reference = {"version": "hull-v1", "groups": {"{\"Fe\":1,\"Mn\":1,\"Na\":1,\"O\":4}": -2.4}}
    if output_directory is not None:
        return run_branch_dataset_preparation(manager, config=config, hull_reference=reference, output_directory=output_directory)
    with TemporaryDirectory() as directory:
        result = run_branch_dataset_preparation(manager, config=config, hull_reference=reference, output_directory=directory)
        return {"report": result["report"], "row": result["dataset"]["rows"][0]}


if __name__ == "__main__":
    print(run_example())
