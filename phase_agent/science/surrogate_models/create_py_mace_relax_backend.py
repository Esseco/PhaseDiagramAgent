"""Adapt the existing py-mace subprocess runner to the relax interface."""

from phase_agent.science.mlip.run_mace_subprocess import run_mace_with_py_mace


def create_py_mace_relax_backend(environment="py-mace"):
    def backend(*, structure, model_path, parameters, work_directory):
        return run_mace_with_py_mace(
            structure,
            model_path=model_path,
            operation="relax",
            work_directory=work_directory,
            parameters=parameters,
            environment=environment,
        )

    return backend
