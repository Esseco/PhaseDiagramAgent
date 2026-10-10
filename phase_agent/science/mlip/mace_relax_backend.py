"""适配 MLIP relax 的 MACE 后端。"""

from phase_agent.science.mlip.run_mace_subprocess import run_mace_with_py_mace


def mace_relax_backend(*, structure, model_path, parameters, work_directory):
    return run_mace_with_py_mace(
        structure,
        model_path=model_path,
        operation="relax",
        work_directory=work_directory,
        parameters=parameters,
    )
