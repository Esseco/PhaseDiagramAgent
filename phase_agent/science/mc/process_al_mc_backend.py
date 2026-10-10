"""适配 MLIP MC 的 Process_AL_MC 后端。"""

from phase_agent.science.mlip.run_mace_subprocess import run_mace_with_py_mace


def process_al_mc_backend(
    *, structure, branch, model_path, segment_budget, checkpoint, parameters, work_directory
):
    result = run_mace_with_py_mace(
        structure,
        model_path=model_path,
        operation="mc",
        work_directory=work_directory,
        parameters=parameters,
        segment_budget=segment_budget,
    )
    result["requested_budget"] = {
        "max_mc_steps": segment_budget,
        "patience_steps": parameters.get("patience_steps"),
        "min_improvement": parameters.get("min_improvement"),
    }
    result["restart_mode"] = "new_segment_from_structure"
    result["strict_chain_resume"] = False
    result["resume_note"] = (
        "Process_AL_MC restarts from the supplied structure; it does not restore the exact Markov-chain state."
        if checkpoint
        else None
    )
    return result
