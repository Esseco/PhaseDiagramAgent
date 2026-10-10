"""把 BOHB 预算动作适配到现有 MLIP+MC 阶段。"""

from phase_agent.science.mc.run_mlip_mc import run_mlip_mc


def create_mlip_mc_bohb_evaluator(
    *,
    manager,
    phase_references,
    model_path,
    backend,
    structure_provider,
    reference_provider,
    result_parser=None,
    parameters=None,
    work_directory_provider=None,
):
    def evaluate(*, branch, action):
        structure = structure_provider(branch)
        work_directory = (
            work_directory_provider(branch, action) if work_directory_provider else None
        )
        result = run_mlip_mc(
            structure,
            branch,
            manager.boundary,
            phase_references,
            segment_budget=int(action["incremental_budget"]),
            backend=backend,
            model_path=model_path,
            parameters={**(parameters or {}), "seed": action["seed"]},
            checkpoint=action.get("checkpoint"),
            work_directory=work_directory,
        )
        parsed = result_parser(result, branch, action) if result_parser else _default_parse(result)
        reference = reference_provider(branch)
        return {
            "status": result["status"],
            "checkpoint": result.get("checkpoint"),
            "actual_cost": result.get("actual_cost"),
            "error": result.get("error"),
            "minimum_energy_per_atom": parsed.get("minimum_energy_per_atom"),
            "composition_group": branch.get("composition_group", branch.get("x")),
            "group_reference_energy_per_atom": reference.get("energy_per_atom"),
            "group_energy_scale": reference.get("scale"),
            "model_version": action["model_version"],
            "hull_reference_version": action["hull_reference_version"],
            "mc_result": result,
        }

    return evaluate


def _default_parse(result):
    outputs = result.get("outputs") or {}
    for key in ("minimum_energy_per_atom", "best_energy_mean_per_atom", "energy_per_atom"):
        if outputs.get(key) is not None:
            return {"minimum_energy_per_atom": outputs[key]}
    return {"minimum_energy_per_atom": None}
