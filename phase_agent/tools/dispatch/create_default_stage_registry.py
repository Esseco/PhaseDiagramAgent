"""注册带统一 task/context 接口的默认五阶段。"""

from .calculation_stage_registry import CalculationStageRegistry
from phase_agent.science.dft.run_relax import run_dft_relax
from phase_agent.science.dft.run_singlepoint import run_dft_singlepoint
from phase_agent.science.mc.run_mlip_mc import run_mlip_mc
from phase_agent.science.mlip.run_relax import run_mlip_relax
from phase_agent.science.structures.run_structure_check import run_structure_check


def create_default_stage_registry():
    registry = CalculationStageRegistry()

    def structure(task, context):
        return context.get("structure") or task.get("structure_path")

    registry.register(
        "simple_check",
        label="简单检查",
        handler=lambda task, context: run_structure_check(
            structure(task, context),
            context["boundary"],
            context["phase_references"],
            expected_branch=context.get("branch"),
            parameters=task.get("parameters"),
        ),
    )
    registry.register(
        "relax_and_feature",
        label="弛豫+特征识别",
        handler=lambda task, context: run_mlip_relax(
            structure(task, context),
            backend=context.get("mlip_backend"),
            model_path=context.get("model_path"),
            parameters=task.get("parameters"),
            work_directory=context.get("work_directory"),
        ),
    )
    registry.register(
        "deep_search",
        label="深度搜索",
        handler=lambda task, context: run_mlip_mc(
            structure(task, context),
            context["branch"],
            context["boundary"],
            context["phase_references"],
            segment_budget=int(task.get("budget", {}).get("steps", 1)),
            backend=context.get("mc_backend"),
            model_path=context.get("model_path"),
            parameters=task.get("parameters"),
            checkpoint=task.get("checkpoint"),
            work_directory=context.get("work_directory"),
        ),
    )
    registry.register(
        "dft_single_point",
        label="DFT 单点能",
        handler=lambda task, context: run_dft_singlepoint(
            structure(task, context),
            backend=context.get("dft_singlepoint_backend"),
            parameters=task.get("parameters"),
            work_directory=context.get("work_directory"),
        ),
    )
    registry.register(
        "dft_relax",
        label="DFT 弛豫",
        handler=lambda task, context: run_dft_relax(
            structure(task, context),
            backend=context.get("dft_relax_backend"),
            parameters=task.get("parameters"),
            work_directory=context.get("work_directory"),
            restart_from=task.get("checkpoint"),
        ),
    )
    return registry
