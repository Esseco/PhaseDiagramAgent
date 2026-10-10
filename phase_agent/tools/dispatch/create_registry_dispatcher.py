"""把阶段注册表包装成 run_pipeline 使用的 dispatcher。"""

from phase_agent.tools.dispatch.dispatch_calculation_stage import execute_registered_stage


def create_registry_dispatcher(registry, context_factory):
    def dispatch(task):
        return execute_registered_stage(task, context_factory(task), registry)

    return dispatch
