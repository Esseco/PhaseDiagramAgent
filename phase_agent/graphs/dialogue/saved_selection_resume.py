"""Prepare one frozen DFT request without entering semantic/model routing."""


def prepare_resume(handler, context):
    from phase_agent.graphs.invocation_context import new_invocation

    invocation = new_invocation()

    def execute():
        missing = object()
        previous = handler.workflow_kwargs.get("frozen_selection_only", missing)
        handler.workflow_kwargs["frozen_selection_only"] = True
        try:
            # This restores the exact selection and its existing template gate.
            # It does not approve a newly generated template on the user's behalf.
            return handler._run(invocation, None, "继续")
        finally:
            if previous is missing:
                handler.workflow_kwargs.pop("frozen_selection_only", None)
            else:
                handler.workflow_kwargs["frozen_selection_only"] = previous

    context.scientific_call = execute
