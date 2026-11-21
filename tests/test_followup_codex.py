from phase_agent.runtime.codex.no_project_model import NoProjectModel
from phase_agent.tools.workflows.propose_followup import propose_after_generation


def test_codex_completed_generation_does_not_launch_project_model():
    frame = {'execution_mode': 'interactive', 'agent_client': NoProjectModel()}
    result = {'status': 'completed', 'state': {}, 'events': [{'action': {'tool': 'generate_branches'},
              'execution': {'status': 'completed', 'result': {'summary': {'registered_structures': 1}}}}]}
    def forbidden(*args, **kwargs):
        raise AssertionError('Codex must decide the next action itself')
    assert propose_after_generation(frame, result, event_loop=forbidden) is result
