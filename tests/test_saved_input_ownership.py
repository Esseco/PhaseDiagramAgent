from phase_agent.tools.workflows.run_event_loop import _load_state


def test_public_state_remains_independent():
    state={'tasks':[{'status':'pending'}]}
    loaded=_load_state(state)
    loaded['tasks'][0]['status']='completed'
    assert state['tasks'][0]['status']=='pending'


def test_owned_transfer_does_not_copy_history():
    class NoCopy(dict):
        def __deepcopy__(self,memo):
            raise AssertionError('owned history must not be recopied')
    state={'tasks':[],'history':NoCopy({'large':'fact'})}
    assert _load_state(state,_owned=True) is state
