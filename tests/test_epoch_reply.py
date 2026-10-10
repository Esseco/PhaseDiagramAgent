from phase_agent.runtime.chat_state_presentation import epoch_stage_line


def test_initial_epoch_zero():
    assert epoch_stage_line({}).startswith('epoch0')
    assert 'epoch0' in epoch_stage_line({},configuring=True)


def test_model_epoch_does_not_use_mc_or_dft_round_number():
    state={'active_model_version':'m2','upload_layout':{'model_rounds':{'m1':1,'m2':3}},'mc_round':9,'dft_round':12}
    assert epoch_stage_line(state).startswith('epoch2')


def test_pending_branch_stage_is_visible():
    state={'pending_execution_policies':{'p':{'agent_proposal':{'recommended_action':'generate_branches'}}}}
    assert 'branch' in epoch_stage_line(state)


def test_active_relax_stage_is_visible():
    state={'tasks':[{'stage':'relax_and_feature','status':'running'}]}
    assert 'Relax' in epoch_stage_line(state)
