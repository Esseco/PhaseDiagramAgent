from phase_agent.analysis.state.model_epoch import model_epoch, model_epoch_label


def test_epoch_uses_persisted_model_round_not_current_activity():
    state = {"upload_layout": {"model_rounds": {"initial": 1, "fine": 2}},
             "iteration": 99, "active_model_version": "fine"}
    assert model_epoch(state, "initial") == "epoch0"
    assert model_epoch(state, "fine") == "epoch1"
    assert model_epoch_label(state, "initial") == "epoch0（initial）"
    assert model_epoch(state, "unknown") is None


def test_report_and_summary_use_same_epoch():
    from tests.test_dft_comparison_csv import add_result
    from phase_agent.runtime.post_dft_presentation import post_dft_lines
    from phase_agent.analysis.state.summarize_model_rounds import summarize_model_rounds
    state = {}
    add_result(state)
    state["upload_layout"] = {"model_rounds": {"m1": 1}}
    assert "epoch0（m1）" in post_dft_lines(state)
    assert summarize_model_rounds(state)[0]["model_epoch"] == "epoch0"
