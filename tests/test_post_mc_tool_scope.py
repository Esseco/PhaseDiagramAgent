from phase_agent.tools.workflows.initial_tool_proposal import _post_mc_allowed_tools


TOOLS = [
    "prepare_local_batch_files",
    "allocate_mc_bohb",
    "generate_branches",
    "check_convergence",
]


def test_post_mc_keeps_relax_preparation_for_unsearched_refresh_branches():
    generation = {
        "post_refresh": True,
        "new_branch_count": 58,
        "first_search_completed": False,
    }
    allowed = _post_mc_allowed_tools(TOOLS, generation, post_mc=True)

    assert "prepare_local_batch_files" in allowed
    assert "allocate_mc_bohb" not in allowed
    assert "generate_branches" not in allowed


def test_post_mc_closes_relax_route_after_refresh_first_search():
    generation = {
        "post_refresh": True,
        "new_branch_count": 58,
        "first_search_completed": True,
    }
    allowed = _post_mc_allowed_tools(TOOLS, generation, post_mc=True)

    assert allowed == ["check_convergence"]


def test_non_post_mc_does_not_change_allowed_tools():
    generation = {
        "post_refresh": True,
        "new_branch_count": 58,
        "first_search_completed": True,
    }
    assert _post_mc_allowed_tools(TOOLS, generation, post_mc=False) == TOOLS
