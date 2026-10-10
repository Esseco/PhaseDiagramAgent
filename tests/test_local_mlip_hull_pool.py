from phase_agent.analysis.phase.update_local_mlip_hull_pool import update_local_mlip_hull_pool
from phase_agent.configuration.defaults.default_layered_search_config import default_layered_search_config
from phase_agent.science.mc.schedule_tiered_mc import schedule_tiered_mc


def test_local_pool_uses_only_recovered_matching_model(tmp_path):
    structure = tmp_path / "final.vasp"
    structure.write_text("downloaded structure", encoding="utf-8")
    config = {"system_config": {"system_id": "layered"}, "mlip": {"version": "m1"}}
    task = {"task_id": "r1", "branch_id": "b1", "structure_id": "s1",
            "stage": "relax_and_feature", "status": "completed", "converged": True,
            "model_version": "m1", "outputs": {"composition": {"Na": 1, "Fe": 1},
            "structure_path": str(structure), "energy": -2.0, "energy_unit": "eV",
            "actual_phase": "O3", "phase_identification": {"status": "identified", "phase": "O3"}}}
    state = {"tasks": [task, {**task, "task_id": "r2", "model_version": "m2"}]}
    path = tmp_path / "pool.json"
    first = update_local_mlip_hull_pool(state, config=config, path=path)
    assert path.is_file()
    pool = first["branch_hull_batches"][first["current_branch_hull_version"]]
    assert len(pool["records"]) == 1
    assert pool["records"][0]["source_task_id"] == "r1"
    second = update_local_mlip_hull_pool(first, config=config, path=path)
    assert second["current_branch_hull_version"] == first["current_branch_hull_version"]


def test_first_mc_bands_and_input_structure():
    policy = default_layered_search_config()["mc_policy"]
    candidates = [{"branch_id": f"b{i}", "structure_id": f"s{i}",
                   "structure_path": f"best{i}.vasp", "relaxed_ehull": gap}
                  for i, gap in enumerate((0.005, 0.015, 0.030, 0.050))]
    result = schedule_tiered_mc(candidates, {}, policy=policy, total_budget=300,
                               seed=4, model_version="m1", hull_reference_version="h1")
    by_branch = {row["branch_id"]: row for row in result["actions"]}
    assert [(by_branch[f"b{i}"]["max_mc_steps"],
             by_branch[f"b{i}"]["patience_steps"]) for i in range(3)] == [
                 (100, 8), (60, 6), (30, 4)]
    assert all(by_branch[f"b{i}"]["structure_path"] == f"best{i}.vasp" for i in range(3))
    assert "b3" not in by_branch or by_branch["b3"]["max_mc_steps"] == 15
