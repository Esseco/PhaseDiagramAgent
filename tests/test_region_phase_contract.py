from copy import deepcopy

import pytest

from phase_agent.decisions.agent.post_dft_review import request_validated_action
from phase_agent.decisions.agent.deepagents_proposal import proposal_material
from phase_agent.tools.policy.region_sampling import errors, policy
from tests.test_on_demand_branch import context, review


def region():
    return {
        "region_id": "r",
        "priority": 1,
        "strategy": "coverage",
        "phase": "O3",
        "min_det_H": 1,
        "max_det_H": 2,
        "na_min": 0.1,
        "na_max": 1.0,
        "reason": "exploration",
        "evidence_refs": ["current"],
    }


@pytest.mark.parametrize("phase", [None, "all", "O1/O3", "o3", ["O1", "O3"], {"name": "O3"}])
def test_invalid_phase_returns_actionable_error_without_guessing(phase):
    row = {**region(), "phase": phase}
    before = deepcopy(row)
    problems = errors({"sampling_mode": "region_adaptive", "regions": [row]})
    assert any(
        "explicit_phase_required" in x and "received=" in x and "O1/O3/P2/P3/OP2" in x
        for x in problems
    )
    assert row == before


def test_model_receives_phase_enum_in_initial_correction_and_transport():
    valid = review()
    valid.update(
        task_key="region-contract",
        target_ids=[],
        budget=0,
        reason="explore",
        expected_purpose="review",
    )
    valid["parameters"].update(sampling_mode="region_adaptive", regions=[region()])
    valid["branch_pool_review"]["choice"] = "supplement"
    bad = deepcopy(valid)
    bad["parameters"]["regions"][0]["phase"] = "O1/O3"
    calls = []

    def client(payload):
        calls.append(payload)
        for material in (payload, proposal_material(payload)):
            item = material["output_contracts"]["generation_region_item"]
            assert item["properties"]["phase"]["enum"] == ["O1", "O3", "P2", "P3", "OP2"]
            assert "phase" in item["required"]
        if len(calls) == 1:
            return deepcopy(bad)
        assert any("O1/O3" in x and "received=" in x for x in payload["validation_errors"])
        assert "single string" in payload["instruction"]
        return deepcopy(valid)

    result = request_validated_action(
        client,
        {
            "allowed_tools": ["generate_branches"],
            "decision_context": {**context(), "branch_count_policy": policy()},
        },
    )
    assert len(calls) == 2
    assert result["parameters"]["regions"][0]["phase"] == "O3"
    assert bad["parameters"]["regions"][0]["phase"] == "O1/O3"


def test_imported_remote_diagnostic_is_not_presented_as_fresh_failure():
    from phase_agent.runtime.waiting_files_presentation import waiting_file_lines

    state = {
        "tasks": [
            {"task_id": "t", "stage": "deep_search", "status": "pending", "relax_mc_bundle_id": "b"}
        ],
        "relax_mc_batches": {
            "b": {
                "import_checksum": "verified",
                "result_summary": {"status": "failed", "reason": "squeue old failure"},
            }
        },
    }
    text = "\n".join(waiting_file_lines(state))
    assert "本地回收校验已完成" in text
    assert "历史控制器诊断" in text and "不是当前作业状态" in text
    assert "squeue old failure" in text and "不要重新上传输入或提交整批" in text
    assert "回传核对失败" not in text


def test_P2_is_valid_in_syntax_and_model_schema():
    from phase_agent.decisions.agent.generation_contracts import GenerationRegion
    row={**region(), "phase":"P2"}
    assert errors({"sampling_mode":"region_adaptive","regions":[row]}) == []
    assert GenerationRegion.model_validate(row).phase == "P2"
    assert "P2" in GenerationRegion.model_json_schema()["properties"]["phase"]["enum"]


@pytest.mark.parametrize("allowed", [{"P2"}, {"O3"}])
def test_P2_preview_still_obeys_project_phase_boundary(monkeypatch,allowed):
    from types import SimpleNamespace
    from pymatgen.core import Lattice, Structure
    import phase_agent.science.structures.boundary_utils as utils
    from phase_agent.tools.policy.region_sampling import preview
    row={**region(),"phase":"P2","min_det_H":1,"max_det_H":1,"na_min":.5,"na_max":1.}
    reference=Structure(Lattice.cubic(5),["Na","Fe","O","O"],[[0,0,0],[.5,.5,.5],[.2,.2,.2],[.8,.8,.8]])
    monkeypatch.setattr(utils,"load_structure",lambda value:reference)
    monkeypatch.setattr(utils,"allowed_phases",lambda value:allowed)
    monkeypatch.setattr(utils,"allowed_phases_at_x",lambda p,x:allowed)
    monkeypatch.setattr(utils,"allowed_supercells",lambda h,p:[[[1,0,0],[0,1,0],[0,0,1]]])
    monkeypatch.setattr(Structure,"make_supercell",lambda *a,**k:pytest.fail("preview must not generate structures"))
    manager=SimpleNamespace(data={"branches":{},"structures":{}},boundary={"P":{},"H":{}})
    params={"sampling_mode":"region_adaptive","regions":[row]}
    if "P2" in allowed:
        assert preview(params,manager,{"P2":reference},{},{})["regions"][0]["legal_cells"] == 1
    else:
        with pytest.raises(ValueError,match="非法或禁用区域"):
            preview(params,manager,{"P2":reference},{},{})
