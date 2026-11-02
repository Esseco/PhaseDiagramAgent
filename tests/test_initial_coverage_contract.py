from types import SimpleNamespace

import pytest
from pymatgen.core import Lattice, Structure

from phase_agent.policies.region_sampling import MODE, preview
from phase_agent.runtime.codex.context import build_codex_rule_audit

H2 = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
H3 = [[3, 0, 0], [0, 1, 0], [0, 0, 1]]


def setup_case():
    base = Structure(
        Lattice.cubic(6),
        ["Na", "Na", "Ni", "Co", "O", "O", "O", "O"],
        [
            [0, 0, 0],
            [0.5, 0, 0],
            [0, 0, 0.5],
            [0.5, 0, 0.5],
            [0, 0, 0.2],
            [0.5, 0, 0.2],
            [0, 0.5, 0.8],
            [0.5, 0.5, 0.8],
        ],
    )
    p3 = base.copy()
    p3.append("Na", [0, 0.5, 0])
    p3.append("Na", [0.5, 0.5, 0])
    refs = {"O1": base, "O3": base, "P3": p3}
    boundary = {
        "P": {"at_x": {"0": ["O1"], "1": ["O3"]}, "intermediate": ["O3", "P3"]},
        "H": {p: [H2, H3] for p in refs},
        "TM_ratio": {"Ni": 1, "Co": 1},
    }
    manager = SimpleNamespace(data={"branches": {}, "structures": {}}, boundary=boundary)
    config = {
        "system": {
            "configuration_space": {"roles": {"T": "fixed"}, "fixed_T_source": "phase_reference"}
        }
    }
    params = {
        "sampling_mode": MODE,
        "regions": [
            {
                "region_id": phase,
                "priority": i + 1,
                "strategy": "coverage",
                "phase": phase,
                "min_det_H": 2,
                "max_det_H": 2,
                "na_min": 0.0,
                "na_max": 1.0,
                "reason": "initial coverage",
                "evidence_refs": [],
            }
            for i, phase in enumerate(refs)
        ],
    }
    return manager, refs, config, params


def test_initial_preview_rejects_split_phase_na_ranges():
    manager, refs, config, params = setup_case()
    params["regions"][1]["na_min"] = 0.5
    params["regions"][2]["na_max"] = 0.5
    with pytest.raises(ValueError, match="initial_coverage_missing_cells"):
        preview(params, manager, refs, config, {})


def test_initial_preview_rejects_parent_strategy():
    manager, refs, config, params = setup_case()
    params["regions"][1]["strategy"] = "composition"
    with pytest.raises(ValueError, match="initial_coverage_requires_coverage"):
        preview(params, manager, refs, config, {})


def test_initial_preview_requires_competitor_for_selected_cell():
    manager, refs, config, params = setup_case()
    params["regions"].pop()
    with pytest.raises(ValueError, match="initial_coverage_missing_cells"):
        preview(params, manager, refs, config, {})


def test_full_initial_preview_is_metadata_only_and_preserves_endpoints(monkeypatch):
    manager, refs, config, params = setup_case()
    monkeypatch.setattr(
        Structure, "make_supercell", lambda *a, **k: pytest.fail("preview expanded a cell")
    )
    result = preview(params, manager, refs, config, {})
    assert result["initial_coverage"]["required"] is True
    assert result["initial_coverage"]["expected_cells"] == 8
    xs = {r["region_id"]: {c["x"] for c in r["cells"]} for r in result["regions"]}
    assert xs == {"O1": {"0"}, "O3": {"1/4", "1/2", "3/4", "1"}, "P3": {"1/4", "1/2", "3/4"}}
    assert all(c["det_H"] == 2 for r in result["regions"] for c in r["cells"])


def test_later_round_can_keep_partial_na_regions():
    manager, refs, config, params = setup_case()
    manager.data["branches"] = {"old": {"P": "O3", "H": H2, "x": "1", "T": ["Ni", "Co"]}}
    params["regions"] = [params["regions"][1]]
    params["regions"][0]["na_min"] = 0.5
    result = preview(params, manager, refs, config, {})
    assert not result.get("initial_coverage", {}).get("required")


def test_codex_audit_rejects_initial_noncoverage_before_model_or_execution():
    action = {
        "tool": "generate_branches",
        "parameters": {"sampling_mode": MODE, "regions": [{"strategy": "composition"}]},
    }
    audit = build_codex_rule_audit(action, {}, {"initial_coverage": {"required": True}})
    assert audit["status"] == "invalid"
    assert "initial_coverage_requires_coverage" in audit["errors"]


def sampled_case():
    from tests.test_region_sampling import Manager

    base, refs, config, params = setup_case()
    manager = Manager()
    manager.boundary = base.boundary
    params["region_preview"] = preview(params, manager, refs, config, {})
    params["seed"] = 42

    def producer(_manager, _refs, **kwargs):
        rid = kwargs["strategy_options"]["_focus"]["region_id"]
        cells = next(
            r["cells"] for r in params["region_preview"]["regions"] if r["region_id"] == rid
        )
        return [
            {"P": rid, "H": c["H"], "x": c["x"], "T": ["Ni", "Co"], "det_H": c["det_H"]}
            for c in cells
        ]

    return manager, refs, params, producer


def test_initial_sampling_covers_all_cells_even_with_same_tm_descriptor():
    from phase_agent.tools.workflows.region_controller import sample_regions

    manager, refs, params, producer = sampled_case()
    chosen, report = sample_regions(
        manager, refs, params, producer=producer, descriptor=lambda r: "same_TM"
    )
    assert len(chosen) == 8
    assert report["initial_coverage"]["complete"]
    assert report["stop_reason"] == "initial_coverage_complete"
    assert manager.data["branches"] == {}


def test_initial_sampling_time_guard_reports_incomplete_grid():
    from phase_agent.tools.workflows.region_controller import sample_regions

    manager, refs, params, producer = sampled_case()
    ticks = iter([0, 0, 301])
    chosen, report = sample_regions(
        manager,
        refs,
        params,
        producer=producer,
        descriptor=lambda r: "same_TM",
        clock=lambda: next(ticks),
    )
    assert len(chosen) == 1
    assert not report["initial_coverage"]["complete"]
    assert report["initial_coverage"]["missing_cells"] == 7
    assert report["status"] == "paused_for_review"


def test_incomplete_initial_grid_cannot_register_or_prepare_inputs(monkeypatch):
    import phase_agent.tools.workflows.run_region_generation as workflow

    manager, refs, params, _ = sampled_case()
    frozen = params["region_preview"]
    from phase_agent.policies.region_sampling import CONTROLS

    params["sampling_controls"] = CONTROLS
    monkeypatch.setattr(workflow, "preview", lambda *a: frozen)
    monkeypatch.setattr(
        workflow,
        "sample_regions",
        lambda *a, **k: (
            [],
            {
                "status": "paused_for_review",
                "initial_coverage": {"required": True, "complete": False},
            },
        ),
    )
    monkeypatch.setattr(
        workflow,
        "run_branch_generation",
        lambda *a, **k: pytest.fail("registered an incomplete grid"),
    )
    result = workflow.run_region_generation(
        action={"parameters": params, "task_key": "initial"},
        context={
            "effective_config": {},
            "event_state": {},
            "manager": manager,
            "phase_references": refs,
        },
    )
    assert result["status"] == "paused_for_review"
    assert result["registered"] == []


def test_initialization_failures_block_automatic_relax_mc_preparation():
    from phase_agent.tools.workflows.scientific_module_scope import prepare_generated_module

    result = {
        "registered": [{"branch_id": "failed-neighbor"}],
        "summary": {
            "sampling_report": {"initial_coverage": {"required": True, "complete": True}},
            "initialization_failed_branches": 1,
        },
    }
    out = prepare_generated_module(
        {"parameters": {"module_execution_scope": {"operation": "prepare_relax_mc_batch"}}},
        result,
        {},
    )
    assert out["status"] == "paused_for_review"
    assert "initial_coverage_incomplete" in out["reason"]


def test_initial_batch_cannot_claim_complete_when_guard_is_too_small():
    manager, refs, config, params = setup_case()
    params["max_new_branches"] = 7
    with pytest.raises(ValueError, match="initial_coverage_exceeds_branch_guard"):
        preview(params, manager, refs, config, {})


def test_report_explains_initial_phase_na_grid_before_approval():
    from phase_agent.tools.workflows.compact_review import branch_lines

    manager, refs, config, params = setup_case()
    params["region_preview"] = preview(params, manager, refs, config, {})
    text = "\n".join(branch_lines({"parameters": params}))
    assert "8 个合法相/H/x" in text
    assert "首轮仅用 coverage" in text
    assert "不按候选Na位点数" in text
    assert "不自动准备 Relax–MC 输入" in text


def test_initial_no_progress_is_not_reported_as_completed():
    from phase_agent.tools.workflows.region_controller import sample_regions

    manager, refs, params, _ = sampled_case()
    chosen, report = sample_regions(
        manager, refs, params, producer=lambda *a, **k: [], descriptor=lambda r: "same_TM"
    )
    assert chosen == []
    assert report["status"] == "paused_for_review"
    assert report["initial_coverage"]["missing_cells"] == 8


def test_explicit_H_sample_keeps_full_na_competition_without_expanding_all_H():
    manager, refs, config, params = setup_case()
    for row in params["regions"]:
        row.update(min_det_H=2, max_det_H=3, supercells=[H3])
    out = preview(params, manager, refs, config, {})
    assert out["initial_coverage"]["expected_cells"] == 12
    assert out["initial_coverage"]["selected_supercells"] == 1
    assert all(cell["H"] == H3 for region in out["regions"] for cell in region["cells"])


def test_H_sample_outside_confirmed_boundary_is_rejected():
    manager, refs, config, params = setup_case()
    params["regions"][0]["supercells"] = [[[9, 0, 0], [0, 1, 0], [0, 0, 1]]]
    with pytest.raises(ValueError, match="selected_supercell_outside_confirmed_boundary"):
        preview(params, manager, refs, config, {})
