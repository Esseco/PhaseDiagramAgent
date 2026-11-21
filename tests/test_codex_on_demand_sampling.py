from copy import deepcopy

from phase_agent.policies.region_sampling import CONTROLS
from phase_agent.tools.workflows.region_controller import sample_regions
from phase_agent.tools.workflows.compact_review import branch_lines
from tests.test_region_sampling import Manager, candidate, params as legacy_params


def params():
    result = legacy_params()
    result["regions"][0]["sample_count"] = 7
    return result


def test_broad_legal_grid_does_not_become_sampling_target():
    p = params()
    p["regions"].append({**p["regions"][0], "region_id": "r2", "priority": 2})
    p["region_preview"] = {
        "regions": [{"region_id": row["region_id"], "cells": [{}] * 1200} for row in p["regions"]]
    }
    calls = []

    def producer(*args, **kwargs):
        calls.append(kwargs["strategy_options"]["_focus"]["region_id"])
        return [candidate(f"{len(calls)}-{i}") for i in range(kwargs["quotas"]["coverage"])]

    chosen, report = sample_regions(
        Manager(), {}, p, producer=producer, descriptor=lambda r: tuple(r["T"])
    )
    assert calls == ["r1", "r2"]
    assert len(chosen) == 14
    assert len(chosen) <= 2 * CONTROLS["batch_attempts"]
    assert report["attempts"] <= 2 * CONTROLS["batch_attempts"]


def test_old_frozen_sampling_controls_require_review():
    from phase_agent.policies.region_sampling import errors

    p = params()
    p["sampling_controls"] = deepcopy(CONTROLS)
    p["sampling_controls"].pop("sampling_policy_version")
    assert "sampling_controls_changed_requires_revision" in errors(p)


def test_review_describes_legal_cells_as_metadata():
    text = "\n".join(
        branch_lines(
            {
                "parameters": {
                    **params(),
                    "region_preview": {
                        "initial_coverage": {
                            "required": True,
                            "selected_supercells": 1,
                            "expected_cells": 1000,
                        }
                    },
                }
            }
        )
    )
    assert "不要求穷举" in text
    assert "不是采样目标" in text
    assert "应覆盖" not in text


def test_initial_contract_calls_for_on_demand_sampling():
    from phase_agent.policies.initial_coverage import INSTRUCTION

    assert "按需求采样" in INSTRUCTION
    assert "不要求穷举" in INSTRUCTION


def test_preflight_uses_same_guard_as_sampling(monkeypatch):
    from phase_agent.analysis.cost.generation_preflight import attach_generation_preflight
    from phase_agent.policies.region_sampling import sampling_limit
    import phase_agent.policies.region_sampling as region

    p = params()
    preview = {"regions": [{"region_id": "r1", "legal_cells": 1200}]}
    monkeypatch.setattr(region, "preview", lambda *args: preview)
    action = {"tool": "generate_branches", "parameters": p}
    result = attach_generation_preflight(
        action,
        {},
        {"manager": None, "phase_references": {}},
        {"budgets": {"total_relative_cost": 1000}},
    )
    assert result["parameters"]["generation_cost_preview"][
        "selected_branch_upper"
    ] == sampling_limit(p)
    assert sampling_limit(p) < 1200


def test_no_candidates_requires_review():
    chosen, report = sample_regions(
        Manager(), {}, params(), producer=lambda *a, **kw: [], descriptor=lambda r: "same"
    )
    assert not chosen
    assert report["status"] == "paused_for_review"
    assert report["stop_reason"] == "paused_no_candidates"


def test_missing_or_excessive_demand_is_rejected():
    from phase_agent.policies.region_sampling import errors

    p = params()
    p["regions"][0].pop("sample_count")
    assert any("sample_count" in e for e in errors(p))
    p["regions"][0]["sample_count"] = 600
    p["regions"].append({**p["regions"][0], "region_id": "r2"})
    assert "requested_samples_exceed_1000" in errors(p)


def test_requested_counts_across_multiple_batches_and_shortfall():
    p = params()
    p["regions"][0]["sample_count"] = 37
    calls = []

    def producer(*args, **kw):
        count = kw["quotas"]["coverage"]
        calls.append(count)
        return [candidate(f"{len(calls)}-{i}") for i in range(count)]

    chosen, report = sample_regions(
        Manager(), {}, p, producer=producer, descriptor=lambda r: "test"
    )
    assert calls == [16, 16, 5]
    assert len(chosen) == 37
    assert report["status"] == "completed"
    assert report["attempts"] == 37
    calls.clear()
    chosen, report = sample_regions(
        Manager(), {}, p, producer=lambda *a, **kw: [candidate("one")], descriptor=lambda r: "test"
    )
    assert len(chosen) == 1
    assert report["regions"][0]["stop_reason"] == "insufficient_candidates"
    assert report["stop_reason"] == "regions_finished_with_shortfall"
    assert report["requested_candidate_count"] == 37
    assert report["unfilled_requested_candidates"] == 36
    assert report["regions"][0]["unfilled"] == 36


def test_metadata_legality_does_not_expand_supercells(monkeypatch):
    from pymatgen.core import Lattice, Structure
    from phase_agent.science.structures.enumerate_legal_frameworks import enumerate_legal_frameworks

    ref = Structure(
        Lattice.cubic(6),
        ["Na", "Co", "O", "O"],
        [[0, 0, 0], [0, 0, 0.5], [0.5, 0, 0.2], [0, 0.5, 0.8]],
    )
    h = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    monkeypatch.setattr(
        Structure,
        "make_supercell",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("expanded metadata")),
    )
    rows = enumerate_legal_frameworks(
        {"P": ["O3"], "H": {"O3": [h]}, "TM_ratio": {"Co": 1}}, {"O3": ref}
    )
    assert rows[0]["atom_count_full"] == 16
    assert rows[0]["allowed_x"] == ["0", "1/4", "1/2", "3/4", "1"]


def test_coverage_samples_across_range_instead_of_low_x_prefix(monkeypatch):
    import phase_agent.science.structures.generate_coverage_branches as m
    from types import SimpleNamespace

    monkeypatch.setattr(
        m,
        "generate_branch_structure",
        lambda *a, **k: {
            "P": k["phase"],
            "H": k["H"],
            "det_H": 1,
            "x": k["x"],
            "T": ["Co"],
            "composition": {},
        },
    )
    monkeypatch.setattr(m, "identify_branch_parameters", lambda s, *a, **k: s)
    xs = [f"{n}/100" for n in range(1, 100)]
    rows = m.generate_coverage_branches(
        SimpleNamespace(data={"branches": {}}, boundary={}),
        [{"P": "O3", "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "allowed_x": xs}],
        {},
        quota=5,
        seed=42,
        preserve_reference_tm=True,
    )
    from fractions import Fraction

    assert len(rows) == 5
    assert max(Fraction(row["x"]) for row in rows) > 0.5


def test_independent_arbitrary_cap_is_rejected():
    import pytest
    from phase_agent.policies.region_sampling import sampling_limit

    p = params()
    p["max_new_branches"] = 100
    with pytest.raises(ValueError, match="branch_limit_must_match_requested_samples"):
        sampling_limit(p)


def test_sampling_scale_guidance_is_shared_by_codex_and_project_policy():
    from phase_agent.policies.initial_coverage import (
        INSTRUCTION as initial,
        SAMPLING_SCALE_INSTRUCTION,
    )
    from phase_agent.policies.region_sampling import policy

    assert SAMPLING_SCALE_INSTRUCTION in initial
    assert SAMPLING_SCALE_INSTRUCTION in policy()["instruction"]
    assert "数百" in SAMPLING_SCALE_INSTRUCTION
    assert "不是固定配额或最低数量" in SAMPLING_SCALE_INSTRUCTION
    assert "须说明" in SAMPLING_SCALE_INSTRUCTION
