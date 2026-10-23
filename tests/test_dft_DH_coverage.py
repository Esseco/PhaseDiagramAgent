from copy import deepcopy
import pytest
from phase_agent.tools.workflows.dft_coverage_review import coverage_evidence, coverage_errors


def candidate(cid, group="g", scope="current_near_hull", h=0):
    return {
        "candidate_id": cid,
        "phase": "O3",
        "dft_sampling_evidence": {
            "status": "available",
            "sampling_evidence_scope": scope,
            "D_candidate_id": group,
            "D_score": 4.5 if group else None,
            "geometry_key": cid,
            "H_single_max": h,
            "H_single_sequence": [h, None],
            "H_D_candidate_union_sequence": [None, 30],
        },
    }


def action(cid, kind="DFT_SINGLE_POINT", reason=None):
    return {"parameters": {"decisions": [{"candidate_id": cid, "action": kind, "reason": reason}]}}


def test_group_deduplicates_members_and_keeps_conditional_values():
    pool = [candidate("a"), candidate("b"), candidate("c", "conditional", "conditional_near_hull")]
    before = deepcopy(pool)
    groups = coverage_evidence(pool)["groups"]
    assert len(groups) == 2 and groups[0]["candidate_ids"] == ["a", "b"]
    assert groups[1]["scope"] == "conditional_near_hull"
    assert groups[1]["H_union_sequence"] == [None, 30] and pool == before
    assert len(coverage_errors(action("a"), pool)) == 1


@pytest.mark.parametrize("kind", ["DFT_SINGLE_POINT", "DFT_RELAX", "DEFER", "REJECT"])
def test_representative_or_reasoned_deferral_suffices_without_minimum_count(kind):
    assert (
        coverage_errors(
            action("a", kind, "reuse verified comparable reference"),
            [candidate("a"), candidate("b")],
        )
        == []
    )


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_unexplained_deferral_and_old_candidate_are_rejected(reason):
    assert coverage_errors(action("a", "DEFER", reason), [candidate("a")])
    assert coverage_errors(action("old"), [candidate("a")])


def test_unknown_and_outside_scope_are_not_treated_as_low_or_required():
    pool = [
        candidate("unknown", None, h=None),
        candidate("zero", None),
        candidate("outside", scope="outside_review_scope"),
        candidate("positive", None, h=2),
    ]
    groups = coverage_evidence(pool)["groups"]
    assert len(groups) == 1 and groups[0]["candidate_ids"] == ["positive"]
    assert groups[0]["D_score"] is None


def test_dft_topic_contains_coverage_review():
    from phase_agent.decisions.agent.summary_request import source_topic

    assert source_topic("dft_DH_coverage_review") == "dft"
