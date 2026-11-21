"""Scientific DFT guards used by the production selection handler."""
import pytest
from phase_agent.configuration.defaults.default_dft_decision_config import default_dft_decision_config
from phase_agent.tools.workflows.validate_dft_agent_decisions import validate_dft_agent_decisions


def validate(decisions, *, budget=2000, duplicate=False, enforcement="advisory"):
    metric = {"candidate_id": "near", "branch_id": "B1", "f_std_max": 1.0,
              "f_std_p95": 0.8, "energy_std": 0.2, "predicted_Ehull": 0.01,
              "estimated_cost": 1}
    if duplicate:
        metric["duplicate_of"] = "original"
    config = default_dft_decision_config()
    config["budget_limits"]["enforcement"] = enforcement
    return validate_dft_agent_decisions(
        {"decisions": decisions, "source": "llm_agent"}, [metric], {},
        config=config, config_version="c1", remaining_budget=budget)


@pytest.mark.parametrize("decision", [
    {"candidate_id": "unknown", "action": "DFT_RELAX"},
    {"candidate_id": "near", "action": "DFT_RELAX", "dft_parameters": {"encut": 300}},
])
def test_unknown_candidate_and_parameter_override_are_rejected(decision):
    result = validate([decision])
    assert not result["valid"]
    assert result["accepted"] == []


def test_duplicate_cannot_become_a_new_dft_task():
    result = validate([{"candidate_id": "near", "action": "DFT_SINGLE_POINT"}], duplicate=True)
    assert result["accepted"] == []
    assert result["rejected"][0]["reason"] == "duplicate_safety_rule"


def test_exhausted_hard_budget_cannot_reserve_a_task():
    result = validate([{"candidate_id": "near", "action": "DFT_SINGLE_POINT"}],
                      budget=0, enforcement="hard")
    assert result["accepted"] == []
    assert result["rejected"][0]["reason"] == "dft_budget"


def _validate_geometry_key(action, sampling, state=None):
    metric = {
        "candidate_id": "near",
        "branch_id": "B1",
        "atom_count": 10,
        "predicted_Ehull": 0.01,
        "dft_sampling_evidence": sampling,
        "geometry_optimization_evidence": {
            "source_task_id": "search-old",
            "residual_force_ev_per_angstrom": 0.2,
            "force_tolerance_ev_per_angstrom": 0.05,
        },
    }
    return validate_dft_agent_decisions(
        {"decisions": [{"candidate_id": "near", "action": action, "reason": "verify"}],
         "source": "llm_agent"},
        [metric], state or {}, config=default_dft_decision_config(),
        config_version="c1", remaining_budget=2000,
    )


@pytest.mark.parametrize("action", ["DFT_SINGLE_POINT", "DFT_RELAX"])
def test_geometry_key_distinguishes_same_sid_and_preserves_legacy_key(action):
    from copy import deepcopy

    legacy_key = f"c1:near:{action}"
    state = {"effective_decisions": {legacy_key: {"status": "completed"}}}
    before = deepcopy(state)
    first = _validate_geometry_key(action, {"geometry_key": "a" * 64}, state)
    second = _validate_geometry_key(action, {"geometry_key": "b" * 64}, state)
    assert first["valid"] and second["valid"]
    assert not first["rejected"] and not second["rejected"]
    assert first["accepted"][0]["task_key"] == legacy_key + ":geometry:" + "a" * 64
    assert second["accepted"][0]["task_key"] == legacy_key + ":geometry:" + "b" * 64
    assert first["accepted"][0]["task_key"] != second["accepted"][0]["task_key"]
    assert first["accepted"][0]["input_structure_sha256"] == "a" * 64
    assert second["accepted"][0]["input_structure_sha256"] == "b" * 64
    assert state == before


@pytest.mark.parametrize("action", ["DFT_SINGLE_POINT", "DFT_RELAX"])
@pytest.mark.parametrize("sampling", [None, {}, {"geometry_key": None}])
def test_geometry_key_absent_retains_legacy_duplicate_detection(action, sampling):
    key = f"c1:near:{action}"
    fresh = _validate_geometry_key(action, sampling)
    assert fresh["valid"] and not fresh["rejected"]
    assert fresh["accepted"][0]["task_key"] == key
    assert fresh["accepted"][0].get("input_structure_sha256") is None
    duplicate = _validate_geometry_key(
        action, sampling, {"effective_decisions": {key: {"status": "completed"}}}
    )
    assert duplicate["accepted"] == []
    assert duplicate["rejected"][0]["reason"] == "duplicate_task_key"


@pytest.mark.parametrize("action", ["DFT_SINGLE_POINT", "DFT_RELAX"])
def test_geometry_key_existing_exact_geometry_is_rejected(action):
    key = f"c1:near:{action}:geometry:" + "a" * 64
    result = _validate_geometry_key(
        action, {"geometry_key": "a" * 64},
        {"effective_decisions": {key: {"status": "completed"}}},
    )
    assert result["accepted"] == []
    assert result["rejected"][0]["reason"] == "duplicate_task_key"


@pytest.mark.parametrize("action", ["DFT_SINGLE_POINT", "DFT_RELAX"])
def test_geometry_key_hex_case_has_one_identity(action):
    result = _validate_geometry_key(action, {"geometry_key": "AB" * 32})
    assert result["valid"] and not result["rejected"]
    assert result["accepted"][0]["task_key"] == f"c1:near:{action}:geometry:" + "ab" * 32
    assert result["accepted"][0]["input_structure_sha256"] == "ab" * 32


@pytest.mark.parametrize("action", ["DFT_SINGLE_POINT", "DFT_RELAX"])
@pytest.mark.parametrize("sha", ["", "bad", "g" * 64, "a" * 63, "a" * 65,
                                "a" * 64 + "\n", 123, True, ["a" * 64]])
def test_geometry_key_invalid_sha_is_rejected(action, sha):
    result = _validate_geometry_key(action, {"geometry_key": sha})
    assert not result["valid"]
    assert result["accepted"] == []
    assert result["errors"]


@pytest.mark.parametrize("action", ["DEFER", "REJECT"])
def test_geometry_key_does_not_change_non_dft_identity(action):
    result = _validate_geometry_key(action, {"geometry_key": "bad"})
    assert result["valid"] and not result["rejected"]
    assert result["accepted"][0]["task_key"] == f"c1:near:{action}"
    assert result["accepted"][0].get("input_structure_sha256") is None


def test_complete_verified_direct_batch_does_not_require_old_audit(tmp_path, monkeypatch):
    import hashlib
    from phase_agent.tools.workflows import direct_extrapolation_sampling

    monkeypatch.setattr(direct_extrapolation_sampling, "valid_direct_receipt", lambda receipt: True)
    frames = []
    for name in ("a", "b"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        frames.append({"candidate_id": name, "structure_path": str(path),
                       "structure_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    state = {
        "dft_audit_candidate_ids": ["old-audit"],
        "qbc_candidates": frames,
        "extrapolation_recovery_cycle": {"status": "awaiting_reviewed_dft_labels",
            "scope_id": "epoch_refresh", "invalid_model_version": "model",
            "direct_source_digest": "digest", "selected": frames},
        "direct_sampling_results": {"epoch_refresh": {"model_version": "model",
            "source_digest": "digest", "selected": frames}},
    }
    metrics = [{"candidate_id": row["candidate_id"], "atom_count": 10} for row in frames]
    metrics.append({"candidate_id": "old-audit", "atom_count": 10})
    def run(ids):
        return validate_dft_agent_decisions(
            {"source": "codex", "decisions": [{"candidate_id": cid, "action": "DFT_SINGLE_POINT"} for cid in ids]},
            metrics, state, config=default_dft_decision_config(), config_version="c1", remaining_budget=2000)
    assert "independent_dft_audit_required" not in run(["a", "b"])["errors"]
    assert "independent_dft_audit_required" in run(["a"])["errors"]
    (tmp_path / "a").write_bytes(b"changed")
    assert "independent_dft_audit_required" in run(["a", "b"])["errors"]
    state.pop("extrapolation_recovery_cycle")
    assert "independent_dft_audit_required" in run(["a", "b"])["errors"]


@pytest.fixture(autouse=True)
def normal_extrapolation_for_unrelated_selection_tests(monkeypatch):
    # These tests isolate decision identity/audit behavior; exact-frame
    # extrapolation checks have their own dedicated regression tests.
    monkeypatch.setattr(
        "phase_agent.evidence.dft_extrapolation.candidate_extrapolation",
        lambda candidate, state: {"status": "normal", "reason": "synthetic"},
    )
