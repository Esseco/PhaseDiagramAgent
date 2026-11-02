from phase_agent.runtime.codex.context import build_codex_rule_audit
from phase_agent.runtime.codex.context import HPC_APPROVAL_POLICY


def test_hpc_operations_require_explicit_manifest_approval():
    assert HPC_APPROVAL_POLICY["upload_requires_explicit_user_approval"] is True
    assert HPC_APPROVAL_POLICY["submit_requires_explicit_user_approval"] is True
    assert "remote_upload_path" in HPC_APPROVAL_POLICY["required_manifest"]
    assert "submit_command" in HPC_APPROVAL_POLICY["required_manifest"]


def _candidate(cid, ehull, qbc, x):
    return {
        "candidate_id": cid,
        "predicted_Ehull": ehull,
        "hull_impact": ehull,
        "x_Na_per_O2": x,
        "phase": "O3",
        "qbc": {"status": "completed", "f_std_max": qbc},
        "dft_sampling_evidence": {"sampling_evidence_scope": "current_near_hull", "H_single_max": 0.0},
    }


def test_dft_rule_audit_records_all_selection_evidence_and_deviations():
    state = {
        "qbc_candidates": [_candidate("S1", 0.0, 0.1, 0.2), _candidate("S2", 0.02, 0.8, 0.8)],
        "dft_audit_candidate_ids": ["S2"],
    }
    action = {
        "tool": "select_dft_candidates",
        "parameters": {
            "decisions": [
                {"candidate_id": "S1", "action": "DFT_SINGLE_POINT", "reason": "hull"},
                {"candidate_id": "S2", "action": "DFT_SINGLE_POINT", "reason": "audit"},
            ]
        },
    }
    audit = build_codex_rule_audit(action, state, {})
    assert audit["status"] == "audited"
    assert audit["evidence_sources"] == ["ehull", "qbc", "diversity", "coverage", "d_h", "independent_audit"]
    assert audit["selected_count"] == 2
    assert audit["independent_audit_selected"] == ["S2"]
    assert "policy_ranked_candidate_ids" in audit


def test_branch_rule_audit_rejects_legacy_quota_shape():
    action = {"tool": "generate_branches", "parameters": {"sampling_mode": "region_adaptive", "quota": 10, "regions": []}}
    audit = build_codex_rule_audit(action, {}, {})
    assert audit["status"] == "invalid"
    assert "quota" in audit["errors"]


def test_codex_transfer_contract_requires_minimal_compressed_reviewed_payloads():
    policy = HPC_APPROVAL_POLICY["data_transfer"]
    assert policy["required_data_only"] is True
    assert policy["compressed_archive_required"] is True
    assert policy["reuse_repository_packaging_and_collection"] is True
    assert policy["inspect_archive_inventory_before_transfer"] is True
    assert policy["approval_does_not_waive_minimization"] is True
    assert policy["preserve_remote_results_and_do_not_recompute"] is True
    assert {"file_count", "compressed_bytes", "uncompressed_bytes", "sha256"} <= set(policy["required_inventory"])
