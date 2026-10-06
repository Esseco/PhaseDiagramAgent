"""Fe/Mn high-spin proxy as a scoped DFT quality gate, not a ground-state proof."""
from scientific_layer.dft.magnetic_diagnostics import diagnose_result_magnetism, LAYERED_PHASES
from scientific_layer.dft.layered_spin_scope import layered_spin_scope


def apply_dft_spin_standard(result, manager=None):
    """Preserve execution status/raw labels; mark target DFT science eligibility.

    This empirical proxy does not compare alternative magnetic-order energies.
    It cannot certify a rigorous ground state. Only known layered oxides with
    Fe/Mn are subject to the gate; unrelated systems retain previous behavior.
    """
    current = diagnose_result_magnetism(result, manager)
    if current.get("stage") not in {"dft_single_point", "dft_relax"}:
        return current
    outputs = current.setdefault("outputs", {})
    report = outputs.get("magnetic_check") or {}
    layered = layered_spin_scope(outputs, manager)
    elements = set(report.get("elements") or [])
    if not elements:
        composition = outputs.get("composition") or {}
        if not composition and manager is not None:
            record = (manager.data.get("structures") or {}).get(current.get("structure_id")) or {}
            branch = (manager.data.get("branches") or {}).get(record.get("branch_id")) or {}
            composition = record.get("composition") or branch.get("composition") or {}
        elements = {element for element, count in composition.items() if count and float(count) > 0}
    target_elements = "O" in elements and bool(elements & {"Fe", "Mn"})
    applies = layered is not False and target_elements
    prior = outputs.get("spin_state_check") or {}
    reasons = current.setdefault("quality_rejection_reasons", [])
    if "checks_passed_before_spin" not in current:
        current["checks_passed_before_spin"] = current.get("checks_passed", True)
    if prior.get("status") in {"unknown", "rejected"} and ("dft_spin_standard_not_passed" in reasons or current["checks_passed_before_spin"] is True):
        if "dft_spin_standard_not_passed" in reasons:
            reasons.remove("dft_spin_standard_not_passed")
        current["checks_passed"] = current["checks_passed_before_spin"] is True and not reasons
    evidence = {"criterion": "layered_fe_mn_high_spin_local_moment_proxy_v1",
                "applies": applies, "magnetic_ground_state_proven": False,
                "evidence_sha256": report.get("evidence_sha256")}
    if not applies:
        evidence.update(status="not_applicable", reason="outside_layered_Fe_Mn_oxide_scope")
    elif layered is None:
        evidence.update(status="unknown", reason="final_layered_scope_unresolved")
        current["checks_passed"] = False
        if "dft_spin_standard_not_passed" not in reasons:
            reasons.append("dft_spin_standard_not_passed")
    elif report.get("status") == "passed" and report.get("checked_atoms", 0) > 0:
        evidence.update(status="passed", reason="expected_local_high_spin_ranges_satisfied")
    else:
        evidence.update(status="rejected" if report.get("status") == "warning" else "unknown",
                        reason=report.get("reason") or "Fe_Mn_spin_evidence_not_accepted")
        current["checks_passed"] = False
        if "dft_spin_standard_not_passed" not in reasons:
            reasons.append("dft_spin_standard_not_passed")
    outputs["spin_state_check"] = evidence
    return current


def spin_standard_passed(row):
    """Pure consumer guard for saved records, with legacy non-target compatibility."""
    report = row.get("spin_state_check") or (row.get("outputs") or {}).get("spin_state_check") or {}
    if not report:
        source = {**row, **(row.get("outputs") or {})}
        raw = source.get("magnetic_moments") or {}
        elements = set(raw.get("elements") or [])
        phase = source.get("actual_phase") or source.get("phase")
        if "O" in elements and elements & {"Fe", "Mn"} and (not phase or phase in LAYERED_PHASES):
            return False  # Newly captured raw evidence must be assessed before science use.
    source = {**row, **(row.get("outputs") or {})}
    phase = source.get("actual_phase") or source.get("phase")
    raw = source.get("magnetic_moments") or source.get("magnetic_check") or {}
    elements = set(raw.get("elements") or [])
    if report.get("applies") is False and phase in LAYERED_PHASES and "O" in elements and elements & {"Fe", "Mn"}:
        return False  # A stale pre-identification exemption is not scientific approval.
    return not report.get("applies") or report.get("status") == "passed"
