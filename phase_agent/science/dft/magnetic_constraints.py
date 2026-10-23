"""Element-wise absolute-moment constraints, independent of raw DFT recovery."""

import math


def magnetic_abs_ranges(manager=None, *, overrides=None):
    """Resolve inclusive |m| intervals in mu_B; partial overrides retain Fe/Mn defaults."""
    from Process_Vasp.results.magnetic_check import RANGES
    from pymatgen.core import Element

    ranges = {key: list(value) for key, value in RANGES.items()}
    system = (manager.data.get("system_config") or {}) if manager is not None else {}
    if overrides is None:
        overrides = system.get("dft_magnetic_abs_ranges", {})
    if not isinstance(overrides, dict):
        raise ValueError("dft_magnetic_abs_ranges must map element symbols to intervals")
    for element, interval in overrides.items():
        if not isinstance(element, str) or not Element.is_valid_symbol(element):
            raise ValueError(f"invalid magnetic constraint element: {element}")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"{element}: expected [minimum, maximum] in mu_B")
        if any(isinstance(value, bool) for value in interval):
            raise ValueError(f"{element}: magnetic range cannot contain booleans")
        low, high = map(float, interval)
        if not all(math.isfinite(value) for value in (low, high)) or not 0 <= low <= high:
            raise ValueError(f"{element}: expected finite 0 <= minimum <= maximum")
        ranges[element] = [low, high]
    return ranges


def check_magnetic_constraints(
    elements,
    moments,
    *,
    ranges,
    explicit_elements=(),
    is_layered_oxide=None,
    noncollinear=False,
    total_moment=None,
):
    """Implicit Fe/Mn defaults are scoped; explicit element ranges apply to any phase."""
    from Process_Vasp import check_layered_oxide_moments

    # Preserve shared validation and diagnostics for the unchanged default path.
    report = check_layered_oxide_moments(
        elements,
        moments,
        is_layered_oxide=is_layered_oxide,
        noncollinear=noncollinear,
        total_moment=total_moment,
    )
    report.update(expected_abs_ranges=ranges, explicit_elements=sorted(explicit_elements))
    # Shared default checks skip unsupported sites; retain their evidence for legacy JSON too.
    if moments is not None:
        report["moments"] = moments
    if not explicit_elements:
        return report
    elements = list(elements)
    implicit = (
        set(elements) & {"Fe", "Mn"} if "O" in elements and is_layered_oxide is not False else set()
    )
    targets = (implicit | set(explicit_elements)) & set(elements)
    unresolved = bool(implicit - set(explicit_elements)) and is_layered_oxide is None
    report.update(
        schema="element-moment-constraints-v1",
        supported_elements=sorted(targets),
        scope_unresolved=unresolved,
        checked_atoms=0,
        reasonable_atoms=[],
        anomalous_atoms=[],
    )
    report.pop("reason", None)
    if not targets:
        report.update(status="skipped", reason="no_constrained_elements")
        return report
    if noncollinear:
        report.update(status="unknown", reason="noncollinear_or_SOC_not_supported")
        return report
    try:
        values = [float(value) for value in moments]
        if len(values) != len(elements) or not all(math.isfinite(value) for value in values):
            raise ValueError("invalid shape or nonfinite moments")
    except (TypeError, ValueError, OverflowError):
        report.update(status="unknown", reason="missing_or_invalid_local_moments")
        return report
    report["moments"] = values
    if unresolved:
        report.update(status="unknown", reason="layered_identity_unavailable")
        return report
    for index, (element, moment) in enumerate(zip(elements, values)):
        if element not in targets:
            continue
        low, high = ranges[element]
        atom = {
            "atom_index": index,
            "element": element,
            "moment": moment,
            "absolute_moment": abs(moment),
            "expected_abs_range": [low, high],
        }
        destination = "reasonable_atoms" if low <= abs(moment) <= high else "anomalous_atoms"
        report[destination].append(atom)
        report["checked_atoms"] += 1
    report["status"] = "warning" if report["anomalous_atoms"] else "passed"
    return report
