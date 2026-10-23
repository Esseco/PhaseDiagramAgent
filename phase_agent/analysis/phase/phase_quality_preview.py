"""Endpoint-relative plot data and fixed-reference influence; no scientific mutations."""

from phase_agent.analysis.phase.phase_quality import (
    _points,
    _hull,
    _at,
    _influence,
    snapshot_evidence_digest,
)
from phase_agent.analysis.phase.endpoint_reference import (
    endpoint_relative_energies,
    PLOT_RULE_VERSION,
)


def build_quality_preview(snapshot, report, *, cluster_ids=(), review_ids=()):
    if snapshot.get("version") != report.get("hull_version") or snapshot.get(
        "model_version"
    ) != report.get("model_version"):
        raise ValueError("相图或模型与审查报告不一致")
    if snapshot_evidence_digest(snapshot) != report.get("snapshot_evidence_digest"):
        raise ValueError("相图记录已改变，请重新审查")
    requested = set(cluster_ids)
    clusters = [c for c in report.get("energy_islands", []) if c["cluster_id"] in requested]
    if len(clusters) != len(requested):
        raise ValueError("离群簇身份无效")
    points = _points(snapshot, [])
    by_key = {p["key"]: p for p in points}
    evidence = {r["geometry_key"]: r for r in report["structures"]}
    removed = {k for c in clusters for k in c["geometry_keys"]}
    selected = [r for r in report["structures"] if r.get("review_id") in set(review_ids)]
    if len(selected) != len(set(review_ids)):
        raise ValueError("审查身份无效")
    removed.update(r["geometry_key"] for r in selected)
    from phase_agent.analysis.phase.phase_quality_endpoints import assert_endpoint_removal

    assert_endpoint_removal(points, report["screening_policy"], removed)
    for key in removed:
        p, row = by_key.get(key), evidence.get(key)
        if (
            not p
            or not row
            or abs(p["energy_mev_per_O2"] - row["energy_mev_per_O2"]) > 1e-6
            or sorted(r["record_id"] for r in p["rows"]) != sorted(row["record_ids"])
        ):
            raise ValueError("结构能量或记录已改变，请重新审查")
    original = _hull([(p["x"], p["energy_mev_per_O2"]) for p in points])
    lift = _influence(points, original, removed)
    if lift is None:
        raise ValueError("假设移除将丢失端点，不能绘制完整对照")
    remaining = [p for p in points if p["key"] not in removed]
    hull = _hull([(p["x"], p["energy_mev_per_O2"]) for p in remaining])
    plot = {"x": [], "eform": [], "phase": [], "is_stable": []}
    for point in remaining:
        for _ in point["rows"]:
            plot["x"].append(point["x"])
            plot["eform"].append(point["energy_mev_per_O2"])
            plot["phase"].append(point["phase"])
            plot["is_stable"].append(
                abs(point["energy_mev_per_O2"] - _at(hull, point["x"])) <= 1e-5
            )
    plot["eform"], reference = endpoint_relative_energies(plot["x"], plot["eform"])
    return {
        "scope": "counterfactual_only_not_rejection",
        "formal_records_modified": False,
        "model_version": snapshot["model_version"],
        "hull_version": snapshot["version"],
        "cluster_ids": sorted(requested),
        "excluded_record_ids": sorted(
            r["record_id"] for p in points if p["key"] in removed for r in p["rows"]
        ),
        "before_count": sum(len(p["rows"]) for p in points),
        "after_count": len(plot["x"]),
        "hull_lift_mev_per_O2": lift,
        "reference": "own_retained_endpoints",
        "endpoint_reference": reference,
        "endpoint_reference_unit": "meV/O2",
        "influence_reference": "fixed_original",
        "plot_rule_version": PLOT_RULE_VERSION,
        "plot_data": plot,
    }
