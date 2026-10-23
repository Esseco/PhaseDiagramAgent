"""Reusable JSON evidence and compact human-readable phase review reports."""

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from Process_AL_MC.structure_check import RULE as STRUCTURE_CHECK_RULE
from phase_agent.analysis.phase.phase_boundary import phase_boundary_rule

from phase_agent.analysis.phase.phase_quality import (
    RULE_VERSION,
    diagnose_phase_diagram,
)
from phase_agent.analysis.phase.phase_quality_policy import effective_policy
from phase_agent.analysis.phase.phase_quality_markdown import review_markdown, review_groups

__all__ = [
    "get_phase_quality_report",
    "quality_summary",
    "priority_group_representatives",
    "analysis_layer_summaries",
    "quality_warning",
    "candidate_quality_evidence",
    "review_markdown",
]


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def get_phase_quality_report(state, *, method="mlip", snapshot=None, policy=None, write=False):
    snapshot = (
        snapshot if snapshot is not None else (state.get("phase_diagrams") or {}).get(method) or {}
    )
    limits = effective_policy(state, policy)
    files = []
    paths = {row.get("structure_path") for row in snapshot.get("entries") or []}
    task_ids = {row.get("source_task_id") for row in snapshot.get("entries") or []} - {None}
    digests = {row.get("structure_sha256") for row in snapshot.get("entries") or []} - {None}
    paths.update(
        task.get("structure_path") or (task.get("inputs") or {}).get("structure_path")
        for task in state.get("tasks") or []
        if task.get("model_version") == state.get("active_model_version")
        and (
            task.get("task_id") in task_ids
            or (task.get("outputs") or {}).get("structure_checksum") in digests
        )
    )
    for raw_path in sorted(paths - {None, ""}):
        if raw_path:
            path = Path(raw_path)
            try:
                stat = path.stat()
                files.append((str(path), stat.st_size, stat.st_mtime_ns))
            except OSError:
                files.append((str(path), None, None))
    fingerprint = _digest(
        {
            "rule": RULE_VERSION,
            "structure_check_rule": STRUCTURE_CHECK_RULE,
            "snapshot": snapshot,
            "policy": limits,
            "policy_control": state.get("phase_quality_policy") or {},
            "phase_boundary": phase_boundary_rule(state),
            "model": state.get("active_model_version"),
            "tasks": state.get("tasks") or [],
            "files": files,
            "human_decisions": state.get("phase_quality_decisions") or [],
            "exclusions": state.get("phase_quality_exclusions") or [],
            "restorations": state.get("phase_quality_restorations") or [],
        }
    )
    stored = snapshot.get("csv_path") or snapshot.get("path")
    folder = Path(stored).resolve().parent if stored else None
    if folder and folder.name == "history":
        folder = folder.parent
    path = folder / "quality" / f"phase_quality_{fingerprint[:20]}.json" if folder else None
    if path and path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            checksum = cached.pop("report_checksum", None)
            if cached.get("input_fingerprint") == fingerprint and checksum == _digest(cached):
                cached["report_checksum"] = checksum
                return cached
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    report = diagnose_phase_diagram(state, method=method, snapshot=snapshot, policy=limits)
    report.update(input_fingerprint=fingerprint, report_path=str(path) if write and path else None)
    report["report_checksum"] = _digest(report)
    if write and path:
        _atomic_text(path, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report


def quality_summary(state, *, method="mlip", snapshot=None, write=False):
    try:
        report = get_phase_quality_report(state, method=method, snapshot=snapshot, write=write)
        groups = review_groups(report)
        number_by_id = {rid: g["number"] for g in groups for rid in g["review_ids"]}
        return {
            k: deepcopy(report[k])
            for k in (
                "status",
                "model_version",
                "hull_version",
                "coverage",
                "partial_refresh",
                "report_path",
                "interpretation",
                "decision_role",
                "recursive_stop_reason",
                "recursive_truncated",
                "recursive_converged",
                "screening_policy",
                "policy_control",
            )
        } | {
            "review_targets": [
                {
                    k: r.get(k)
                    for k in (
                        "review_id",
                        "structure_ids",
                        "disposition_label",
                        "review_pending",
                        "hull_lift_mev_per_O2",
                        "group_hull_lift_mev_per_O2",
                        "joint_hull_lift_mev_per_O2",
                        "energy_island_ids",
                        "H_mev_per_O2",
                        "D_mev_per_O2",
                        "tier",
                        "conditional_only",
                        "endpoint_protected",
                        "preview_excluded",
                        "near_current_hull",
                        "distance_to_current_hull_mev_per_O2",
                        "sampling_evidence_scope",
                        "quality_sampling_evidence_eligible",
                    )
                }
                | {"review_number": number_by_id.get(r["review_id"])}
                for r in report["structures"]
                if r["review_recommended"]
            ][:100],
            "restoration_targets": deepcopy(report.get("restoration_targets") or [])[-20:],
            "pending_human_count": report.get("pending_human_count", 0),
            "review_groups": [
                {k: g[k] for k in ("number", "phase", "x", "H", "D", "review_ids", "disposition")}
                for g in groups[:30]
            ],
            "analysis_layers": analysis_layer_summaries(report),
            "priority_structures": priority_group_representatives(report),
            "energy_islands": [
                {
                    k: c.get(k)
                    for k in (
                        "cluster_id",
                        "phase",
                        "x",
                        "record_ids",
                        "review_ids",
                        "reference_record_ids",
                        "reference_count",
                        "reference_status",
                        "gap_to_lowest_reference_mev_per_O2",
                        "hull_lift_mev_per_O2",
                        "disposition",
                    )
                }
                for c in report.get("energy_islands", [])[:20]
            ],
            "dft_sampling_advice": {
                "decision_role": report["decision_role"],
                "instruction": "仅当前近凸包或递归后进入近凸包的结构获得本模块选点优先证据，远点D大不自动提权。正式Ehull取已执行移除后的当前正式凸包，条件距离不能替换predicted_Ehull。选点读取dft_sampling_evidence.D_score（固定快照、无量纲）及明确前置假设的H序列；旧D_mev_per_O2只供原审查规则，不混用阈值。D或QBC任一突出都可提出核查，二者不直接相加；H补充相图影响，兼顾相/Na/环境及已有DFT覆盖，无固定名额或新增D cutoff。QBC不是实际误差，缺失不是零，不自动提交。",
            },
        }
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        return {
            "status": "diagnostic_unavailable",
            "screening_policy": effective_policy(state),
            "policy_control": {
                k: state.get("phase_quality_policy", {}).get(k, v)
                for k, v in (("revision", 0), ("manual_fields", []))
            },
            "reason": str(error),
            "interpretation": "诊断未完成，原数据保留；不能将未知当作零影响。",
        }


def priority_group_representatives(report, limit=8):
    """Bounded context must not be crowded out by many members of one group."""
    seen, seen_clusters, result = set(), set(), []
    for row in sorted(
        report["structures"],
        key=lambda r: (
            -(r.get("dft_evidence_priority_mev_per_O2") or 0),
            -max(
                r.get("group_hull_lift_mev_per_O2") or 0, r.get("joint_hull_lift_mev_per_O2") or 0
            ),
        ),
    ):
        if (
            row["review_recommended"]
            and row["group_id"] not in seen
            and not seen_clusters.intersection(row.get("energy_island_ids", []))
        ):
            result.append(deepcopy(row))
            seen.add(row["group_id"])
            seen_clusters.update(row.get("energy_island_ids", []))
    return result[:limit]


def analysis_layer_summaries(report):
    result = []
    for layer in report.get("analysis_layers") or []:
        summary = {
            k: deepcopy(layer[k])
            for k in (
                "analysis_layer",
                "assumed_removed_group_ids",
                "joint_group_ids",
                "joint_step_lift_mev_per_O2",
                "cumulative_lift_mev_per_O2",
                "conditional",
                "applied_to_preview",
                "endpoint_blocked_keys",
            )
        }
        joint = layer["joint_cases"]
        summary["combination_stats"] = deepcopy(joint["stats"])
        summary["combination_examples"] = [
            {k: case[k] for k in ("mode", "order", "hull_lift_mev_per_O2", "record_ids")}
            for case in joint["cases"][:5]
        ]
        result.append(summary)
    return result


def quality_warning(state, *, method="mlip", snapshot=None):
    summary = quality_summary(state, method=method, snapshot=snapshot, write=True)
    if summary["status"] == "diagnostic_unavailable":
        return "相图核查未完成：" + summary["reason"]
    lines = [
        "相图核查："
        + ("建议人工审查" if summary["status"] == "review_required" else "未发现达到阈值的凸包影响")
        + f"；H>{summary['screening_policy']['direct_removal_mev_per_O2']:g}直接移除，H>{summary['screening_policy']['hull_shift_mev_per_O2']:g}或D>{summary['screening_policy']['outlier_gap_mev_per_O2']:g}且结构模块确认有问题时移除；端点基准保护优先，第二档交人工。"
    ]
    if summary.get("analysis_layers"):
        lines.append(
            f"已检查最多5项联合移除及{len(summary['analysis_layers'])}轮条件递归；{'按H/D截止参数收敛' if summary['recursive_converged'] else '未收敛，需要继续核查'}。后层不代表当前凸包的独立影响。"
        )
    for r in summary["priority_structures"][:3]:
        lift = r["group_hull_lift_mev_per_O2"]
        value = f"{lift:.3f} meV/O₂" if lift is not None else "无法完整估计（端点缺少替代）"
        lines.append(
            f"审查#{r['review_number']}：{','.join(r['structure_ids'])}，组影响 {value}，同相低能参考 {r['reference_count']} 个。"
        )
    if summary["report_path"]:
        lines.append(f"核查报告：`{summary['report_path']}`")
    return "\n".join(lines)


def candidate_quality_evidence(state, candidates, *, report=None):
    """Attach exact-geometry/current-model evidence; never alter Ehull/QBC/order/count."""
    from phase_agent.analysis.convergence.independent_evidence import geometry_key

    if not candidates:
        return []
    if report is None:
        try:
            report = get_phase_quality_report(state)
        except (OSError, ValueError, TypeError, KeyError):
            report = {}
    rows = report.get("structures") or []
    from phase_agent.analysis.phase.phase_quality_disposition import filter_excluded_candidates

    result = deepcopy(filter_excluded_candidates(state, candidates))
    model = state.get("active_model_version")
    snapshot = (state.get("phase_diagrams") or {}).get("mlip") or {}
    sampling, sampling_error = {}, None
    if report.get("model_version") == model and report.get("hull_version") == snapshot.get(
        "version"
    ):
        from phase_agent.analysis.phase.phase_dft_sampling import fixed_snapshot_sampling_evidence

        try:
            sampling = fixed_snapshot_sampling_evidence(snapshot, report)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            sampling_error = str(error)
    for candidate in result:
        candidate["phase_quality_signal"] = {"status": "unavailable", "decision_made": False}
        candidate["dft_sampling_evidence"] = {
            "status": "unavailable",
            "decision_made": False,
            "reason": sampling_error or "No exact current model/hull/geometry binding",
        }
        current_hull = ((state.get("phase_diagrams") or {}).get("mlip") or {}).get("version")
        if (
            report.get("model_version") != model
            or candidate.get("model_version", model) != model
            or report.get("hull_version") != current_hull
            or candidate.get("phase_diagram_version", current_hull) != current_hull
        ):
            continue
        key = geometry_key(candidate)
        path = candidate.get("structure_path")
        if key is None and path:
            key = geometry_key({"structure": path})
        raw_hash = None
        if path:
            try:
                raw_hash = sha256(Path(path).read_bytes()).hexdigest()
            except OSError as error:
                candidate["dft_sampling_evidence"].update(
                    reason=f"structure_file_unavailable: {path}: {error}",
                    structure_file_status="unavailable",
                )
                continue
        matches = [
            r
            for r in rows
            if (key and r["geometry_key"] == key)
            or (raw_hash and raw_hash in r["structure_sha256"])
        ]
        if not matches:
            evidence = sampling.get(raw_hash or key)
            if evidence:
                candidate["dft_sampling_evidence"] = {
                    **deepcopy(evidence),
                    "quality_priority_eligible": False,
                    "sampling_evidence_scope": "outside_review_scope",
                    "endpoint_protected": False,
                    "structure_review": "not_assessed",
                }
        if len(matches) == 1:
            evidence = sampling.get(matches[0]["geometry_key"])
            if evidence:
                candidate["dft_sampling_evidence"] = {
                    **deepcopy(evidence),
                    "quality_priority_eligible": matches[0].get(
                        "quality_sampling_evidence_eligible", False
                    ),
                    "sampling_evidence_scope": matches[0].get("sampling_evidence_scope"),
                    "endpoint_protected": matches[0].get("endpoint_protected", False),
                    "structure_review": matches[0].get("framework_status", "unknown"),
                }
            candidate["phase_quality_signal"] = {
                **deepcopy(matches[0]),
                "status": "available",
                "model_version": model,
                "hull_version": report["hull_version"],
                "decision_made": False,
                "decision_role": "supporting_evidence_alongside_ehull_and_qbc",
                "formal_ehull_reference": "current_formal_hull_after_executed_exclusions",
                "conditional_distance_unit": "meV/O2",
                "conditional_distance_role": "diagnostic_only_not_predicted_Ehull",
            }
    return result
