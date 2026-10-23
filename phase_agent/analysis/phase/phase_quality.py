"""Read-only, fixed-reference hull sensitivity; influence is not prediction error."""

from collections import defaultdict
from hashlib import sha256
import json
from math import isfinite
from statistics import median

DEFAULT_POLICY = {
    "group_span_mev_per_O2": 10.0,
    "reference_window_mev_per_O2": 50.0,
    "hull_shift_mev_per_O2": 50.0,
    "outlier_gap_mev_per_O2": 50.0,
    "direct_removal_mev_per_O2": 100.0,
    "recursive_max_layers": None,
    "joint_max_order": 5,
}
RULE_VERSION = "phase-quality-HD-near-hull-v8"


def snapshot_evidence_digest(snapshot):
    return sha256(
        json.dumps(
            {k: snapshot.get(k) for k in ("entries", "model_version", "version")},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()


def screening_policy(policy=None):
    result = {**DEFAULT_POLICY, **(policy or {})}
    if set(result) != set(DEFAULT_POLICY) or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or v <= 0
        for k, v in result.items()
        if k != "recursive_max_layers"
    ):
        raise ValueError("无效的相图核查参数")
    depth = result["recursive_max_layers"]
    if depth is not None and (isinstance(depth, bool) or not isinstance(depth, int) or depth < 1):
        raise ValueError("recursive_max_layers 须为正整数或None（按截止参数收敛）")
    if not isinstance(result["joint_max_order"], int) or result["joint_max_order"] > 5:
        raise ValueError("joint_max_order须为1到5的整数")
    if result["direct_removal_mev_per_O2"] < result["hull_shift_mev_per_O2"]:
        raise ValueError("H直接移除阈值不能小于H审查阈值")
    return result


def _hull(points):
    lowest = {}
    for x, y in points:
        lowest[x] = min(y, lowest.get(x, y))
    hull = []
    for p in sorted(lowest.items()):
        while len(hull) >= 2:
            a, b = hull[-2:]
            if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 1e-14:
                break
            hull.pop()
        hull.append(p)
    return hull


def _at(hull, x):
    if not hull or x < hull[0][0] - 1e-9 or x > hull[-1][0] + 1e-9:
        return None
    for a, b in zip(hull, hull[1:]):
        if a[0] - 1e-9 <= x <= b[0] + 1e-9:
            return a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0])
    return hull[0][1] if abs(x - hull[0][0]) < 1e-9 else None


def _influence(points, hull, removed):
    alternate = _hull([(p["x"], p["energy_mev_per_O2"]) for p in points if p["key"] not in removed])
    # Losing an endpoint loses the domain: do not report zero or a partial maximum.
    if (
        not alternate
        or alternate[0][0] > hull[0][0] + 1e-9
        or alternate[-1][0] < hull[-1][0] - 1e-9
    ):
        return None
    xs = sorted({x for x, _ in hull + alternate})
    return max(0.0, max(_at(alternate, x) - _at(hull, x) for x in xs))


def _identity(row, index):
    from phase_agent.analysis.convergence.independent_evidence import geometry_key

    # Canonical in-memory geometry, otherwise stored final geometry checksum.
    key = geometry_key(row)
    return key or f"unknown-{index}"


def _points(snapshot, issues):
    points, frameworks = {}, set()
    for index, row in enumerate(snapshot.get("entries") or []):
        try:
            composition = row["composition"]
            units = float(composition["O"]) / 2
            if units <= 0:
                raise ValueError("nonpositive oxygen")
            normalized = tuple(
                sorted((k, round(float(v) / units, 9)) for k, v in composition.items() if float(v))
            )
            x = round(float(composition.get("Na", 0)) / units, 9)
            energy = float(row["eform_per_O2"]) * 1000
            if x < 0 or not isfinite(x) or not isfinite(energy):
                raise ValueError("nonfinite coordinate")
            if row.get("source_version") not in (None, snapshot.get("model_version")):
                raise ValueError("record model mismatch")
            frameworks.add(tuple((k, v) for k, v in normalized if k != "Na"))
            key = _identity(row, index)
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as error:
            issues.append(f"记录 {row.get('record_id')} 无法诊断：{error}")
            continue
        point = points.setdefault(
            key,
            {
                "key": key,
                "x": x,
                "composition_key": normalized,
                "phase": row.get("phase"),
                "energy_mev_per_O2": energy,
                "rows": [],
            },
        )
        if (
            point["composition_key"] != normalized
            or abs(point["energy_mev_per_O2"] - energy) > 1e-4
            or point["phase"] != row.get("phase")
        ):
            raise ValueError("同一精确几何的组成、相或能量记录冲突，需先人工核对")
        point["rows"].append(row)
    if len(frameworks) > 1:
        raise ValueError("混合了不同骨架组分，不能用一维 Na 凸包进行核查")
    if issues:
        raise ValueError("；".join(issues))
    return list(points.values())


def _geometry_evidence(point, state):
    from phase_agent.analysis.phase.phase_quality_geometry import inspect_geometry

    from phase_agent.analysis.phase.phase_framework import inspect_framework

    row = point["rows"][0]
    result = {
        "status": "unavailable",
        "interpretation": "配位变化可能是相变或重构，不证明能量错误；保留并人工核查",
    }
    try:
        result.update(status="observed", final=inspect_geometry(row))
    except (OSError, ValueError, TypeError, KeyError) as error:
        result["reason"] = str(error)
        return result
    # Bind initial/final observations to a source task and this model, never structure_id alone.
    task_ids = {r.get("source_task_id") for r in point["rows"]} - {None}
    digests = {r.get("structure_sha256") for r in point["rows"]} - {None}
    for task in reversed(state.get("tasks") or []):
        output = task.get("outputs") or {}
        digest = (output.get("phase_identification") or {}).get("structure_sha256") or output.get(
            "structure_checksum"
        )
        if task.get("model_version") != state.get("active_model_version") or not (
            task.get("task_id") in task_ids or digest in digests
        ):
            continue
        path = task.get("structure_path") or (task.get("inputs") or {}).get("structure_path")
        if path and path != row.get("structure_path"):
            try:
                initial = inspect_geometry(
                    {
                        "structure_path": path,
                        "composition": row["composition"],
                        "phase": task.get("phase") or row.get("phase"),
                    }
                )
                result["framework"] = inspect_framework(row, path)
                result["initial"] = initial
                result["initial_structure_path"] = str(path)
                result["volume_change_fraction"] = (
                    result["final"]["volume_per_O2_A3"] / initial["volume_per_O2_A3"] - 1
                )
                result["non_six_fraction_change"] = (
                    result["final"]["non_six_fraction"] - initial["non_six_fraction"]
                )
            except (OSError, ValueError, TypeError, KeyError) as error:
                result["initial_unavailable"] = "初始结构检查未完成：" + str(error)
        break
    if "initial" not in result:
        result.setdefault(
            "initial_unavailable", "缺少绑定的弛豫前结构，不能声称发生了配位/骨架变化"
        )
    result.setdefault(
        "framework",
        {
            "status": "uncertain",
            "auto_confirmed": False,
            "reason": result.get("initial_unavailable", "缺少绑定初始结构"),
        },
    )
    if result["framework"].get("auto_confirmed"):
        result["interpretation"] = (
            "严格骨架判据确认本项目层状骨架局部破坏；仅当前高影响的完整移除集合可有备份地排除，不宣判重构相的物理能量错误"
        )
    return result


def _energy_groups(points, limits):
    pools = defaultdict(list)
    for p in points:
        pools[(p["composition_key"], p["phase"])].append(p)
    groups = {}
    for pool in pools.values():
        current = []
        for p in sorted(pool, key=lambda p: (p["energy_mev_per_O2"], p["key"])):
            if (
                current
                and p["energy_mev_per_O2"] - current[0]["energy_mev_per_O2"]
                > limits["group_span_mev_per_O2"] + 1e-8
            ):
                gid = (
                    "G-"
                    + sha256("|".join(sorted(p["key"] for p in current)).encode()).hexdigest()[:12]
                )
                groups[gid] = current
                current = []
            current.append(p)
        if current:
            gid = (
                "G-" + sha256("|".join(sorted(p["key"] for p in current)).encode()).hexdigest()[:12]
            )
            groups[gid] = current
    return groups


def _diagnose_layer(
    state, *, method="mlip", snapshot=None, policy=None, geometry_cache=None, include_keys=()
):
    snapshot = (
        snapshot if snapshot is not None else (state.get("phase_diagrams") or {}).get(method) or {}
    )
    from phase_agent.analysis.phase.phase_quality_policy import effective_policy

    limits = effective_policy(state, policy)
    if snapshot.get("status") != "completed" or not snapshot.get("entries"):
        raise ValueError("缺少完整的相图快照")
    if method == "mlip" and snapshot.get("model_version") != state.get("active_model_version"):
        raise ValueError("相图与当前模型版本不一致")
    issues = []
    points = _points(snapshot, issues)
    hull = _hull([(p["x"], p["energy_mev_per_O2"]) for p in points])
    if len(hull) < 2:
        raise ValueError("至少需要两个不同 Na 含量")
    pools = defaultdict(list)
    for p in points:
        pools[(p["composition_key"], p["phase"])].append(p)
    groups = _energy_groups(points, limits)
    records = []
    for group_id, members in groups.items():
        # Only groups touching the lower hull need counterfactual diagnostics.
        if not any(
            abs(p["energy_mev_per_O2"] - _at(hull, p["x"])) <= 1e-5 or p["key"] in include_keys
            for p in members
        ):
            continue
        keys = {p["key"] for p in members}
        group_lift = _influence(points, hull, keys)
        peers = [
            p
            for p in pools[(members[0]["composition_key"], members[0]["phase"])]
            if p["key"] not in keys
        ]
        reference = []
        if peers:
            minimum = min(p["energy_mev_per_O2"] for p in peers)
            reference = [
                p["energy_mev_per_O2"]
                for p in peers
                if p["energy_mev_per_O2"] <= minimum + limits["reference_window_mev_per_O2"] + 1e-8
            ]
        for p in members:
            lift = _influence(points, hull, {p["key"]})
            gap = min(reference) - p["energy_mev_per_O2"] if reference else None
            review = (
                group_lift is None
                or group_lift > limits["hull_shift_mev_per_O2"]
                or p["key"] in include_keys
            )
            # A low-energy island (>= grouping resolution below all other groups),
            # or missing reference evidence, warrants secondary geometry inspection.
            # Every actionable geometry needs its own evidence before rejection.
            inspect = review
            rows = p["rows"]
            other_phase = [
                q["energy_mev_per_O2"]
                for q in points
                if q["composition_key"] == p["composition_key"] and q["phase"] != p["phase"]
            ]
            records.append(
                {
                    "geometry_key": p["key"],
                    "group_id": group_id,
                    "group_size": len(members),
                    "record_ids": sorted(str(r["record_id"]) for r in rows if r.get("record_id")),
                    "structure_ids": sorted(
                        {r["structure_id"] for r in rows if r.get("structure_id")}
                    ),
                    "structure_paths": sorted(
                        {r["structure_path"] for r in rows if r.get("structure_path")}
                    ),
                    "structure_sha256": sorted(
                        {r["structure_sha256"] for r in rows if r.get("structure_sha256")}
                    ),
                    "x": p["x"],
                    "phase": p["phase"],
                    "energy_mev_per_O2": p["energy_mev_per_O2"],
                    "distance_to_current_hull_mev_per_O2": p["energy_mev_per_O2"]
                    - _at(hull, p["x"]),
                    "hull_lift_mev_per_O2": lift,
                    "group_hull_lift_mev_per_O2": group_lift,
                    "priority_mev_per_O2": lift,
                    "group_priority_mev_per_O2": group_lift,
                    "review_recommended": review,
                    "reference_count": len(reference),
                    "gap_to_lowest_reference_mev_per_O2": gap,
                    "gap_to_reference_median_mev_per_O2": median(reference) - p["energy_mev_per_O2"]
                    if reference
                    else None,
                    "gap_to_lowest_other_phase_mev_per_O2": min(other_phase)
                    - p["energy_mev_per_O2"]
                    if other_phase
                    else None,
                    "reference_status": "available"
                    if len(reference) >= 5
                    else "insufficient_references",
                    "geometry": _cached_geometry(p, state, geometry_cache)
                    if inspect
                    else {"status": "not_triggered", "group_representative_key": members[0]["key"]},
                }
            )
    records.sort(
        key=lambda r: (
            r["group_priority_mev_per_O2"] is not None,
            -(r["group_priority_mev_per_O2"] or 0),
            r["group_id"],
            -(r["priority_mev_per_O2"] or 0),
            r["geometry_key"],
        )
    )
    for number, row in enumerate(records, 1):
        row["review_number"] = number
    return {
        "rule_version": RULE_VERSION,
        "status": "review_required"
        if any(r["review_recommended"] for r in records)
        else "screened_no_findings",
        "model_version": snapshot.get("model_version"),
        "hull_version": snapshot.get("version"),
        "screening_policy": limits,
        "structures": records,
        "coverage": {
            "entries": len(snapshot["entries"]),
            "unique_geometries": len(points),
            "hull_groups": len({r["group_id"] for r in records}),
            "geometry_inspected": sum(r["geometry"]["status"] == "observed" for r in records),
        },
        "partial_refresh": bool((snapshot.get("refresh_coverage") or {}).get("partial")),
        "candidate_policy": "preserve_and_review",
        "decision_role": "supporting_evidence_alongside_ehull_and_qbc",
        "interpretation": "H为固定原能量参考下的凸包最大抬升；D为同相同组分低能簇与上方参考的断层。两者不相加、不解释为误差概率。当前H超过直接移除阈值，或H/D超审查阈值且公共结构模块确认有问题，归第一档；其他高证据归第二档正式保留、右图假设排除打×。第二档DFT辅助证据主要看D，并结合Ehull/QBC。诊断只读，未知不等于零，不提交计算。",
    }


def _cached_geometry(point, state, cache):
    if cache is None:
        return _geometry_evidence(point, state)
    if point["key"] not in cache:
        cache[point["key"]] = _geometry_evidence(point, state)
    return cache[point["key"]]


def diagnose_phase_diagram(state, *, method="mlip", snapshot=None, policy=None):
    """Follow one monotonic, endpoint-safe chain until the H/D criteria find no new set."""
    from phase_agent.analysis.phase.phase_boundary import assert_snapshot_boundary
    from phase_agent.analysis.phase.phase_hull_sensitivity import joint_cases
    from phase_agent.analysis.phase.phase_energy_islands import energy_islands
    from phase_agent.analysis.phase.phase_quality_endpoints import (
        endpoint_guards,
        retained_keys,
        select_preview_keys,
    )
    from phase_agent.analysis.phase.phase_quality_restore import restored_records

    snapshot = (
        snapshot if snapshot is not None else (state.get("phase_diagrams") or {}).get(method) or {}
    )
    assert_snapshot_boundary(state, snapshot)
    from phase_agent.analysis.phase.phase_quality_policy import effective_policy

    limits = effective_policy(state, policy)
    cache = {}
    original = _points(snapshot, [])
    if not original:
        raise ValueError("缺少可核查的相图记录")
    original_hull = _hull([(p["x"], p["energy_mev_per_O2"]) for p in original])
    keep = retained_keys(state, snapshot, limits, original)
    restored = restored_records(state)
    keep.update(
        p["key"]
        for p in original
        if p["rows"]
        and all((snapshot.get("model_version"), r["record_id"]) in restored for r in p["rows"])
    )
    baseline_guards = endpoint_guards(original, limits)
    active = original
    excluded_keys, excluded_groups, layers = set(), [], []
    by_key = {}
    combined = []
    report = None
    stop = "depth_limit"
    # Every applied pass removes >=1 exact geometry. One final pass proves convergence.
    budget = limits["recursive_max_layers"] or len(original) + 1
    for depth in range(budget):
        active_rows = [row for p in active for row in p["rows"]]
        active_groups = _energy_groups(active, limits)
        joint = joint_cases(
            active,
            active_groups,
            threshold=limits["hull_shift_mev_per_O2"],
            order=limits["joint_max_order"],
            escalation_threshold=limits["direct_removal_mev_per_O2"],
            strict_review=True,
        )
        islands = energy_islands(active, active_groups, limits)
        include_keys = {key for case in joint["cases"] for key in case["geometry_keys"]}
        include_keys.update(key for cluster in islands for key in cluster["geometry_keys"])
        current = _diagnose_layer(
            state,
            method=method,
            snapshot={**snapshot, "entries": active_rows},
            policy=limits,
            geometry_cache=cache,
            include_keys=include_keys,
        )
        if report is None:
            report = current
            report["energy_islands"] = islands
            report["policy_control"] = {
                k: state.get("phase_quality_policy", {}).get(k, v)
                for k, v in (("revision", 0), ("manual_fields", []), ("history", []))
            }
        current_rows = current["structures"]
        keys, blocked = select_preview_keys(
            active,
            limits,
            joint["cases"],
            islands,
            keep=keep,
        )
        groups = {
            gid: {p["key"] for p in members}
            for gid, members in active_groups.items()
            if any(p["key"] in keys for p in members)
        }
        for candidate in current_rows:
            key = candidate["geometry_key"]
            candidate["endpoint_protected"] = key in baseline_guards
            candidate["endpoint_protection_reason"] = baseline_guards.get(key)
            candidate["D_mev_per_O2"] = max(
                (
                    c["gap_to_lowest_reference_mev_per_O2"]
                    for c in islands
                    if key in c["geometry_keys"]
                    and c["distance_to_hull_mev_per_O2"] <= limits["hull_shift_mev_per_O2"]
                ),
                default=0.0,
            )
        for row in current_rows:
            key = row["geometry_key"]
            if key not in by_key:
                row["analysis_layer"] = depth
                row["assumed_removed_group_ids"] = list(excluded_groups)
                row["conditional_evidence"] = []
                if depth:
                    conditional = {
                        "analysis_layer": depth,
                        "assumed_removed_group_ids": list(excluded_groups),
                        "hull_lift_mev_per_O2": row["hull_lift_mev_per_O2"],
                        "group_hull_lift_mev_per_O2": row["group_hull_lift_mev_per_O2"],
                        "D_mev_per_O2": row["D_mev_per_O2"],
                        "distance_to_conditional_hull_mev_per_O2": row[
                            "distance_to_current_hull_mev_per_O2"
                        ],
                        "joint_hull_lift_mev_per_O2": max(
                            (
                                c["hull_lift_mev_per_O2"]
                                for c in joint["cases"]
                                if key in c["geometry_keys"]
                            ),
                            default=0.0,
                        ),
                    }
                    row["conditional_evidence"].append(conditional)
                    group_keys = {
                        r["geometry_key"] for r in current_rows if r["group_id"] == row["group_id"]
                    }
                    row["hull_lift_mev_per_O2"] = _influence(original, original_hull, {key})
                    row["group_hull_lift_mev_per_O2"] = _influence(
                        original, original_hull, group_keys
                    )
                    row["priority_mev_per_O2"] = row["hull_lift_mev_per_O2"]
                    row["group_priority_mev_per_O2"] = row["group_hull_lift_mev_per_O2"]
                    row["D_mev_per_O2"] = max(
                        (
                            c["gap_to_lowest_reference_mev_per_O2"]
                            for c in report["energy_islands"]
                            if key in c["geometry_keys"]
                        ),
                        default=0.0,
                    )
                    row["distance_to_current_hull_mev_per_O2"] = row["energy_mev_per_O2"] - _at(
                        original_hull, row["x"]
                    )
                by_key[key] = row
                combined.append(row)
            elif depth and row["review_recommended"]:
                by_key[key]["conditional_evidence"].append(
                    {
                        "analysis_layer": depth,
                        "assumed_removed_group_ids": list(excluded_groups),
                        "hull_lift_mev_per_O2": row["hull_lift_mev_per_O2"],
                        "group_hull_lift_mev_per_O2": row["group_hull_lift_mev_per_O2"],
                        "D_mev_per_O2": row["D_mev_per_O2"],
                        "distance_to_conditional_hull_mev_per_O2": row[
                            "distance_to_current_hull_mev_per_O2"
                        ],
                        "joint_hull_lift_mev_per_O2": max(
                            (
                                c["hull_lift_mev_per_O2"]
                                for c in joint["cases"]
                                if key in c["geometry_keys"]
                            ),
                            default=0.0,
                        ),
                    }
                )
                by_key[key]["review_recommended"] = True
                if (
                    by_key[key]["geometry"]["status"] == "not_triggered"
                    and row["geometry"]["status"] != "not_triggered"
                ):
                    by_key[key]["geometry"] = row["geometry"]
        current_hull = _hull([(p["x"], p["energy_mev_per_O2"]) for p in active])
        step = _influence(active, current_hull, keys) if keys else 0.0
        cumulative = (
            _influence(original, original_hull, excluded_keys | keys)
            if keys
            else _influence(original, original_hull, excluded_keys)
        )
        layers.append(
            {
                "analysis_layer": depth,
                "assumed_removed_group_ids": list(excluded_groups),
                "joint_group_ids": sorted(groups),
                "joint_geometry_keys": sorted(keys),
                "endpoint_blocked_keys": sorted(blocked),
                "applied_to_preview": False,
                "joint_step_lift_mev_per_O2": step,
                "cumulative_lift_mev_per_O2": cumulative,
                "support_geometry_keys": [r["geometry_key"] for r in current_rows],
                "joint_cases": joint,
                "energy_islands": islands,
                "conditional": bool(depth),
                "scope": "counterfactual_only_not_rejection",
            }
        )
        if any(p["key"].startswith("unknown-") for p in active):
            stop = "unknown_geometry_identity"
            break
        if not keys:
            stop = "no_more_removable_high_influence_groups"
            break
        if step is None:
            stop = "joint_removal_loses_endpoint_domain"
            break
        if depth + 1 == budget:
            break
        if keys.intersection(excluded_keys) or not keys <= {p["key"] for p in active}:
            stop = "no_progress_guard"
            break
        excluded_keys.update(keys)
        layers[-1]["applied_to_preview"] = True
        excluded_groups.extend(g for g in sorted(groups) if g not in excluded_groups)
        active = [p for p in original if p["key"] not in excluded_keys]
    for number, row in enumerate(combined, 1):
        row["review_number"] = number
    report["applied_exclusions"] = [
        {
            k: entry.get(k)
            for k in (
                "source",
                "reason",
                "model_version",
                "hull_version",
                "record_ids",
                "structure_ids",
                "backup_path",
                "review_ids",
                "evidence",
            )
        }
        for entry in state.get("phase_quality_exclusions") or []
        if entry.get("model_version") == state.get("active_model_version")
    ]
    excluded_ids = {
        r.get("record_id")
        for r in (state.get("phase_records") or [])
        + (state.get("phase_quality_archived_records") or [])
        if str(r.get("status", "")).startswith("excluded_")
    }
    report["restoration_targets"] = [
        {
            k: entry.get(k)
            for k in ("review_ids", "record_ids", "structure_ids", "reason", "backup_path")
        }
        for entry in report["applied_exclusions"]
        if entry.get("review_ids") and excluded_ids.intersection(entry["record_ids"])
    ]
    report["applied_restorations"] = state.get("phase_quality_restorations") or []
    report["structures"] = combined
    report["snapshot_evidence_digest"] = snapshot_evidence_digest(snapshot)
    report["analysis_layers"] = layers
    report["preview_excluded_geometry_keys"] = sorted(excluded_keys)
    report["preview_final_support_keys"] = layers[-1]["support_geometry_keys"]
    for row in combined:
        row["preview_excluded"] = row["geometry_key"] in excluded_keys
    report["recursive_stop_reason"] = stop
    report["recursive_converged"] = stop == "no_more_removable_high_influence_groups"
    report["recursive_truncated"] = not report["recursive_converged"]
    report["status"] = (
        "review_required"
        if any(r["review_recommended"] for r in combined)
        else "screened_no_findings"
    )
    report["coverage"]["geometry_inspected"] = sum(
        v["status"] == "observed" for v in cache.values()
    )
    report["interpretation"] += (
        "联合与递归只是假设敏感性分析；分别搜索当前H审查阈值和直接移除阈值，低阈值不会截断高阈值搜索。当前凸包附近距离使用H审查窗口；沿同一条假设排除链重查新暴露点，按H/D截止参数收敛，保留最终复核结果。端点同相替代参考不足时基准簇优先保留。条件H/D不能授权正式删除；原、组、联合、条件影响分开，不相加。"
    )
    from phase_agent.analysis.phase.phase_quality_disposition import annotate_dispositions

    annotate_dispositions(report, state, snapshot=snapshot, points=original)
    review_items = defaultdict(list)
    for row in report["structures"]:
        if row["review_pending"]:
            review_items[(row["phase"], row["x"])].append(row)
    report["review_items"] = [
        {
            "phase": phase,
            "x": x,
            "geometry_keys": [r["geometry_key"] for r in members],
            "H": _influence(original, original_hull, {r["geometry_key"] for r in members}),
            "D": max(
                (r.get("D_mev_per_O2") for r in members if r.get("D_mev_per_O2") is not None),
                default=None,
            ),
        }
        for (phase, x), members in review_items.items()
    ]
    return report
