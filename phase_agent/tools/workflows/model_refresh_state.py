"""Build immutable refresh previews from registered, on-disk structures."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from phase_agent.analysis.feedback.model_refresh_plan import build_model_refresh_plan
from phase_agent.tools.budget.estimate_stage_cost import estimate_stage_cost


def _hd_reference(snapshot, report):
    from math import isfinite
    from phase_agent.analysis.phase.phase_quality import _hull, _at, snapshot_evidence_digest

    if (
        report.get("hull_version") != snapshot.get("version")
        or report.get("model_version") != snapshot.get("model_version")
        or report.get("snapshot_evidence_digest") != snapshot_evidence_digest(snapshot)
    ):
        raise ValueError("refresh_HD_report_snapshot_mismatch")
    limits = report["screening_policy"]
    from phase_agent.analysis.phase.phase_quality_preview import build_quality_preview

    excluded_keys = set(report.get("preview_excluded_geometry_keys") or [])
    selected = [r for r in report.get("structures") or [] if r.get("geometry_key") in excluded_keys]
    preview = build_quality_preview(snapshot, report, review_ids=[r["review_id"] for r in selected])
    high = set(preview["excluded_record_ids"])
    for row in report.get("structures") or []:
        if row.get("analysis_layer", 0) != 0:
            continue
        h, d = row.get("hull_lift_mev_per_O2"), row.get("D_mev_per_O2")
        if (
            isinstance(h, (float, int))
            and not isinstance(h, bool)
            and isfinite(h)
            and h > limits["hull_shift_mev_per_O2"]
        ) or (
            isinstance(d, (float, int))
            and not isinstance(d, bool)
            and isfinite(d)
            and d > limits["outlier_gap_mev_per_O2"]
        ):
            high.update(row.get("record_ids") or [])
    entries = snapshot.get("entries") or []
    removed = set(preview["excluded_record_ids"])
    remaining = [row for row in entries if row.get("record_id") not in removed]
    if not entries or not remaining:
        raise ValueError("refresh_HD_reference_empty")
    points = [(float(r["x_Na_per_O2"]), float(r["energy_per_O2"])) for r in remaining]
    hull = _hull(points)
    result = {}
    for row in entries:
        if row.get("record_id") in high:
            continue
        baseline = _at(hull, float(row["x_Na_per_O2"]))
        total_atoms = sum(float(v) for v in row["composition"].values())
        oxygen = float(row["composition"].get("O", 0))
        if baseline is None or total_atoms <= 0 or oxygen <= 0:
            raise ValueError("refresh_HD_reference_invalid_composition")
        result[row["record_id"]] = max(
            0.0, (float(row["energy_per_O2"]) - baseline) * oxygen / (2 * total_atoms)
        )
    return high, result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def refresh_preview(state, manager, config):
    refresh = state["model_refresh"]
    old = refresh["old_model_version"]
    candidates = []
    represented = set()
    structures = manager.data.get("structures") or {}

    def append(entry, *, historical=False, mixed=False, hd_required=False):
        sid = entry.get("structure_id")
        if sid not in structures:
            raise ValueError(f"刷新结构未登记：{sid}")
        path = Path(entry.get("structure_path") or structures[sid].get("source_path") or "")
        if not path.is_file():
            raise ValueError(f"刷新结构文件不存在：{sid}: {path}")
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if entry.get("ehull") is not None:
            if entry.get("ehull_unit") != "eV/atom":
                raise ValueError(f"刷新Ehull缺少明确的eV/atom单位：{sid}")
            if not historical and not mixed and entry.get("source_version") not in {None, old}:
                raise ValueError(f"刷新Ehull来源模型不一致：{sid}")
            expected = entry.get("structure_sha256")
            if expected and expected != checksum:
                raise ValueError(f"旧相图结构内容已改变，需要先核对能量证据：{sid}")
        record = structures[sid]
        composition = entry.get("composition") or record.get("composition") or {}
        atom_count = sum(composition.values())
        if not atom_count:
            raise ValueError(f"刷新结构缺少组成：{sid}")
        oxygen = float(composition.get("O", 0))
        if oxygen <= 0:
            raise ValueError(f"刷新结构不是含氧体系：{sid}")
        if historical and any(r["structure_sha256"] == checksum for r in candidates):
            return  # A more recent old-model assessment already revisited this geometry.
        candidates.append(
            {
                "structure_id": sid,
                "branch_id": record["branch_id"],
                "structure_path": str(path.resolve()),
                "structure_sha256": checksum,
                "atom_count": atom_count,
                "phase": entry.get("phase"),
                "x_Na_per_O2": float(composition.get("Na", 0)) * 2 / oxygen,
                "ehull": None if historical else entry.get("ehull"),
                "source_version": old,
                "energy_source_version": entry.get("source_version") if mixed else old,
                "historical_near_hull": historical,
                "hd_required": hd_required,
            }
        )
        represented.add(sid)

    diagram = refresh.get("old_diagram") or {}
    if diagram.get("model_version") not in {None, old}:
        raise ValueError("刷新旧相图版本不一致")
    latest_policy = (
        state.get("mlip_energy_policy") or (config.get("model_refresh") or {}).get("energy_policy")
    ) == "latest_valid_per_structure_v1"
    if latest_policy:
        from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram

        if not state.get("phase_records"):
            raise ValueError("新 epoch 刷新缺少可核验的历史相图记录；不能用单模型或空数据替代")
        latest = update_phase_diagram(
            state["phase_records"],
            active_model_version=old,
            output_state=state,
            latest_per_structure=True,
            latest_only=True,
        )["diagrams"]["mlip"]
        if latest.get("status") != "completed":
            raise ValueError(
                "最新结构能量谱系未解决，不能生成刷新预览："
                + str(latest.get("issues") or latest.get("reason"))
            )
        from phase_agent.analysis.phase.phase_quality_reports import get_phase_quality_report

        review_state = {
            **state,
            "active_model_version": old,
            "phase_diagrams": {**(state.get("phase_diagrams") or {}), "mlip": latest},
        }
        report = get_phase_quality_report(review_state, snapshot=latest)
        high, reference_ehull = _hd_reference(latest, report)
        for entry in latest.get("entries") or []:
            is_high = entry.get("record_id") in high
            selected_entry = (
                entry
                if is_high
                else {
                    **entry,
                    "ehull": reference_ehull[entry["record_id"]],
                    "ehull_unit": "eV/atom",
                }
            )
            append(selected_entry, mixed=True, hd_required=is_high)
    else:
        for entry in diagram.get("entries") or []:
            append(entry)
        for version, prior in (state.get("phase_diagrams_by_model") or {}).items():
            if version in {old, refresh["new_model_version"]}:
                continue
            for entry in prior.get("entries") or []:
                if entry.get("ehull") is not None and float(entry["ehull"]) <= 0.010:
                    append(entry, historical=True)
    missing = sorted(set(structures) - represented)
    for sid in missing:
        append({"structure_id": sid})
    plan = build_model_refresh_plan(
        candidates, old_version=old, new_version=refresh["new_model_version"]
    )
    if latest_policy:
        plan["energy_policy"] = "latest_valid_per_structure_v1"
        plan["energy_source_models"] = latest.get("source_model_versions", [])
        plan["mixed_model_versions"] = latest.get("mixed_model_versions", False)
        plan["hd_refresh_reference"] = {
            "hull_version": latest["version"],
            "report_digest": report["snapshot_evidence_digest"],
            "thresholds": report["screening_policy"],
            "high_record_ids": sorted(high),
            "reference_scope": "counterfactual_selection_only",
        }
    if not plan["candidates"]:
        raise ValueError("累计结构池为空，不能生成刷新方案")
    from phase_agent.analysis.cost.calibrate_relative_cost import calibrate_relative_cost

    factor, samples = calibrate_relative_cost(state, "relax_and_feature")
    for row in plan["candidates"]:
        # Relax is an explicit conservative upper bound until single-point
        # observations are available. It is not presented as a measured runtime.
        row["relative_cost"] = (
            estimate_stage_cost(
                "relax_and_feature", atom_count=row["atom_count"], budgets=config["budgets"]
            )["value"]
            * factor
        )
    plan["cost_calibration"] = {"factor": factor, "measured_samples": samples}
    from phase_agent.analysis.cost.predict_runtime import predict_runtime

    timing = [
        predict_runtime("relax_and_feature", atom_count=r["atom_count"], state=state)
        for r in plan["candidates"]
    ]
    plan["runtime_budget"] = {
        "status": "historical_relax_upper_bound"
        if timing and all(t.get("elapsed_seconds") is not None for t in timing)
        else "unavailable",
        "serial_seconds": sum(t["elapsed_seconds"] for t in timing)
        if timing and all(t.get("elapsed_seconds") is not None for t in timing)
        else None,
        "note": "弛豫同类实测的规模粗估；单点暂用弛豫上界，不含排队，不是并行完成时间",
    }
    plan["initial_cost"] = sum(r["relative_cost"] for r in plan["candidates"])
    plan["maximum_supplemental_cost"] = (
        0.0
        if (plan.get("policy") or {}).get("force_followup_version") == 2
        else sum(r["relative_cost"] for r in plan["candidates"] if r["operation"] == "predict")
    )
    plan["maximum_total_cost"] = plan["initial_cost"] + plan["maximum_supplemental_cost"]
    plan["cost_basis"] = "size_aware_relax_upper_bound_for_single_point; runtime_unmeasured"
    plan["model"] = deepcopy(state.get("active_model") or config.get("mlip") or {})
    if plan["model"].get("version") != refresh["new_model_version"]:
        raise ValueError("刷新模型路径与激活版本不一致")
    plan["relax_parameters"] = deepcopy((config.get("mlip") or {}).get("relax_parameters") or {})
    plan["checksum"] = fingerprint(plan)
    return plan


def current_model_refresh(state):
    """Return refresh facts only when they belong to the active model epoch."""
    refresh = state.get("model_refresh") or {}
    active = state.get("active_model_version")
    return refresh if active and refresh.get("new_model_version") == active else {}


def refresh_active(state):
    return bool(state.get("model_refresh") and state["model_refresh"].get("status") != "completed")
