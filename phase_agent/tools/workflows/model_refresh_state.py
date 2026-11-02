"""Build immutable refresh previews from registered, on-disk structures."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from phase_agent.analysis.feedback.model_refresh_plan import build_model_refresh_plan
from phase_agent.tools.budget.estimate_stage_cost import estimate_stage_cost


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def refresh_preview(state, manager, config):
    refresh = state["model_refresh"]
    old = refresh["old_model_version"]
    candidates = []
    represented = set()
    structures = manager.data.get("structures") or {}

    def append(entry, *, historical=False, mixed=False):
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
            }
        )
        represented.add(sid)

    diagram = refresh.get("old_diagram") or {}
    if diagram.get("model_version") not in {None, old}:
        raise ValueError("刷新旧相图版本不一致")
    latest_policy = (config.get("model_refresh") or {}).get(
        "energy_policy"
    ) == "latest_valid_per_structure_v1"
    if latest_policy:
        from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram

        latest = update_phase_diagram(
            state["phase_records"],
            active_model_version=old,
            output_state=state,
            latest_per_structure=True,
        )["diagrams"]["mlip"]
        if latest.get("status") != "completed":
            raise ValueError(
                "最新结构能量谱系未解决，不能生成刷新预览："
                + str(latest.get("issues") or latest.get("reason"))
            )
        for entry in latest.get("entries") or []:
            append(entry, mixed=True)
    else:
        for entry in diagram.get("entries") or []:
            append(entry)
        for version, prior in (state.get("phase_diagrams_by_model") or {}).items():
            if version in {old, refresh["new_model_version"]}:
                continue
            for entry in prior.get("entries") or []:
                if entry.get("ehull") is not None and float(entry["ehull"]) <= 0.010:
                    append(entry, historical=True)
    for sid in sorted(set(structures) - represented):
        append({"structure_id": sid})
    plan = build_model_refresh_plan(
        candidates, old_version=old, new_version=refresh["new_model_version"]
    )
    if latest_policy:
        plan["energy_policy"] = "latest_valid_per_structure_v1"
        plan["energy_source_models"] = latest.get("source_model_versions", [])
        plan["mixed_model_versions"] = latest.get("mixed_model_versions", False)
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
