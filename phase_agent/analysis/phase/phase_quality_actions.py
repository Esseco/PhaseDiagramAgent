"""Backed-up exclusions and explicit evidence-bound human retain/remove decisions."""

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil

from phase_agent.analysis.phase.phase_quality_reports import get_phase_quality_report, _digest
from phase_agent.analysis.phase.phase_quality_review import exclude_records
from phase_agent.tools.step_runner.file_protocol import write_json


def backup_selection(state, manager, record_ids, structure_ids, root, *, source, reason):
    """Back up only selected records and files; keep raw task products in place."""
    records = [
        deepcopy(r) for r in state.get("phase_records") or [] if r.get("record_id") in record_ids
    ]
    structures = {sid: deepcopy(manager.data["structures"][sid]) for sid in structure_ids}
    identity = {
        "model": state.get("active_model_version"),
        "record_ids": sorted(record_ids),
        "source": source,
        "reason": reason,
    }
    folder = Path(root) / ("quality-" + _digest(identity)[:20])
    manifest_path = folder / "audit.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("identity") != identity:
            raise ValueError("备份身份冲突")
        for item in previous["files"]:
            if item.get("status") == "unavailable":
                continue
            if sha256(Path(item["backup_path"]).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("已有备份校验失败，禁止排除")
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    task_ids = {r.get("source_task_id") for r in records} - {None}
    initial_paths = {
        t.get("structure_path") or (t.get("inputs") or {}).get("structure_path")
        for t in state.get("tasks") or []
        if t.get("task_id") in task_ids
    }
    paths = (
        initial_paths
        | {r.get("structure_path") for r in records}
        | {r.get("source_path") for r in structures.values()}
    )
    for raw in sorted(paths - {None, ""}):
        path = Path(raw)
        if not path.is_file():
            files.append({"original_path": str(path), "status": "unavailable"})
            continue
        digest = sha256(path.read_bytes()).hexdigest()
        if source == "auto_remove" and any(
            r.get("structure_path") == raw and r.get("structure_sha256") != digest for r in records
        ):
            raise ValueError("自动排除前最终结构已改变，需重新检查")
        target = folder / "structures" / (digest + path.suffix)
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        if sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("结构备份校验失败")
        files.append({"original_path": str(path), "backup_path": str(target), "sha256": digest})
    write_json(
        manifest_path,
        {
            "identity": identity,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "records": records,
            "structures": structures,
            "branches": {
                s["branch_id"]: deepcopy(manager.data["branches"][s["branch_id"]])
                for s in structures.values()
            },
            "files": files,
        },
    )
    return folder


def apply_quality_review(
    state,
    manager,
    report,
    *,
    review_ids,
    decision,
    reason,
    backup_root=None,
    directory=None,
    human_requested=False,
):
    """Human may retain uncertain results; auto removal requires a recomputed policy rule."""
    if decision not in {"retain", "remove", "auto_remove", "restore"} or not str(reason).strip():
        raise ValueError("需要有效处理决定及原因")
    if decision != "auto_remove" and human_requested is not True:
        raise ValueError("人工处理需要明确用户指令")
    if decision == "restore":
        from phase_agent.analysis.phase.phase_quality_restore import restore_quality_records

        return restore_quality_records(
            state,
            manager,
            review_ids=review_ids,
            reason=reason,
            backup_root=backup_root,
            directory=directory,
            human_requested=human_requested,
        )
    actual = get_phase_quality_report(state)
    if (
        report.get("input_fingerprint") != actual["input_fingerprint"]
        or report.get("structures") != actual["structures"]
    ):
        raise ValueError("相图或结构证据已改变，请重新审查")
    requested = set(review_ids)
    selected = [r for r in actual["structures"] if r["review_id"] in requested]
    if not requested or len(selected) != len(requested):
        raise ValueError("审查身份无效或已过期")
    if decision == "auto_remove" and not all(r["automatic_exclusion_eligible"] for r in selected):
        raise ValueError(
            "未满足当前H直接移除阈值，或H/D超阈值且公共结构模块严格确认异常，不能自动排除"
        )
    if decision == "auto_remove":
        keys = {r["geometry_key"] for r in selected}
        covered = {
            r["geometry_key"]
            for r in selected
            if r.get("automatic_exclusion_reason") == "single_impact_above_100"
            or (
                (r["geometry"].get("framework") or {}).get("auto_confirmed") is True
                and (r.get("D_mev_per_O2") or 0)
                > actual["screening_policy"]["outlier_gap_mev_per_O2"]
                and r.get("near_current_hull")
            )
        }
        for case in actual["analysis_layers"][0]["joint_cases"]["cases"] + actual.get(
            "energy_islands", []
        ):
            case_keys = set(case["geometry_keys"])
            lift = case.get("hull_lift_mev_per_O2")
            if (
                case_keys <= keys
                and lift is not None
                and lift + 1e-7 >= actual["screening_policy"]["hull_shift_mev_per_O2"]
            ):
                covered.update(case_keys)
        if covered != keys:
            raise ValueError("不能拆开高影响联合集合执行自动排除")
    if decision == "retain":
        if any(r.get("automatic_exclusion_reason") == "single_impact_above_100" for r in selected):
            raise ValueError("单结构H超过当前直接移除阈值，不能暂不处理；可明确人工恢复")
        current = deepcopy(state)
        for row in selected:
            current.setdefault("phase_quality_decisions", []).append(
                {
                    "review_id": row["review_id"],
                    "decision": "retain",
                    "reason": reason,
                    "model_version": report["model_version"],
                    "hull_version": report["hull_version"],
                    "record_ids": row["record_ids"],
                    "geometry_key": row["geometry_key"],
                    "screening_policy": deepcopy(report["screening_policy"]),
                    "human_requested": True,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        return {
            "status": "retained",
            "state": current,
            "continue_dft_selection": True,
            "excluded_record_ids": [],
            "submitted": False,
        }
    if backup_root is None:
        raise ValueError("排除前必须配置备份目录")
    record_ids = {record_id for r in selected for record_id in r["record_ids"]}
    model = state.get("active_model_version")
    records = [
        r
        for r in state.get("phase_records") or []
        if r.get("record_id") in record_ids
        and r.get("status", "completed") == "completed"
        and r.get("energy_method") == "mlip"
        and (r.get("model_version") or r.get("source_version")) == model
    ]
    if {r.get("record_id") for r in records} != record_ids:
        raise ValueError("所选结构无法精确绑定当前模型的有效MLIP记录")
    structure_ids = {r["structure_id"] for r in records if r.get("structure_id")}
    folder = backup_selection(
        state, manager, record_ids, structure_ids, backup_root, source=decision, reason=reason
    )
    return exclude_records(
        state,
        manager,
        record_ids,
        structure_ids,
        directory=directory,
        entry={
            "model_version": model,
            "hull_version": report["hull_version"],
            "screening_policy": deepcopy(report["screening_policy"]),
            "policy_revision": report.get("policy_control", {}).get("revision", 0),
            "reason": reason,
            "source": decision,
            "human_requested": human_requested,
            "review_ids": sorted(requested),
            "backup_path": str(folder),
            "evidence": [
                {
                    k: deepcopy(r.get(k))
                    for k in (
                        "geometry_key",
                        "record_ids",
                        "structure_ids",
                        "structure_paths",
                        "automatic_exclusion_reason",
                        "framework_status",
                        "geometry",
                        "hull_lift_mev_per_O2",
                        "group_hull_lift_mev_per_O2",
                        "H_mev_per_O2",
                        "D_mev_per_O2",
                        "tier",
                        "review_id",
                        "review_number",
                    )
                }
                for r in selected
            ],
        },
    )


def automatic_round_review(state, manager, *, bundle_id, backup_root, directory=None):
    """Repeat current-evidence actions to cutoff convergence; never act on conditional H/D."""
    from phase_agent.analysis.phase.phase_quality import RULE_VERSION

    batch = (state.get("relax_mc_batches") or {}).get(bundle_id) or {}
    summary = batch.get("result_summary") or {}
    model = state.get("active_model_version")
    snapshot = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if (
        batch.get("status") != "completed"
        or not batch.get("import_checksum")
        or batch.get("model_version") != model
        or summary.get("status") != "completed"
        or summary.get("model_version") != model
        or (batch.get("plan_checksum") and summary.get("plan_checksum") != batch["plan_checksum"])
        or snapshot.get("status") != "completed"
        or snapshot.get("model_version") != model
        or any(
            t.get("relax_mc_bundle_id") == bundle_id
            and t.get("status") not in {"completed", "failed", "timeout", "cancelled"}
            for t in state.get("tasks") or []
        )
    ):
        return {"status": "not_ready", "state": state, "submitted": False}
    identity = _digest(
        {
            "bundle": bundle_id,
            "import": batch["import_checksum"],
            "model": state.get("active_model_version"),
            "rule": RULE_VERSION,
            "policy": state.get("phase_quality_policy") or {},
        }
    )
    if identity in (state.get("phase_quality_round_actions") or {}):
        return {"status": "already_reviewed", "state": state, "submitted": False}
    shadow = deepcopy(manager)
    current, removed = state, []
    for _ in range(len(snapshot.get("entries", [])) + 1):
        report = get_phase_quality_report(current, write=True)
        ids = [r["review_id"] for r in report["structures"] if r["automatic_exclusion_eligible"]]
        if not ids:
            break
        applied = apply_quality_review(
            current,
            shadow,
            report,
            review_ids=ids,
            decision="auto_remove",
            reason="自动处置第一档：当前H超过直接移除阈值，或H/D超过审查阈值且公共结构模块确认异常；数值、阈值和结构依据见审计",
            backup_root=backup_root,
            directory=directory,
        )
        current = applied["state"]
        if not applied["excluded_record_ids"] or set(removed).intersection(
            applied["excluded_record_ids"]
        ):
            raise ValueError("自动审查未产生新增排除，停止以避免重复执行")
        removed.extend(applied["excluded_record_ids"])
    current = deepcopy(current)
    current.setdefault("phase_quality_round_actions", {})[identity] = {
        "bundle_id": bundle_id,
        "rule_version": RULE_VERSION,
        "excluded_record_ids": removed,
        "final_hull_version": current["phase_diagrams"]["mlip"]["version"],
    }
    manager.data = shadow.data
    return {
        "status": "completed",
        "state": current,
        "excluded_record_ids": removed,
        "submitted": False,
    }
