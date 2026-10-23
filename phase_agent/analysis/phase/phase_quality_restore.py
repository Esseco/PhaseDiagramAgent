"""Explicit human restoration of exact, backed-up exclusions; no silent undo."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil


def restored_records(state):
    rows = {r.get("record_id"): r for r in (state or {}).get("phase_records") or []}
    protected = set()
    for event in (state or {}).get("phase_quality_restorations") or []:
        for record_id, digest in event["sha_by_record"].items():
            row = rows.get(record_id) or {}
            model = row.get("model_version") or row.get("source_version")
            if model == event["model_version"] and row.get("structure_sha256") == digest:
                protected.add((model, record_id))
    return protected


def restore_quality_records(
    state, manager, *, review_ids, reason, backup_root, directory=None, human_requested=False
):
    from phase_agent.analysis.phase.phase_boundary import (
        phase_boundary_rule,
        outside_phase_boundary,
    )
    from phase_agent.analysis.phase.phase_quality_actions import backup_selection
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram

    if human_requested is not True or not str(reason).strip() or backup_root is None:
        raise ValueError("恢复需要明确人工指令、原因和备份目录")
    requested, selected = set(review_ids), {}
    model = state.get("active_model_version")
    for review_id in requested:
        entry = next(
            (
                e
                for e in reversed(state.get("phase_quality_exclusions") or [])
                if review_id in (e.get("review_ids") or [])
            ),
            None,
        )
        if not entry or entry.get("model_version") != model or not entry.get("backup_path"):
            raise ValueError("恢复编号不属于当前模型或缺少审计备份")
        evidence = next(
            (e for e in entry.get("evidence") or [] if e.get("review_id") == review_id), {}
        )
        ids = evidence.get("record_ids")
        if not ids:
            if (
                len(entry.get("review_ids") or []) != 1
                and not set(entry["review_ids"]) <= requested
            ):
                raise ValueError("历史编号无法精确映射记录，禁止扩大恢复范围")
            ids = entry["record_ids"]
        manifest = json.loads(
            (Path(entry["backup_path"]) / "audit.json").read_text(encoding="utf-8")
        )
        originals = {r["record_id"]: r for r in manifest["records"]}
        for record_id in ids:
            if record_id not in originals:
                raise ValueError("备份缺少待恢复记录")
            selected[record_id] = (originals[record_id], manifest)
    if not selected:
        raise ValueError("未选择恢复项")
    current_rows = {
        r["record_id"]: r
        for r in (state.get("phase_quality_archived_records") or [])
        + (state.get("phase_records") or [])
    }
    rule, files = phase_boundary_rule(state), []
    structure_ids = set()
    for record_id, (original, manifest) in selected.items():
        row = current_rows.get(record_id)
        if not row or not str(row.get("status", "")).startswith("excluded_"):
            raise ValueError("记录已恢复或不再处于排除状态")
        if (row.get("model_version") or row.get("source_version")) != model:
            raise ValueError("不能恢复其他模型的排除结果")
        ignored = {"status", "phase_quality_review"}
        if {k: v for k, v in row.items() if k not in ignored} != {
            k: v for k, v in original.items() if k not in ignored
        }:
            raise ValueError("记录科学内容已变化，不能用旧备份覆盖")
        sid = row.get("structure_id")
        structure = manager.data["structures"].get(sid) or manifest["structures"].get(sid)
        if not structure:
            raise ValueError("历史结构登记缺失，需先人工核对")
        branch = manager.data["branches"][structure["branch_id"]]
        if outside_phase_boundary(row, rule) or outside_phase_boundary(
            {
                "composition": structure.get("composition") or row["composition"],
                "phase": branch["P"],
            },
            rule,
        ):
            raise ValueError("恢复会违反当前相/组分边界，请先明确修改边界")
        structure_ids.add(sid)
        digest, raw = row.get("structure_sha256"), row.get("structure_path")
        if isinstance(digest, str) and len(digest) == 64:
            target = Path(raw) if raw else None
            if target and target.is_file():
                if sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError("结构文件已改变，不能恢复该能量记录")
            else:
                item = next(
                    (
                        f
                        for f in manifest["files"]
                        if f.get("original_path") == raw and f.get("sha256") == digest
                    ),
                    None,
                )
                if (
                    not target
                    or not item
                    or sha256(Path(item["backup_path"]).read_bytes()).hexdigest() != digest
                ):
                    raise ValueError("缺少可验证的结构备份")
                storage = (state.get("confirmed_config") or {}).get("storage") or {}
                project = Path(storage.get("workspace_root") or Path(backup_root).parent).resolve()
                if not target.resolve().is_relative_to(project):
                    raise ValueError("不能自动写回项目外的结构路径")
                files.append((Path(item["backup_path"]), target))
    backup_manager = deepcopy(manager)
    for record_id, (_, manifest) in selected.items():
        sid = current_rows[record_id]["structure_id"]
        backup_manager.data["structures"].setdefault(sid, deepcopy(manifest["structures"][sid]))
    backup_state = deepcopy(state)
    backup_state["phase_records"] = list(current_rows.values())
    backup = backup_selection(
        backup_state,
        backup_manager,
        set(selected),
        structure_ids,
        backup_root,
        source="human_restore",
        reason=reason,
    )
    for source, target in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    current, shadow = deepcopy(state), deepcopy(manager)
    event = {
        "model_version": model,
        "review_ids": sorted(requested),
        "record_ids": sorted(selected),
        "sha_by_record": {key: value[0].get("structure_sha256") for key, value in selected.items()},
        "reason": reason,
        "human_requested": True,
        "backup_path": str(backup),
        "instruction": "人工恢复覆盖同一模型/记录/几何的自动移除；新模型或几何改变后重新评估",
    }
    current.setdefault("phase_quality_restorations", []).append(event)
    current["phase_records"] = [
        r for r in current.get("phase_records", []) if r["record_id"] not in selected
    ] + [deepcopy(current_rows[rid]) for rid in selected]
    current["phase_quality_archived_records"] = [
        r
        for r in current.get("phase_quality_archived_records", [])
        if r["record_id"] not in selected
    ]
    for row in current["phase_records"]:
        if row["record_id"] in selected:
            row["status"] = "completed"
            row["phase_quality_review"] = {"disposition": "human_restored", "reason": reason}
    for sid in structure_ids:
        if sid not in shadow.data["structures"]:
            shadow.data["structures"][sid] = deepcopy(backup_manager.data["structures"][sid])
        structure = shadow.data["structures"][sid]
        branch = shadow.data["branches"][structure["branch_id"]]
        branch["structure_ids"] = sorted(set(branch["structure_ids"]) | {sid})
    previous = (current.get("phase_diagrams") or {}).get("mlip") or {}
    updated = update_phase_diagram(
        current["phase_records"],
        active_model_version=model,
        output_state=current,
        output_directory=directory,
        parent_versions={"mlip": previous.get("version")},
    )
    current.setdefault("phase_diagrams", {}).update(updated["diagrams"])
    current.setdefault("phase_diagrams_by_model", {}).update(updated["mlip_by_version"])
    current["current_branch_hull_by_model"], current["current_branch_hull_version"] = {}, None
    current.pop("current_state_snapshot", None)
    current.pop("agent_state_summary", None)
    current["coverage"] = shadow.coverage()
    shadow.data.setdefault("phase_quality_restoration_log", []).append(deepcopy(event))
    manager.data = shadow.data
    return {
        "status": "restored",
        "state": current,
        "restored_record_ids": sorted(selected),
        "continue_dft_selection": True,
        "submitted": False,
    }
