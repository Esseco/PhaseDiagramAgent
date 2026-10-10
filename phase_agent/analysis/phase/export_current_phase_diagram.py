"""Export the saved current hull without advancing the search workflow."""

from pathlib import Path

from phase_agent.analysis.phase.export_phase_diagram_csv import export_phase_diagram_csv
from phase_agent.analysis.phase.phase_snapshot_paths import phase_snapshot_directory


def export_current_phase_diagram(state, *, directory=None, method="mlip"):
    if method not in {"mlip", "dft", "combined"}:
        raise ValueError("相图方法只能是 mlip、dft 或 combined")
    snapshot = (state.get("phase_diagrams") or {}).get(method) or {}
    if method == "combined":
        if snapshot.get("model_version") != state.get("active_model_version"):
            raise ValueError("综合相图与当前模型不一致，请由 Agent 更新")
        path = snapshot.get("csv_path")
        if snapshot.get("status") in {"completed", "partial"} and path and Path(path).is_file():
            return Path(path), len(snapshot.get("entries") or []), snapshot["version"]
        raise ValueError("综合相图尚无已保存 CSV，请先由 Agent 回收并分析；导出命令不计算校正")
    if snapshot.get("status") != "completed" or not snapshot.get("entries"):
        raise ValueError(f"当前 {method.upper()} 相图尚无可导出的完整数据")
    if any(
        "eform_per_O2" not in row
        or "ehull" not in row
        or not row.get("phase")
        or row.get("phase_identification_status") != "identified"
        for row in snapshot["entries"]
    ):
        raise ValueError("当前相图是旧格式或未完成相识别，请先回收结果并更新相图")
    if method == "mlip" and snapshot.get("model_version") != state.get("active_model_version"):
        raise ValueError("当前 MLIP 模型版本与相图版本不一致，请先更新相图")
    version = snapshot.get("version")
    if not version:
        raise ValueError("当前相图缺少版本号")
    stored_path = snapshot.get("csv_path")
    if stored_path and Path(stored_path).is_file():
        return Path(stored_path), len(snapshot["entries"]), version
    archive_path = snapshot.get("archive_csv_path")
    if archive_path and Path(archive_path).is_file():
        from phase_agent.analysis.phase.publish_current_csv import publish_current_csv

        current = publish_current_csv(snapshot, archive_path)
        return current, len(snapshot["entries"]), version
    if directory is None:
        if not snapshot.get("path"):
            raise ValueError("未配置相图输出目录")
        parent = Path(snapshot["path"]).parent
        if parent.name == "history":
            directory = (
                parent.parent.parent.parent
                if parent.parent.name == "phase_diagrams"
                else parent.parent.parent
            )
        elif parent.name == "dft":
            directory = parent.parent
        elif parent.parent.name == "mlip":
            directory = parent.parent.parent
        elif parent.parent.name == "phase_diagrams":
            directory = parent.parent
        else:
            directory = parent
    target_dir = phase_snapshot_directory(
        directory, method, snapshot.get("model_version"), state=state
    )
    path = target_dir / f"phase_diagram_{method}_{version}.csv"
    if not path.is_file():
        export_phase_diagram_csv(snapshot, path)
    from phase_agent.analysis.phase.publish_current_csv import publish_current_csv

    current = publish_current_csv(snapshot, path)
    return current, len(snapshot["entries"]), version
