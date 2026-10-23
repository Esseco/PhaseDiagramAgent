"""Local phase diagram artifacts selected by the dialogue model."""

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from math import isfinite
from pathlib import Path

from phase_agent.analysis.phase.export_current_phase_diagram import export_current_phase_diagram
from phase_agent.analysis.phase.phase_snapshot_paths import phase_snapshot_directory
from phase_agent.analysis.phase.endpoint_reference import (
    endpoint_relative_energies,
    PLOT_RULE_VERSION,
)


def _current_snapshot(state, method):
    snapshot = (state.get("phase_diagrams") or {}).get(method) or {}
    from phase_agent.analysis.phase.phase_boundary import assert_snapshot_boundary

    assert_snapshot_boundary(state, snapshot)
    if (
        method in {"mlip", "combined"}
        and snapshot
        and (
            not state.get("active_model_version")
            or snapshot.get("model_version") != state["active_model_version"]
        )
    ):
        raise ValueError("已保存相图与当前模型版本不一致，请先回收并更新相图")
    return snapshot


def _location(state, snapshot, directory, method):
    stored = snapshot.get("csv_path") or snapshot.get("path")
    if stored:
        parent = Path(stored).resolve().parent
        return parent.parent if parent.name == "history" else parent
    if directory is None:
        raise ValueError("未配置相图输出目录，且没有已登记的相图文件")
    return (
        phase_snapshot_directory(
            directory, method, snapshot.get("model_version"), state=deepcopy(state)
        )
        .resolve()
        .parent
    )


def phase_diagram_reply(state, *, operation, method="mlip", directory=None, formats=("png",)):
    if operation not in {"plot", "locate", "export_csv"} or method not in {
        "mlip",
        "dft",
        "combined",
    }:
        raise ValueError("无效的相图对话操作")
    try:
        snapshot = (state.get("phase_diagrams") or {}).get(method) or {}
        if operation == "locate":
            folder = _location(state, snapshot, directory, method)
            lines = [f"{method.upper()} 相图文件夹：`{folder}`"]
            try:
                _current_snapshot(state, method)
            except ValueError as error:
                lines.append(str(error))
            if not folder.is_dir():
                lines.append("该目录尚未生成。")
            for key, label in (("csv_path", "CSV"), ("path", "快照")):
                if snapshot.get(key):
                    path = Path(snapshot[key]).resolve()
                    lines.append(f"{label}：`{path}`（{'存在' if path.is_file() else '尚不存在'}）")
            if not snapshot:
                lines.append("当前尚无已登记的相图数据。")
            if folder.is_dir():
                plots = sorted((folder / "plots").glob("phase_diagram_*.*"))
                if plots:
                    lines.append(f"已保存绘图目录：`{folder / 'plots'}`")
            return "\n".join(lines)
        if operation == "export_csv":
            path, count, version = export_current_phase_diagram(
                state, directory=directory, method=method
            )
            return f"当前 {method.upper()} 相图 CSV（版本 {version}，{count} 个结构）：`{path.resolve()}`"
        return _plot(state, _current_snapshot(state, method), method, directory, formats)
    except (OSError, ValueError, TypeError, ImportError) as error:
        return f"当前 {method.upper()} 相图{'未绘制' if operation == 'plot' else 'CSV未导出' if operation == 'export_csv' else '位置无法确认'}：{error}。"


def _plot(state, snapshot, method, directory, formats):
    if (
        isinstance(formats, str)
        or not formats
        or any(value not in {"png", "pdf", "svg"} for value in formats)
    ):
        raise ValueError("绘图格式仅支持非空的 png/pdf/svg 列表")
    formats = tuple(dict.fromkeys(formats))
    # Validate all scientific coordinates before creating any artifacts.
    points = []
    for entry in snapshot.get("entries") or []:
        composition = entry.get("composition") or {}
        oxygen = float(composition.get("O") or 0)
        sodium = float(composition.get("Na") or 0)
        energy = float(entry.get("eform_per_O2", "nan"))
        if oxygen <= 0 or sodium < 0 or not all(map(isfinite, (oxygen, sodium, energy))):
            raise ValueError("相图缺少有限的 Na/O 组成或 eform_per_O2 形成能，不能绘制")
        points.append((2 * sodium / oxygen, energy, entry))
    if not points:
        raise ValueError("尚无可绘制的已验证相图数据")
    from Process_Phasediagram import plot_phase_diagram
    from phase_agent.analysis.state.model_epoch import model_epoch, model_epoch_label
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # Export on a copy: publication must not modify workflow state or approvals.
    csv_path, count, version = export_current_phase_diagram(
        deepcopy(state), directory=directory, method=method
    )
    folder = csv_path.resolve().parent / "plots"
    figure = Figure(figsize=(6.5, 4.8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    relative, _ = endpoint_relative_energies(
        [x for x, _, _ in points], [1000 * energy for _, energy, _ in points]
    )
    plot_phase_diagram(
        [x for x, _, _ in points],
        relative,
        [row.get("phase") for _, _, row in points],
        [row.get("is_stable") for _, _, row in points],
        phase_hulls=True,
        unstable_alpha=0.65,
        ylabel="Formation energy (meV/O2)",
        ax=axes,
    )
    model_version = snapshot.get("model_version") or state.get("active_model_version")
    epoch_label = model_epoch_label(state, model_version)
    epoch = model_epoch(state, model_version) or "epoch unrecorded"
    axes.set_title(f"{method.upper()} | {epoch} | {version}")
    folder.mkdir(parents=True, exist_ok=True)
    name = (
        "phase_diagram_"
        + sha256(f"{version}:{PLOT_RULE_VERSION}".encode()).hexdigest()[:16]
        + "_phase_hulls"
    )
    paths = []
    for extension in formats:
        path = folder / f"{name}.{extension}"
        buffer = BytesIO()
        figure.savefig(buffer, format=extension, dpi=300)
        path.write_bytes(buffer.getvalue())
        paths.append(path)
    from phase_agent.analysis.phase.phase_quality_reports import quality_warning

    diagnostic_snapshot = {**snapshot, "csv_path": str(csv_path)}
    warning = quality_warning(state, method=method, snapshot=diagnostic_snapshot)
    comparison_note = ""
    if method == "mlip":
        from phase_agent.analysis.phase.phase_quality_reports import get_phase_quality_report
        from phase_agent.analysis.phase.phase_quality_comparison import publish_quality_comparison

        try:
            report = get_phase_quality_report(state, snapshot=diagnostic_snapshot, write=True)
            compared = publish_quality_comparison(state, report, directory=folder, formats=formats)
            if compared["status"] == "completed":
                comparison_note = (
                    "\n左右对照（右图第一/第二档假设排除并打×，第二档正式保留）：\n"
                    + "\n".join(f"`{p}`" for p in compared["plot_paths"])
                )
            else:
                comparison_note = "\n对照图未生成：" + compared["reason"]
        except (OSError, ValueError, TypeError, KeyError) as error:
            comparison_note = "\n对照图核查未完成：" + str(error)
    return (
        f"已绘制当前 {method.upper()} 相图（版本 {version}，{count} 个结构；分相已开启）。\n"
        + f"数据所属模型：{epoch_label}。绘图：Process_Phasediagram（phase_hulls=True）。\n"
        + "形成能按本图保留集合的端点归一化；H使用固定原始参考。\n"
        + f"文件夹：`{folder}`\n"
        + "\n".join(f"{path.suffix[1:].upper()}：`{path}`" for path in paths)
        + f"\n数据 CSV：`{csv_path.resolve()}`\n"
        + warning
        + comparison_note
    )
