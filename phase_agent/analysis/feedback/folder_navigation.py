"""Folder-level navigation: list all outputs without reading scientific data."""

from pathlib import Path
import re
from os.path import relpath
from urllib.parse import quote

START = "<!-- analysis-folder-navigation:start -->"
END = "<!-- analysis-folder-navigation:end -->"


def _link(path, base):
    relative = path.relative_to(base).as_posix()
    label = path.name.replace("[", "\\[").replace("]", "\\]")
    return f"[{label}](<{quote(relative, safe='/')}>)"


def _category(name):
    if name == "phase_diagram":
        return "相图与按顺序登记的分析步骤"
    if name == "dft" or name.startswith("Search-group-"):
        return "DFT：各批次结果、磁矩与累计误差"
    if name == "Round_Error":
        return "本轮模型与 DFT 同帧误差图"
    if name == "KFold_Error":
        return "微调训练 K 折误差图"
    if name == "training":
        return "微调训练：K折能量与受力对比"
    if name == "dft_error" or name == "round_metrics.csv":
        return "本模型DFT误差与轮次汇总"
    if name in {"phase_diagrams", "DFT_Selection", "Historical_Reports"}:
        return "历史兼容文件：保留原路径，不代表当前结果"
    if name in {"diagnostics", "direct_sampling", "extrapolation_review", "recovery"}:
        return "采样、外推与恢复证据"
    return "其他输出（全部保留）"


def _write_section(path, body):
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

    original = path.read_text(encoding="utf-8") if path.is_file() else ""
    if START in original:
        if original.count(START) != 1 or original.count(END) != 1:
            raise ValueError("ambiguous analysis navigation markers")
        before, tail = original.split(START, 1)
        _, after = tail.split(END, 1)
        text = before + START + "\n" + body + "\n" + END + after
    else:
        text = original.rstrip() + "\n\n" + START + "\n" + body + "\n" + END + "\n"
    _publish_bytes(path, text.lstrip().encode("utf-8"))


def publish_folder_navigation(directory):
    """Only enumerate two directory levels; do not move or classify scientific validity."""
    root = Path(directory)
    if not root.is_dir():
        return
    from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory

    from phase_agent.analysis.feedback.physical_layout import managed_alias

    metadata = Path(relpath(analysis_metadata_directory(root), root)).as_posix()
    lines = [
        "# 分析结果目录",
        "",
        "从 latest 查看当前结果及有效性状态；按 epoch 回看过程。文件存在不代表已通过科学验收。",
        "",
        "- 当前结果：[latest](latest/README.md)",
        f"- 全部登记产物：[完整索引](<{metadata}/output_index.md>)",
        f"- 跨轮汇总：[round_summary.csv](<{metadata}/round_summary.csv>)",
        "",
        "## 各轮与跨轮数据",
        "",
    ]
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if child.name in {"README.md", "latest"} or managed_alias(root, child):
            continue
        if child.is_symlink() or child.is_junction():
            lines.append(f"- {_link(child, root)}（链接，未展开）")
            continue
        if child.is_dir() and re.fullmatch(r"epoch\d+", child.name):
            lines.append(f"- [{child.name}](<{child.name}/README.md>)：本轮全部输出")
            sections = {}
            for entry in sorted(child.iterdir(), key=lambda p: p.name):
                if entry.name != "README.md" and not managed_alias(root, entry):
                    sections.setdefault(_category(entry.name), []).append(entry)
            detail = ["## 文件夹导航（不删减任何输出）", ""]
            for category, entries in sections.items():
                detail += [f"### {category}", ""]
                for entry in entries:
                    detail.append("- " + _link(entry, child))
                    if (
                        entry.is_dir()
                        and not entry.is_symlink()
                        and not entry.is_junction()
                        and (
                            entry.name in {"dft", "diagnostics", "phase_diagram", "dft_error"}
                            or entry.name.startswith("Search-group-")
                        )
                    ):
                        for sub in sorted(entry.iterdir(), key=lambda p: p.name):
                            if not managed_alias(root, sub):
                                detail.append("  - " + _link(sub, child))
                detail.append("")
            _write_section(child / "README.md", "\n".join(detail))
        else:
            purpose = (
                "跨轮当前表格与附属信息" if child.name == "out_data" else "保留的历史归档或其他输出"
            )
            lines.append(f"- {_link(child, root)}：{purpose}")
    lines += ["", "隐藏兼容入口只保留旧路径可读性，不是额外数据副本；实际历史文件集中在 history。"]
    _write_section(root / "README.md", "\n".join(lines))
    history = root / "history"
    if history.is_dir():
        entries = ["# 完整历史与内部数据", "", "所有文件保留；当前结果请查看 ../latest。", ""]
        for item in sorted(history.iterdir()):
            if item.name != "README.md":
                entries.append("- " + _link(item, history))
                if item.is_dir() and not item.is_symlink() and not item.is_junction():
                    for sub in sorted(item.iterdir()):
                        entries.append("  - " + _link(sub, history))
        _write_section(history / "README.md", "\n".join(entries))
