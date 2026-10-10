"""Build versioned SVG charts from validated project state, never from LLM text."""

from __future__ import annotations

from collections import Counter
from html import escape

from phase_agent.tools.step_runner.file_protocol import content_id


def build_webui_charts(state: dict) -> dict:
    datasets = {
        "convex_hull": _hull_points(state),
        "phase_soc_coverage": _coverage_points(state),
        "mlip_dft_error": _error_points(state),
        "tasks_and_budget": _task_budget_points(state),
    }
    version = content_id(
        {
            "config": state.get("confirmed_config_version"),
            "model": state.get("active_model_version"),
            "hull": state.get("active_hull_version"),
            "datasets": datasets,
        },
        "charts",
    )
    charts = {}
    for name, rows in datasets.items():
        charts[name] = {
            "name": name,
            "data_version": version,
            "data": rows,
            "svg": _svg(name, rows, version),
        }
    return {"data_version": version, "charts": charts}


def _hull_points(state):
    phase = state.get("phase_diagrams") or state.get("phase_diagram_state") or {}
    diagrams = phase.get("diagrams", phase) if isinstance(phase, dict) else {}
    rows = []
    for method in ("mlip", "dft"):
        for item in (diagrams.get(method) or {}).get("entries") or []:
            x = _composition_x(item.get("composition"))
            y = item.get("ehull")
            if x is not None and _number(y):
                rows.append(
                    {
                        "label": method,
                        "x": float(x),
                        "y": float(y),
                        "record_id": item.get("record_id"),
                    }
                )
    return rows


def _coverage_points(state):
    value = (
        state.get("coverage_summary") or state.get("coverage_report") or state.get("coverage") or {}
    )
    rows = []
    if isinstance(value, dict):
        source = value.get("cells") or value.get("by_phase_soc") or value
        if isinstance(source, dict):
            for label, item in source.items():
                amount = (
                    item.get("fraction", item.get("coverage")) if isinstance(item, dict) else item
                )
                if _number(amount):
                    rows.append({"label": str(label), "value": float(amount)})
        elif isinstance(source, list):
            value = source
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                amount = item.get("fraction", item.get("coverage"))
                if _number(amount):
                    rows.append(
                        {
                            "label": str(item.get("phase_soc") or item.get("label") or len(rows)),
                            "value": float(amount),
                        }
                    )
    return rows[:40]


def _error_points(state):
    validation = state.get("mlip_validation") or state.get("model_validation_summary") or {}
    rows = []
    candidates = validation.get("by_branch") if isinstance(validation, dict) else None
    if isinstance(candidates, dict):
        for branch, item in candidates.items():
            value = item.get("mae_eV_per_atom", item.get("mae")) if isinstance(item, dict) else item
            if _number(value):
                rows.append({"label": str(branch), "value": float(value)})
    for item in state.get("validation_history") or []:
        value = item.get("mae_eV_per_atom", item.get("mae")) if isinstance(item, dict) else None
        if _number(value):
            rows.append(
                {"label": str(item.get("model_version") or len(rows)), "value": float(value)}
            )
    return rows[-40:]


def _task_budget_points(state):
    counts = Counter(row.get("status", "unknown") for row in state.get("tasks") or [])
    rows = [
        {"label": f"task:{key}", "value": float(value)} for key, value in sorted(counts.items())
    ]
    used = (state.get("budget_usage") or {}).get("total_relative_cost")
    reserved = state.get("reserved_relative_cost")
    remaining = state.get("budget_remaining")
    for label, value in (
        ("budget:used", used),
        ("budget:reserved", reserved),
        ("budget:remaining", remaining),
    ):
        if _number(value):
            rows.append({"label": label, "value": float(value)})
    return rows


def _svg(name, rows, version):
    width, height, margin = 680, 260, 42
    title = name.replace("_", " ")
    if not rows:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="120" '
            f'viewBox="0 0 {width} 120"><rect width="100%" height="100%" fill="#fff"/>'
            f'<text x="20" y="35" font-family="sans-serif" font-size="18">{escape(title)}</text>'
            f'<text x="20" y="72" font-family="sans-serif" fill="#666">No validated data</text>'
            f'<text x="20" y="100" font-family="monospace" font-size="10">{escape(version)}</text></svg>'
        )
    bar_mode = not all("x" in row and "y" in row for row in rows)
    if bar_mode:
        maximum = max([abs(float(row.get("value", 0))) for row in rows] + [1.0])
        step = max(1, (width - 2 * margin) / max(1, len(rows)))
        shapes = []
        for index, row in enumerate(rows):
            value = float(row.get("value", 0))
            bar = (height - 2 * margin) * abs(value) / maximum
            x = margin + index * step
            y = height - margin - bar
            shapes.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(2, step * 0.72):.1f}" height="{bar:.1f}" fill="#3976af"><title>{escape(str(row.get("label")))}: {value:g}</title></rect>'
            )
    else:
        xs = [float(row["x"]) for row in rows]
        ys = [float(row["y"]) for row in rows]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmin == xmax:
            xmax = xmin + 1
        if ymin == ymax:
            ymax = ymin + 1
        shapes = []
        for row in rows:
            x = margin + (float(row["x"]) - xmin) / (xmax - xmin) * (width - 2 * margin)
            y = height - margin - (float(row["y"]) - ymin) / (ymax - ymin) * (height - 2 * margin)
            row_x, row_y = row.get("x"), row.get("y")
            shapes.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#3976af"><title>{escape(str(row.get("label")))} x={row_x} y={row_y}</title></circle>'
            )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="#fff"/><text x="20" y="25" font-family="sans-serif" font-size="18">{escape(title)}</text>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#888"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#888"/>',
        *shapes,
        f'<text x="{margin}" y="{height - 8}" font-family="monospace" font-size="9">{escape(version)}</text></svg>',
    ]
    return "".join(parts)


def _composition_x(value):
    if _number(value):
        return value
    if isinstance(value, dict):
        for key in ("x", "Na", "Li", "alkali_fraction"):
            if _number(value.get(key)):
                return value[key]
    return None


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)
