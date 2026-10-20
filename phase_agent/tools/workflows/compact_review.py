"""Small human inventories projected from existing signed scientific records."""

from collections import Counter
from pathlib import Path

from phase_agent.tools.relax_mc_batch.files import read
from phase_agent.tools.workflows.branch_review_view import raw_record


def value(item):
    if item is None:
        return "未记录"
    if isinstance(item, float):
        return f"{item:.6g}"
    return str(item).replace("\n", " ").replace("|", "/")


def interval(low, high):
    if low is None and high is not None:
        return f"≤{value(high)}"
    if high is None and low is not None:
        return f"≥{value(low)}"
    return value(low) if low == high else f"{value(low)}–{value(high)}"


def cell_size(row):
    from phase_agent.tools.workflows.generation_artifacts import determinant

    return row.get("supercell_det") or determinant(row.get("H"))


def branch_lines(record):
    actual = record.get("actual")
    if actual is not None:
        rows = [r for r in actual.get("branches") or [] if r.get("is_new", True)]
        counts = Counter(
            (value(r.get("phase")), value(r.get("Na_fraction")), value(cell_size(r))) for r in rows
        )
        return [
            f"新增 branch：{value(actual.get('new_branch_count'))}；登记初态：{value(actual.get('registered_structure_count'))}",
            "相 | Na/O₂ | 超胞倍数 | branch数",
            *(f"{p} | {x} | {h} | {n}" for (p, x, h), n in sorted(counts.items())),
        ]
    plan, params = record.get("plan") or {}, record.get("parameters") or {}
    regions = params.get("regions")
    if regions == []:
        return ["计划新增 branch：0"]
    allocations = regions if regions is not None else plan.get("allocations") or []
    lines = ["Branch 计划（尚未生成）"]
    if plan.get("candidate_quota") is not None:
        lines.append(
            f"候选配额：{value(plan['candidate_quota'])}；branch上限：{value(plan.get('selected_branch_upper'))}"
        )
    lines.append("相 | Na/O₂范围 | 超胞倍数范围 | 候选配额")
    for row in allocations:
        from phase_agent.tools.workflows.branch_review_view import allocation_phases

        phases = (
            ",".join(str(p) for p in allocation_phases(plan, row)) or "全部允许相"
            if row.get("phase") == "all"
            else value(row.get("phase"))
        )
        na = (
            interval(row.get("na_min"), row.get("na_max"))
            if row.get("na_min") is not None
            else value(row.get("x", plan.get("Na_fractions")))
        )
        size = interval(row.get("min_det_H"), row.get("max_det_H"))
        lines.append(
            f"{phases} | {na} | {size} | {value(row['quota']) if 'quota' in row else '生成后统计'}"
        )
    if not allocations:
        from phase_agent.tools.workflows.branch_review_view import phase_rules

        lines.extend(
            f"{value(r.get('相'))} | {value(r.get('钠含量_x', r.get('允许钠含量_x')))} | {value(plan.get('max_det_H'))} | 未逐相分配"
            for r in phase_rules(plan)
        )
    return lines


def dft_lines(record):
    selected = [
        r
        for r in record.get("decisions") or []
        if r.get("action") in {"DFT_SINGLE_POINT", "DFT_RELAX"}
    ]
    counts = Counter(r.get("action") for r in record.get("decisions") or [])
    states = {
        "awaiting_approval": "待审批",
        "prepared": "已生成输入",
        "tasks_created": "已登记任务",
        "input_preparation_failed": "输入准备受阻",
    }
    lines = [
        f"DFT 选点：{states.get(record.get('status'), value(record.get('status')))}",
        f"选中 {len(selected)} 点：静态 {counts['DFT_SINGLE_POINT']}，Relax {counts['DFT_RELAX']}；暂缓 {counts['DEFER']}，不选 {counts['REJECT']}",
        "结构编号 | 相 | Na/O₂ | 超胞倍数 | 计算类型",
    ]
    for row in selected:
        calculation = "静态" if row["action"] == "DFT_SINGLE_POINT" else "Relax"
        lines.append(
            f"{value(row.get('candidate_id'))} | {value(row.get('phase'))} | {value(row.get('Na_fraction'))} | {value(cell_size(row))} | {calculation}"
        )
    return lines


def review_artifacts(path, *, kind, record=None, state=None):
    """Keep the authoritative JSON intact; publish/update only its small TXT view."""
    path = Path(path)
    if kind not in {"branch", "dft"}:
        raise ValueError("unsupported_review_kind")
    from phase_agent.tools.workflows.readonly_views import is_read_only

    if is_read_only():
        if not path.is_file():
            return {}
        target = path.with_suffix(".txt")
        return {"json": str(path), "text": str(target if target.is_file() else path)}
    record = raw_record(read(path) if record is None else record)
    lines = branch_lines(record) if kind == "branch" else dft_lines(record)
    config = (state or {}).get("confirmed_config") or {}
    config = config.get("config") or config
    boundary = (
        (record.get("plan") or {}).get("boundary")
        or config.get("boundary")
        or (config.get("system") or {}).get("boundary")
        or {}
    )
    if boundary.get("TM_ratio"):
        lines.insert(
            1,
            "TM配比："
            + "，".join(f"{k}:{value(v)}" for k, v in sorted(boundary["TM_ratio"].items())),
        )
    text = "\n".join(lines) + "\n"
    target = path.with_suffix(".txt")
    if not target.is_file() or target.read_text(encoding="utf-8") != text:
        target.write_text(text, encoding="utf-8")
    return {"json": str(path), "text": str(target)}
