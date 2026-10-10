"""Compact evidence diagnostics, never an approval or scientific validation."""


def evidence_reference_lines(action):
    check = action.get("evidence_reference_check") or {}
    status = check.get("status")
    if status == "unknown_references":
        missing = check.get("missing") or []
        return [
            "依据提醒：有未匹配的证据引用："
            + "、".join(missing[:5])
            + (f"（共 {len(missing)} 项）" if len(missing) > 5 else "")
            + "；需核对方案依据。"
        ]
    if status == "invalid_reference_format":
        return ["依据提醒：证据引用格式无效，尚未核对。"]
    if status == "no_references":
        return ["依据提醒：方案未提供结构化证据引用。"]
    if status == "references_found":
        refs = action.get("evidence_refs") or []
        return [f"依据：已匹配 {len(set(refs))} 项引用；不代表科学校验通过。"]
    return []
