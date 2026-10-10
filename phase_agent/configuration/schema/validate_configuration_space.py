"""Validate the user-defined roles of layered-oxide configuration variables."""


def validate_configuration_space(system: dict) -> dict:
    space = system.get("configuration_space")
    if space is None:  # Existing saved configurations keep their original behavior.
        return {"valid": True, "errors": []}
    if not isinstance(space, dict):
        return {"valid": False, "errors": ["system.configuration_space 必须是对象"]}
    roles = space.get("roles")
    if not isinstance(roles, dict) or set(roles) != {"H", "P", "x", "T", "N"}:
        return {"valid": False, "errors": ["roles 必须分别定义 H、P、x、T、N"]}
    errors = []
    for name in ("H", "P", "x", "T"):
        if roles[name] not in {"fixed", "branch"}:
            errors.append(f"{name} 当前只支持 fixed 或 branch")
    if roles["N"] != "internal":
        errors.append("N 当前由 branch 内 Na/空位搜索处理，必须为 internal")
    if roles["T"] == "fixed" and space.get("fixed_T_source") != "phase_reference":
        errors.append("固定 T 必须明确 fixed_T_source=phase_reference")
    if roles["T"] == "branch" and space.get("fixed_T_source") not in (None, ""):
        errors.append("T 为 branch 变量时不得指定 fixed_T_source")
    fixed = space.get("fixed_values") or {}
    if not isinstance(fixed, dict):
        errors.append("fixed_values 必须是对象")
    else:
        for name in ("H", "P", "x"):
            if roles[name] == "fixed" and fixed.get(name) is None:
                errors.append(f"{name} 固定时必须提供 fixed_values.{name}")
    boundary = system.get("boundary") or {}
    if isinstance(fixed, dict):
        from phase_agent.science.structures.boundary_utils import allowed_phases, normalize_H, det_H

        if roles["P"] == "fixed" and fixed.get("P") is not None and boundary.get("P"):
            if str(fixed["P"]).upper() not in allowed_phases(boundary["P"]):
                errors.append("fixed_values.P is outside system.boundary.P")
        if roles["x"] == "fixed" and fixed.get("x") is not None:
            from fractions import Fraction

            try:
                x = Fraction(str(fixed["x"]))
                if not 0 <= x <= 1:
                    errors.append("fixed_values.x must be within Na/O2 [0,1]")
            except (ValueError, ZeroDivisionError):
                errors.append("fixed_values.x must be a finite Na/O2 fraction")
        if roles["H"] == "fixed" and fixed.get("H") is not None:
            try:
                normalized = normalize_H(fixed["H"])
                det_H(normalized)
                matrices = boundary.get("H") or {}
                phases = [str(fixed.get("P")).upper()] if roles["P"] == "fixed" else matrices
                if matrices and not any(
                    normalized == normalize_H(h)
                    for phase in phases
                    for h in matrices.get(phase, [])
                ):
                    errors.append("fixed_values.H is outside system.boundary.H")
            except (ValueError, TypeError):
                errors.append("fixed_values.H must be a valid integer supercell matrix")
    return {"valid": not errors, "errors": errors}
