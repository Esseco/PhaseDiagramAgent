"""Calculate an explicitly labelled atom-count-aware proxy cost."""


def calculate_size_aware_cost(*, atom_count, evaluation_count, config: dict, initial_state_count=1) -> dict:
    missing = [name for name, value in (("atom_count", atom_count), ("evaluation_count", evaluation_count)) if value is None]
    if missing:
        return {"status": "unknown", "kind": "proxy_relative", "value": None, "missing": missing}
    atoms, evaluations, states = int(atom_count), float(evaluation_count), int(initial_state_count)
    if atoms <= 0 or evaluations < 0 or states <= 0:
        raise ValueError("atom_count/initial_state_count 必须为正，evaluation_count 不能为负")
    reference = float(config.get("reference_atoms", 1.0)); exponent = float(config.get("atom_exponent", 1.0)); scale = float(config.get("scale", 1.0))
    if reference <= 0 or exponent < 0 or scale < 0:
        raise ValueError("成本模型参数无效")
    value = (atoms / reference) ** exponent * evaluations * states * scale
    return {"status": "estimated", "kind": "proxy_relative", "value": float(value), "atom_count": atoms, "evaluation_count": evaluations, "initial_state_count": states, "model": {"reference_atoms": reference, "atom_exponent": exponent, "scale": scale}, "missing": []}
