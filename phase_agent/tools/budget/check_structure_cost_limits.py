"""Close the supercell search space with size and proxy-cost limits."""


def check_structure_cost_limits(
    *, atom_count=None, det_H=None, proxy_cost=None, limits: dict
) -> dict:
    reasons = []
    for name, value in (
        ("max_atoms", atom_count),
        ("max_det_H", det_H),
        ("max_proxy_cost_per_task", proxy_cost),
    ):
        limit = limits.get(name)
        if limit is not None and value is None:
            reasons.append(f"unknown_{name.removeprefix('max_')}")
        elif limit is not None and float(value) > float(limit):
            reasons.append(name)
    return {
        "allowed": not reasons,
        "reasons": reasons,
        "inputs": {"atom_count": atom_count, "det_H": det_H, "proxy_cost": proxy_cost},
    }
