"""按组分计算可比较损失，避免只追逐全局最低能。"""


def calculate_bohb_loss(branch: dict, result: dict, *, objective: dict) -> dict:
    if objective.get('name') == 'relaxed_batch_hull_gap':
        outputs = result.get('outputs') or {}
        value = result.get('minimum_energy_per_atom')
        if value is None and outputs.get('energy_unit') == 'eV' and outputs.get('energy') is not None and outputs.get('atom_count'):
            value = float(outputs['energy']) / float(outputs['atom_count'])
        reference = branch.get('hull_reference_energy_per_atom')
        if value is None or reference is None:
            return {'status': 'unknown', 'loss': None, 'missing': ['actual_mc_energy_or_hull_reference']}
        return {'status': 'completed', 'loss': float(value)-float(reference),
                'group': str(branch.get('composition_group')), 'basis': objective}
    group = result.get(objective.get("group_key", "composition_group"), branch.get(objective.get("group_key", "composition_group")))
    value = result.get(objective.get("value_key", "minimum_energy_per_atom"))
    reference = result.get(objective.get("reference_key", "group_reference_energy_per_atom"))
    scale = result.get(objective.get("scale_key", "group_energy_scale"))
    missing = [name for name, item in (("composition_group", group), ("objective_value", value), ("fixed_reference", reference), ("normalization_scale", scale)) if item is None]
    if missing or scale == 0:
        if scale == 0:
            missing.append("nonzero_normalization_scale")
        return {"status": "unknown", "loss": None, "group": group, "missing": missing, "basis": objective}
    loss = (float(value) - float(reference)) / float(scale)
    return {"status": "completed", "loss": loss, "group": str(group), "components": {"value": float(value), "reference": float(reference), "scale": float(scale)}, "basis": objective, "missing": []}
