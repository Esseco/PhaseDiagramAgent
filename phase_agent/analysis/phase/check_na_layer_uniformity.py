"""Use the installed Py-Code layer check for a final structure."""

from __future__ import annotations

from pathlib import Path


def check_na_layer_uniformity(structure_path, composition, *, target_layers=3, structure_data=None):
    """Return an observed result or an explicit reason it is unavailable."""
    rule = f"Process_Vasp.structure.check_layer_equal(Na, target_layers={target_layers}, EL_Equal=False)"
    if not (composition or {}).get("Na"):
        return {"uniform": None, "status": "not_applicable", "rule": rule}
    try:
        if structure_data is None and (not structure_path or not Path(structure_path).is_file()):
            return {"uniform": None, "status": "missing_final_structure", "rule": rule}
    except (OSError, TypeError, ValueError):
        return {"uniform": None, "status": "invalid_structure_path", "rule": rule}
    try:
        from pymatgen.core import Structure
        from Process_Vasp.structure import check_layer_equal
    except ImportError:
        return {"uniform": None, "status": "py_code_unavailable", "rule": rule}
    try:
        structure = (
            Structure.from_dict(structure_data)
            if structure_data is not None
            else Structure.from_file(str(structure_path))
        )
        uniform = check_layer_equal(
            structure, element="Na", target_layers=target_layers, EL_Equal=False
        )
    except Exception as error:
        return {
            "uniform": None,
            "status": f"structure_check_failed:{type(error).__name__}",
            "rule": rule,
        }
    return {"uniform": None, "occupied": bool(uniform), "status": "checked", "rule": rule}


def cached_na_layer_check(check, row, cache):
    """Reuse checks only within one hull build after verifying exact file bytes."""
    import hashlib

    path = row.get("structure_path")
    data = row.get("structure")
    composition = row.get("composition") or {}
    if cache is None or data is not None or not path or not composition.get("Na"):
        return check(path, composition, structure_data=data)
    try:
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError):
        return check(row.get("structure_path"), composition, structure_data=data)
    key = (id(check), str(path), digest)
    if key in cache:
        return dict(cache[key])
    result = check(row.get("structure_path"), composition, structure_data=data)
    try:
        if (
            result.get("status") == "checked"
            and hashlib.sha256(path.read_bytes()).hexdigest() == digest
        ):
            cache[key] = dict(result)
    except OSError:
        pass
    return result
