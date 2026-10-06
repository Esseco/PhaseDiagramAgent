"""Use the installed Py-Code layer check for a final structure."""

from __future__ import annotations

from pathlib import Path


def check_na_layer_uniformity(structure_path, composition, *, target_layers=3, structure_data=None):
    """Return an observed result or an explicit reason it is unavailable."""
    rule = f"Process_Vasp.structure.check_layer_equal(Na, target_layers={target_layers})"
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
        structure = Structure.from_dict(structure_data) if structure_data is not None else Structure.from_file(str(structure_path))
        uniform = check_layer_equal(
            structure, element="Na", target_layers=target_layers, EL_Equal=True)
    except Exception as error:
        return {"uniform": None, "status": f"structure_check_failed:{type(error).__name__}",
                "rule": rule}
    return {"uniform": bool(uniform), "status": "checked", "rule": rule}
