"""Read only metadata needed for normalization from an existing structure."""

from scientific_layer.structures.boundary_utils import load_structure


def read_structure_metadata(path) -> dict:
    if not path:
        return {"status": "unknown", "atom_count": None, "missing": ["structure_path"]}
    try:
        structure = load_structure(path)
    except Exception as error:
        return {"status": "unknown", "atom_count": None, "missing": ["readable_structure"], "error": str(error)}
    return {"status": "completed", "atom_count": len(structure), "missing": []}
