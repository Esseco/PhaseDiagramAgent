"""Load the genuine final result structure from JSON, or legacy files."""
import hashlib
import json
from pathlib import Path


def load_result_structure(outputs):
    from pymatgen.core import Structure
    from monty.json import MontyEncoder
    if outputs.get("final_frame_valid") is False:
        raise ValueError("actual final frame is invalid; earlier training frames are not a substitute")
    if (outputs.get("final_frame_index") is not None and
            outputs.get("training_frame_index") != outputs["final_frame_index"]):
        raise ValueError("structure does not correspond to the actual final frame")
    payload = outputs.get("structure")
    if payload is not None:
        structure = Structure.from_dict(payload)
        content = {"lattice": structure.lattice.matrix.tolist(),
                   "sites": [{"species": site.as_dict()["species"],
                              "abc": site.frac_coords.tolist(), "properties": site.properties}
                             for site in structure], "charge": structure.charge}
        digest = hashlib.sha256(json.dumps(content, sort_keys=True, cls=MontyEncoder,
                               ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        return structure, digest
    value = outputs.get("structure_path") or outputs.get("final_structure_path")
    if not value or not Path(value).is_file():
        raise FileNotFoundError("final_structure_missing")
    path = Path(value)
    return Structure.from_file(path), hashlib.sha256(path.read_bytes()).hexdigest()
