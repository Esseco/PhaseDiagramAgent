"""Scoped VASP symmetry remediation inside the approved retry handler."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


def symmetry_retry_settings(original):
    if original.get("stage") not in {"dft_single_point", "dft_relax"}:
        raise ValueError("symmetry_repair_requires_dft_task")
    directory = Path(original["input_path"]).parent
    workflow = json.loads((directory / "workflow.json").read_text(encoding="utf-8"))
    parameters = deepcopy(original.get("parameters") or {})
    stage = "static" if original["stage"] == "dft_single_point" else "relax"
    incar = parameters.setdefault("incar_settings", {})
    incar[stage] = {
        **workflow.get("incar_settings", {}).get(stage, {}),
        **incar.get(stage, {}),
        "ISYM": -1,
        "SYMPREC": 1e-5,
    }
    parameters.setdefault("kpoints_settings", deepcopy(workflow.get("kpoints_settings", {})))
    script = (directory / "submit_gpu.sh").read_text(encoding="utf-8")
    script, ranks = re.subn(
        r"(?m)^#SBATCH (?:-n\s+|--ntasks[= ])\d+\s*$", "#SBATCH --ntasks=1", script
    )
    script, per_node = re.subn(
        r"(?m)^#SBATCH --ntasks-per-node[= ]\d+\s*$", "#SBATCH --ntasks-per-node=1", script
    )
    if ranks != 1 or per_node != 1 or not re.search(r"(?m)^#SBATCH --gres=gpu:1\s*$", script):
        raise ValueError("ambiguous_original_mpi_gpu_resources")
    if "python3 workflow.py" not in script:
        raise ValueError("original_workflow_entry_missing")
    return parameters, script


def apply_retry_settings(task, parameters, script):
    task["parameters"] = parameters
    task["reviewed_submit_script"] = script
    task["retry_repair_profile"] = "vasp_symmetry_single_gpu"
    task["upload_operation_id"] = hashlib.sha256(task["task_key"].encode()).hexdigest()[:12]
