"""从主流程安全调用 py-mace 环境。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_mace_with_py_mace(
    structure,
    *,
    model_path,
    operation,
    work_directory,
    parameters=None,
    segment_budget=None,
    environment="py-mace",
) -> dict:
    work = Path(work_directory)
    work.mkdir(parents=True, exist_ok=True)
    model = Path(model_path)
    if not model.exists():
        return {"status": "not_configured", "error": f"MACE model not found: {model}"}
    structure_path = work / "input.vasp"
    if isinstance(structure, (str, Path)):
        structure_path = Path(structure)
    elif hasattr(structure, "to"):
        structure.to(filename=structure_path)
    else:
        return {"status": "failed", "error": "structure must be a path or pymatgen Structure"}
    job = {
        "operation": operation,
        "structure_path": str(structure_path),
        "model_path": str(model),
        "output_directory": str(work),
        "parameters": dict(parameters or {}),
        "segment_budget": segment_budget,
    }
    job_path, result_path = work / "mace_job.json", work / "mace_result.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    worker = Path(__file__).with_name("mace_worker.py")
    command = (
        [sys.executable]
        if environment == "current"
        else ["conda", "run", "-n", environment, "python"]
    )
    completed = subprocess.run(
        [*command, str(worker), str(job_path), str(result_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode or not result_path.exists():
        return {
            "status": "failed",
            "error": completed.stderr or completed.stdout,
            "returncode": completed.returncode,
        }
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("structure_path"):
        from pymatgen.core import Structure

        result["structure"] = Structure.from_file(result["structure_path"])
    result["model_name"] = "mh-1"
    result["model_path"] = str(model)
    return result
