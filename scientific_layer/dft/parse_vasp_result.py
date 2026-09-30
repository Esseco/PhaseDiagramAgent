"""Parse one directly submitted VASP calculation into the shared result format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_vasp_result(directory, *, exit_code=0, parser=None):
    root = Path(directory)
    task = _read_json(root / "task.json")
    if int(exit_code) != 0:
        return _base(task, "failed", error=f"vasp exited with code {exit_code}")
    from monty.os.path import zpath
    vasprun_path = Path(zpath(root / "vasprun.xml"))
    if not vasprun_path.is_file():
        return _base(task, "failed", error="vasprun.xml missing")
    try:
        if parser is None:
            from pymatgen.io.vasp.outputs import Vasprun
            parsed = Vasprun(str(vasprun_path), parse_potcar_file=False)
        else:
            parsed = parser(vasprun_path)
        converged = bool(parsed.converged)
        # Export one portable uncompressed structure for the existing collector.
        contcar = Path(zpath(root / "CONTCAR"))
        exported = root / "CONTCAR"
        if contcar.is_file() and contcar != exported:
            import gzip
            import shutil
            with gzip.open(contcar, "rb") as source, exported.with_name("CONTCAR.part").open("wb") as target:
                shutil.copyfileobj(source, target)
            exported.with_name("CONTCAR.part").replace(exported)
        outputs = {
            "energy": float(parsed.final_energy),
            "energy_unit": "eV",
            "converged": converged,
            "dft_code": "VASP",
            "dft_version": getattr(parsed, "vasp_version", None),
            "ionic_steps": len(getattr(parsed, "ionic_steps", []) or []),
            "electronic_steps": sum(
                len(item.get("electronic_steps") or [])
                for item in (getattr(parsed, "ionic_steps", []) or [])
            ),
            "structure_path": str(root / "CONTCAR") if (root / "CONTCAR").is_file() else None,
        }
        result = _base(task, "completed" if converged else "failed")
        result.update({"converged": converged, "outputs": outputs})
        if not converged:
            result["error"] = "VASP calculation did not converge"
        return result
    except Exception as error:
        return _base(task, "failed", error=f"VASP parse failed: {type(error).__name__}: {error}")


def write_vasp_result(directory, result_path, *, exit_code=0, parser=None):
    result = parse_vasp_result(directory, exit_code=exit_code, parser=parser)
    path = Path(result_path)
    if not path.is_absolute():
        path = Path(directory) / path
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    marker = path.with_name("task.finished.json")
    marker_temporary = marker.with_suffix(marker.suffix + ".tmp")
    marker_temporary.write_text(
        json.dumps({
            "task_id": result.get("task_id"), "task_key": result.get("task_key"),
            "status": result.get("status"), "result_file": path.name,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker_temporary.replace(marker)
    return result


def _base(task, status, error=None):
    return {
        "task_id": task.get("task_id"),
        "task_key": task.get("task_key"),
        "structure_id": task.get("structure_id"),
        "stage": task.get("stage"),
        "status": status,
        "converged": False if status == "failed" else None,
        "actual_cost": None,
        "result_path": str(Path(task.get("result_path") or ".").parent),
        "error": error,
    }


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--exit-code", type=int, default=0)
    arguments = parser.parse_args(argv)
    write_vasp_result(
        arguments.directory, arguments.result_path, exit_code=arguments.exit_code
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
