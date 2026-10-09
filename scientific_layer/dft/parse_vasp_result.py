"""Parse one directly submitted VASP calculation into the shared result format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_vasp_result(directory, *, exit_code=0, parser=None):
    root = Path(directory)
    task = _read_json(root / "task.json")
    from monty.os.path import zpath
    vasprun_path = Path(zpath(root / "vasprun.xml"))
    if not vasprun_path.is_file():
        error = f"vasp exited with code {exit_code}; vasprun.xml missing" if int(exit_code) else "vasprun.xml missing"
        return _base(task, "failed", error=error)
    try:
        xml_complete = True
        if parser is None:
            from pymatgen.io.vasp.outputs import Vasprun
            from xml.etree.ElementTree import ParseError
            xml_errors = (ParseError,)
            try:
                from lxml.etree import XMLSyntaxError
            except ImportError:
                pass
            else:
                xml_errors += (XMLSyntaxError,)
            options = dict(parse_potcar_file=False, parse_dos=False,
                           parse_eigen=False, parse_projected_eigen=False)
            try:
                parsed = Vasprun(str(vasprun_path), **options)
            except xml_errors:
                parsed = Vasprun(str(vasprun_path), exception_on_bad_xml=False, **options)
                xml_complete = False
        else:
            parsed = parser(vasprun_path)
        converged = bool(parsed.converged) and int(exit_code) == 0 and xml_complete
        outputs = {
            "energy": float(parsed.final_energy),
            "energy_unit": "eV",
            "converged": converged,
            "xml_complete": xml_complete,
            "dft_code": "VASP",
            "dft_version": getattr(parsed, "vasp_version", None),
            "ionic_steps": len(getattr(parsed, "ionic_steps", []) or []),
            "electronic_steps": sum(
                len(item.get("electronic_steps") or [])
                for item in (getattr(parsed, "ionic_steps", []) or [])
            ),
            "final_frame_index": len(getattr(parsed, "ionic_steps", []) or []) - 1,
        }
        result = _base(task, "completed" if converged else "failed")
        from scientific_layer.dft.vasp_training_labels import extract_training_frames, final_training_labels
        extracted = extract_training_frames(parsed)
        outputs.update(training_frames=extracted["frames"],
                       rejected_training_frames=extracted["rejected"],
                       ionic_converged=bool(getattr(parsed, "converged_ionic", parsed.converged)))
        try:
            outputs.update(final_training_labels(parsed))
            outputs["energy"] = outputs["training_energy"]
            outputs["final_frame_valid"] = True
        except (KeyError, TypeError, ValueError) as error:
            outputs.update(training_ready=False, final_frame_valid=False, training_error=str(error))
        from scientific_layer.dft.magnetic_diagnostics import extract_magnetic_diagnostics
        outputs["magnetic_moments"] = extract_magnetic_diagnostics(root, parsed)
        result.update({"converged": converged, "outputs": outputs})
        if not converged:
            result["error"] = ("VASP XML incomplete; verified frames retained for training" if not xml_complete else
                               f"vasp exited with code {exit_code}" if int(exit_code) else
                               "VASP calculation did not converge; verified electronic frames retained for training")
        return result
    except Exception as error:
        return _base(task, "failed", error=f"VASP parse failed: {type(error).__name__}: {error}")


def write_vasp_result(directory, result_path, *, exit_code=0, parser=None):
    result = parse_vasp_result(directory, exit_code=exit_code, parser=parser)
    from scientific_layer.dft.write_training_dataset import write_training_dataset
    write_training_dataset(directory, result)
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
