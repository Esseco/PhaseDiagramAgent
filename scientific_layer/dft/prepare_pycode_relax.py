"""Reuse the installed Py-Code atomate2 relax generator without running VASP."""
from pathlib import Path
from scientific_layer.dft.incar_policy import layered_oxide_incar


def prepare_pycode_relax(task, *, manager, generator=None):
    if task.get("stage") != "dft_relax":
        raise ValueError("Py-Code 接口仅支持结构优化 dft_relax；请确认优化方案")
    parameters = dict(task.get("parameters") or {})
    unknown = set(parameters) - {"incar_settings", "kpoints_settings"}
    if unknown:
        raise ValueError(f"未支持的 DFT 参数：{sorted(unknown)}")
    record = manager.data.get("structures", {}).get(task.get("structure_id")) or {}
    source = Path(task.get("structure_path") or record.get("source_path") or "")
    if not source.is_file():
        raise FileNotFoundError(f"DFT 所选结构不存在：{source}")
    incar = dict(parameters.get("incar_settings") or {})
    incar["relax"] = {**layered_oxide_incar(incar.get("relax")), "GGA": None}
    if any(stage != "relax" for stage in incar):
        raise ValueError("仅允许 relax 阶段 INCAR 参数")
    kpoints = dict(parameters.get("kpoints_settings") or {})
    if any(stage != "relax" for stage in kpoints):
        raise ValueError("仅允许 relax 阶段 KPOINTS 参数")
    if generator is None:
        from Process_Vasp.generation import generate_atomate_input
        generator = generate_atomate_input
    directory = Path(task["work_directory"])
    generator(directory, source, calculation="relax", incar_settings=incar,
              kpoints_settings=kpoints, job_name="DFT-relax")
    workflow = directory / "workflow.py"
    workflow.rename(directory / "atomate_relax.py")
    workflow.write_text('''"""Run one Py-Code relaxation and export the task result."""
import runpy
from pathlib import Path
from execution_layer.remote.finalize_vasp import finalize_vasp

root = Path(__file__).resolve().parent
exit_code = 0
try:
    runpy.run_path(str(root / "atomate_relax.py"), run_name="__main__")
except BaseException:
    exit_code = 1
    raise
finally:
    outputs = list((root / "runs").rglob("vasprun.xml*"))
    if len(outputs) > 1:
        raise RuntimeError("Expected exactly one relaxation output")
    calculation = outputs[0].parent if outputs else root
    result = finalize_vasp(root, exit_code=exit_code, calculation_directory=calculation)
    if result["status"] != "completed" and exit_code == 0:
        raise RuntimeError("DFT relaxation failed or did not converge")
''', encoding="utf-8", newline="\n")
    return {"generator": "atomate", "backend": "pycode_atomate2_relax",
            "status": "pending", "calculation_directory": str(directory)}
