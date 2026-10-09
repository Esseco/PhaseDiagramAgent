from types import SimpleNamespace
import pytest
from scientific_layer.dft.prepare_pycode_relax import prepare_pycode_relax


def test_relax_uses_selected_path_and_gga_none(tmp_path):
    source = tmp_path / "selected.vasp"
    source.write_text("selected")
    target = tmp_path / "task"
    received = {}
    def generator(directory, structure, **kwargs):
        received.update(kwargs)
        assert structure == source
        directory.mkdir()
        (directory / "workflow.py").write_text("pass")
    output = prepare_pycode_relax({"stage": "dft_relax", "structure_path": str(source),
        "work_directory": str(target)}, manager=SimpleNamespace(data={}), generator=generator)
    assert received["calculation"] == "relax"
    assert received["incar_settings"]["relax"]["GGA"] is None
    settings = received["incar_settings"]["relax"]
    assert settings["ALGO"] == "Normal"
    assert {key: settings[key] for key in ("AMIX", "BMIX", "AMIX_MAG", "BMIX_MAG")} == {
        "AMIX": 0.2, "BMIX": 0.0001, "AMIX_MAG": 0.8, "BMIX_MAG": 0.0001}
    assert output["backend"] == "pycode_atomate2_relax"
    compile((target / "workflow.py").read_text(), "workflow.py", "exec")


def test_single_point_is_not_relaxed(tmp_path):
    source = tmp_path / "selected.vasp"
    source.write_text("selected")
    received = {}
    def generator(directory, structure, **kwargs):
        received.update(kwargs)
        directory.mkdir()
        (directory / "workflow.py").write_text("pass")
    result = prepare_pycode_relax({"stage": "dft_single_point", "structure_path": str(source),
        "work_directory": str(tmp_path / "task")}, manager=SimpleNamespace(data={}), generator=generator)
    assert received["calculation"] == "static"
    assert received["incar_settings"]["static"]["NSW"] == 0
    assert received["incar_settings"]["static"]["IBRION"] == -1
    assert received["incar_settings"]["static"]["ALGO"] == "Normal"
    assert result["backend"] == "pycode_atomate2_static"


def test_legacy_atomate_incar_policy(tmp_path):
    from scientific_layer.dft.create_atomate_workflow import generate_dft_workflow_with_atomate
    from pymatgen.io.vasp.inputs import Incar
    def writer(workflow, directory):
        Incar({"ENCUT": 520, "ALGO": "Fast"}).write_file(directory / "INCAR")
        return [str(directory / "INCAR")]
    result = generate_dft_workflow_with_atomate(None, calculation_type="singlepoint",
        work_directory=tmp_path, workflow_factory=lambda *args, **kwargs: {}, input_writer=writer)
    assert result["status"] == "pending"
    incar = Incar.from_file(tmp_path / "INCAR")
    assert incar["ALGO"] == "Normal"
    assert incar["AMIX_MAG"] == 0.8
    assert incar["ENCUT"] == 520


def test_real_pycode_generator_supports_static(tmp_path):
    import json
    from pymatgen.core import Lattice, Structure
    from Process_Vasp.generation import generate_atomate_input
    source = tmp_path / "test.vasp"
    Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]).to(filename=source, fmt="poscar")
    directory = tmp_path / "static"
    generate_atomate_input(directory, source, calculation="static",
                          incar_settings={"static": {"NSW": 0, "ALGO": "Normal"}})
    assert json.loads((directory / "workflow.json").read_text())["calculation"] == "static"
    script = (directory / "workflow.py").read_text()
    compile(script, "workflow.py", "exec")
    assert ('from atomate_runner import main' in script
            or 'from Process_Vasp.workflows.atomate_runner import main' in script)
    runner_path = directory / "atomate_runner.py"
    if not runner_path.is_file():
        import inspect
        from pathlib import Path
        from Process_Vasp.workflows import atomate_runner
        runner_path = Path(inspect.getfile(atomate_runner))
    runner = runner_path.read_text(encoding="utf-8")
    assert '"static": ["static"]' in runner
    compile(runner, "atomate_runner.py", "exec")
