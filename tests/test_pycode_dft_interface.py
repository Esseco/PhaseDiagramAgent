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


def test_single_point_is_not_silently_changed():
    with pytest.raises(ValueError, match="dft_relax"):
        prepare_pycode_relax({"stage": "dft_single_point"}, manager=None)


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
