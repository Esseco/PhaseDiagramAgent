from phase_agent.analysis.phase.branch_relax_hull import rank_relaxed_branches
from tests.test_mc_phase_diagram_ehull import _evidence


def test_path_alias_preserves_current_diagram_ehull(tmp_path):
    pool, diagram, candidate = _evidence()
    structure = tmp_path / "final.vasp"
    structure.write_text("structure")
    intermediate = tmp_path / "alias"
    intermediate.mkdir()
    pool["records"][0]["structure_path"] = str(structure)
    diagram["entries"][0]["structure_path"] = str(intermediate / ".." / "final.vasp")
    ranked, missing = rank_relaxed_branches([candidate], pool, phase_diagram=diagram)
    assert missing == [] and ranked[0]["relaxed_ehull"] == 0.018
    diagram["entries"][0]["normalized_total_energy"] = -3
    assert rank_relaxed_branches([candidate], pool, phase_diagram=diagram) == ([], ["B1"])


def test_different_files_with_identical_content_are_not_aliases(tmp_path):
    pool, diagram, candidate = _evidence()
    for name in ["one.vasp", "two.vasp"]:
        (tmp_path / name).write_text("same content")
    pool["records"][0]["structure_path"] = str(tmp_path / "one.vasp")
    diagram["entries"][0]["structure_path"] = str(tmp_path / "two.vasp")
    assert rank_relaxed_branches([candidate], pool, phase_diagram=diagram) == ([], ["B1"])
    diagram["entries"][0]["structure_path"] = str(tmp_path / "missing.vasp")
    assert rank_relaxed_branches([candidate], pool, phase_diagram=diagram) == ([], ["B1"])
