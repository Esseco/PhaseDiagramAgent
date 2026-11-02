from fractions import Fraction

from phase_agent.science.structures.generate_composition_branches import _neighbouring_x


def test_adjacent_legal_concentrations_and_endpoints():
    values = ["0", "1/4", "1/2", "3/4", "1"]
    assert _neighbouring_x("1/2", values) == {Fraction(1, 4), Fraction(3, 4)}
    assert _neighbouring_x("0", values) == {Fraction(1, 4)}
    assert _neighbouring_x("1", values) == {Fraction(3, 4)}
    assert _neighbouring_x("1/2", ["1/2"]) == set()


def test_focused_region_cannot_skip_a_nearer_legal_concentration(monkeypatch):
    from types import SimpleNamespace
    from importlib import import_module

    module = import_module("phase_agent.science.structures.generate_composition_branches")
    enumerator = import_module("phase_agent.science.structures.enumerate_legal_frameworks")
    matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    parent = {"P": "O3", "H": matrix, "x": "1/2", "T": ["Ni"]}
    manager = SimpleNamespace(boundary={}, data={"branches": {"parent": parent}})
    full = {"P": "O3", "H": matrix, "allowed_x": ["0", "1/4", "1/2", "3/4", "1"]}
    focused = {**full, "allowed_x": ["0", "1/4"]}
    monkeypatch.setattr(enumerator, "enumerate_legal_frameworks", lambda *args: [full])
    monkeypatch.setattr(module, "generate_branch_structure", lambda *args, **kwargs: kwargs)
    monkeypatch.setattr(
        module,
        "identify_branch_parameters",
        lambda s, *args, **kwargs: {
            "P": s["phase"],
            "H": s["H"],
            "x": s["x"],
            "T": ["Ni"],
            "det_H": 1,
            "composition": {},
        },
    )
    results = module.generate_composition_branches(
        manager, [focused], {}, parent_branch_ids=["parent"], quota=10, seed=1
    )
    assert [row["x"] for row in results] == ["1/4"]
