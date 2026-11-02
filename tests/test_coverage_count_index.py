from types import SimpleNamespace

from phase_agent.science.structures import generate_coverage_branches as coverage


def test_coverage_prefers_sparse_cells_then_less_populated_phase(monkeypatch):
    h = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    branches = {
        "a": {"P": "O3", "H": h, "x": "0"},
        "b": {"P": "O3", "H": h, "x": "0"},
        "c": {"P": "P3", "H": h, "x": "1"},
    }
    frameworks = [{"P": p, "H": h, "allowed_x": ["0", "1"]} for p in ("O3", "P3")]
    monkeypatch.setattr(coverage, "generate_branch_structure", lambda boundary, **kw: kw)
    monkeypatch.setattr(
        coverage,
        "identify_branch_parameters",
        lambda structure, *args, **kw: {
            "P": structure["phase"],
            "H": structure["H"],
            "x": structure["x"],
            "det_H": 1,
            "T": ["Ni"],
            "composition": {},
        },
    )
    results = coverage.generate_coverage_branches(
        SimpleNamespace(data={"branches": branches}, boundary={}),
        frameworks,
        {},
        quota=4,
        seed=7,
        preserve_reference_tm=True,
    )
    assert [(row["P"], row["x"]) for row in results] == [
        ("P3", "0"),
        ("O3", "1"),
        ("P3", "1"),
        ("O3", "0"),
    ]
