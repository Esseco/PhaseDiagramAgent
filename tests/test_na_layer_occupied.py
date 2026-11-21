from unittest.mock import patch

from pymatgen.core import Lattice, Structure
from phase_agent.analysis.phase.check_na_layer_uniformity import check_na_layer_uniformity


def test_layer_rule_allows_unequal_na_counts():
    structure = Structure(Lattice.cubic(10), ["Na", "Na", "Na", "Na"], [[0, 0, .1], [.5, .5, .1], [0, 0, .4], [0, 0, .7]])
    result = check_na_layer_uniformity(None, {"Na": 4}, structure_data=structure.as_dict())
    assert result["status"] == "checked"
    assert result["occupied"] is True
    assert result["uniform"] is None


def test_layer_rule_reports_missing_na_layer_and_exempts_na_zero():
    structure = Structure(Lattice.cubic(10), ["Na", "Na"], [[0, 0, .1], [0, 0, .5]])
    result = check_na_layer_uniformity(None, {"Na": 2}, structure_data=structure.as_dict())
    assert result["status"] == "checked" and result["occupied"] is False
    empty = check_na_layer_uniformity(None, {"Cr": 1, "O": 2})
    assert empty["status"] == "not_applicable"
