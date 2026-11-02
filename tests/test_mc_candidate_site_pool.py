from fractions import Fraction

import pytest
from pymatgen.core import Lattice, Structure

from phase_agent.science.structures.build_mc_full_na_template import build_mc_full_na_template
from Process_AL_MC.MC_sample import LayeredOxide_MCOrderingClass


H = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
BOUNDARY = {"P": ["P3"], "H": {"P3": [H]}, "TM_ratio": {"Ni": 1}}


def _pool():
    return Structure(Lattice.cubic(5), ["Na", "Na", "Ni", "O", "O"],
                     [[0, 0, 0], [.5, .5, 0], [0, 0, .5], [.25, .25, .25], [.75, .75, .75]])


def test_extra_na_sites_remain_available_and_mc_keeps_target_count():
    branch = {"branch_id": "P", "P": "P3", "H": H, "x": "1/2", "T": ["Ni", "Ni"]}
    full = build_mc_full_na_template(branch, BOUNDARY, {"P3": _pool()})
    assert full.composition["Na"] == 4
    assert full.composition["Ni"] == 2 and full.composition["O"] == 4
    current = full.copy()
    na = [i for i, site in enumerate(current) if site.specie.symbol == "Na"]
    current.remove_sites(na[1:])
    assert Fraction(int(current.composition["Na"]) * 2, int(current.composition["O"])) == Fraction(1, 2)
    mc = LayeredOxide_MCOrderingClass.__new__(LayeredOxide_MCOrderingClass)
    mc.full_na_structure = full
    mc.structure = current
    mc.na_template = full
    mc.na_site_indices = [i for i, site in enumerate(full) if site.specie.symbol == "Na"]
    mc.na_occupation = mc._build_occ_from_input_structure()
    assert mc.na_occupation.count("Na") == 1 and mc.na_occupation.count("Vac") == 3
    mc.max_duplicate_trials = 10
    mc.visited = {mc._na_key(mc.na_occupation)}
    mc.duplicate_rejections = 0
    trial, _, occupancy = mc._trial_na()
    assert trial.composition["Na"] == 1
    assert occupancy.count("Na") == 1 and occupancy.count("Vac") == 3


def test_extra_na_sites_preserve_fixed_tm_and_validate_h_and_t():
    config = {"system": {"configuration_space": {"roles": {"T": "fixed"}}}}
    branch = {"branch_id": "P", "P": "P3", "H": H, "x": "1/2"}
    full = build_mc_full_na_template(branch, BOUNDARY, {"P3": _pool()}, config=config)
    assert full.composition["Na"] == 4 and full.composition["Ni"] == 2
    bad = {**branch, "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    with pytest.raises(ValueError, match="允许范围"):
        build_mc_full_na_template(bad, BOUNDARY, {"P3": _pool()}, config=config)
    variable = {"branch": {"roles": {"T": "branch"}}}
    bad_t = {**branch, "T": ["Fe", "Fe"]}
    with pytest.raises(ValueError):
        build_mc_full_na_template(bad_t, BOUNDARY, {"P3": _pool()}, config=variable)
