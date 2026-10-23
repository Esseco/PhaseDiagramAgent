import json

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from Process_AL_MC.structure_check import (
    LayeredFrameworkReference,
    check_structure,
    compact_structure_check,
)


def shell_pair(delta=-0.35):
    height = 2.8 / np.sqrt(6) / 10
    initial = Structure(
        Lattice.hexagonal(2.8, 10),
        ["Co", "O", "O"],
        [[0, 0, 0.5], [1 / 3, 2 / 3, 0.5 - height], [2 / 3, 1 / 3, 0.5 + height]],
    )
    initial.make_supercell([2, 2, 1])
    final = initial.copy()
    final.translate_sites(4, [0, 0, delta], frac_coords=False)
    return initial, final


def no_angle_screen(*args, **kwargs):
    pytest.fail("angle screen must be skipped")


def test_cutoff_hint_passes_reuses_neighbors_and_survives_compaction(monkeypatch):
    initial, final = shell_pair()
    reference = LayeredFrameworkReference(initial)
    calls = []
    original = Structure.get_neighbor_list

    def counted(structure, radius, *args, **kwargs):
        calls.append(radius)
        return original(structure, radius, *args, **kwargs)

    monkeypatch.setattr(Structure, "get_neighbor_list", counted)
    result = check_structure(final, reference=reference, check_framework=True)
    assert result["status"] == "passed" and not result["failed_checks"]
    framework = result["framework"]
    assert framework["warnings"] == ["cutoff_sensitive_coordination"]
    assert framework["local_coordination"]["status"] == "cutoff_sensitive"
    assert framework["cutoff_sensitivity"]["consistent"]
    assert calls == [0.6, max(reference.cutoffs)]
    compact = compact_structure_check(result)
    assert compact["framework"]["warnings"] == framework["warnings"]
    assert "final_counts" not in compact["framework"]["local_coordination"]
    json.dumps(compact, allow_nan=False)


@pytest.mark.parametrize(
    "kind",
    [
        "preserved",
        "middle_changed",
        "confirmed_damage",
        "topology",
        "initial_not_O6",
        "missing_reference",
    ],
)
def test_other_outcomes_skip_angle_screen(monkeypatch, kind):
    initial, final = shell_pair()
    reference = LayeredFrameworkReference(initial)
    if kind == "preserved":
        final = initial
    elif kind == "middle_changed":
        _, final = shell_pair(-0.6)
    elif kind == "confirmed_damage":
        _, final = shell_pair(-0.9)
    elif kind == "topology":
        final = initial.copy()
        final.scale_lattice(initial.volume * 1.12**3)
    elif kind == "initial_not_O6":
        reference._initial_coordination[:, 0] = 5
    monkeypatch.setattr(reference, "_inspect_cutoff_sensitivity", no_angle_screen)
    result = check_structure(
        final,
        reference=None if kind == "missing_reference" else reference,
        check_framework=True,
    )
    expected = (
        "passed"
        if kind == "preserved"
        else "invalid"
        if kind == "confirmed_damage"
        else "uncertain"
    )
    assert result["status"] == expected
    assert "warnings" not in result.get("framework", {})


def test_strong_angle_distortion_remains_uncertain_not_rejected():
    initial, final = shell_pair(0)
    for i, site in enumerate(final):
        if site.specie.symbol == "O":
            coords = site.frac_coords.copy()
            coords[2] = 0.5 + (coords[2] - 0.5) * 0.3
            final.replace(i, site.specie, coords)
    final.translate_sites(4, [0, 0, -1.2], frac_coords=False)
    result = check_structure(
        final, reference=LayeredFrameworkReference(initial), check_framework=True
    )
    assert result["status"] == "uncertain" and not result["failed_checks"]
    sensitivity = result["framework"]["cutoff_sensitivity"]
    assert sensitivity["max_angle_rms_deg"] > 12 and not sensitivity["consistent"]
