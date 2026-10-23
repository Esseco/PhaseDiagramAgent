from copy import deepcopy
import json

import numpy as np
import pytest

from phase_agent.analysis.phase.endpoint_reference import (
    endpoint_relative_energies,
    PLOT_RULE_VERSION,
)
from phase_agent.analysis.phase.build_composition_hull_entries import build_composition_hull_entries
from phase_agent.analysis.phase.phase_quality import diagnose_phase_diagram
from phase_agent.analysis.phase.phase_quality_preview import build_quality_preview
from phase_agent.analysis.phase.phase_quality_comparison import publish_quality_comparison
from phase_agent.analysis.phase.phase_round_analysis import publish_relax_mc_analysis
from phase_agent.runtime.phase_diagram_chat import phase_diagram_reply
from tests.test_phase_energy_islands import masked_endpoint
from tests.test_phase_diagram_chat import diagram
from tests.test_phase_round_analysis import registered


def test_one_global_endpoint_chord_for_all_phases_and_duplicates():
    x = [1, 0, 0.5, 0, 1]
    energy = [50, 120, -20, 100, 60]
    relative, reference = endpoint_relative_energies(x, energy)
    assert relative == pytest.approx([0, 20, -95, 0, 10])
    assert reference == {"x": [0, 1], "energy": [100, 50]}
    # Adding any affine reference changes neither formation energy nor phase ordering.
    shifted, _ = endpoint_relative_energies(x, np.array(energy) + 800 + 70 * np.array(x))
    assert shifted == pytest.approx(relative)


@pytest.mark.parametrize(
    "x,energy", [([0], [0]), ([0, 1], [0, float("nan")]), ([0, 1], [0]), ([0, 1], [0, 1j])]
)
def test_invalid_or_missing_endpoint_data_is_explicit(x, energy):
    with pytest.raises(ValueError):
        endpoint_relative_energies(x, energy)


def test_observed_domain_and_crosses_use_explicit_reference_without_extrapolation():
    values, reference = endpoint_relative_energies([0.2, 0.5, 0.8], [20, -30, 80])
    assert values == pytest.approx([0, -80, 0])
    cross, _ = endpoint_relative_energies([0.2], [0], reference=reference)
    assert cross == pytest.approx([-20])
    with pytest.raises(ValueError):
        endpoint_relative_energies([0], [0], reference=reference)


def test_existing_entry_builder_uses_original_total_energies_and_preserves_ehull():
    records = [
        dict(
            record_id=str(i),
            composition={"Na": x, "Co": 1, "O": 2},
            normalized_total_energy=e,
            original_energy=e,
            phase="O3",
            phase_identification_status="identified",
        )
        for i, (x, e) in enumerate([(0, -10), (0.5, -11), (0.5, -10.5), (1, -9)])
    ]
    entries = build_composition_hull_entries(records)
    assert [r["eform_per_O2"] for r in entries] == pytest.approx([0, -1.5, -1, 0])
    assert entries[2]["ehull_per_O2"] == pytest.approx(0.5)
    assert entries[2]["ehull"] == pytest.approx(0.5 / 3.5)
    assert [r["normalized_total_energy"] for r in entries] == [-10, -11, -10.5, -9]


def test_rebased_preview_keeps_H_D_and_formal_state():
    state = masked_endpoint()
    before = deepcopy(state)
    report = diagnose_phase_diagram(state)
    cluster = next(c for c in report["energy_islands"] if len(c["record_ids"]) == 4)
    preview = build_quality_preview(
        state["phase_diagrams"]["mlip"], report, cluster_ids=[cluster["cluster_id"]]
    )
    for x in (0, 1):
        assert min(
            e for y, e in zip(preview["plot_data"]["x"], preview["plot_data"]["eform"]) if y == x
        ) == pytest.approx(0)
    assert preview["hull_lift_mev_per_O2"] == pytest.approx(196)
    assert preview["endpoint_reference"]["energy"] == pytest.approx([196, 0])
    assert preview["reference"] == "own_retained_endpoints"
    assert preview["influence_reference"] == "fixed_original"
    assert diagnose_phase_diagram(state)["structures"] == report["structures"]
    assert state == before


@pytest.mark.parametrize("formats", [None, ["png", "pdf", "svg"]])
def test_comparison_transforms_crosses_and_publishes_reference_metadata(tmp_path, formats):
    state = masked_endpoint()
    report = diagnose_phase_diagram(state)
    before = deepcopy(state)
    options = {} if formats is None else {"formats": formats}
    result = publish_quality_comparison(state, report, directory=tmp_path, **options)
    assert result["plot_rule_version"] == PLOT_RULE_VERSION
    assert result["left_endpoint_reference"]["energy"] == pytest.approx([0, 0])
    right_ends = result["right_endpoint_reference"]["energy"]
    assert right_ends == pytest.approx([196, 0])
    crosses = [
        (x, e) for c in result["cross_plot_data"].values() for x, e in zip(c["x"], c["eform"])
    ]
    assert any(x == 0 and abs(e + 196) < 1e-7 for x, e in crosses)
    assert state == before
    assert {p.rsplit(".", 1)[1] for p in result["plot_paths"]} == set(formats or ["png"])
    from PIL import Image
    with Image.open(next(p for p in result["plot_paths"] if p.endswith(".png"))) as image:
        assert image.info["dpi"] == pytest.approx((300, 300), abs=.02)


def test_normal_plot_adapter_also_rebases_and_keeps_saved_snapshot(tmp_path, monkeypatch):
    import Process_Phasediagram

    state = diagram()
    for row in state["phase_diagrams"]["mlip"]["entries"]:
        x = row["composition"]["Na"]
        row["eform_per_O2"] += 0.14 * (1 - x)
    before = deepcopy(state)
    received = []
    original = Process_Phasediagram.plot_phase_diagram

    def record(x, energy, phases, stable, **kwargs):
        received.append(energy)
        return original(x, energy, phases, stable, **kwargs)

    monkeypatch.setattr(Process_Phasediagram, "plot_phase_diagram", record)
    reply = phase_diagram_reply(state, operation="plot", directory=tmp_path)
    assert received[0] == pytest.approx([0, -200, 0])
    assert "端点归一化" in reply and state == before


def test_old_plot_receipt_is_republished_without_changing_review_ids(tmp_path, monkeypatch):
    import phase_agent.runtime.phase_diagram_chat as chat

    image = tmp_path / "plot.png"
    image.write_bytes(b"fixture")
    calls = []

    def plot(*args, **kwargs):
        calls.append(1)
        return f"PNG：`{image}`"

    monkeypatch.setattr(chat, "phase_diagram_reply", plot)
    state = registered(tmp_path)
    first = publish_relax_mc_analysis(state, "b1")
    receipt = json.loads(open(first["receipt_path"], encoding="utf-8").read())
    receipt.pop("plot_rule_version")
    with open(first["receipt_path"], "w", encoding="utf-8") as stream:
        json.dump(receipt, stream)
    republished = publish_relax_mc_analysis(state, "b1")
    assert len(calls) == 2 and republished["plot_rule_version"] == PLOT_RULE_VERSION
    assert (
        json.loads(open(first["review_report_path"], encoding="utf-8").read())["structures"]
        == diagnose_phase_diagram(state)["structures"]
    )
