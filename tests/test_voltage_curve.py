import json
from pathlib import Path

import pytest

from phase_agent.analysis.phase.voltage_curve import build_voltage_curve, write_voltage_outputs


def snapshot():
    return {
        "status": "completed",
        "model_version": "m1",
        "version": "h1",
        "entries": [
            {
                "record_id": "r0",
                "x_Na_per_O2": 0.0,
                "hull_energy_per_O2": -4.0,
                "is_stable": True,
                "phase": "O3",
            },
            {
                "record_id": "r1",
                "x_Na_per_O2": 0.5,
                "hull_energy_per_O2": -4.5,
                "is_stable": True,
                "phase": "P3",
            },
            {
                "record_id": "r2",
                "x_Na_per_O2": 1.0,
                "hull_energy_per_O2": -4.6,
                "is_stable": True,
                "phase": "O3",
            },
        ],
    }


def test_voltage_curve_uses_stable_hull_segments_and_reference():
    status, rows = build_voltage_curve(
        snapshot(),
        {"confirmed_config": {"energy_conventions": {"voltage_references": {"Na": {"value": -1.5, "source": "test"}}}}},
    )
    assert status["status"] == "completed"
    assert [row["voltage_V"] for row in rows] == pytest.approx([-0.5, -1.3])
    assert rows[0]["start_record_id"] == "r0"


def test_voltage_outputs_use_default_reference_without_config(tmp_path):
    info = write_voltage_outputs(snapshot(), {}, tmp_path / "voltage.png", tmp_path / "voltage.csv", tmp_path / "voltage_status.json")
    assert info["status"] == "completed"
    assert info["reference"]["defaulted"] is True
    assert info["reference"]["value_eV_per_Na"] == -1.288
    assert Path(info["plot_path"]).is_file()
    assert Path(info["csv_path"]).read_text(encoding="utf-8-sig").splitlines()[0].startswith("status,")
    assert json.loads(Path(info["status_path"]).read_text(encoding="utf-8"))["status"] == "completed"

