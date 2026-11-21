from pathlib import Path
import csv
import json
from phase_agent.analysis.feedback.kfold_error_trend import publish_kfold_error_trend


def test_verified_mae_by_resulting_epoch_and_unknowns(tmp_path):
    root = tmp_path / "analysis_outputs"
    root.mkdir()
    state = {"upload_layout": {"model_rounds": {"base": 1, "model1": 2, "model2": 3}}}

    def job(name, version, e, f):
        return {
            "job_id": name,
            "source_epoch": "epoch0",
            "model_version": version,
            "fingerprint": "f" * 64,
            "status": "completed",
            "metrics": {
                "energy": {"mae": e, "mae_unit": "meV/atom", "structures": 4},
                "force": {"mae": f, "mae_unit": "meV/Å", "points": 12},
            },
        }

    rows = [
        job("j1", "model1", 5, 50),
        job("j2", "model2", 3, 70),
        job("unknown", "not_registered", 1, 2),
        {"job_id": "missing", "status": "unavailable", "reason": "missing evidence"},
    ]
    result = publish_kfold_error_trend(state, root, rows)
    assert [(row["epoch"], row["energy_mae_meV_per_atom"]) for row in result["rows"][:2]] == [
        ("epoch1", 5),
        ("epoch2", 3),
    ]
    assert result["rows"][2]["status"] == "unavailable"
    assert result["rows"][3]["status"] == "unavailable"
    for kind in ("E", "F"):
        assert (root / "latest" / f"KFold_Error_{kind}_MAE_by_epoch.png").read_bytes()[
            :8
        ] == b"\x89PNG\r\n\x1a\n"
    with (root / "latest/KFold_Error_MAE_by_epoch.csv").open(encoding="utf-8-sig", newline="") as f:
        assert len(list(csv.DictReader(f))) == 4
    assert (
        json.loads((root / "latest/KFold_Error_MAE_by_epoch.json").read_text(encoding="utf8"))[
            "evaluation_type"
        ]
        == "kfold_out_of_fold"
    )
    assert not list((root / "latest").glob("*.pdf"))
