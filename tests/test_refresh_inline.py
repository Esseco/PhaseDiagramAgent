import json
import pytest
from phase_agent.science.mlip.refresh_inline import run_inline_refresh


def job(tmp_path, threshold=0.1):
    structure = tmp_path / "input.vasp"
    structure.write_text("identity", encoding="utf8")
    return {
        "structure_path": str(structure),
        "output_directory": str(tmp_path),
        "model_version": "m1",
        "model_path": "/m",
        "parameters": {"refresh_force_threshold": threshold},
    }


def predictor(value, calls):
    def run(job):
        calls.append("predict")
        return {
            "status": "completed",
            "energy": -1.0,
            "forces": [[value, 0, 0]],
            "forces_unit": "eV/angstrom",
            "single_point_completed": True,
            "structure_path": job["structure_path"],
            "structure_checksum": __import__("hashlib")
            .sha256(__import__("pathlib").Path(job["structure_path"]).read_bytes())
            .hexdigest(),
            "mlip_version": job["model_version"],
            "atom_count": 1,
        }

    return run


def test_single_point_below_threshold_no_relax(tmp_path):
    calls = []
    result = run_inline_refresh(
        job(tmp_path), predictor(0.1, calls), lambda _: calls.append("relax")
    )
    assert calls == ["predict"] and result["single_point_completed"]
    assert result["refresh_force_followup"]["relaxed"] is False


def test_relax_above_threshold_and_cache_after_failure(tmp_path):
    calls = []

    def fail(_):
        calls.append("relax")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        run_inline_refresh(job(tmp_path), predictor(0.11, calls), fail)

    def relax(_):
        calls.append("relax")
        return {
            "status": "completed",
            "optimizer_converged": True,
            "energy": -2.0,
            "structure_path": "final",
        }

    result = run_inline_refresh(job(tmp_path), predictor(0.11, calls), relax)
    assert calls == ["predict", "relax", "relax"]
    assert result["converged"] is True and result["refresh_force_followup"]["relaxed"] is True
    assert result["refresh_force_followup"]["initial_prediction"]["energy"] == -1.0


def test_cached_prediction_identity_change_is_rejected(tmp_path):
    calls = []
    run_inline_refresh(job(tmp_path), predictor(0.01, calls), lambda _: None)
    changed = job(tmp_path)
    changed["model_version"] = "other"
    with pytest.raises(ValueError, match="identity"):
        run_inline_refresh(changed, predictor(0.01, calls), lambda _: None)


def test_failed_optimizer_does_not_report_completion(tmp_path):
    calls = []
    with pytest.raises(ValueError, match="converged"):
        run_inline_refresh(
            job(tmp_path),
            predictor(0.2, calls),
            lambda _: {"status": "completed", "optimizer_converged": False},
        )
