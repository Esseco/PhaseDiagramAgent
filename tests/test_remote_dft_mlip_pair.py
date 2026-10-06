"""Portable HPC pairs, exact-frame validation and metadata-only supplementation."""
from copy import deepcopy
import json
import pytest
from tests.test_round_model_recovery import dft_result
from execution_layer.remote.predict_dft_final_frame import write_prediction, supplement_results
from execution_layer.remote.dft_mlip_pair import load_pair, validate_prediction
from execution_layer.remote.integrity import file_checksum, verify_result
from execution_layer.workflows.create_dft_comparison_evaluator import create_dft_comparison_evaluator
from execution_layer.state.dft_result_refresh import validated_magnetic_refresh


def fixture(tmp_path):
    model = tmp_path / "m1.model"
    model.write_text("synthetic weights")
    return {"version": "m1", "model_path": str(model)}, dft_result()


def predict(*args, **kwargs):
    return {"status": "completed", "energy": -7., "energy_unit": "eV",
            "forces": [[.2, .3, .4]] * 2, "forces_unit": "eV/angstrom"}


def save_result(root, result):
    path = root / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    marker = {key: result.get(key) for key in ("task_id", "task_key", "model_version", "status")}
    marker["result_checksum"] = file_checksum(path)
    (root / "task.finished.json").write_text(json.dumps(marker), encoding="utf-8")


def test_transport_and_local_evaluator_use_remote_only(tmp_path, monkeypatch):
    model, result = fixture(tmp_path)
    prediction = write_prediction(tmp_path, result, model, predictor=predict)
    assert prediction["status"] == "completed"
    save_result(tmp_path, result)
    verified = verify_result(tmp_path / "result.json", tmp_path / "task.finished.json", result)
    assert verified["valid"]
    monkeypatch.setattr("scientific_layer.mlip.run_mace_subprocess.run_mace_with_py_mace",
                        lambda *a, **kw: pytest.fail("local inference forbidden"))
    config = {"mlip": {"version": "m1", "model_path": "/HPC/not-local.model"}}
    evaluated = create_dft_comparison_evaluator(config)(result=verified["result"], structure_id="S-D1", manager=None)
    assert evaluated["energy"] == -7.
    assert evaluated["model_sha256"] == file_checksum(model["model_path"])


def test_missing_remote_pair_explicit_no_local_fallback(tmp_path):
    model, result = fixture(tmp_path)
    with pytest.raises(ValueError, match="remote mlip_result.json unavailable"):
        create_dft_comparison_evaluator({"mlip": model})(result=result, structure_id="S-D1", manager=None)


def test_tampered_file_wrong_frame_and_model_are_rejected(tmp_path):
    model, result = fixture(tmp_path)
    prediction = write_prediction(tmp_path, result, model, predictor=predict)
    bad = {**prediction, "final_frame_index": 1}
    with pytest.raises(ValueError, match="final-frame mismatch"):
        validate_prediction(result, bad)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_prediction(result, prediction, "0" * 64)
    (tmp_path / "mlip_result.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_pair(result, tmp_path)


def test_prediction_failure_does_not_fail_or_discard_dft(tmp_path):
    model, result = fixture(tmp_path)
    model["model_path"] = str(tmp_path / "missing.model")
    prediction = write_prediction(tmp_path, result, model, predictor=predict)
    assert prediction["status"] == "failed"
    assert result["status"] == "completed" and result["outputs"]["energy"] == -8.
    assert load_pair(result, tmp_path)["status"] == "failed"


def test_supplement_preserves_dft_and_is_cached_and_refreshable(tmp_path):
    model, result = fixture(tmp_path)
    prior = deepcopy(result)
    save_result(tmp_path, result)
    calls = []
    def predictor(*a, **kw):
        calls.append(kw)
        return predict()
    assert supplement_results(tmp_path, model, predictor=predictor)[0]["status"] == "completed"
    assert supplement_results(tmp_path, model, predictor=predictor)[0]["status"] == "cached"
    assert len(calls) == 1
    verified = verify_result(tmp_path / "result.json", tmp_path / "task.finished.json", prior)
    assert verified["valid"]
    incoming = verified["result"]
    refresh = validated_magnetic_refresh(prior, incoming)
    assert refresh["status"] == "refresh"
    assert refresh["result"]["outputs"]["energy"] == prior["outputs"]["energy"]
    assert refresh["result"]["outputs"]["remote_mlip_prediction"]["status"] == "completed"
    assert validated_magnetic_refresh(refresh["result"], incoming)["status"] == "unchanged"
