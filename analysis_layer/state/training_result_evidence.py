"""Read lightweight training reports; never activate or certify model files."""
import csv
import hashlib
import json
import math
from pathlib import Path


def training_result_evidence(state):
    reports = []
    for job_id, job in (state.get("remote_finetune_jobs") or {}).items():
        if job.get("status") == "abandoned" or not job.get("directory"):
            continue
        directory = Path(job["directory"]) / "results"
        marker = directory / "training.finished.json"
        if not marker.is_file():
            continue
        report = {"job_id": job_id, "source_model_version": job.get("original_model_version"),
                  "workflow_status": job.get("status"), "activated": job.get("activated") is True,
                  "source": str(directory), "status": "invalid_report"}
        try:
            paths = [marker, directory / "kfold_metrics.csv", directory / "models.json"]
            raw = []
            for path in paths:
                if path.stat().st_size > 2_000_000:
                    raise ValueError("training report exceeds size limit")
                raw.append(path.read_bytes())
            finished = json.loads(raw[0])
            rows = list(csv.DictReader(raw[1].decode("utf-8-sig").splitlines()))
            models = json.loads(raw[2])
            if finished.get("status") != "completed":
                raise ValueError("training completion not reported")
            if finished.get("original_model_version") != job.get("original_model_version"):
                raise ValueError("source model version mismatch")
            totals = [row for row in rows if row.get("fold") == "all_out_of_fold"]
            if len(totals) != 1 or not isinstance(models, list):
                raise ValueError("missing unique out-of-fold metrics or model manifest")
            total = totals[0]
            count = int(total["structures"])
            if count <= 0 or count != finished.get("out_of_fold_structures"):
                raise ValueError("out-of-fold count mismatch")
            if len(models) != finished.get("committee_count"):
                raise ValueError("committee manifest count mismatch")
            metrics = {key: float(total[key]) for key in (
                "energy_MAE_meV_per_atom", "energy_RMSE_meV_per_atom",
                "force_MAE_meV_per_A", "force_RMSE_meV_per_A")}
            if any(not math.isfinite(value) or value < 0 for value in metrics.values()):
                raise ValueError("invalid training metrics")
            report.update(status="completed_report_not_model_validation", structures=count,
                metrics=metrics, evaluation_type=total.get("evaluation_type"),
                committee_count=len(models),
                model_files_verified=False,
                report_id="training-report:" + hashlib.sha256(b"".join(raw)).hexdigest()[:16],
                limitation="Fold validation can select checkpoints; final full-data members and QBC need independent checks. No automatic activation.")
        except (OSError, ValueError, KeyError, TypeError) as error:
            report["error"] = str(error)
        reports.append(report)
    return reports
