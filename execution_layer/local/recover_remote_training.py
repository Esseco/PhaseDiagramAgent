"""Validate lightweight returned training artifacts; never load or activate models."""
import csv
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path, PurePosixPath

FILES = ("training.finished.json", "models.json", "kfold_metrics.csv",
         "energy_comparison.csv", "force_comparison.csv")
INACTIVE = {"abandoned", "cancelled", "superseded"}


def _csv(path, columns):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name} 缺少必要列")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} 无数据")
    for row in rows:
        for column in columns:
            if not math.isfinite(float(row[column])):
                raise ValueError(f"{path.name} 含非有限数值")
    return rows


def inspect_training_results(job):
    """Read-only inspection, including legacy manifests with unresolved paths."""
    root = Path(job["directory"]) / "results"
    if not root.is_dir() or not any((root / name).exists() for name in FILES):
        return None
    try:
        missing = [name for name in FILES if not (root / name).is_file()]
        if missing:
            raise ValueError("缺少 " + "、".join(missing))
        marker = json.loads((root / FILES[0]).read_text(encoding="utf-8-sig"))
        models = json.loads((root / FILES[1]).read_text(encoding="utf-8-sig"))
        version = job["original_model_version"]
        if marker.get("status") != "completed" or marker.get("original_model_version") != version:
            raise ValueError("完成标记状态或原模型版本不匹配")
        if not isinstance(models, list) or not models:
            raise ValueError("模型清单为空或格式错误")
        ids = [row.get("model_id") for row in models]
        expected = {row["model_id"] for row in job.get("report", {}).get("committees", [])}
        if len(set(ids)) != len(ids) or not all(ids) or (expected and set(ids) != expected):
            raise ValueError("模型成员与已登记训练方案不匹配")
        if sum(row.get("is_main_model") is True for row in models) != 1:
            raise ValueError("主模型必须唯一")
        if marker.get("committee_count") != sum(row.get("is_committee_member") is True for row in models):
            raise ValueError("committee 数量不匹配")
        blocked = []
        for row in models:
            if row.get("source_model_version") != version:
                raise ValueError("模型清单原模型版本不匹配")
            if row.get("training_round") != Path(job["directory"]).name:
                raise ValueError("模型清单训练轮次不匹配")
            remote = row.get("remote_model_path") or row.get("model_path") or ""
            if not PurePosixPath(remote).is_absolute() or ".." in PurePosixPath(remote).parts:
                blocked.append(f"{row['model_id']} 缺少超算绝对模型路径")
            digest = row.get("sha256", "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
                blocked.append(f"{row['model_id']} 缺少有效模型SHA256")
        metric_columns = ("energy_MAE_meV_per_atom", "energy_RMSE_meV_per_atom",
                          "force_MAE_meV_per_A", "force_RMSE_meV_per_A")
        metrics = _csv(root / "kfold_metrics.csv", metric_columns)
        energies = _csv(root / "energy_comparison.csv", ("E_DFT_eV_per_atom", "E_MLIP_eV_per_atom"))
        forces = _csv(root / "force_comparison.csv", ("F_DFT_eV_per_A", "F_MLIP_eV_per_A"))
        totals = [row for row in metrics if row.get("fold") == "all_out_of_fold"]
        if len(totals) != 1 or int(totals[0]["structures"]) != len(energies) or int(totals[0]["force_components"]) != len(forces):
            raise ValueError("K折总计与逐点数据数量不一致")
        if marker.get("out_of_fold_structures") != len(energies):
            raise ValueError("完成标记与能量数据数量不一致")
        folds = {row.get("fold") for row in energies}
        if None in folds or len(folds) != marker.get("folds") or {r.get("fold") for r in forces} != folds:
            raise ValueError("K折划分与完成标记不一致")
        expected_folds = {r["model_id"] for r in job.get("report", {}).get("main_model", {}).get("fold_jobs", [])}
        if expected_folds and folds != expected_folds:
            raise ValueError("K折成员与已登记训练方案不匹配")
        fold_rows = [row for row in metrics if row.get("fold") not in {"all_out_of_fold", "current_round_out_of_fold"}]
        if len(fold_rows) != len(folds) or {r.get("fold") for r in fold_rows} != folds:
            raise ValueError("K折指标缺少逐折记录或包含重复记录")
        fingerprint = hashlib.sha256()
        for name in FILES:
            fingerprint.update(name.encode())
            with (root / name).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    fingerprint.update(chunk)
        return {"status": "metadata_required" if blocked else "validation_required",
                "directory": str(root), "fingerprint": fingerprint.hexdigest(),
                "models": models, "kfold_metrics": totals[0], "issues": blocked,
                "activated": False}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        return {"status": "invalid_results", "directory": str(root), "issues": [str(error)]}


def training_result_message(result):
    if result["status"] == "invalid_results":
        return "训练结果待修正：" + "；".join(result["issues"]) + "。未重新训练或激活。"
    text = "训练结果已回传；K折指标与逐点CSV已校验，模型尚未独立验证或激活。"
    metric = result["kfold_metrics"]
    text += (f"\nK折汇总：能量 MAE/RMSE {float(metric['energy_MAE_meV_per_atom']):.1f}/"
             f"{float(metric['energy_RMSE_meV_per_atom']):.1f} meV/atom；受力 "
             f"{float(metric['force_MAE_meV_per_A']):.1f}/{float(metric['force_RMSE_meV_per_A']):.1f} meV/Å。")
    if result["issues"]:
        return text + "\n模型清单需补齐超算绝对路径及SHA256；在超算inputs目录运行 python collect_training_results.py --manifest-only，回传models.json即可，无需重训或下载模型。"
    return text + "\n下一步需接入远端模型独立验证接口，再进行激活审批；目前仅完成结果登记，不自动激活或重训。K折结果不等于全数据主模型的独立测试。"


def recover_remote_training(state):
    current = deepcopy(state)
    reports = []
    for job in (current.get("remote_finetune_jobs") or {}).values():
        if job.get("status") in INACTIVE or job.get("activated"):
            continue
        result = inspect_training_results(job)
        if result is None:
            continue
        job["returned_results"] = result
        if result["status"] != "invalid_results":
            job["status"] = "results_received"
        elif job.get("status") == "results_received":
            job["status"] = "results_invalid"
        reports.append(result)
    return current, reports
