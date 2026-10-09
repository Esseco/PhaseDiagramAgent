"""Prepare portable MACE inputs after approval; never train on the local host."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import shlex
import re
import yaml

from scientific_layer.training.prepare_mace_finetune import prepare_mace_finetune
from execution_layer.remote.write_training_submission import training_environment


def prepare_remote_finetune(state, config, *, replacement_directory=None):
    current = deepcopy(state)
    settings = deepcopy(config.get("mlip_finetune") or {})
    if not settings.get("committee"):
        settings["committee"] = [{"seed": 2026 + i, "bootstrap_seed": 3026 + i} for i in range(4)]
    root = config.get("upload_batches_directory")
    base = current.get("active_model") or config.get("mlip") or {}
    model_path = base.get("model_path")
    if not root or not model_path or not settings.get("committee"):
        return {"status": "not_configured", "state": current,
                "reason": "微调需配置上传目录、原轮次远端模型路径和训练成员；未训练。"}
    try:
        environment, environment_source = training_environment(config)
    except ValueError as error:
        return {"status": "not_configured", "state": current, "reason": str(error)}
    from scientific_layer.training.cumulative_training_records import cumulative_training_records
    records = cumulative_training_records(current)
    from scientific_layer.training.prepare_main_model import grouped_folds
    try:
        grouped_folds(records, settings)
    except ValueError as error:
        return {"status": "insufficient_new_data", "state": current,
                "reason": f"无法可靠进行分组5折：{error}；未生成输入、未训练。"}
    digest = hashlib.sha256(json.dumps(["training-inputs-v4", records, settings, base, environment], sort_keys=True, default=str).encode()).hexdigest()[:16]
    version = base.get("version") or base.get("name") or current.get("active_model_version") or "unknown-model"
    # The digest names one immutable approved dataset/settings combination.
    existing = (current.get("remote_finetune_jobs") or {}).get(digest)
    from execution_layer.remote.model_upload_directory import model_upload_directory
    epoch_root = model_upload_directory(root, current, version)
    prior = [job for job in (current.get("remote_finetune_jobs") or {}).values()
             if job.get("original_model_version") == version
             and job.get("status") not in {"superseded", "cancelled", "abandoned"}]
    if existing and Path(existing["directory"]).is_dir():
        if existing.get("status") == "results_received":
            from execution_layer.local.recover_remote_training import training_result_message
            return {"status": "training_results_received", "state": current,
                    "reason": training_result_message(existing["returned_results"])}
        submission = Path(existing["directory"]) / "inputs"
        if not submission.is_dir():
            submission = Path(existing["directory"])
        return {"status": "awaiting_remote_training", "state": current,
                "reason": f"微调轮已准备：{existing['directory']}；完整上传该轮，在{submission}内sbatch GPU.sh一次，回传同轮results。未重复生成、本机训练或追加轮次。"}
    for job_key, job in (current.get("remote_finetune_jobs") or {}).items():
        if job not in prior:
            continue
        if not Path(job["directory"]).is_dir():
            return {"status": "confirmation_required", "state": current,
                    "reason": f"训练记录与文件不一致：{job['directory']} 不存在。需先确认恢复或废弃旧方案；未递增编号、未生成。"}
        if job is not existing and job.get("status") in {"inputs_prepared", "awaiting_remote_training"}:
            current["finetune_input_conflict"] = {"job_key": job_key, "directory": job["directory"],
                                                  "model_version": version}
            return {"status": "confirmation_required", "state": current,
                    "reason": f"已有未完成训练输入：{job['directory']}；参数或数据已改变，请先确认是否废弃并重新生成原轮输入。未追加训练轮次、未覆盖。"}
    indices = []
    for job in prior:
        match = re.fullmatch(r"MLIP-finetune-round-(\d+)", Path(job["directory"]).name)
        if not match:
            return {"status": "confirmation_required", "state": current,
                    "reason": "训练记录仍使用旧目录名称，需先核对轮次；未生成。"}
        indices.append(int(match[1]))
    directory = Path(existing["directory"]) if existing else epoch_root / f"MLIP-finetune-round-{max(indices, default=0)+1:04d}"
    if replacement_directory is not None:
        replacement = Path(replacement_directory).resolve()
        abandoned = [job for job in (current.get("remote_finetune_jobs") or {}).values()
                     if job.get("status") == "abandoned" and job.get("original_model_version") == version
                     and Path(job["directory"]).resolve() == replacement]
        if (not abandoned or replacement.parent != epoch_root.resolve()
                or not re.fullmatch(r"MLIP-finetune-round-\d+", replacement.name)):
            raise ValueError("未经确认的微调替换目录")
        directory = replacement
    if directory.exists():
        return {"status": "not_configured", "state": current,
                "reason": f"训练目录已存在但未登记：{directory}；需先确认是否重新生成，未覆盖。"}
    settings.setdefault("training", {})["foundation_model"] = model_path
    settings["training"].setdefault("amsgrad", True)
    settings["training"].setdefault("scaling", "rms_forces_scaling")
    if base.get("mace_head"):
        settings["training"]["foundation_head"] = base["mace_head"]
    settings["training"].pop("minimum_new_dft_records", None)
    settings["split"] = {"train": 1.0, "valid": 0.0, "test": 0.0}
    # Final committee must include every accepted structure, not bootstrap subsamples.
    for member in settings["committee"]:
        member.pop("bootstrap_seed", None)
    inputs = directory / "inputs"
    report = prepare_mace_finetune(records, inputs, config=settings)
    report["dataset_scope"] = {"cumulative_records": len(records),
                               "current_round_records": sum(r["is_current_round"] for r in records),
                               "data_ids": [r.get("data_id") for r in records]}
    report["remote_environment"] = {"name": environment, "source": environment_source}
    report["original_model_version"] = version
    from analysis_layer.state.model_epoch import model_epoch
    report["epoch"] = model_epoch(current, version)
    report["training_round"] = directory.name
    if not report["split_counts"]["train"]:
        return {"status": "insufficient_new_data", "state": current,
                "reason": "按branch分组后训练/验证/测试集有空集；请补充代表性DFT，未训练。"}
    for member in report["committees"]:
        parameters = member["parameters"]
        for split in ("train", "valid", "test"):
            parameters[f"{split}_file"] = "../_shared_data/train.xyz"
        parameters.pop("test_file", None)
        parameters["valid_fraction"] = 0.0
        parameters.setdefault("patience", 20)
        Path(member["config_path"]).write_text(yaml.safe_dump(parameters, sort_keys=False), encoding="utf-8")
    from scientific_layer.training.prepare_main_model import prepare_main_model
    try:
        main_settings = deepcopy(settings)
        index = settings.get("main_model_index", 0)
        if type(index) is not int or not 0 <= index < len(settings["committee"]):
            raise ValueError("main_model_index超出committee范围")
        main_settings["training"].update({key: value for key, value in settings["committee"][index].items()
                                          if key != "bootstrap_seed"})
        report["main_model"] = prepare_main_model(records, inputs, main_settings)
    except ValueError as error:
        return {"status": "insufficient_new_data", "state": current,
                "reason": f"训练输入尚未就绪：{error}；未训练或提交。"}
    command = "python" if environment == "current" else f"conda run --no-capture-output -n {shlex.quote(environment)} python"
    lines = ["#!/bin/bash", "set -euo pipefail", 'cd "$(dirname "$0")"']
    for member in report["committees"]:
        lines.append(f"(cd {shlex.quote(member['model_id'])} && {command} -m mace.cli.run_train --config mace_train.yaml)")
    for member in report["main_model"]["fold_jobs"]:
        lines.append(f"(cd {shlex.quote(member['model_id'])} && {command} -m mace.cli.run_train --config mace_train.yaml)")
    evaluator = Path(__file__).parents[1] / "remote/collect_training_results.py"
    (inputs / "collect_training_results.py").write_text(evaluator.read_text(encoding="utf-8"), encoding="utf-8")
    lines.append(f"{command} collect_training_results.py")
    (inputs / "training_plan.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    # The portable training plan is the authoritative report; the lower-level
    # preparation report is superseded before this directory is published.
    (inputs / "finetune_report.json").unlink(missing_ok=True)
    (inputs / "run_training.sh").write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
    from execution_layer.remote.write_training_submission import write_training_submission
    write_training_submission(inputs, environment)
    (directory / "results").mkdir(exist_ok=True)
    current.setdefault("remote_finetune_jobs", {})[digest] = {"directory": str(directory),
        "status": "inputs_prepared", "original_model_version": version, "report": report,
        "submitted": False, "activated": False}
    return {"status": "awaiting_remote_training", "state": current,
        "reason": f"微调轮：{directory}；完整上传该轮目录，在inputs内sbatch GPU.sh一次。回传该轮results中的指标CSV、models.json和完成标记即可，模型留在超算inputs。未本机训练/提交；后续仍需独立验证与单独激活审批。"}
