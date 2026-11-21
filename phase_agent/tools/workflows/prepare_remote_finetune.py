"""Prepare portable MACE inputs after approval; never train on the local host."""

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import re
import yaml

from phase_agent.science.training.prepare_mace_finetune import prepare_mace_finetune
from phase_agent.tools.remote.write_training_submission import training_environment
from phase_agent.runtime.turn_process import timed_call


def prepare_remote_finetune(
    state,
    config,
    *,
    replacement_directory=None,
    training_action=None,
    transition_policy=None,
    transition_approval_reason=None,
):
    current = timed_call("finetune_state_copy", deepcopy, state, category="training_inputs")
    from phase_agent.tools.workflows.manual_training_coexistence import blocking_tasks

    if blocking_tasks(current, training_action):
        return {
            "status": "tasks_awaiting_results",
            "state": current,
            "reason": "model_training_waits_for_inflight_tasks",
        }
    settings = deepcopy(config.get("mlip_finetune") or {})
    from phase_agent.policies.training_transition import (
        resolve_training_transition,
        same_training_transition,
    )

    transition = resolve_training_transition(
        config,
        requested=transition_policy,
        approval_reason=transition_approval_reason,
    )
    if not settings.get("committee"):
        settings["committee"] = [{"seed": 2026 + i, "bootstrap_seed": 3026 + i} for i in range(4)]
    root = config.get("upload_batches_directory")
    base = current.get("active_model") or config.get("mlip") or {}
    from phase_agent.analysis.convergence.training_provenance import model_with_training_provenance

    base = model_with_training_provenance(current, base)
    # The active model identifies the search epoch; every training run starts
    # from the original foundation model recorded in the confirmed config.
    initial_mlip = (current.get("confirmed_config") or {}).get("mlip") or {}
    foundation_path = (settings.get("training") or {}).get("foundation_model")
    model_path = initial_mlip.get("model_path") or foundation_path
    if not model_path and not current.get("active_model"):
        model_path = (config.get("mlip") or {}).get("model_path")
    if not root or not model_path or not settings.get("committee"):
        return {
            "status": "not_configured",
            "state": current,
            "reason": "微调需配置上传目录、原轮次远端模型路径和训练成员；未训练。",
        }
    try:
        environment, environment_source = training_environment(config)
    except ValueError as error:
        return {"status": "not_configured", "state": current, "reason": str(error)}
    from phase_agent.science.training.cumulative_training_records import (
        cumulative_training_records,
    )

    records = timed_call(
        "finetune_dataset", cumulative_training_records, current, category="training_inputs"
    )
    holdout_policy = current.get("training_preflight_policy")
    holdout_records, holdout_plan = [], None
    if holdout_policy:
        from phase_agent.science.training.training_preflight_selection import (
            selected_records, split_independent_holdout,
        )

        records = selected_records(current, holdout_policy)
        records, holdout_records, holdout_plan = split_independent_holdout(
            records, base, settings, holdout_policy
        )
        from phase_agent.science.training.training_preflight_selection import (
            group_equivalent_training_records,
        )

        records, cv_grouping = group_equivalent_training_records(records, settings)
        settings["group_keys"] = ["independent_group_id"]
        holdout_plan["cv_grouping"] = cv_grouping
    from phase_agent.science.training.prepare_main_model import grouped_folds

    try:
        timed_call(
            "finetune_group_validation",
            grouped_folds,
            records,
            settings,
            category="training_inputs",
        )
    except ValueError as error:
        return {
            "status": "insufficient_new_data",
            "state": current,
            "reason": f"无法可靠进行分组5折：{error}；未生成输入、未训练。",
        }
    digest_payload = ["training-inputs-v5", records, settings, base, model_path, environment]
    if holdout_plan is not None:
        digest_payload.extend(["independent-holdout-v1", holdout_policy, holdout_plan])
    # Keep the historical digest stable when the legacy manual path is used;
    # an explicit transition is part of the immutable handoff identity.
    if transition is not None:
        digest_payload.extend(["post-training-transition-v1", transition])
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    version = (
        base.get("version")
        or base.get("name")
        or current.get("active_model_version")
        or "unknown-model"
    )
    # The digest names one immutable approved dataset/settings combination.
    existing = (current.get("remote_finetune_jobs") or {}).get(digest)
    if existing and not same_training_transition(
        existing.get("post_training_transition"), transition
    ):
        return {
            "status": "confirmation_required",
            "state": current,
            "reason": "已有训练交接的模型切换策略不同；未修改不可变训练记录。",
        }
    from phase_agent.tools.remote.model_upload_directory import model_upload_directory

    epoch_root = model_upload_directory(root, current, version)
    prior = [
        job
        for job in (current.get("remote_finetune_jobs") or {}).values()
        if job.get("original_model_version") == version
        and job.get("status") not in {"superseded", "cancelled", "abandoned"}
    ]
    if existing and Path(existing["directory"]).is_dir():
        if existing.get("status") == "results_received":
            from phase_agent.tools.local.recover_remote_training import (
                training_result_message,
            )

            return {
                "status": "training_results_received",
                "state": current,
                "reason": training_result_message(existing["returned_results"]),
            }
        submission = Path(existing["directory"]) / "inputs"
        if not submission.is_dir():
            submission = Path(existing["directory"])
        from phase_agent.tools.remote.submission_storage import restore_storage_directory
        from phase_agent.tools.remote.write_training_submission import prepare_training_handoff

        restore_storage_directory(Path(existing["directory"]) / "inputs", "inputs")
        handoff = existing.get("handoff")
        if not handoff or not Path(handoff["upload_archive"]).is_file():
            handoff = prepare_training_handoff(Path(existing["directory"]), submission)
            existing["handoff"] = handoff
        return {
            "status": "awaiting_remote_training",
            "handoffs": [deepcopy(handoff)],
            "submitted": False,
            "state": current,
            "reason": f"上传文件夹：{handoff['folder']}；上传后执行 sbatch submit_all.sh。完成后把远端 return-results 文件夹传回 {handoff.get('return_directory') or handoff['results_directory']}，再说“继续”。无需手动解压；已提交的不要重复提交。",
        }
    for job_key, job in (current.get("remote_finetune_jobs") or {}).items():
        if job not in prior:
            continue
        if not Path(job["directory"]).is_dir():
            return {
                "status": "confirmation_required",
                "state": current,
                "reason": f"训练记录与文件不一致：{job['directory']} 不存在。需先确认恢复或废弃旧方案；未递增编号、未生成。",
            }
        if job is not existing and job.get("status") in {
            "inputs_prepared",
            "awaiting_remote_training",
        }:
            current["finetune_input_conflict"] = {
                "job_key": job_key,
                "directory": job["directory"],
                "model_version": version,
            }
            return {
                "status": "confirmation_required",
                "state": current,
                "reason": f"已有未完成训练输入：{job['directory']}；参数或数据已改变，请先确认是否废弃并重新生成原轮输入。未追加训练轮次、未覆盖。",
            }
    indices = []
    for job in prior:
        match = re.fullmatch(r"MLIP-finetune-round-(\d+)", Path(job["directory"]).name)
        if not match:
            return {
                "status": "confirmation_required",
                "state": current,
                "reason": "训练记录仍使用旧目录名称，需先核对轮次；未生成。",
            }
        indices.append(int(match[1]))
    directory = (
        Path(existing["directory"])
        if existing
        else epoch_root / f"MLIP-finetune-round-{max(indices, default=0) + 1:04d}"
    )
    if replacement_directory is not None:
        replacement = Path(replacement_directory).resolve()
        abandoned = [
            job
            for job in (current.get("remote_finetune_jobs") or {}).values()
            if job.get("status") == "abandoned"
            and job.get("original_model_version") == version
            and Path(job["directory"]).resolve() == replacement
        ]
        if (
            not abandoned
            or replacement.parent != epoch_root.resolve()
            or not re.fullmatch(r"MLIP-finetune-round-\d+", replacement.name)
        ):
            raise ValueError("未经确认的微调替换目录")
        directory = replacement
    if directory.exists():
        return {
            "status": "not_configured",
            "state": current,
            "reason": f"训练目录已存在但未登记：{directory}；需先确认是否重新生成，未覆盖。",
        }
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
    report = timed_call(
        "finetune_committee_files",
        prepare_mace_finetune,
        records,
        inputs,
        config=settings,
        category="training_inputs",
    )
    if holdout_plan is not None:
        from ase.io import write
        from phase_agent.science.training.prepare_mace_finetune import _to_atoms

        test_path = inputs / "_shared_data" / "test.xyz"
        write(test_path, [_to_atoms(row, settings.get("labels", {})) for row in holdout_records], format="extxyz")
        report["split_counts"]["test"] = len(holdout_records)
        report["split_groups"]["test"] = holdout_plan["groups"]
        report["independent_holdout"] = {
            **holdout_plan, "status": "reserved_not_evaluated",
            "count": len(holdout_records), "relative_path": "_shared_data/test.xyz",
            "policy": deepcopy(holdout_policy),
        }
    from phase_agent.analysis.convergence.independent_evidence import training_snapshot

    report["training_snapshot"] = training_snapshot(records, base)
    report["dataset_scope"] = {
        "cumulative_records": len(records),
        "current_round_records": sum(r["is_current_round"] for r in records),
        "data_ids": [r.get("data_id") for r in records],
    }
    report["remote_environment"] = {"name": environment, "source": environment_source}
    report["original_model_version"] = version
    report["post_training_transition"] = deepcopy(transition)
    from phase_agent.evidence.model_epoch import model_epoch

    report["epoch"] = model_epoch(current, version)
    report["training_round"] = directory.name
    if not report["split_counts"]["train"]:
        return {
            "status": "insufficient_new_data",
            "state": current,
            "reason": "按branch分组后训练/验证/测试集有空集；请补充代表性DFT，未训练。",
        }
    for member in report["committees"]:
        parameters = member["parameters"]
        for split in ("train", "valid", "test"):
            parameters[f"{split}_file"] = "../_shared_data/train.xyz"
        parameters.pop("test_file", None)
        parameters["valid_fraction"] = 0.0
        parameters.setdefault("patience", 20)
        Path(member["config_path"]).write_text(
            yaml.safe_dump(parameters, sort_keys=False), encoding="utf-8"
        )
    from phase_agent.science.training.prepare_main_model import prepare_main_model

    try:
        main_settings = deepcopy(settings)
        index = settings.get("main_model_index", 0)
        if type(index) is not int or not 0 <= index < len(settings["committee"]):
            raise ValueError("main_model_index超出committee范围")
        main_settings["training"].update(
            {
                key: value
                for key, value in settings["committee"][index].items()
                if key != "bootstrap_seed"
            }
        )
        report["main_model"] = timed_call(
            "finetune_cv_files",
            prepare_main_model,
            records,
            inputs,
            main_settings,
            category="training_inputs",
        )
    except ValueError as error:
        return {
            "status": "insufficient_new_data",
            "state": current,
            "reason": f"训练输入尚未就绪：{error}；未训练或提交。",
        }
    if holdout_plan is not None:
        report["main_model"]["independent_holdout_reserved"] = True
        report["main_model"]["independent_test"] = False
        report["main_model"]["note"] = "委员会和交叉验证仅使用训练部分；独立测试集已保留，尚未评估。"
    models = [
        row["model_id"] for row in [*report["committees"], *report["main_model"]["fold_jobs"]]
    ]
    report["execution"] = {
        "mode": "slurm_array",
        "training_models": models,
        "gpus_per_job": 1,
        "max_parallel_training_jobs": min(4, len(models)),
        "collection_dependency": "afterany",
        "collection_gpu_jobs": 1,
    }
    evaluator = Path(__file__).parents[1] / "remote/collect_training_results.py"
    (inputs / "collect_training_results.py").write_text(
        evaluator.read_text(encoding="utf-8"), encoding="utf-8"
    )
    bundle_source = Path(__file__).parents[1] / "remote/transfer_bundle.py"
    (inputs / "transfer_bundle.py").write_text(
        bundle_source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    worker_source = Path(__file__).parents[1] / "remote/run_training_task.py"
    (inputs / worker_source.name).write_text(
        worker_source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (inputs / "training_plan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    # The portable training plan is the authoritative report; the lower-level
    # preparation report is superseded before this directory is published.
    (inputs / "finetune_report.json").unlink(missing_ok=True)
    (inputs / "run_training.sh").write_text(
        '#!/bin/bash\nset -euo pipefail\ncd "$(dirname "$0")"\n'
        ': "${SLURM_ARRAY_TASK_ID:?Submit the training array through submit_all.sh}"\n'
        "python run_training_task.py\n",
        encoding="utf-8",
        newline="\n",
    )
    from phase_agent.tools.remote.write_training_submission import (
        write_training_submission,
    )

    timed_call(
        "finetune_submission_script",
        write_training_submission,
        inputs,
        environment,
        category="training_inputs",
    )
    (directory / "results").mkdir(exist_ok=True)
    from phase_agent.tools.remote.write_training_submission import prepare_training_handoff

    handoff = timed_call(
        "finetune_delivery_bundle",
        prepare_training_handoff,
        directory,
        inputs,
        category="training_inputs",
    )
    current.setdefault("remote_finetune_jobs", {})[digest] = {
        "upload_archive": handoff["upload_archive"],
        "directory": str(directory),
        "status": "inputs_prepared",
        "original_model_version": version,
        "report": report,
        "submitted": False,
        "activated": False,
    }
    if transition is not None:
        current["remote_finetune_jobs"][digest]["post_training_transition"] = transition
    current["remote_finetune_jobs"][digest]["handoff"] = handoff
    from phase_agent.tools.remote.submission_layout import write_submission_index

    timed_call(
        "finetune_submission_index",
        write_submission_index,
        root,
        current,
        category="training_inputs",
    )
    return {
        "status": "awaiting_remote_training",
        "handoffs": [deepcopy(handoff)],
        "submitted": False,
        "state": current,
        "reason": f"上传文件夹：{handoff['folder']}；上传后执行 sbatch submit_all.sh。完成后把对应批次 _output 文件夹传回 {handoff['return_directory']}，再说“继续”。无需手动解压；已提交的不要重复提交。",
    }
