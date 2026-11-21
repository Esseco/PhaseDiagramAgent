"""Publish K-fold out-of-fold training figures from registered returned evidence."""

import csv
import hashlib
import json
from pathlib import Path
import re

from phase_agent.analysis.feedback.export_dft_products import _publish_bytes
from phase_agent.evidence.model_epoch import model_epoch
from phase_agent.tools.local.recover_remote_training import FILES


def _verified(job, key):
    result = job.get("returned_results") or {}
    fingerprint = result.get("fingerprint") or ""
    if len(fingerprint) != 64 or result.get("status") not in {
        "validation_required",
        "metadata_required",
    }:
        raise ValueError("训练结果未登记或未通过回收校验")
    root = Path(job["directory"]) / "results"
    selected = job.get("model_manifest_file") or (job.get("training_handoff") or {}).get(
        "manifest_result"
    )
    manifest = Path(selected) if selected else root / "models.json"
    if manifest.resolve().parent != root.resolve():
        raise ValueError("模型清单路径不属于该训练结果")
    paths = {name: root / name for name in FILES}
    paths["models.json"] = manifest
    digest = hashlib.sha256()
    for name in FILES:
        digest.update(name.encode())
        with paths[name].open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    if digest.hexdigest() != fingerprint:
        raise ValueError("训练文件与已登记指纹不一致")
    version = (job.get("training_handoff") or {}).get("candidate_model_version")
    expected = f"remote-{key}-{fingerprint[:12]}"
    if version and version != expected:
        raise ValueError("训练任务与模型版本绑定不一致")
    marker = json.loads(paths["training.finished.json"].read_text(encoding="utf-8-sig"))
    if marker.get("status") != "completed" or marker.get("original_model_version") != job.get(
        "original_model_version"
    ):
        raise ValueError("训练完成标记或来源模型不一致")

    def rows(name):
        with paths[name].open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    energies, forces, metrics = (
        rows("energy_comparison.csv"),
        rows("force_comparison.csv"),
        rows("kfold_metrics.csv"),
    )
    totals = [r for r in metrics if r.get("fold") == "all_out_of_fold"]
    if (
        len(totals) != 1
        or int(totals[0]["structures"]) != len(energies)
        or int(totals[0]["force_components"]) != len(forces)
        or marker.get("out_of_fold_structures") != len(energies)
    ):
        raise ValueError("折外点数与训练汇总不一致")
    folds = {r.get("fold") for r in energies}
    if (
        None in folds
        or len(folds) != marker.get("folds")
        or {r.get("fold") for r in forces} != folds
    ):
        raise ValueError("折外数据折数不一致")
    energy = [
        {
            "comparison_status": "completed",
            "task_id": r["task_id"],
            "dft_energy_eV_per_atom": float(r["E_DFT_eV_per_atom"]),
            "mlip_energy_eV_per_atom": float(r["E_MLIP_eV_per_atom"]),
        }
        for r in energies
    ]
    force = [
        {
            "comparison_status": "completed",
            "task_id": r["task_id"],
            "dft_force_eV_per_A": float(r["F_DFT_eV_per_A"]),
            "mlip_force_eV_per_A": float(r["F_MLIP_eV_per_A"]),
        }
        for r in forces
    ]
    return fingerprint, version, energy, force


def publish_training_parity(state, directory):
    from phase_agent.analysis.feedback.dft_parity_plots import export_parity_plots

    root = Path(directory).resolve()
    if root.name != "analysis_outputs" or not root.is_dir():
        raise ValueError("分析输出目录未登记")
    from phase_agent.evidence.dft_extrapolation import failed_dft_task_ids

    failed_ids = failed_dft_task_ids(state)
    active = state.get("active_model_version")
    outcomes = []
    candidates = []
    for key, job in (state.get("remote_finetune_jobs") or {}).items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(key)):
            continue
        epoch = model_epoch(state, job.get("original_model_version"))
        row = {"job_id": key, "source_epoch": epoch, "status": "unavailable",
               "model_version": (job.get("training_handoff") or {}).get("candidate_model_version")}
        try:
            if not epoch:
                raise ValueError("无法确认训练来源轮次")
            fingerprint, version, energies, forces = _verified(job, key)
            excluded = sorted({r["task_id"] for r in energies if r["task_id"] in failed_ids})
            energies = [r for r in energies if r["task_id"] not in failed_ids]
            forces = [r for r in forces if r["task_id"] not in failed_ids]
            row["excluded_extrapolation_failed_task_ids"] = excluded
            round_name = Path(job["directory"]).name
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", round_name) or round_name in {".", ".."}:
                raise ValueError("训练轮次名称不可安全用于目录")
            parent = root / epoch / "KFold_Error"
            folder = parent / round_name
            number = 2
            while folder.exists():
                identity = folder / "training_identity.json"
                if (
                    identity.is_file()
                    and json.loads(identity.read_text(encoding="utf8")).get("job_id") == key
                ):
                    break
                folder = parent / f"{round_name}-{number:02d}"
                number += 1
            if not energies or not forces:
                if folder.is_dir():
                    import uuid

                    archive = (
                        root / "history" / "training_parity" / ("filtered-" + uuid.uuid4().hex)
                    )
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    folder.rename(archive)
                raise ValueError("剔除外推失败结构后没有完整折外配对")
            label = f"K-fold out-of-fold | {epoch} | {key}"
            if excluded:
                from phase_agent.analysis.feedback.dft_parity_plots import archive_filtered_parity

                archive_filtered_parity(root, folder, energies, forces, label)
            plot = export_parity_plots(energies, forces, folder, label)
            if plot.get("status") != "completed":
                raise ValueError(plot.get("reason", "绘图失败"))
            _publish_bytes(
                folder / "training_identity.json",
                (
                    json.dumps(
                        {"job_id": key, "fingerprint": fingerprint}, ensure_ascii=False, indent=2
                    )
                    + "\n"
                ).encode("utf8"),
            )
            row.update(
                status="completed",
                model_version=version,
                fingerprint=fingerprint,
                directory=str(folder),
                metrics=plot["metrics"],
            )
            _publish_bytes(
                folder / "README.md",
                (
                    f"# K折折外预测 | {epoch} | {key}\n\n训练方法的折外评估，不是最终全数据模型的独立测试；图中剔除已确认外推失败的结构，原折模型没有重新训练。\n\n[能量](energy_parity.png) · [受力](force_parity.png) · [指标](parity_metrics.json)\n"
                ).encode("utf8"),
            )
            if version == active and job.get("activated") is True:
                candidates.append((row, energies, forces))
        except (OSError, ValueError, KeyError, TypeError) as error:
            row["reason"] = str(error)
        outcomes.append(row)
    latest = root / "latest" / "KFold_Error"
    old_status_path = root / "latest" / "training_parity_status.json"
    try:
        old_status = json.loads(old_status_path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        old_status = {}
    current_identity = candidates[0][0]["fingerprint"] if len(candidates) == 1 else None
    if latest.is_dir() and (
        not current_identity or old_status.get("fingerprint") != current_identity
    ):
        import uuid

        archive = root / "history" / "training_parity" / ("previous-" + uuid.uuid4().hex)
        archive.parent.mkdir(parents=True, exist_ok=True)
        latest.rename(archive)
    status = {
        "status": "unavailable",
        "model_version": active,
        "reason": "当前模型没有唯一匹配且已校验的微调折外结果",
    }
    if len(candidates) == 1:
        row, energies, forces = candidates[0]
        label = f"K-fold out-of-fold | {row['source_epoch']} | {row['job_id']}"
        if row.get("excluded_extrapolation_failed_task_ids"):
            from phase_agent.analysis.feedback.dft_parity_plots import archive_filtered_parity

            archive_filtered_parity(root, latest, energies, forces, label)
        plot = export_parity_plots(energies, forces, latest, label)
        if plot.get("status") == "completed":
            status = {
                "status": "completed",
                "model_version": active,
                "job_id": row["job_id"],
                "source_epoch": row["source_epoch"],
                "fingerprint": row["fingerprint"],
                "metrics": plot["metrics"],
            }
    _publish_bytes(
        root / "latest" / "training_parity_status.json",
        (json.dumps(status, ensure_ascii=False, indent=2) + "\n").encode("utf8"),
    )
    report = {
        "latest": status,
        "jobs": outcomes,
        "interpretation": "K折折外预测；非最终模型的独立测试",
    }
    _publish_bytes(
        root / "training_parity_index.json",
        (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf8"),
    )
    from phase_agent.analysis.feedback.kfold_error_trend import publish_kfold_error_trend

    report["trend"] = publish_kfold_error_trend(state, root, outcomes)
    lines = [
        "# 微调训练 K 折折外对比图",
        "",
        "[按模型 epoch 的能量误差](latest/KFold_Error_E_MAE_by_epoch.png) · [受力误差](latest/KFold_Error_F_MAE_by_epoch.png) · [数据和不可用原因](latest/KFold_Error_MAE_by_epoch.csv)",
        "",
        "K 折折外预测评估训练方法，不是最终全数据模型的独立测试；图中剔除已确认外推失败的结构，原折模型没有重新训练。",
        "",
    ]
    if status["status"] == "completed":
        lines.append(
            "当前模型：[能量](latest/KFold_Error/energy_parity.png) · [受力](latest/KFold_Error/force_parity.png)"
        )
    else:
        lines.append("当前模型不可用：" + str(status["reason"]))
    for item in outcomes:
        lines.append("")
        title = f"{item.get('source_epoch') or '轮次未知'} · {item['job_id']}"
        if item["status"] == "completed":
            folder = Path(item["directory"]).relative_to(root).as_posix()
            lines.append(
                f"- {title}：[能量]({folder}/energy_parity.png) · [受力]({folder}/force_parity.png) · [说明]({folder}/README.md)"
            )
        else:
            lines.append(f"- {title}：未生成；{item.get('reason', '未知原因')}")
    _publish_bytes(root / "training_parity.md", ("\n".join(lines) + "\n").encode("utf8"))
    return report
