"""Publish version/round-scoped DFT datasets and plotting CSVs, without inference."""

import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import uuid

from phase_agent.analysis.feedback.dft_comparison_tables import (
    ENERGY_FIELDS,
    FORCE_FIELDS,
    METRIC_FIELDS,
    build_comparison_tables,
    comparison_metrics,
)
from phase_agent.analysis.phase.phase_snapshot_paths import model_output_directory
from phase_agent.analysis.feedback.dft_export_reuse import (
    report_manifest,
    report_version,
    reusable_metrics,
)
from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory
from phase_agent.evidence.model_epoch import model_epoch


def dft_product_path(root, name):
    category = (
        "training"
        if name == "training.json"
        else "comparisons"
        if name
        in {"energy_comparison.csv", "force_comparison.csv", "metrics.csv", "mlip_dft_metrics.json"}
        else "diagnostics"
    )
    return Path(root) / category / name


def export_dft_products(state, directory, *, check_reuse_only=False):
    """Refresh saved products only; keep old directories and unchanged files intact."""
    if directory is None:
        return True if check_reuse_only else None
    from phase_agent.science.dft.spin_acceptance import spin_standard_passed
    from phase_agent.evidence.dft_extrapolation import failed_dft_task_ids

    failed_ids = failed_dft_task_ids(state)
    groups = {}
    for record in state.get("dft_dataset_records", []):
        groups.setdefault(_scope_key(record["round_scope"]), []).append(record)
    planned, destinations = [], set()
    for key, records in sorted(groups.items()):
        identifier = hashlib.sha256(key.encode()).hexdigest()[:16]
        scope = json.loads(key)
        tasks = _round_tasks(state, scope, records)
        root = _product_directory(directory, state, scope, tasks, identifier)
        if root in destinations:
            raise ValueError(f"conflicting DFT scopes share output directory: {root}")
        destinations.add(root)
        planned.append((key, identifier, scope, records, tasks, root))
    exports, version_summaries = {}, {}
    previous = state.get("dft_result_exports") or {}
    report_cache = analysis_metadata_directory(directory) / "dft_report_versions"
    for key, identifier, scope, records, tasks, root in planned:
        records = sorted(records, key=lambda row: str(row["task_id"]))
        comparisons = [
            row
            for row in state.get("dft_mlip_comparisons", [])
            if _scope_key(row["round_scope"]) == key
        ]
        eligible_records = [row for row in records if row.get("task_id") not in failed_ids]
        excluded_ids = sorted(
            {row["task_id"] for row in records if row.get("task_id") in failed_ids}
        )
        version = scope.get("model_version") or "unknown-model"
        epoch = model_epoch(state, version)
        version_root = model_output_directory(directory, version, state=state)
        training = sorted(
            [
                row
                for row in state.get("dft_training_records", [])
                if _scope_key(row["round_scope"]) == key
                and row.get("checks_passed", True) is True
                and row.get("task_id") not in failed_ids
                and spin_standard_passed(row)
            ],
            key=lambda row: str(row.get("data_id")),
        )
        pending = [
            row["task_id"]
            for row in tasks
            if row.get("status") in {"pending", "running"}
            and row.get("recovery_wait_waived") is not True
        ]
        expected = len({row["task_id"] for row in tasks} | {row["task_id"] for row in records})
        paths = [
            dft_product_path(root, name)
            for name in (
                "training.json",
                "dft_records.json",
                "mlip_dft_metrics.json",
                "energy_comparison.csv",
                "force_comparison.csv",
                "magnetic_moments.csv",
                "metrics.csv",
            )
        ]
        manifest = report_cache / f"{identifier}.json"
        try:
            source_version = report_version(
                str(root),
                str(version_root),
                scope,
                epoch,
                records,
                comparisons,
                training,
                excluded_ids,
                expected,
                sorted(pending),
            )
        except (TypeError, ValueError):
            source_version = None
        entry = previous.get(identifier)
        saved_metrics = reusable_metrics(root, paths, entry, source_version, manifest, paths[2])
        if isinstance(saved_metrics, dict):
            try:
                metric_rows = _metric_rows(saved_metrics, root.name)
            except (ValueError, TypeError, KeyError):
                pass
            else:
                for row in metric_rows:
                    row["epoch"] = epoch
                version_summaries.setdefault(version_root, []).extend(metric_rows)
                exports[identifier] = entry
                continue
        if check_reuse_only:
            return False
        energies, forces = build_comparison_tables(
            eligible_records, comparisons, round_name=root.name
        )
        for row in energies + forces:
            row["epoch"] = epoch
        from phase_agent.analysis.feedback.dft_magnetic_tables import magnetic_rows, MAGNETIC_FIELDS

        metrics = comparison_metrics(energies, forces, scope)
        metrics.update(
            recovered_tasks=len(records),
            expected_tasks=expected,
            pending_tasks=len(pending),
            pending_task_ids=sorted(pending),
            excluded_extrapolation_failed_task_ids=excluded_ids,
        )
        metric_rows = _metric_rows(metrics, root.name)
        for row in metric_rows:
            row["epoch"] = epoch
        magnetism = magnetic_rows(records)
        for row in magnetism:
            row.update(
                epoch=epoch, search_group_index=scope.get("search_group_index"), dft_round=root.name
            )
        for name, payload in {
            "training.json": training,
            "dft_records.json": records,
            "mlip_dft_metrics.json": metrics,
        }.items():
            _publish_bytes(
                dft_product_path(root, name),
                (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
                    "utf-8"
                ),
            )
        for name, fields, rows in (
            ("energy_comparison.csv", ENERGY_FIELDS, energies),
            ("force_comparison.csv", FORCE_FIELDS, forces),
            ("magnetic_moments.csv", MAGNETIC_FIELDS, magnetism),
            ("metrics.csv", METRIC_FIELDS, metric_rows),
        ):
            _publish_csv(dft_product_path(root, name), fields, rows)
        from phase_agent.analysis.feedback.dft_magnetic_plots import export_magnetic_plots

        magnetic_plots = export_magnetic_plots(magnetism, root / "diagnostics" / "magnetic_plots")
        from phase_agent.analysis.feedback.dft_parity_plots import export_parity_plots

        label = f"{epoch or 'Epoch unrecorded'} ({version})"
        if excluded_ids:
            from phase_agent.analysis.feedback.dft_parity_plots import archive_filtered_parity

            archive_filtered_parity(
                directory, root / "comparisons" / "plots", energies, forces, label
            )
        plots = export_parity_plots(energies, forces, root / "comparisons" / "plots", label)
        version_summaries.setdefault(version_root, []).extend(metric_rows)
        exports[identifier] = {
            "directory": str(root),
            "round_scope": scope,
            "parity_plots": plots,
            "magnetic_plots": magnetic_plots,
            "records": len(records),
            "matched_structures": metrics["matched_structures"],
            "expected_tasks": expected,
            "pending_tasks": len(pending),
            "energy_csv_path": str(dft_product_path(root, "energy_comparison.csv")),
            "force_csv_path": str(dft_product_path(root, "force_comparison.csv")),
            "magnetic_csv_path": str(dft_product_path(root, "magnetic_moments.csv")),
            "metrics_csv_path": str(dft_product_path(root, "metrics.csv")),
            "round_metrics_csv_path": str(version_root / "round_metrics.csv"),
        }
        receipt = report_manifest(root, paths, exports[identifier], source_version)
        if receipt is not None:
            try:
                _publish_bytes(manifest, receipt)
            except OSError:
                pass  # Reuse is optional; the verified report has already been published.
    if check_reuse_only:
        return set(exports) == set(previous) and all(
            item.get("round_metrics_csv_path") and Path(item["round_metrics_csv_path"]).is_file()
            for item in exports.values()
        )
    for root, rows in version_summaries.items():
        _publish_csv(
            root / "round_metrics.csv",
            METRIC_FIELDS,
            sorted(
                rows,
                key=lambda row: (str(row["search_group_index"]), row["dft_round"], row["metric"]),
            ),
        )
    state["dft_result_exports"] = exports
    if Path(directory).name == "analysis_outputs":
        _write_round_parity_index(state, directory, exports)
    from phase_agent.evidence.summarize_model_rounds import (
        summarize_model_rounds,
        ROUND_SUMMARY_FIELDS,
    )

    state["model_round_summary"] = summarize_model_rounds(state)
    summary_path = analysis_metadata_directory(directory) / "round_summary.csv"
    _publish_csv(summary_path, ROUND_SUMMARY_FIELDS, state["model_round_summary"])
    state["round_summary_csv_path"] = str(summary_path)
    if Path(directory).name == "analysis_outputs":
        from phase_agent.analysis.feedback.round_error_trend import publish_round_error_trend

        publish_round_error_trend(state, directory)
    publish_output_catalog(state, directory, action="dft")


def _write_round_parity_index(state, directory, exports):
    """Describe only actually verified, same-frame predictions for each DFT round."""
    from os.path import relpath
    from phase_agent.evidence.model_epoch import model_epoch

    root = Path(directory)
    active = state.get("active_model_version")
    lines = [
        "# 本轮 MLIP 与 DFT 同帧对比",
        "",
        "使用每轮选出并实际回收的 DFT 结构及对应轮次的冻结 MLIP 预测。",
        "DFT 弛豫任务使用已核验最终帧；未完成配对的结构不进入图。",
        "这不是 K 折误差图，也不自动代表独立测试。",
        "",
        f"当前模型：`{active or '未登记'}`",
        "",
    ]
    for item in sorted(exports.values(), key=lambda row: str(row.get("directory"))):
        scope = item.get("round_scope") or {}
        model = scope.get("model_version")
        epoch = model_epoch(state, model) or "轮次未登记"
        plot = item.get("parity_plots") or {}
        path = Path(item["directory"])
        prefix = Path(relpath(path, root)).as_posix()
        target = None
        if epoch != "轮次未登记":
            group = scope.get("search_group_index")
            group_name = (
                f"Search-group-{int(group):04d}" if group is not None else "Search-group-unassigned"
            )
            target = root / epoch / "Round_Error" / group_name / path.name
        if target is not None and plot.get("status") != "completed" and target.is_dir():
            history = root / "history" / epoch / "round_error_unavailable"
            history.mkdir(parents=True, exist_ok=True)
            target.rename(history / uuid.uuid4().hex)
        if target is not None and plot.get("status") == "completed":
            for name in (
                "energy_parity.png",
                "force_parity.png",
            ):
                source = path / "comparisons" / "plots" / name
                if source.is_file():
                    _publish_bytes(target / name, source.read_bytes())
            _publish_bytes(
                target / "README.md",
                (
                    f"# 本轮 MLIP 与 DFT 同帧比较\n\n模型：{model}\n\n数据来源：{path}\n\nDFT 弛豫使用最终帧。\n"
                ).encode("utf8"),
            )
            prefix = Path(relpath(target, root)).as_posix()
        lines.append(f"## {epoch} · {path.name}")
        lines.append(
            f"模型：`{model or '未登记'}`；已配对 {item.get('matched_structures', 0)} / 已回收 {item.get('records', 0)}；待回收 {item.get('pending_tasks', 0)}。"
        )
        if plot.get("status") == "completed":
            plot_prefix = prefix if epoch != "轮次未登记" else prefix + "/comparisons/plots"
            lines.append(
                f"[能量图](<{plot_prefix}/energy_parity.png>) · [受力图](<{plot_prefix}/force_parity.png>)"
            )
        else:
            lines.append("未生成：" + str(plot.get("reason") or "缺少已核验同帧预测"))
        lines.append("")
    if not exports:
        lines.append("尚无登记的 DFT 结果；需要先回收已完成结果和同帧模型预测。")
    _publish_bytes(root / "round_dft_parity.md", ("\n".join(lines) + "\n").encode("utf8"))


def publish_output_catalog(state, directory, *, action="refresh", epoch=None):
    """Refresh complete analysis bundles and publish navigation after a verified update."""
    if directory is None:
        return
    from phase_agent.analysis.feedback.analysis_publication import publish_analysis_products

    publish_analysis_products(state, directory)
    if Path(directory).name == "analysis_outputs":
        from phase_agent.analysis.feedback.training_parity import publish_training_parity

        publish_training_parity(state, directory)
    try:
        from phase_agent.analysis.feedback.out_data import export_out_data

        state["out_data_export"] = export_out_data(state, directory, action=action, epoch=epoch)
    except Exception as error:
        # Derived convenience exports must never invalidate verified scientific publications.
        state["out_data_export"] = {
            "status": "not_ready",
            "directory": str(Path(directory).resolve() / "out_data"),
            "reason": f"{type(error).__name__}: {error}",
        }
    from phase_agent.analysis.feedback.output_catalog import output_catalog, catalog_markdown

    from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory

    path = analysis_metadata_directory(directory) / "output_index.md"
    from phase_agent.analysis.feedback.workspace_guide import publish_epoch_guides

    publish_epoch_guides(directory, state)
    organization_path = analysis_metadata_directory(directory) / "analysis_organization.json"
    if organization_path.is_file():
        from phase_agent.analysis.feedback.analysis_output_organization import write_latest_guide

        try:
            saved = json.loads(organization_path.read_text(encoding="utf8"))
            write_latest_guide(directory, state, saved.get("dft_selections") or {})
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    _publish_bytes(
        path,
        catalog_markdown(
            output_catalog(state), Path(directory), state=state, index_directory=path.parent
        ).encode("utf-8"),
    )
    state.pop("output_index_csv_path", None)
    state["output_index_path"] = str(path)
    from phase_agent.analysis.feedback.folder_navigation import publish_folder_navigation

    if Path(directory).name == "analysis_outputs":
        from phase_agent.analysis.feedback.physical_layout import organize_physical_layout

        organize_physical_layout(directory)
    publish_folder_navigation(directory)


def _scope_key(scope):
    return json.dumps(scope, sort_keys=True)


def _round_tasks(state, scope, records):
    known = {row["task_id"] for row in records}
    tasks = []
    for task in state.get("tasks", []):
        if task.get("task_id") in known:
            tasks.append(task)
        elif (
            scope.get("upload_operation_id")
            and task.get("stage") in {"dft_single_point", "dft_relax"}
            and all(
                task.get(key) == scope.get(key)
                for key in (
                    "upload_operation_id",
                    "model_version",
                    "search_group_index",
                    "parent_relax_round",
                )
            )
        ):
            tasks.append(task)
    return tasks


def _product_directory(directory, state, scope, tasks, identifier):
    version = str(scope.get("model_version") or "unknown-model")
    if version in {".", ".."}:
        raise ValueError("invalid MLIP version for DFT output directory")
    version_root = model_output_directory(directory, version, state=state)
    round_names, group_indices = set(), set()
    for task in tasks:
        for field in ("input_path", "calculation_result_path", "result_path"):
            for part in PurePosixPath(str(task.get(field) or "").replace("\\", "/")).parts:
                if re.fullmatch(r"DFT-round-\d+_[A-Za-z0-9_.-]+", part):
                    round_names.add(part)
                match = re.fullmatch(r"Search-group-(\d+)", part)
                if match:
                    group_indices.add(int(match[1]))
    group = scope.get("search_group_index")
    if group is not None:
        if int(group) <= 0:
            raise ValueError("DFT search group index must be positive")
        group_indices.add(int(group))
    if len(round_names) > 1 or len(group_indices) > 1:
        raise ValueError("conflicting saved Search-group/DFT-round lineage")
    group = next(iter(group_indices), None)
    group_name = f"Search-group-{group:04d}" if group is not None else "Search-group-unassigned"
    if round_names:
        round_name = next(iter(round_names))
        operation = scope.get("upload_operation_id")
        if (
            operation
            and re.fullmatch(r"[A-Za-z0-9_.-]+", str(operation))
            and round_name.split("_", 1)[1] != str(operation)
        ):
            raise ValueError("DFT round directory does not match saved upload operation")
    else:
        operation = scope.get("upload_operation_id")
        registry_key = (
            f"{version}:Search-group-{group:04d}:DFT" if group is not None else f"{version}:DFT"
        )
        index = (
            ((state.get("upload_layout") or {}).get("operations") or {}).get(registry_key) or {}
        ).get(operation)
        round_name = (
            f"DFT-round-{int(index):04d}_{operation}"
            if index and operation and re.fullmatch(r"[A-Za-z0-9_.-]+", str(operation))
            else f"DFT-round-unassigned-{identifier}"
        )
    from phase_agent.analysis.feedback.output_paths import dft_round_directory

    saved = (state.get("dft_result_exports") or {}).get(identifier) or {}
    registered = saved.get("directory")
    legacy = version_root / group_name / round_name
    current = version_root / "dft" / group_name / round_name
    if registered and Path(registered) in {legacy, current}:
        other = current if Path(registered) == legacy else legacy
        if other.exists() and other.resolve() != Path(registered).resolve():
            raise ValueError("duplicate DFT round destinations require explicit review")
        return Path(registered)
    return dft_round_directory(version_root, group_name, round_name)


def _metric_rows(metrics, round_name):
    scope = metrics["round_scope"]
    complete = (
        metrics["matched_structures"] == metrics["recovered_tasks"] == metrics["expected_tasks"]
        and not metrics["pending_tasks"]
    )
    identity = {
        "model_version": scope.get("model_version"),
        "search_group_index": scope.get("search_group_index"),
        "dft_round": round_name,
        "upload_operation_id": scope.get("upload_operation_id"),
        "parent_relax_round": scope.get("parent_relax_round"),
        "recovered_tasks": metrics["recovered_tasks"],
        "expected_tasks": metrics["expected_tasks"],
        "pending_tasks": metrics["pending_tasks"],
        "matched_structures": metrics["matched_structures"],
        "not_evaluated_structures": len(metrics["not_evaluated"]),
        "status": "completed" if complete else "partial",
    }
    return [
        {
            **identity,
            "metric": name,
            "mae": metrics[name]["mae"],
            "rmse": metrics[name]["rmse"],
            "unit": metrics[name]["mae_unit"],
            "sample_count": metrics[name]["components"]
            if name == "forces"
            else metrics["matched_structures"],
            "averaging": "atomic Cartesian components"
            if name == "forces"
            else "equal structure weights",
        }
        for name in ("energy_total", "energy_per_atom", "forces")
    ]


def _publish_csv(path, fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _publish_bytes(path, stream.getvalue().encode("utf-8-sig"))


def _publish_bytes(path, data):
    from phase_agent.tools.remote.transfer_bundle import local_path

    path = local_path(path)
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
