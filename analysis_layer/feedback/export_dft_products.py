"""Publish version/round-scoped DFT datasets and plotting CSVs, without inference."""
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import uuid

from analysis_layer.feedback.dft_comparison_tables import (
    ENERGY_FIELDS, FORCE_FIELDS, METRIC_FIELDS, build_comparison_tables, comparison_metrics,
)
from analysis_layer.phase.phase_snapshot_paths import model_output_directory


def dft_product_path(root, name):
    category = ("training" if name == "training.json" else "comparisons" if name in {
        "energy_comparison.csv", "force_comparison.csv", "metrics.csv", "mlip_dft_metrics.json"} else "diagnostics")
    return Path(root) / category / name


def export_dft_products(state, directory):
    """Refresh saved products only; keep old directories and unchanged files intact."""
    if directory is None:
        return
    from scientific_layer.dft.spin_acceptance import spin_standard_passed
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
    for key, identifier, scope, records, tasks, root in planned:
        comparisons = [row for row in state.get("dft_mlip_comparisons", [])
                       if _scope_key(row["round_scope"]) == key]
        energies, forces = build_comparison_tables(records, comparisons, round_name=root.name)
        from analysis_layer.state.model_epoch import model_epoch
        epoch = model_epoch(state, scope.get("model_version") or "unknown-model")
        for row in energies + forces:
            row["epoch"] = epoch
        from analysis_layer.feedback.dft_magnetic_tables import magnetic_rows, MAGNETIC_FIELDS
        metrics = comparison_metrics(energies, forces, scope)
        pending = [row["task_id"] for row in tasks if row.get("status") in {"pending", "running"}
                   and row.get("recovery_wait_waived") is not True]
        expected = len({row["task_id"] for row in tasks} | {row["task_id"] for row in records})
        metrics.update(recovered_tasks=len(records), expected_tasks=expected,
                       pending_tasks=len(pending), pending_task_ids=sorted(pending))
        metric_rows = _metric_rows(metrics, root.name)
        for row in metric_rows:
            row["epoch"] = epoch
        training = sorted([row for row in state.get("dft_training_records", [])
                           if _scope_key(row["round_scope"]) == key
                           and row.get("checks_passed", True) is True
                           and spin_standard_passed(row)], key=lambda row: str(row.get("data_id")))
        magnetism = magnetic_rows(records)
        for row in magnetism:
            row.update(epoch=epoch, search_group_index=scope.get("search_group_index"), dft_round=root.name)
        for name, payload in {"training.json": training, "dft_records.json": sorted(records, key=lambda row: str(row["task_id"])),
                              "mlip_dft_metrics.json": metrics}.items():
            _publish_bytes(dft_product_path(root, name), (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        for name, fields, rows in (("energy_comparison.csv", ENERGY_FIELDS, energies),
                                   ("force_comparison.csv", FORCE_FIELDS, forces),
                                   ("magnetic_moments.csv", MAGNETIC_FIELDS, magnetism),
                                   ("metrics.csv", METRIC_FIELDS, metric_rows)):
            _publish_csv(dft_product_path(root, name), fields, rows)
        version_root = model_output_directory(directory, scope.get("model_version") or "unknown-model", state=state)
        from analysis_layer.feedback.dft_parity_plots import export_parity_plots
        version = scope.get("model_version") or "unknown-model"
        plots = export_parity_plots(energies, forces, root / "comparisons" / "plots",
                                   f"{model_epoch(state, version) or 'Epoch unrecorded'} ({version})")
        version_summaries.setdefault(version_root, []).extend(metric_rows)
        exports[identifier] = {"directory": str(root), "round_scope": scope,
                               "parity_plots": plots,
                               "records": len(records), "matched_structures": metrics["matched_structures"],
                               "expected_tasks": expected, "pending_tasks": len(pending),
                               "energy_csv_path": str(dft_product_path(root, "energy_comparison.csv")),
                               "force_csv_path": str(dft_product_path(root, "force_comparison.csv")),
                               "magnetic_csv_path": str(dft_product_path(root, "magnetic_moments.csv")),
                               "metrics_csv_path": str(dft_product_path(root, "metrics.csv")),
                               "round_metrics_csv_path": str(version_root / "round_metrics.csv")}
    for root, rows in version_summaries.items():
        _publish_csv(root / "round_metrics.csv", METRIC_FIELDS, sorted(
            rows, key=lambda row: (str(row["search_group_index"]), row["dft_round"], row["metric"])))
    state["dft_result_exports"] = exports
    from analysis_layer.state.summarize_model_rounds import summarize_model_rounds, ROUND_SUMMARY_FIELDS
    state["model_round_summary"] = summarize_model_rounds(state)
    summary_path = Path(directory) / "round_summary.csv"
    _publish_csv(summary_path, ROUND_SUMMARY_FIELDS, state["model_round_summary"])
    state["round_summary_csv_path"] = str(summary_path)
    publish_output_catalog(state, directory)


def publish_output_catalog(state, directory):
    if directory is None:
        return
    from analysis_layer.feedback.output_catalog import output_catalog, catalog_markdown
    path = Path(directory) / "output_index.md"
    _publish_bytes(path, catalog_markdown(output_catalog(state), Path(directory)).encode("utf-8"))
    state.pop("output_index_csv_path", None)
    state["output_index_path"] = str(path)


def _scope_key(scope):
    return json.dumps(scope, sort_keys=True)


def _round_tasks(state, scope, records):
    known = {row["task_id"] for row in records}
    tasks = []
    for task in state.get("tasks", []):
        if task.get("task_id") in known:
            tasks.append(task)
        elif (scope.get("upload_operation_id") and task.get("stage") in {"dft_single_point", "dft_relax"}
              and all(task.get(key) == scope.get(key) for key in (
                  "upload_operation_id", "model_version", "search_group_index", "parent_relax_round"))):
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
        if operation and re.fullmatch(r"[A-Za-z0-9_.-]+", str(operation)) and round_name.split("_", 1)[1] != str(operation):
            raise ValueError("DFT round directory does not match saved upload operation")
    else:
        operation = scope.get("upload_operation_id")
        registry_key = f"{version}:Search-group-{group:04d}:DFT" if group is not None else f"{version}:DFT"
        index = (((state.get("upload_layout") or {}).get("operations") or {}).get(registry_key) or {}).get(operation)
        round_name = (f"DFT-round-{int(index):04d}_{operation}" if index and operation
                      and re.fullmatch(r"[A-Za-z0-9_.-]+", str(operation)) else f"DFT-round-unassigned-{identifier}")
    return version_root / group_name / round_name


def _metric_rows(metrics, round_name):
    scope = metrics["round_scope"]
    complete = (metrics["matched_structures"] == metrics["recovered_tasks"] == metrics["expected_tasks"]
                and not metrics["pending_tasks"])
    identity = {"model_version": scope.get("model_version"), "search_group_index": scope.get("search_group_index"),
                "dft_round": round_name, "upload_operation_id": scope.get("upload_operation_id"),
                "parent_relax_round": scope.get("parent_relax_round"),
                "recovered_tasks": metrics["recovered_tasks"], "expected_tasks": metrics["expected_tasks"],
                "pending_tasks": metrics["pending_tasks"], "matched_structures": metrics["matched_structures"],
                "not_evaluated_structures": len(metrics["not_evaluated"]),
                "status": "completed" if complete else "partial"}
    return [{**identity, "metric": name, "mae": metrics[name]["mae"], "rmse": metrics[name]["rmse"],
             "unit": metrics[name]["mae_unit"],
             "sample_count": metrics[name]["components"] if name == "forces" else metrics["matched_structures"],
             "averaging": "atomic Cartesian components" if name == "forces" else "equal structure weights"}
            for name in ("energy_total", "energy_per_atom", "forces")]


def _publish_csv(path, fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _publish_bytes(path, stream.getvalue().encode("utf-8-sig"))


def _publish_bytes(path, data):
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
