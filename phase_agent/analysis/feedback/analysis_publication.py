"""Publish MLIP hull pairs and model-bound cumulative DFT diagnostics using existing science."""

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

from phase_agent.analysis.feedback.versioned_products import (
    digest,
    publish_bundle,
    publication_status,
    write_json,
)
from phase_agent.evidence.model_epoch import model_epoch


def publish_analysis_products(state, directory):
    """Derived artifact failures never invalidate verified scientific result imports."""
    if directory is None:
        return
    root = Path(directory).resolve()
    state["analysis_output_directory"] = str(root)
    model = state.get("active_model_version")
    epoch = model_epoch(state, model)
    publications = state.setdefault("analysis_publications", {})
    recovery = state.get("extrapolation_recovery_cycle") or {}
    invalid_active = (
        recovery.get("status") == "awaiting_reviewed_dft_labels"
        and not recovery.get("initial_setup")
        and model
        and model == recovery.get("invalid_model_version")
    )
    for kind, builder in (("phase_diagram", _phase), ("dft_error", _dft)):
        if kind == "phase_diagram" and invalid_active:
            continue  # Keep the verified previous latest until the replacement model is ready.
        try:
            publications[kind] = builder(state, root)
            if kind == "phase_diagram":
                state.setdefault("phase_epoch_publications", {})[model] = publications[kind]
                # A completed Relax/MC import is published under its own receipt below,
                # after its diagnostics are registered, rather than as a duplicate update.
                from phase_agent.analysis.feedback.analysis_events import ACTIVE_RELAX

                pending_relax = ACTIVE_RELAX.get() is not None
                if not pending_relax:
                    try:
                        from phase_agent.analysis.feedback.analysis_events import phase_event

                        phase_event(state, root, publications[kind])
                        (state.get("analysis_step_diagnostics") or {}).pop("phase", None)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        state.setdefault("analysis_step_diagnostics", {})["phase"] = str(error)
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            try:
                publications[kind] = publication_status(
                    root, kind, model, epoch, status="not_ready", reason=reason
                )
            except OSError:
                publications[kind] = {
                    "status": "not_ready",
                    "model_version": model,
                    "epoch": epoch,
                    "reason": reason,
                }

    cycles = [*(state.get("extrapolation_recovery_history") or []), recovery]
    invalid_models = {v for cycle in cycles for v in cycle.get("invalidated_model_versions") or []}
    for version, snapshot in (state.get("phase_diagrams_by_model") or {}).items():
        if version == model or version in invalid_models or not model_epoch(state, version):
            continue
        epoch_name = model_epoch(state, version)
        try:
            if snapshot.get("model_version") != version:
                raise ValueError("Historical hull model binding mismatch")
            historical = deepcopy(state)
            historical["active_model_version"] = (
                version  # Rendering view only; never activate a model.
            )
            historical["active_model"] = (
                (state.get("model_registry") or {}).get(version) or {}
            ).get("model") or {}
            historical["phase_diagrams"] = {"mlip": deepcopy(snapshot)}
            historical["tasks"] = [
                t for t in historical.get("tasks") or [] if t.get("model_version") == version
            ]
            result = _phase(historical, root, current=False)
        except Exception as error:
            result = {
                "status": "not_ready",
                "epoch": epoch_name,
                "model_version": version,
                "reason": f"{type(error).__name__}: {error}",
            }
            try:
                write_json(root / epoch_name / "phase_diagram" / "latest_status.json", result)
            except OSError:
                pass
        state.setdefault("phase_epoch_publications", {})[version] = result


def _model_digest(state, model):
    entry = (state.get("model_registry") or {}).get(model) or {}
    if entry.get("comparison_binding_error"):
        raise ValueError(entry["comparison_binding_error"])
    active = state.get("active_model") or {} if model == state.get("active_model_version") else {}
    return entry.get("comparison_model_sha256") or active.get("sha256")


def _phase(state, root, *, current=True):
    from phase_agent.tools.relax_mc_batch.completion_gate import require_complete_imports

    require_complete_imports(state)

    from phase_agent.runtime.phase_diagram_chat import _plot, _current_snapshot
    from phase_agent.analysis.phase.phase_quality_reports import (
        get_phase_quality_report,
        review_markdown,
    )
    from phase_agent.analysis.phase.endpoint_reference import PLOT_RULE_VERSION
    from phase_agent.analysis.phase.phase_quality import RULE_VERSION
    from phase_agent.analysis.phase.export_phase_diagram_csv import export_phase_diagram_csv
    from phase_agent.analysis.phase.voltage_curve import (
        VOLTAGE_RULE_VERSION,
        voltage_reference,
        write_voltage_outputs,
    )

    model = state.get("active_model_version")
    epoch = model_epoch(state, model)
    snapshot = _current_snapshot(state, "mlip")
    if (
        not epoch
        or snapshot.get("status") != "completed"
        or snapshot.get("validity", "valid") != "valid"
    ):
        raise ValueError("Current model epoch or verified complete MLIP hull is unavailable")
    report = get_phase_quality_report(state, snapshot=snapshot)
    identity = {
        "input_digest": digest(snapshot),
        "quality_fingerprint": report["input_fingerprint"],
        "hull_version": snapshot.get("version"),
        "model_sha256": _model_digest(state, model),
        "plot_rule": PLOT_RULE_VERSION,
        "quality_rule": RULE_VERSION,
        "voltage_rule": VOLTAGE_RULE_VERSION,
        "voltage_reference": voltage_reference(state)[0],
    }

    if not current:
        identity["publication_basis"] = "saved_historical_hull_current_review_rules"

    def build(folder):
        artifacts = {}
        _plot(
            deepcopy(state),
            snapshot,
            "mlip",
            root,
            ("png",),
            output_directory=folder,
            artifacts=artifacts,
            strict=True,
        )
        for key, name in (("main", "main.png"), ("hd", "hd_cross.png")):
            sources = artifacts.get(key) or []
            if len(sources) != 1:
                raise ValueError(f"Missing {key} figure")
            source = Path(sources[0])
            source.rename(folder / name)
        write_json(folder / "phase_diagram.json", snapshot)
        export_phase_diagram_csv(snapshot, folder / "phase_diagram.csv")
        write_voltage_outputs(
            snapshot,
            state,
            folder / "voltage_curve.png",
            folder / "voltage_curve.csv",
            folder / "voltage_curve_status.json",
        )
        write_json(folder / "quality_report.json", report)
        (folder / "report.md").write_text(
            f"# MLIP phase diagram | {epoch} | {model}\n\nHull: {snapshot.get('version')}\n\n"
            "[Main](main.png) · [HD comparison](hd_cross.png) · [Data](phase_diagram.csv) · "
            "[Voltage](voltage_curve.png) · [Voltage data](voltage_curve.csv)\n\n"
            "HD crosses are a counterfactual review; they do not remove formal records.\n\n"
            + review_markdown(report, bundle_id=f"{epoch}:{snapshot.get('version')}"),
            encoding="utf-8",
        )

    publication = publish_bundle(
        root,
        kind="phase_diagram",
        epoch=epoch,
        model=model,
        identity=identity,
        required=(
            "main.png",
            "hd_cross.png",
            "phase_diagram.json",
            "phase_diagram.csv",
            "voltage_curve.png",
            "voltage_curve.csv",
            "voltage_curve_status.json",
            "quality_report.json",
            "report.md",
        ),
        build=build,
        current=current,
    )
    status_path = Path(publication["directory"]) / "voltage_curve_status.json"
    if status_path.is_file():
        publication["voltage_curve"] = json.loads(status_path.read_text(encoding="utf-8"))
    return publication


def _unique_records(records):
    unique = {}
    for record in records:
        key = record.get("data_id") or (record.get("task_id"), record.get("training_frame_index"))
        token = digest(key)
        if token in unique and unique[token] != record:
            raise ValueError(f"Conflicting DFT dataset identity: {key}")
        unique[token] = record
    return [unique[key] for key in sorted(unique)]


def _archive_failed_latest(root, epoch, *, include_current, failed_ids):
    """Do not leave a known-failed data bundle at a visible latest path."""
    import uuid

    root = Path(root).resolve()
    from phase_agent.analysis.feedback.output_paths import epoch_product_directory

    targets = [epoch_product_directory(root, epoch, "dft_error") / "latest"]
    if include_current:
        targets.append(root / "latest" / "dft_error")
    for target in targets:
        if target.is_symlink() or not target.is_dir() or not target.resolve().is_relative_to(root):
            continue
        try:
            manifest = json.loads((target / "dataset_manifest.json").read_text(encoding="utf8"))
        except (OSError, ValueError):
            continue
        if not any(record.get("task_id") in failed_ids for record in manifest.get("records") or []):
            continue
        archive = root / "history" / "filtered_dft_latest" / uuid.uuid4().hex
        archive.parent.mkdir(parents=True, exist_ok=True)
        target.rename(archive)


def _dft(state, root):
    from phase_agent.analysis.feedback.dft_comparison_tables import (
        ENERGY_FIELDS,
        FORCE_FIELDS,
        build_comparison_tables,
        comparison_metrics,
    )
    from phase_agent.analysis.feedback.dft_parity_plots import export_parity_plots, PLOT_VERSION
    from phase_agent.analysis.feedback.export_dft_products import _publish_csv

    from phase_agent.evidence.dft_extrapolation import failed_dft_task_ids

    failed_ids = failed_dft_task_ids(state)
    groups = defaultdict(list)
    for record in state.get("dft_dataset_records") or []:
        model = record.get("model_version") or (record.get("round_scope") or {}).get(
            "model_version"
        )
        if model_epoch(state, model):
            groups[model].append(record)
    active = state.get("active_model_version")
    by_model = state.setdefault("dft_epoch_publications", {})
    for model, rows in sorted(groups.items()):
        epoch = model_epoch(state, model)
        try:
            excluded_ids = sorted(
                {r.get("task_id") for r in rows if r.get("task_id") in failed_ids}
            )
            records = _unique_records([r for r in rows if r.get("task_id") not in failed_ids])
            comparisons = [
                c
                for c in state.get("dft_mlip_comparisons") or []
                if (c.get("round_scope") or {}).get("model_version") == model
            ]
            comparisons = _unique_records(comparisons)
            energies, forces = build_comparison_tables(
                records, comparisons, round_name="epoch cumulative"
            )
            for row in energies + forces:
                row["epoch"] = epoch
            metrics = comparison_metrics(energies, forces, {"model_version": model, "epoch": epoch})
            fingerprints = {r["model_sha256"] for r in energies if r.get("model_sha256")}
            frozen_digest = _model_digest(state, model)
            if len(fingerprints) > 1 or (
                fingerprints and frozen_digest and fingerprints != {frozen_digest}
            ):
                raise ValueError("Conflicting model fingerprints within one epoch")
            identity = {
                "input_digest": digest([records, comparisons]),
                "plot_rule": PLOT_VERSION,
                "model_sha256": frozen_digest or next(iter(fingerprints), None),
                "evaluation_type": "original_round_model_same_frame",
                "table_rule": 1,
            }

            def build(folder):
                write_json(
                    folder / "dataset_manifest.json",
                    {
                        "epoch": epoch,
                        "model_version": model,
                        "records": records,
                        "comparisons": comparisons,
                        "excluded_extrapolation_failed_task_ids": excluded_ids,
                    },
                )
                write_json(folder / "metrics.json", metrics)
                _publish_csv(folder / "energy_comparison.csv", ENERGY_FIELDS, energies)
                _publish_csv(folder / "force_comparison.csv", FORCE_FIELDS, forces)
                competing = state.get("mp_competing_dft_entries") or []
                if not competing and state.get("mp_competing_repo_path"):
                    from phase_agent.analysis.phase.mp_global_ehull import CompetingPhaseRepo

                    elems = {str(el) for rec in records for el in (rec.get("composition") or {})}
                    if elems:
                        try:
                            competing = CompetingPhaseRepo(state["mp_competing_repo_path"]).lookup(
                                elems, state.get("mp_competing_dft_parameters") or {}
                            )
                        except (OSError, ValueError):
                            competing = []
                if competing:
                    from phase_agent.analysis.phase.mp_global_ehull import (
                        compute_gs_global_dft_ehull,
                        export_gs_global_dft_ehull_csv,
                    )

                    gs_recs = [
                        r
                        for r in records
                        if r.get("is_ground_state") is True
                        or r.get("selection_mode") == "phase_diagram_gs"
                    ] or records
                    ehull_res = compute_gs_global_dft_ehull(gs_recs, competing)
                    if ehull_res.get("status") == "evaluated":
                        export_gs_global_dft_ehull_csv(
                            ehull_res, folder / "gs_global_dft_ehull.csv"
                        )
                plots = export_parity_plots(energies, forces, folder, f"{epoch} ({model})")
                if plots.get("status") != "completed":
                    raise ValueError(plots.get("reason", "DFT parity plots unavailable"))
                # Paths in the renderer's cache receipt must survive staging-directory cleanup.
                if (folder / "parity_metrics.json").is_file():
                    import json

                    parity = json.loads(
                        (folder / "parity_metrics.json").read_text(encoding="utf-8")
                    )
                    parity["files"] = {k: Path(v).name for k, v in parity["files"].items()}
                    write_json(folder / "parity_metrics.json", parity)
                (folder / "report.md").write_text(
                    f"# DFT same-frame diagnostic | {epoch} | {model}\n\n"
                    f"Matched structures: {metrics['matched_structures']}; pending/unmatched: {len(metrics['not_evaluated'])}; excluded confirmed extrapolation failures: {len(excluded_ids)}.\n\n"
                    "Cumulative verified pairs, deduplicated by dataset identity. Energy uses equal structure weights; forces pool atomic Cartesian components.\n\n"
                    "This is an original-round frozen-model diagnostic. Independent-test status requires separate training-membership evidence.\n\n"
                    "[Energy](energy_parity.png) · [Forces](force_parity.png) · [Metrics](metrics.json)\n",
                    encoding="utf-8",
                )

            by_model[model] = publish_bundle(
                root,
                kind="dft_error",
                epoch=epoch,
                model=model,
                identity=identity,
                required=(
                    "dataset_manifest.json",
                    "metrics.json",
                    "energy_comparison.csv",
                    "force_comparison.csv",
                    "energy_parity.png",
                    "force_parity.png",
                    "report.md",
                ),
                build=build,
                current=model == active,
            )
        except Exception as error:
            if failed_ids:
                _archive_failed_latest(
                    root, epoch, include_current=model == active, failed_ids=failed_ids
                )
            result = {
                "status": "not_ready",
                "epoch": epoch,
                "model_version": model,
                "reason": f"{type(error).__name__}: {error}",
            }
            from phase_agent.analysis.feedback.output_paths import epoch_product_directory

            write_json(
                epoch_product_directory(root, epoch, "dft_error") / "latest_status.json", result
            )
            by_model[model] = result
    result = by_model.get(active) if active in groups else None
    if result and result["status"] == "completed":
        return result
    raise ValueError(
        (result or {}).get("reason")
        or "No current-model same-frame DFT pairs; historical data requires actual matching predictions"
    )
