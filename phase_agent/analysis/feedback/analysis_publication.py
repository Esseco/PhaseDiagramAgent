"""Publish MLIP hull pairs and model-bound cumulative DFT diagnostics using existing science."""

from collections import defaultdict
from copy import deepcopy
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
    for kind, builder in (("phase_diagram", _phase), ("dft_error", _dft)):
        try:
            publications[kind] = builder(state, root)
            if kind == "phase_diagram":
                state.setdefault("phase_epoch_publications", {})[model] = publications[kind]
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

    for version, snapshot in (state.get("phase_diagrams_by_model") or {}).items():
        if version == model or not model_epoch(state, version):
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
        write_json(folder / "quality_report.json", report)
        (folder / "report.md").write_text(
            f"# MLIP phase diagram | {epoch} | {model}\n\nHull: {snapshot.get('version')}\n\n"
            "[Main](main.png) · [HD comparison](hd_cross.png) · [Data](phase_diagram.csv)\n\n"
            "HD crosses are a counterfactual review; they do not remove formal records.\n\n"
            + review_markdown(report, bundle_id=f"{epoch}:{snapshot.get('version')}"),
            encoding="utf-8",
        )

    return publish_bundle(
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
            "quality_report.json",
            "report.md",
        ),
        build=build,
        current=current,
    )


def _unique_records(records):
    unique = {}
    for record in records:
        key = record.get("data_id") or (record.get("task_id"), record.get("training_frame_index"))
        token = digest(key)
        if token in unique and unique[token] != record:
            raise ValueError(f"Conflicting DFT dataset identity: {key}")
        unique[token] = record
    return [unique[key] for key in sorted(unique)]


def _dft(state, root):
    from phase_agent.analysis.feedback.dft_comparison_tables import (
        ENERGY_FIELDS,
        FORCE_FIELDS,
        build_comparison_tables,
        comparison_metrics,
    )
    from phase_agent.analysis.feedback.dft_parity_plots import export_parity_plots, PLOT_VERSION
    from phase_agent.analysis.feedback.export_dft_products import _publish_csv

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
            records = _unique_records(rows)
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
                    },
                )
                write_json(folder / "metrics.json", metrics)
                _publish_csv(folder / "energy_comparison.csv", ENERGY_FIELDS, energies)
                _publish_csv(folder / "force_comparison.csv", FORCE_FIELDS, forces)
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
                    f"Matched structures: {metrics['matched_structures']}; pending/unmatched: {len(metrics['not_evaluated'])}.\n\n"
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
            result = {
                "status": "not_ready",
                "epoch": epoch,
                "model_version": model,
                "reason": f"{type(error).__name__}: {error}",
            }
            write_json(root / epoch / "dft_error" / "latest_status.json", result)
            by_model[model] = result
    result = by_model.get(active) if active in groups else None
    if result and result["status"] == "completed":
        return result
    raise ValueError(
        (result or {}).get("reason")
        or "No current-model same-frame DFT pairs; historical data requires actual matching predictions"
    )
