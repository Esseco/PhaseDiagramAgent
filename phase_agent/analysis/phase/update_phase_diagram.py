"""按统一能量口径分别更新 MLIP 与 DFT 凸包。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition

from phase_agent.analysis.phase.check_na_layer_uniformity import check_na_layer_uniformity
from phase_agent.analysis.phase.export_phase_diagram_csv import export_phase_diagram_csv
from phase_agent.analysis.phase.phase_snapshot_paths import phase_snapshot_directory


def update_phase_diagram(
    records: list[dict[str, Any]],
    *,
    energy_basis: str = "total",
    output_directory: str | Path | None = None,
    parent_versions: dict[str, str] | None = None,
    active_model_version: str | None = None,
    output_state: dict | None = None,
    include_dft: bool = False,
    latest_per_structure: bool = False,
) -> dict[str, Any]:
    """分别构建 mlip/dft 相图，保留原能量并保存可追踪版本。"""
    if energy_basis not in {"total", "per_atom", "per_formula_unit"}:
        raise ValueError("energy_basis 必须是 total/per_atom/per_formula_unit")
    output = {"energy_basis": energy_basis, "diagrams": {}, "mlip_by_version": {}}
    refresh_plan = ((output_state or {}).get("model_refresh") or {}).get("plan") or {}
    latest_mode = (
        latest_per_structure or refresh_plan.get("energy_policy") == "latest_valid_per_structure_v1"
    )
    latest_normalized = []
    from phase_agent.analysis.phase.phase_boundary import (
        phase_boundary_rule,
        outside_phase_boundary,
    )

    phase_rule = phase_boundary_rule(output_state)
    excluded = {
        (review.get("model_version"), record_id)
        for review in (output_state or {}).get("phase_quality_exclusions") or []
        for record_id in review.get("record_ids") or []
    }
    from phase_agent.analysis.phase.phase_quality_restore import restored_records

    excluded -= restored_records(output_state)
    from phase_agent.analysis.convergence.mlip_relaxation import relaxation_failure

    tasks = {t.get("task_id"): t for t in (output_state or {}).get("tasks") or []}
    versions = sorted(
        {
            str(item.get("model_version") or item.get("source_version") or "unknown")
            for item in records
            if item.get("energy_method") == "mlip"
        }
    )
    groups = [("mlip", version) for version in versions] + ([("dft", None)] if include_dft else [])
    for method, model_version in groups:
        selected = [
            item
            for item in records
            if str(item.get("energy_method", "")).lower() == method
            and item.get("status", "completed") == "completed"
            and (
                method != "mlip"
                or (item.get("model_version") or item.get("source_version"), item.get("record_id"))
                not in excluded
            )
            and (
                method != "mlip"
                or str(item.get("model_version") or item.get("source_version") or "unknown")
                == model_version
            )
        ]
        selected.sort(
            key=lambda item: (str(item.get("record_id") or ""), str(item.get("structure_id") or ""))
        )
        normalized, rejected = [], []
        for item in selected:
            try:
                if method == "mlip" and (
                    item.get("checks_passed", True) is not True
                    or relaxation_failure(tasks.get(item.get("source_task_id")) or item)
                ):
                    raise ValueError("MLIP relaxation quality standard not passed")
                if method == "dft":
                    from phase_agent.science.dft.spin_acceptance import spin_standard_passed

                    if item.get("checks_passed", True) is not True or not spin_standard_passed(
                        item
                    ):
                        raise ValueError("DFT quality/spin standard not passed")
                if item.get("energy_unit") != "eV":
                    raise ValueError("相图记录 energy_unit 必须为 'eV'")
                if outside_phase_boundary(item, phase_rule):
                    raise ValueError("phase_outside_composition_boundary")
                composition = Composition(item["composition"])
                energy = _total_energy(item, composition, energy_basis)
                normalized.append(
                    {
                        **item,
                        "original_energy": item["energy"],
                        "normalized_total_energy": energy,
                        "composition": composition.as_dict(),
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                rejected.append({"record_id": item.get("record_id"), "reason": str(error)})
        if method == "mlip" and latest_mode:
            latest_normalized.extend(normalized)
        snapshot = _build_snapshot(
            method,
            normalized,
            rejected,
            energy_basis,
            (parent_versions or {}).get(method),
            phase_rule=phase_rule,
        )
        if method == "mlip":
            snapshot["model_version"] = model_version
            snapshot["energy_basis_id"] = f"mlip:{model_version}:{energy_basis}"
            snapshot["version"] = hashlib.sha256(
                f"{model_version}:{snapshot['version']}".encode()
            ).hexdigest()[:12]
            snapshot["parent_version"] = (parent_versions or {}).get(f"mlip:{model_version}")
            output["mlip_by_version"][model_version] = snapshot
        else:
            output["diagrams"][method] = snapshot
        refresh = (output_state or {}).get("model_refresh") or {}
        if (
            method == "mlip"
            and model_version == refresh.get("new_model_version")
            and refresh.get("plan")
        ):
            plan = refresh["plan"]
            successful = {
                t.get("input_structure_sha256")
                for t in (output_state or {}).get("tasks") or []
                if t.get("model_refresh_id") == plan["checksum"]
                and t.get("status") == "completed"
                and (
                    t.get("converged") is True
                    or (t.get("outputs") or {}).get("single_point_completed") is True
                )
                and t.get("checks_passed", True) is True
            }
            refreshed_count = len(successful)
            snapshot["refresh_coverage"] = {
                "refreshed": refreshed_count,
                "deferred": max(0, plan["total_unique"] - refreshed_count),
                "partial": refreshed_count < plan["total_unique"],
            }
        if output_directory is not None:
            root = Path(output_directory)
            output_version = (
                model_version
                or active_model_version
                or (versions[0] if len(versions) == 1 else None)
            )
            directory = phase_snapshot_directory(root, method, output_version, state=output_state)
            if output_state is not None:
                from phase_agent.evidence.model_epoch import model_epoch

                snapshot["epoch"] = model_epoch(output_state, output_version)
            filename = f"phase_diagram_{method}_{snapshot['version']}"
            path = directory / f"{filename}.json"
            legacy_paths = [root / f"{filename}.json"]
            if method == "mlip":
                legacy_paths.insert(0, root / directory.parent.name / f"{filename}.json")
                legacy_paths.insert(0, root / "mlip" / directory.parent.name / f"{filename}.json")
            else:
                legacy_paths.insert(0, root / "dft" / f"{filename}.json")
            if not path.is_file():
                # Keep unchanged historical snapshots in place; do not duplicate them.
                path = next((candidate for candidate in legacy_paths if candidate.is_file()), path)
            if path.is_file():
                # The same input must reuse the original immutable snapshot.
                stored = json.loads(path.read_text(encoding="utf-8"))
                if stored.get("version") == snapshot["version"]:
                    stored["path"] = str(path)
                    snapshot.clear()
                    snapshot.update(stored)
                    archive = snapshot.get("archive_csv_path")
                    if archive and Path(archive).is_file():
                        from phase_agent.analysis.phase.publish_current_csv import (
                            publish_current_csv,
                        )

                        publish_current_csv(snapshot, archive)
                    elif not snapshot.get("entries"):
                        _publish_empty_current_csv(snapshot, directory, filename)
                    continue
            directory.mkdir(parents=True, exist_ok=True)
            csv_path = directory / f"{filename}.csv"
            if snapshot.get("status") == "completed" and snapshot.get("entries"):
                export_phase_diagram_csv(snapshot, csv_path)
                from phase_agent.analysis.phase.publish_current_csv import publish_current_csv

                publish_current_csv(snapshot, csv_path)
            elif not snapshot.get("entries"):
                _publish_empty_current_csv(snapshot, directory, filename)
            snapshot["path"] = str(path)
            path.write_text(
                json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    chosen = active_model_version or (versions[0] if len(versions) == 1 else None)
    output["diagrams"]["mlip"] = output["mlip_by_version"].get(chosen) or {
        "method": "mlip",
        "model_version": chosen,
        "status": "unknown",
        "reason": "active_model_version_required"
        if chosen is None
        else "no_results_for_active_model",
        "entries": [],
    }
    if latest_mode:
        from phase_agent.analysis.phase.latest_mlip_lineage import latest_mlip_records

        rounds = ((output_state or {}).get("upload_layout") or {}).get("model_rounds") or {}
        try:
            latest, issues, aliases = latest_mlip_records(
                latest_normalized, (output_state or {}).get("tasks") or [], rounds, return_aliases=True
            )
        except (OSError, ValueError, TypeError) as error:
            latest, issues, aliases = [], [{"reason": str(error)}], {}
        if issues:
            snapshot = {
                "method": "mlip",
                "model_version": chosen,
                "status": "unknown",
                "reason": "latest_structure_lineage_unresolved",
                "issues": issues,
                "entries": [],
                "energy_policy": "latest_valid_per_structure_v1",
                "version": hashlib.sha256(json.dumps(issues, sort_keys=True).encode()).hexdigest()[
                    :12
                ],
            }
        else:
            for record in latest:
                record["source_version"] = record.get("model_version") or record.get(
                    "source_version"
                )
            snapshot = _build_snapshot(
                "mlip",
                latest,
                [],
                energy_basis,
                (parent_versions or {}).get("mlip"),
                phase_rule=phase_rule,
            )
            sources = sorted({r["source_version"] for r in latest})
            provenance = [(r.get("record_id"), r["source_version"]) for r in latest]
            snapshot["version"] = hashlib.sha256(
                json.dumps(
                    [snapshot["version"], chosen, sorted(provenance), sorted(aliases.items())],
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:12]
            snapshot["equivalent_record_aliases"] = aliases
            active_snapshot = output["mlip_by_version"].get(chosen) or {}
            if active_snapshot.get("refresh_coverage"):
                snapshot["refresh_coverage"] = active_snapshot["refresh_coverage"].copy()
            snapshot.update(
                model_version=chosen,
                energy_policy="latest_valid_per_structure_v1",
                energy_basis_id=f"mlip:latest-per-structure:{energy_basis}",
                source_model_versions=sources,
                mixed_model_versions=len(sources) > 1,
                independent_model_comparability="not_established",
            )
        if output_directory is not None:
            directory = phase_snapshot_directory(
                Path(output_directory), "mlip", chosen, state=output_state
            )
            from phase_agent.evidence.model_epoch import model_epoch

            snapshot["epoch"] = model_epoch(output_state or {}, chosen)
            if snapshot.get("status") == "completed" and snapshot.get("entries"):
                directory.mkdir(parents=True, exist_ok=True)
                filename = f"phase_diagram_mlip_{snapshot['version']}"
                path = directory / f"{filename}.json"
                archive = directory / f"{filename}.csv"
                if path.is_file():
                    stored = json.loads(path.read_text(encoding="utf-8"))
                    if stored.get("version") != snapshot["version"]:
                        raise ValueError("mixed_hull_snapshot_identity_mismatch")
                    snapshot = stored
                else:
                    export_phase_diagram_csv(snapshot, archive)
                    snapshot["path"] = str(path)
                    path.write_text(
                        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                from phase_agent.analysis.phase.publish_current_csv import publish_current_csv

                if archive.is_file():
                    publish_current_csv(snapshot, archive)
            else:
                _publish_empty_current_csv(
                    snapshot, directory, f"phase_diagram_mlip_{snapshot['version']}"
                )
        output["diagrams"]["mlip"] = snapshot
    return output


def _publish_empty_current_csv(snapshot, directory, filename):
    """Do not leave old, now-ineligible points in the discoverable current CSV."""
    current = directory.parent / "phase_diagram.csv"
    if not current.is_file():
        return  # No bootstrap CSV for a system with no phase data yet.
    archive = directory / f"{filename}.csv"
    if not archive.is_file():
        export_phase_diagram_csv({**snapshot, "entries": []}, archive)
    from phase_agent.analysis.phase.publish_current_csv import publish_current_csv

    publish_current_csv(snapshot, archive)


def _total_energy(item, composition, basis):
    energy = float(item["energy"])
    if basis == "per_atom":
        return energy * composition.num_atoms
    if basis == "per_formula_unit":
        units = item.get("formula_units")
        if not isinstance(units, (int, float)) or units <= 0:
            raise ValueError("per_formula_unit 需要正数 formula_units")
        return energy * units
    return energy


def _build_snapshot(method, records, rejected, basis, parent, *, phase_rule=None):
    payload = [
        {
            "record_id": item.get("record_id"),
            "structure_id": item.get("structure_id"),
            "structure_path": item.get("structure_path"),
            "structure_sha256": item.get("structure_sha256"),
            "phase": item.get("phase"),
            "composition": item["composition"],
            "energy": item["normalized_total_energy"],
        }
        for item in records
    ]
    identity = {
        "method": method,
        "basis": basis,
        "algorithm": "na_eform_hull_v1",
        "entries": payload,
    }
    if phase_rule is not None:
        identity["phase_boundary_rule"] = phase_rule
    version = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    snapshot = {
        "method": method,
        "energy_basis": basis,
        "energy_basis_id": f"{method}:{basis}",
        "version": version,
        "parent_version": parent,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entries": [],
        "rejected": rejected,
        "status": "completed",
    }
    if phase_rule is not None:
        snapshot["phase_boundary_rule"] = phase_rule
    if not records:
        snapshot.update({"status": "unknown", "reason": "no_identified_energy_records"})
        return snapshot
    try:
        elements = {element for item in records for element in item["composition"]}
        elemental_endpoints = {
            next(iter(item["composition"])) for item in records if len(item["composition"]) == 1
        }
        if records and not elements.issubset(elemental_endpoints):
            from phase_agent.analysis.phase.build_composition_hull_entries import (
                build_composition_hull_entries,
            )

            snapshot["entries"] = build_composition_hull_entries(records)
            snapshot["hull_domain"] = "observed_Na_endpoints"
            return snapshot
        entries = [
            PDEntry(
                item["composition"],
                item["normalized_total_energy"],
                name=item.get("record_id"),
            )
            for item in records
        ]
        diagram = PhaseDiagram(entries)
        for item, entry in zip(records, entries, strict=True):
            na_check = check_na_layer_uniformity(
                item.get("structure_path"),
                item["composition"],
                structure_data=item.get("structure"),
            )
            snapshot["entries"].append(
                {
                    "record_id": item.get("record_id"),
                    "structure_id": item.get("structure_id"),
                    "structure_path": item.get("structure_path"),
                    "structure": item.get("structure"),
                    "phase": item.get("phase"),
                    "composition": item["composition"],
                    "original_energy": item["original_energy"],
                    "normalized_total_energy": item["normalized_total_energy"],
                    "ehull": float(diagram.get_e_above_hull(entry)),
                    "ehull_unit": "eV/atom",
                    "is_stable": entry in diagram.stable_entries,
                    "source_version": item.get("source_version"),
                    "na_layer_uniform": na_check["uniform"],
                    "na_layer_status": na_check["status"],
                    "na_layer_rule": na_check["rule"],
                }
            )
    except Exception as error:
        if isinstance(error, ValueError) and "至少需要两个不同 Na 含量" in str(error):
            snapshot.update(
                {"status": "unknown", "reason": "insufficient_Na_endpoints", "error": str(error)}
            )
        else:
            snapshot.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
    return snapshot
