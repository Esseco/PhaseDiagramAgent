"""One simple factual row per saved MLIP version/round, using its own DFT pairs."""

from phase_agent.analysis.feedback.dft_comparison_tables import (
    build_comparison_tables,
    comparison_metrics,
)
from phase_agent.evidence.dft_round_status import dft_recovery_rounds
from phase_agent.evidence.model_epoch import model_epoch

ROUND_SUMMARY_FIELDS = (
    "model_epoch",
    "mlip_round",
    "model_version",
    "model_sha256",
    "generation_batches",
    "branches_proposed",
    "branches_selected",
    "branches_registered",
    "branch_count_status",
    "relax_tasks",
    "relax_completed",
    "mc_branches",
    "mc_tasks",
    "mc_completed",
    "dft_selected_structures",
    "dft_tasks",
    "dft_recovered",
    "dft_successful",
    "dft_failed",
    "dft_recovery_ratio",
    "dft_pending",
    "dft_wait_waived",
    "dft_spin_passed",
    "dft_spin_rejected",
    "dft_spin_unknown",
    "matched_dft_tasks",
    "energy_mae_eV",
    "energy_rmse_eV",
    "energy_mae_eV_per_atom",
    "energy_rmse_eV_per_atom",
    "force_mae_eV_per_A",
    "force_rmse_eV_per_A",
    "force_component_count",
    "gs_matched_dft_tasks",
    "gs_energy_mae_eV_per_atom",
    "gs_force_mae_eV_per_A",
    "sample_matched_dft_tasks",
    "sample_energy_mae_eV_per_atom",
    "sample_force_mae_eV_per_A",
    "error_status",
)


def summarize_model_rounds(state):
    tasks = state.get("tasks") or []
    datasets = state.get("dft_dataset_records") or []
    generations = state.get("generation_history") or []
    versions = {
        row.get("model_version")
        for row in [*tasks, *datasets, *generations]
        if row.get("model_version")
    }
    saved_rounds = (state.get("upload_layout") or {}).get("model_rounds") or {}
    versions.update(saved_rounds)
    recovery = dft_recovery_rounds(state)
    output = []
    for version in sorted(versions, key=lambda v: (saved_rounds.get(v, float("inf")), v)):
        selected_tasks = [row for row in tasks if row.get("model_version") == version]
        records = [row for row in datasets if row.get("model_version") == version]
        comparisons = [
            row
            for row in state.get("dft_mlip_comparisons") or []
            if (row.get("round_scope") or {}).get("model_version") == version
        ]
        fingerprints = {(r.get("mlip_prediction") or {}).get("model_sha256") for r in records}
        fingerprints.discard(None)
        if len(fingerprints) > 1:
            raise ValueError(
                f"different comparison model fingerprints under version {version}; cannot merge round errors"
            )
        energies, forces = build_comparison_tables(
            records, comparisons, round_name=f"MLIP:{version}"
        )
        metrics = comparison_metrics(energies, forces, {"model_version": version})
        tasks_by_id = {row.get("task_id"): row for row in selected_tasks if row.get("task_id")}
        gs_ids = {
            row.get("task_id")
            for row in records
            if (tasks_by_id.get(row.get("task_id")) or {}).get("is_ground_state") is True
            or (tasks_by_id.get(row.get("task_id")) or {}).get("selection_mode")
            == "phase_diagram_gs"
            or row.get("is_ground_state") is True
            or row.get("selection_mode") == "phase_diagram_gs"
        }
        gs_e, gs_f = build_comparison_tables(
            [r for r in records if r.get("task_id") in gs_ids],
            [c for c in comparisons if c.get("task_id") in gs_ids],
            round_name=f"MLIP:{version}:GS",
        )
        sp_e, sp_f = build_comparison_tables(
            [r for r in records if r.get("task_id") not in gs_ids],
            [c for c in comparisons if c.get("task_id") not in gs_ids],
            round_name=f"MLIP:{version}:sample",
        )
        gs_metrics = comparison_metrics(gs_e, gs_f, {"model_version": version, "subset": "gs"})
        sp_metrics = comparison_metrics(sp_e, sp_f, {"model_version": version, "subset": "sample"})
        gen_rows, unknown = _generations_for_model(generations, tasks, version)
        unknown = unknown or "generation_history" not in state
        summary = {
            "model_epoch": model_epoch(state, version),
            "mlip_round": saved_rounds.get(version),
            "model_version": version,
            "model_sha256": ((state.get("model_registry") or {}).get(version) or {}).get(
                "comparison_model_sha256"
            )
            or next(iter(fingerprints), None),
            "generation_batches": len(gen_rows),
            "branch_count_status": "unknown_legacy_lineage" if unknown else "recorded",
        }
        for name, source in (
            ("branches_proposed", "proposed_branches"),
            ("branches_selected", "selected_branches"),
            ("branches_registered", "registered_branches"),
        ):
            values = [(row.get("summary") or {}).get(source) for row in gen_rows]
            summary[name] = None if unknown or any(v is None for v in values) else sum(values)
        relax = [t for t in selected_tasks if t.get("stage") == "relax_and_feature"]
        mc = [t for t in selected_tasks if t.get("stage") == "deep_search"]
        dft = [t for t in selected_tasks if t.get("stage") in {"dft_single_point", "dft_relax"}]
        rounds = [r for r in recovery if r["scope"].get("model_version") == version]
        received = sum(r["recovered_tasks"] for r in rounds)
        summary.update(
            relax_tasks=len(relax),
            relax_completed=sum(t.get("status") == "completed" for t in relax),
            mc_branches=len({t["branch_id"] for t in mc if t.get("branch_id")}),
            mc_tasks=len(mc),
            mc_completed=sum(t.get("status") == "completed" for t in mc),
            dft_selected_structures=len({t["structure_id"] for t in dft if t.get("structure_id")}),
            dft_tasks=len(dft),
            dft_recovered=received,
            dft_successful=sum(r["successful_tasks"] for r in rounds),
            dft_failed=sum(r["failed_tasks"] for r in rounds),
            dft_recovery_ratio=received / len(dft) if dft else None,
            dft_pending=sum(len(r["pending_task_ids"]) for r in rounds),
            dft_wait_waived=sum(len(r["waived_task_ids"]) for r in rounds),
            dft_spin_passed=sum(
                (r.get("spin_state_check") or {}).get("status") == "passed" for r in records
            ),
            dft_spin_rejected=sum(
                (r.get("spin_state_check") or {}).get("status") == "rejected" for r in records
            ),
            dft_spin_unknown=sum(
                (r.get("spin_state_check") or {}).get("status") == "unknown" for r in records
            ),
            matched_dft_tasks=metrics["matched_structures"],
            energy_mae_eV=metrics["energy_total"]["mae"],
            energy_rmse_eV=metrics["energy_total"]["rmse"],
            energy_mae_eV_per_atom=metrics["energy_per_atom"]["mae"],
            energy_rmse_eV_per_atom=metrics["energy_per_atom"]["rmse"],
            force_mae_eV_per_A=metrics["forces"]["mae"],
            force_rmse_eV_per_A=metrics["forces"]["rmse"],
            force_component_count=metrics["forces"]["components"],
            gs_matched_dft_tasks=gs_metrics["matched_structures"],
            gs_energy_mae_eV_per_atom=gs_metrics["energy_per_atom"]["mae"],
            gs_force_mae_eV_per_A=gs_metrics["forces"]["mae"],
            sample_matched_dft_tasks=sp_metrics["matched_structures"],
            sample_energy_mae_eV_per_atom=sp_metrics["energy_per_atom"]["mae"],
            sample_force_mae_eV_per_A=sp_metrics["forces"]["mae"],
            error_status="completed"
            if dft and metrics["matched_structures"] == len(dft)
            else "partial",
        )
        output.append(summary)
    return output


def _generations_for_model(generations, tasks, version):
    selected, unknown = [], False
    for row in generations:
        declared = row.get("model_version")
        if not declared:
            registered_ids = set(row.get("registered_ids") or [])
            evidence = {
                t.get("model_version")
                for t in tasks
                if t.get("structure_id") in registered_ids and t.get("model_version")
            }
            declared = next(iter(evidence)) if len(evidence) == 1 else None
        if declared == version:
            selected.append(row)
        elif declared is None:
            unknown = True
    return selected, unknown
