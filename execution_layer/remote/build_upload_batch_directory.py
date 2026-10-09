"""Build readable upload paths without changing stable task or batch IDs."""

from __future__ import annotations

from pathlib import Path
import re


STAGE_NAMES = {
    "relax_and_feature": ("Relax-screening", "Relax-submission"),
    "deep_search": ("MC-search", "MC-sampling"),
    "dft_single_point": ("DFT-single-point", "DFT-single-point-submission"),
    "dft_relax": ("DFT-relax", "DFT-relax-submission"),
    "mlip_training": ("MLIP-training", "MLIP-training-submission"),
    "mlip_validation": ("MLIP-validation", "MLIP-validation-submission"),
}


def build_upload_batch_directory(root, state, *, batch_id, stage, model_version,
                                 operation_id=None, segment_index=None, search_group_index=None,
                                 parent_relax_round=None, model_refresh_round=None, refresh_operation=None):
    """Return and register a stable, readable directory for one new batch."""
    layout = state.setdefault("upload_layout", {})
    model_rounds = layout.setdefault("model_rounds", {})
    for batch in state.get("slurm_batches") or []:
        prior_model = str(batch.get("model_version") or "unversioned")
        if prior_model not in model_rounds:
            saved_round = batch.get("mlip_round")
            model_rounds[prior_model] = int(saved_round) if saved_round else (
                max((int(v) for v in model_rounds.values()), default=0) + 1)
    model_label = str(model_version or "unversioned")
    if model_label not in model_rounds:
        model_rounds[model_label] = max((int(v) for v in model_rounds.values()), default=0) + 1
    mlip_round = int(model_rounds[model_label])
    group, prefix = STAGE_NAMES.get(stage, (_safe(stage or "other"), "submission"))
    prior = [row for row in state.get("slurm_batches") or []
             if int(row.get("mlip_round") or 0) == mlip_round
             and row.get("calculation_group") == group]
    submission_index = max((int(row.get("submission_index") or 0) for row in prior), default=0) + 1
    round_directory = f"epoch{mlip_round-1}_{_safe(model_label)}"
    batch_directory = f"{prefix}-{submission_index:04d}_{batch_id}"
    stage_directory = Path(root) / round_directory / group
    if search_group_index is not None:
        stage_directory = Path(root) / round_directory / f"Search-group-{int(search_group_index):04d}"
        metadata_group = int(search_group_index)
    metadata = {
        "mlip_round": mlip_round,
        "calculation_group": group,
        "submission_index": submission_index,
        "model_label": model_label,
    }
    if model_refresh_round is not None:
        if refresh_operation not in {"relax", "predict"}:
            raise ValueError("invalid model refresh operation")
        label = "Relax" if refresh_operation == "relax" else "Single-point"
        stage_directory = Path(root) / round_directory / f"Model-refresh-{int(model_refresh_round):04d}" / label
        metadata.update(model_refresh_round=int(model_refresh_round), refresh_operation=refresh_operation,
                        operation_directory=str(stage_directory))
        return stage_directory / "inputs" / batch_directory, metadata
    if search_group_index is not None:
        metadata['search_group_index'] = metadata_group
    if operation_id:
        operations = layout.setdefault("operations", {})
        scope_group = "DFT" if stage in {"dft_single_point", "dft_relax"} else group
        scope = (f"{model_label}:Search-group-{int(search_group_index):04d}:{scope_group}"
                 if search_group_index is not None else f"{model_label}:{scope_group}")
        registry = operations.setdefault(scope, {})
        if operation_id not in registry:
            scoped_prior = [row for row in prior if search_group_index is None
                            or row.get('search_group_index') == search_group_index]
            prior_indices = [int(row.get("operation_index") or 0) for row in scoped_prior]
            legacy_index = 1 if any(not row.get("operation_id") for row in scoped_prior) else 0
            index = max([*registry.values(), *prior_indices, legacy_index]) + 1
            registry[operation_id] = index
        operation_index = registry[operation_id]
        label = (f"DFT-round-{operation_index:04d}" if scope_group == "DFT"
                 else f"allocation-{operation_index:04d}")
        if stage == "deep_search":
            label += f"_segment-{int(segment_index or 0) + 1:02d}"
        label += f"_{_safe(operation_id)}"
        if search_group_index is not None:
            if stage == 'deep_search':
                parent = int(parent_relax_round or 1)
                label = f"Relax-{parent:04d}_MC-round-{int(segment_index or 0) + 1:04d}"
                metadata['parent_relax_round'] = parent
                metadata['parent_relax_directory'] = str(stage_directory / f'Relax-{parent:04d}')
            elif stage == 'relax_and_feature':
                label = f"Relax-{operation_index:04d}"
            elif stage not in {'dft_single_point', 'dft_relax'}:
                stage_directory /= group
            if stage in {'dft_single_point', 'dft_relax'}:
                stage_directory /= label
                label = group
        stage_directory /= label
        metadata.update(operation_id=operation_id, operation_index=operation_index,
                        operation_directory=str(stage_directory))
    return stage_directory / "inputs" / batch_directory, metadata


def _safe(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return cleaned[:64] or "unnamed"
