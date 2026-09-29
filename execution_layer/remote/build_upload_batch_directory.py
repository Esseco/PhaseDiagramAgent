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


def build_upload_batch_directory(root, state, *, batch_id, stage, model_version):
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
    round_directory = f"MLIP-round-{mlip_round:04d}_{_safe(model_label)}"
    batch_directory = f"{prefix}-{submission_index:04d}_{batch_id}"
    metadata = {
        "mlip_round": mlip_round,
        "calculation_group": group,
        "submission_index": submission_index,
        "model_label": model_label,
    }
    return Path(root) / round_directory / group / batch_directory, metadata


def _safe(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return cleaned[:64] or "unnamed"
