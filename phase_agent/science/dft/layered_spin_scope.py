"""Resolve final-structure scope independently of human-readable system IDs."""

LAYERED_PHASES = {"O1", "O2", "O3", "OP2", "P2", "P3"}


def layered_spin_scope(outputs, manager=None):
    evidence = outputs.get("phase_identification") or {}
    phase = (
        outputs.get("actual_phase")
        or evidence.get("phase")
        or outputs.get("observed_phase")
        or evidence.get("observed_phase")
    )
    if phase and phase != "X" and evidence.get("status") != "unknown":
        return phase in LAYERED_PHASES
    # An attempted but inconclusive final classification is not a non-layered result.
    if evidence.get("status") == "unknown":
        return None
    system = (manager.data.get("system_config") or {}) if manager is not None else {}
    explicit = system.get("is_layered_oxide")
    if isinstance(explicit, bool):
        return explicit
    prior = outputs.get("magnetic_check") or {}
    return (
        prior.get("is_layered_oxide") if isinstance(prior.get("is_layered_oxide"), bool) else None
    )
