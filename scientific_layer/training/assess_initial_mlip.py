"""Assess a new system's initial MLIP from small endpoint/intermediate probes."""


def assess_initial_mlip(records, *, required_roles=("endpoint", "intermediate"), limits=None):
    settings = {"max_failure_fraction": 0.5, "max_energy_error_ev_per_atom": None}
    settings.update(limits or {})
    completed = [row for row in records if row.get("status") == "completed"]
    roles = {row.get("phase_role") for row in completed}
    missing = sorted(set(required_roles) - roles)
    failures = [row for row in records if row.get("status") in {"failed", "timeout"}]
    fraction = len(failures) / len(records) if records else None
    errors = [abs(float(row["error_ev_per_atom"])) for row in completed
              if row.get("error_ev_per_atom") is not None]
    checks = {"required_phase_roles": not missing,
              "failure_fraction": fraction is not None and fraction <= settings["max_failure_fraction"],
              "energy_error": settings["max_energy_error_ev_per_atom"] is None or
                              bool(errors) and max(errors) <= settings["max_energy_error_ev_per_atom"]}
    return {"status": "passed" if all(checks.values()) else "insufficient_evidence" if missing or not records else "anomaly_detected",
            "checks": checks, "missing_phase_roles": missing, "failure_fraction": fraction,
            "max_energy_error_ev_per_atom": max(errors) if errors else None, "record_count": len(records)}
