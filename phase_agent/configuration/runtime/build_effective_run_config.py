"""Build one effective runtime config from a confirmed scientific snapshot."""

from copy import deepcopy


RUNTIME_KEYS = {
    "structure_directory",
    "state_path",
    "ledger_path",
    "branch_energy_pool_ledger_path",
    "phase_diagram_directory",
    "work_directory",
    "task_versions",
    "mlip",
    "deepseek",
    "approval_directory",
    "local_action_directory",
    "upload_batches_directory",
}


def build_effective_run_config(config_session: dict, runtime_config: dict | None = None) -> dict:
    """Overlay runtime-only settings, while confirmed values remain authoritative."""
    snapshot = config_session.get("confirmed_snapshot") or {}
    if config_session.get("status") != "confirmed" or not snapshot.get("config"):
        raise ValueError("configuration_not_confirmed")
    confirmed = deepcopy(snapshot["config"])
    supplied = deepcopy(runtime_config or {})
    effective = deepcopy(confirmed)
    for key, value in supplied.items():
        if key in RUNTIME_KEYS:
            if key == "mlip" and isinstance(value, dict):
                # Runtime defaults contain empty model paths. Do not let those
                # defaults erase the confirmed remote model path or parameters.
                merged_mlip = deepcopy(effective.get("mlip") or {})
                for model_key, model_value in value.items():
                    if model_key == "relax_parameters":
                        # Relax settings affect result identity. Only the confirmed
                        # snapshot may supply them; a local runtime default must
                        # not turn completed Relax results into new tasks.
                        continue
                    if model_value is None or model_value == "":
                        continue
                    if isinstance(model_value, (dict, list, tuple)) and not model_value:
                        continue
                    merged_mlip[model_key] = deepcopy(model_value)
                effective[key] = merged_mlip
            else:
                effective[key] = value
    if "system" in confirmed:
        effective["system_config"] = deepcopy(confirmed["system"])
    for key in ("budgets", "dft", "convergence", "agent", "frozen_parameters"):
        if key in confirmed:
            effective[key] = deepcopy(confirmed[key])
    if "bohb" in effective:
        effective["bohb"]["budget_limits"] = deepcopy(effective.get("budgets") or {})
    calculation = confirmed.get("calculation") or {}
    effective.update(deepcopy(confirmed.get("run") or {}))
    if calculation.get("mlip_version") is not None:
        effective.setdefault("mlip", {})["version"] = calculation["mlip_version"]
    enabled = set(calculation.get("enabled_stages") or [])
    workflow = (effective.get("system_config") or {}).get("calculation_workflow") or {}
    if enabled:
        for stage in workflow.get("stages") or []:
            stage["enabled"] = stage.get("name") in enabled
    quotas = (confirmed.get("generation_actions") or {}).get("quotas") or {}
    if quotas:
        effective["total_quota"] = sum(int(value) for value in quotas.values())
    effective["config_version"] = snapshot["config_version"]
    effective["config_hash"] = snapshot.get("config_hash")
    effective["runtime_keys"] = sorted(key for key in supplied if key in RUNTIME_KEYS)
    effective["ignored_runtime_keys"] = sorted(key for key in supplied if key not in RUNTIME_KEYS)
    return effective
