"""Decide whether accumulated checked DFT data can trigger retraining."""


def check_mlip_retrain_trigger(dft_records: list[dict], *, requested_action: str, config: dict) -> dict:
    valid = [item for item in dft_records if item.get("status") == "completed" and item.get("converged") is True and item.get("checks_passed", True) and item.get("energy") is not None]
    threshold = int((config.get("retrain") or {}).get("minimum_new_dft_records", 10))
    ready = len(valid) >= threshold
    return {"action": "RETRAIN_MLIP" if ready and requested_action == "RETRAIN_MLIP" else "CONTINUE_DATA_COLLECTION", "ready": ready, "valid_new_dft_count": len(valid), "minimum_new_dft_records": threshold, "reason": "threshold_met_and_requested" if ready and requested_action == "RETRAIN_MLIP" else "insufficient_data_or_not_requested"}
