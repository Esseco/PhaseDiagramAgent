"""Bound extrapolation evidence for the exact DFT candidate geometry."""

import hashlib
import math
from pathlib import Path


def candidate_extrapolation(candidate, state):
    """Return normal, failed or unknown; absence never proves safety."""
    path = candidate.get("structure_path")
    digest = candidate.get("structure_sha256") or (
        candidate.get("dft_sampling_evidence") or {}
    ).get("geometry_key")
    model = candidate.get("model_version") or state.get("active_model_version")
    if not path or not model:
        return {"status": "unknown", "reason": "exact_geometry_or_model_missing"}
    try:
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest and actual != digest:
            return {"status": "unknown", "reason": "geometry_checksum_mismatch"}
    except OSError:
        return {"status": "unknown", "reason": "geometry_unavailable"}
    digest = digest or actual
    flags = set()
    if candidate.get("extrapolation_failure") is True:
        flags.add("failed")
    for task in state.get("tasks") or []:
        if task.get("model_version") != model:
            continue
        output = task.get("outputs") or {}
        if output.get("structure_checksum") == digest:
            monitor = output.get("extrapolation_monitor") or {}
            review = task.get("extrapolation_review") or {}
            if (
                monitor.get("extrapolation_failure") is True
                or review.get("status") == "extrapolation_failure"
                or task.get("result_disposition") == "excluded_extrapolation_failure"
            ):
                flags.add("failed")
            score = monitor.get("final_feature_difference")
            threshold = monitor.get("feature_difference_threshold", 1.0)
            if (
                task.get("status") == "completed"
                and monitor.get("extrapolation_failure") is False
                and monitor.get("failures") == []
                and type(score) in (int, float)
                and math.isfinite(score)
                and type(threshold) in (int, float)
                and threshold in (1.0, 3.0)
                and (
                    not (task.get("parameters") or {}).get("model_refresh_operation")
                    or threshold == 1.0
                )
                and 0 <= score <= threshold
            ):
                flags.add("normal")
        observations = output.get("relaxation_observations") or []
        steps = output.get("actual_mc_steps")
        observations_complete = (
            task.get("status") == "completed"
            and output.get("relaxation_observations_complete") is True
            and type(steps) is int
            and steps >= 0
            and len(observations) == steps + 1
        )
        for observation in observations:
            if (
                not observations_complete
                or observation.get("structure_checksum") != digest
                or observation.get("monitor_complete") is not True
                or observation.get("identity_scheme") != "periodic_input_v1"
            ):
                continue
            flag = observation.get("extrapolation_failure")
            if flag is True:
                flags.add("failed")
            elif flag is False:
                flags.add("normal")
        for failure in output.get("extrapolation_failures") or []:
            if failure.get("structure_checksum") == digest:
                flags.add("failed")
    if "failed" in flags:
        return {"status": "failed", "reason": "confirmed_extrapolation_failure"}
    if "normal" in flags:
        return {"status": "normal", "reason": "exact_frame_monitor"}
    for batch in (state.get("remote_dft_sampling") or {}).values():
        if batch.get("status") != "accepted" or batch.get("model_version") != model:
            continue
        if any(item.get("structure_sha256") == digest for item in batch.get("selected") or []):
            return {"status": "normal", "reason": "accepted_remote_dft_sampling"}
    return {"status": "unknown", "reason": "exact_frame_monitor_unavailable"}


def known_failed_frames(state):
    """Index only explicitly failed exact frames, retaining model provenance."""
    result = set()
    for task in state.get("tasks") or []:
        model = task.get("model_version")
        if not model:
            continue
        output = task.get("outputs") or {}
        monitor = output.get("extrapolation_monitor") or {}
        if (
            monitor.get("extrapolation_failure") is True
            or (task.get("extrapolation_review") or {}).get("status") == "extrapolation_failure"
            or task.get("result_disposition") == "excluded_extrapolation_failure"
        ) and output.get("structure_checksum"):
            result.add((model, output["structure_checksum"]))
        for failure in output.get("extrapolation_failures") or []:
            if failure.get("structure_checksum"):
                result.add((model, failure["structure_checksum"]))
        for observation in output.get("relaxation_observations") or []:
            if (
                observation.get("monitor_complete") is True
                and observation.get("extrapolation_failure") is True
                and observation.get("structure_checksum")
            ):
                result.add((model, observation["structure_checksum"]))
    return result


def failed_dft_task_ids(state):
    """Match failed frames to DFT inputs; never infer from a reused structure ID."""
    failed = known_failed_frames(state)
    if not failed:
        return set()
    matches = set()
    for task in state.get("tasks") or []:
        if task.get("stage") not in {"dft_single_point", "dft_relax"}:
            continue
        model = task.get("extrapolation_model_version") or task.get("model_version")
        digest = task.get("input_structure_sha256")
        if not digest and task.get("structure_path"):
            try:
                digest = hashlib.sha256(Path(task["structure_path"]).read_bytes()).hexdigest()
            except OSError:
                continue
        if (model, digest) in failed:
            matches.add(task.get("task_id"))
    return matches - {None}
