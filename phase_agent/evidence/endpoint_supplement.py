"""Register validated historical endpoint evidence through the project Agent."""

from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import io
import json
import math
import re
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def register_endpoint_supplement(state, csv_path, archive_path, *, csv_sha256, archive_sha256):
    """Validate all bindings before publishing immutable evidence and updating state."""
    csv_bytes = Path(csv_path).read_bytes()
    archive_bytes = Path(archive_path).read_bytes()
    if _sha(csv_bytes) != csv_sha256 or _sha(archive_bytes) != archive_sha256:
        raise ValueError("endpoint_source_checksum_mismatch")
    model = state.get("active_model_version")
    refresh = state.get("model_refresh") or {}
    if not model or model != refresh.get("new_model_version"):
        raise ValueError("endpoint_model_mismatch")
    if state.get("pending_execution_policies"):
        raise ValueError("pending_execution_policy_blocks_endpoint_registration")
    previous = state.get("endpoint_supplement")
    if previous:
        if (
            previous.get("csv_sha256") == csv_sha256
            and previous.get("archive_sha256") == archive_sha256
            and previous.get("model_version") == model
        ):
            if (supplement_facts(state) or {}).get("invalid"):
                old = Path(previous["directory"]).resolve()
                new = Path(csv_path).resolve().parent
                if (old.exists() or Path(archive_path).resolve().parent != new
                    or Path(csv_path).name != "evidence.csv"
                    or Path(archive_path).name != "frames.zip"
                    or new.name != old.name
                    or not new.is_relative_to(Path(state["analysis_output_directory"]).resolve())):
                    raise ValueError("endpoint_existing_registration_invalid")

                def relocate(value):
                    if isinstance(value, dict):
                        return {key: relocate(item) for key, item in value.items()}
                    if isinstance(value, list):
                        return [relocate(item) for item in value]
                    if isinstance(value, str):
                        try:
                            return str(new / Path(value).relative_to(old))
                        except ValueError:
                            pass
                    return value

                updated = relocate(deepcopy(state))
                if (supplement_facts(updated) or {}).get("invalid"):
                    raise ValueError("endpoint_existing_registration_invalid")
                return updated, {"status": "relocated", "task_count": previous["task_count"],
                                 "directory": str(new), "previous_directory": str(old)}
            return state, {"status": "already_registered", "task_count": previous["task_count"]}
        raise ValueError("endpoint_supplement_already_registered")
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
    if not rows or len(rows) != len({r["task_id"] for r in rows}):
        raise ValueError("endpoint_duplicate_or_empty")
    scope = set(refresh.get("task_ids") or [])
    tasks = {t.get("task_id"): t for t in state.get("tasks") or []}
    if state.get("extrapolation_review_summary"):
        flagged = next(
            (
                t
                for t in tasks.values()
                if (t.get("extrapolation_review") or {}).get("status") == "extrapolation_failure"
            ),
            None,
        )
        if not flagged:
            raise ValueError("endpoint_prior_evidence_missing")
        prior = Path(flagged["extrapolation_review"]["source"]).read_bytes()
        if _sha(prior) != state["extrapolation_review_summary"].get("evidence_sha256"):
            raise ValueError("endpoint_prior_evidence_invalid")
        prior_ids = {r["task_id"] for r in csv.DictReader(io.StringIO(prior.decode("utf-8-sig")))}
        if prior_ids & {r["task_id"] for r in rows}:
            raise ValueError("endpoint_overlaps_prior_evidence")
    payloads = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as z:
        manifest = json.loads(z.read("transfer_manifest.json"))
        entries = manifest.get("files") or []
        bypath = {r["path"]: r for r in entries}
        if (
            len(entries) != len(bypath)
            or len(entries) != len(rows)
            or manifest.get("kind") != "results"
        ):
            raise ValueError("endpoint_archive_manifest_invalid")
        if (
            set(z.namelist()) != set(bypath) | {"transfer_manifest.json"}
            or len(z.namelist()) != len(bypath) + 1
        ):
            raise ValueError("endpoint_archive_members_invalid")
        for row in rows:
            tid = row["task_id"]
            task = tasks.get(tid) or {}
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", tid):
                raise ValueError("endpoint_task_id_invalid")
            if task.get("extrapolation_review") or (task.get("outputs") or {}).get(
                "extrapolation_failures"
            ):
                raise ValueError("endpoint_task_already_has_evidence:" + tid)
            score = float(row["score"])
            if (
                tid not in scope
                or task.get("model_version") != model
                or task.get("structure_id") != row["structure_id"]
                or not math.isfinite(score)
                or score < 0
                or row.get("error")
                or row["classification"]
                != ("extrapolation_failure" if score > 1 else "below_threshold")
            ):
                raise ValueError("endpoint_task_or_score_invalid:" + tid)
            path = PurePosixPath(row["archive_path"])
            parts = path.parts
            if (
                len(parts) != 3
                or parts[-1] != "initial_relaxed.vasp"
                or any(p in {".", ".."} for p in parts)
            ):
                raise ValueError("endpoint_archive_path_invalid")
            inp = Path(task["input_path"])
            directory = inp.parent
            if directory.name != parts[-2] or directory.parent.name != parts[-3]:
                raise ValueError("endpoint_input_path_mismatch:" + tid)
            record = json.loads(inp.read_text(encoding="utf-8"))
            initial_sha = _sha((directory / "initial.vasp").read_bytes())
            rp = Path(task["result_path"])
            result_bytes = rp.read_bytes()
            result = json.loads(result_bytes)
            marker = json.loads((rp.parent / "task.finished.json").read_text(encoding="utf-8"))
            if (
                record.get("task_id") != tid
                or record.get("task_key") != task.get("task_key")
                or result.get("task_id") != tid
                or result.get("task_key") != task.get("task_key")
                or marker.get("task_id") != tid
                or marker.get("task_key") != task.get("task_key")
                or marker.get("result_checksum") != _sha(result_bytes)
                or marker.get("status") != result.get("status")
                or result.get("status") != "failed"
                or result.get("error") != "ValueError: refresh_relaxation_not_converged"
                or initial_sha != row["initial_sha256"]
                or initial_sha != task.get("input_structure_sha256")
                or initial_sha != record.get("input_structure_sha256")
            ):
                raise ValueError("endpoint_task_identity_invalid:" + tid)
            entry = bypath.get(row["archive_path"])
            if not entry or entry["sha256"] != row["final_sha256"]:
                raise ValueError("endpoint_frame_manifest_invalid:" + tid)
            if (
                z.getinfo(row["archive_path"]).file_size != entry["size"]
                or entry["size"] > 10_000_000
            ):
                raise ValueError("endpoint_frame_size_invalid:" + tid)
            frame = z.read(row["archive_path"])
            if _sha(frame) != entry["sha256"]:
                raise ValueError("endpoint_frame_checksum_invalid:" + tid)
            from ase.io import read

            atoms = read(io.StringIO(frame.decode("utf8")), format="vasp")
            initial_atoms = read(directory / "initial.vasp", format="vasp")
            if sorted(atoms.get_chemical_symbols()) != sorted(initial_atoms.get_chemical_symbols()):
                raise ValueError("endpoint_composition_mismatch:" + tid)
            payloads.append((task, row, frame, score))
    from phase_agent.analysis.feedback.output_paths import analysis_epoch_directory

    output_root = Path(state["analysis_output_directory"]).resolve()
    name = "supplement-" + csv_sha256[:16]
    legacy = output_root / "endpoint_supplements" / name
    if legacy.exists():
        # Preserve an already-written pre-layout receipt after an interrupted registration.
        destination = legacy
    else:
        destination = analysis_epoch_directory(output_root, state, model) / "endpoint_supplements" / name
    base = destination.parent
    base.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (
            _sha((destination / "evidence.csv").read_bytes()) != csv_sha256
            or _sha((destination / "frames.zip").read_bytes()) != archive_sha256
            or any(
                _sha((destination / "frames" / (t["task_id"] + ".vasp")).read_bytes())
                != row["final_sha256"]
                for t, row, frame, score in payloads
                if score > 1
            )
        ):
            raise ValueError("endpoint_existing_destination_conflict")
    else:
        with tempfile.TemporaryDirectory(prefix=".endpoint-", dir=base) as tmp:
            stage = Path(tmp) / "data"
            stage.mkdir()
            (stage / "evidence.csv").write_bytes(csv_bytes)
            (stage / "frames.zip").write_bytes(archive_bytes)
            (stage / "frames").mkdir()
            for task, row, frame, score in payloads:
                if score > 1:
                    (stage / "frames" / (task["task_id"] + ".vasp")).write_bytes(frame)
            stage.rename(destination)
    failures = 0
    for task, row, frame, score in payloads:
        if score <= 1:
            continue
        failures += 1
        outputs = task.setdefault("outputs", {})
        if outputs is None:
            outputs = task["outputs"] = {}
        evidence = {
            "structure_path": str(destination / "frames" / (task["task_id"] + ".vasp")),
            "structure_checksum": row["final_sha256"],
            "extrapolation_monitor": {
                "extrapolation_failure": True,
                "final_feature_difference": score,
                "trigger": "historical_final",
                "force_jump_evidence_available": False,
            },
        }
        outputs.setdefault("extrapolation_failures", []).append(evidence)
    state["endpoint_supplement"] = {
        "model_version": model,
        "csv_sha256": csv_sha256,
        "archive_sha256": archive_sha256,
        "directory": str(destination),
        "task_count": len(rows),
        "failure_count": failures,
        "task_ids": [r["task_id"] for r in rows],
    }
    return state, {
        "status": "registered",
        "task_count": len(rows),
        "failure_count": failures,
        "directory": str(destination),
    }


def supplement_facts(state):
    """Revalidate persisted evidence before exposing counts."""
    item = state.get("endpoint_supplement")
    if not item:
        return None
    try:
        directory = Path(item["directory"]).resolve()
        from phase_agent.analysis.feedback.output_paths import analysis_epoch_directory

        output_root = Path(state["analysis_output_directory"]).resolve()
        unassigned = analysis_epoch_directory(output_root, state, item.get("model_version")).resolve()
        if not directory.is_relative_to(output_root) and not directory.is_relative_to(unassigned):
            raise ValueError("endpoint_directory_invalid")
        csv_bytes = (directory / "evidence.csv").read_bytes()
        if (
            _sha(csv_bytes) != item["csv_sha256"]
            or _sha((directory / "frames.zip").read_bytes()) != item["archive_sha256"]
            or item["model_version"] != state["active_model_version"]
        ):
            raise ValueError("endpoint_integrity_invalid")
        rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
        ids = [r["task_id"] for r in rows]
        if ids != item["task_ids"] or len(ids) != item["task_count"] or len(set(ids)) != len(ids):
            raise ValueError("endpoint_rows_invalid")
        taskmap = {t.get("task_id"): t for t in state.get("tasks") or []}
        failures = 0
        for row in rows:
            task = taskmap[row["task_id"]]
            score = float(row["score"])
            if (
                task.get("model_version") != item["model_version"]
                or not math.isfinite(score)
                or score < 0
                or row.get("error")
                or row["classification"]
                != ("extrapolation_failure" if score > 1 else "below_threshold")
            ):
                raise ValueError("endpoint_row_invalid")
            if score > 1:
                failures += 1
                frame = directory / "frames" / (row["task_id"] + ".vasp")
                if _sha(frame.read_bytes()) != row["final_sha256"]:
                    raise ValueError("endpoint_frame_invalid")
                matches = (task.get("outputs") or {}).get("extrapolation_failures") or []
                if not any(
                    x.get("structure_path") == str(frame)
                    and x.get("structure_checksum") == row["final_sha256"]
                    for x in matches
                ):
                    raise ValueError("endpoint_task_binding_invalid")
        if failures != item["failure_count"]:
            raise ValueError("endpoint_count_invalid")
        return {"task_ids": set(ids), "failure_count": failures, "task_count": len(ids)}
    except (OSError, KeyError, TypeError, ValueError, csv.Error):
        return {"invalid": True}
