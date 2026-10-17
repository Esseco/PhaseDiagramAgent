"""Compute-node collection only: no scheduler calls and no scientific execution."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import zipfile


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(root, name):
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ":" in name
        or "\\" in name
        or any(p in {"", ".", ".."} for p in name.split("/"))
    ):
        raise ValueError("unsafe_recovery_path:" + name)
    target = root.joinpath(*path.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("recovery_path_outside_source")
    return target


def unpack(root):
    work = root / "work"
    with zipfile.ZipFile(root / "inputs.zip") as archive:
        if archive.getinfo("transfer_manifest.json").file_size > 2 * 1024**2:
            raise ValueError("oversized_collection_manifest")
        manifest = json.loads(archive.read("transfer_manifest.json"))
        rows = manifest["files"]
        if (
            manifest.get("kind") != "inputs"
            or {r["path"] for r in rows} != {"recovery_plan.json", "_transfer.py"}
            or len(archive.infolist()) != 3
        ):
            raise ValueError("unexpected_collection_payload")
        payloads = {}
        for row in rows:
            info = archive.getinfo(row["path"])
            if info.file_size > 2 * 1024**2 or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("unsafe_collection_payload")
            content = archive.read(row["path"])
            if len(content) != row["size"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise ValueError("collection_input_checksum_mismatch")
            target = safe(work, row["path"])
            if target.exists() and target.read_bytes() != content:
                raise ValueError("collection_work_payload_changed")
            payloads[target] = content
    work.mkdir(exist_ok=True)
    for target, content in payloads.items():
        if not target.exists():
            target.write_bytes(content)
    plan = json.loads((work / "recovery_plan.json").read_text(encoding="utf-8"))
    if plan.get("schema") != 1 or plan.get("mode") != "collect_only" or not plan.get("tasks"):
        raise ValueError("not_a_collection_only_plan")
    spec = importlib.util.spec_from_file_location("collection_transfer", work / "_transfer.py")
    transfer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transfer)
    return work, plan, transfer


def terminal(directory, expected):
    result, marker = directory / "result.json", directory / "task.finished.json"
    if not result.is_file() or not marker.is_file():
        return None
    data = json.loads(result.read_text(encoding="utf-8"))
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if (key == "task_checksum" and receipt.get(key) != value) or (
            key != "task_checksum"
            and value is not None
            and (data.get(key) != value or receipt.get(key) != value)
        ):
            raise ValueError("original_task_identity_mismatch:" + key)
    if receipt.get("result_checksum") != checksum(result) or receipt.get("status") != data.get(
        "status"
    ):
        raise ValueError("original_result_checksum_or_status_mismatch")
    if data.get("status") not in {"completed", "failed", "timeout", "cancelled"}:
        return None
    return data


def artifacts(source, directory, task, data):
    files = {
        "result.json": directory / "result.json",
        "task.finished.json": directory / "task.finished.json",
    }
    outputs = data.get("outputs") or {}
    worker = safe(source, task["input_directory"])
    for key, name, digest_key in [
        ("training_file", "training.json", "training_checksum"),
        ("mlip_result_file", "mlip_result.json", "mlip_result_checksum"),
    ]:
        if outputs.get(key):
            if outputs[key] != name:
                raise ValueError("unexpected_scientific_artifact:" + key)
            candidate = directory / name
            if not candidate.is_file():
                candidate = worker / name
            if not candidate.is_file() or checksum(candidate) != outputs.get(digest_key):
                raise ValueError("required_scientific_artifact_invalid:" + name)
            files[name] = candidate
    if outputs.get("structure") is None:
        value = outputs.get("structure_path") or outputs.get("final_structure_path")
        if value:
            name = Path(value).name
            if Path(name).suffix.lower() not in {".vasp", ".poscar"} and name != "CONTCAR":
                raise ValueError("unexpected_structure_artifact")
            candidate = directory / name
            if not candidate.is_file():
                candidate = worker / name
            if not candidate.is_file() or (
                outputs.get("structure_checksum")
                and checksum(candidate) != outputs["structure_checksum"]
            ):
                raise ValueError("required_structure_missing_or_invalid")
            files[name] = candidate
    names = {"checkpoint.json", "checkpoint.json.gz"}
    if data.get("checkpoint"):
        names.add(Path(data["checkpoint"]).name)
    for name in names:
        for parent in (directory, worker):
            if (parent / name).is_file():
                files[name] = parent / name
                break
    for path in files.values():
        if not path.resolve().is_relative_to(source):
            raise ValueError("scientific_artifact_outside_original_package")
    return files


def collect(root):
    work, plan, transfer = unpack(root)
    # The original folder must sit beside this delivery. No guessed filesystem scan.
    supplied = os.environ.get("PHASE_RECOVERY_SOURCE")
    if supplied and not Path(supplied).is_absolute():
        raise ValueError("PHASE_RECOVERY_SOURCE_must_be_absolute")
    source = Path(supplied).resolve() if supplied else safe(root.parent, plan["source_folder"])
    if source == root or not (source / "inputs.zip").is_file():
        raise FileNotFoundError(
            "original_package_missing: place this input beside " + plan["source_folder"]
        )
    if checksum(source / "inputs.zip") != plan["source_archive_sha256"]:
        raise ValueError("original_input_archive_changed")
    report = {
        "mode": "collect_only",
        "source_folder": plan["source_folder"],
        "recovered_task_ids": [],
        "missing_task_ids": [],
        "errors": [],
        "science_submissions": 0,
    }
    receipt = source / ".submitted_jobs.json"
    if receipt.is_file():
        submitted = json.loads(receipt.read_text(encoding="utf-8"))
        if submitted.get("archive_sha256") != plan["source_archive_sha256"]:
            raise ValueError("original_dispatch_receipt_archive_mismatch")
        report["original_submission_states"] = {
            name: row.get("status") for name, row in (submitted.get("jobs") or {}).items()
        }
    else:
        report["original_submission_states"] = "unknown; never submit missing tasks"
    output = root.with_name(root.name[:-6] + "_output")
    with tempfile.TemporaryDirectory(prefix="gather-", dir=work) as temporary:
        staging = Path(temporary)
        for task in plan["tasks"]:
            tid = task["expected"]["task_id"]
            try:
                candidates = []
                for name in (task["result_directory"], task["input_directory"]):
                    directory = safe(source, name)
                    data = terminal(directory, task["expected"])
                    if data is not None:
                        candidates.append((directory, data))
                if not candidates:
                    report["missing_task_ids"].append(tid)
                    continue
                if len({checksum(d / "result.json") for d, _ in candidates}) != 1:
                    raise ValueError("conflicting_original_terminal_results")
                directory, data = candidates[0]
                selected = artifacts(source, directory, task, data)
                destination = safe(staging, task["result_directory"])
                destination.mkdir(parents=True)
                for name, path in selected.items():
                    shutil.copyfile(path, destination / name)
                # Completion/data may change while collecting: verify the snapshot.
                terminal(destination, task["expected"])
                artifacts(
                    staging,
                    destination,
                    {**task, "input_directory": task["result_directory"]},
                    data,
                )
                report["recovered_task_ids"].append(tid)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                # Do not transport an incomplete task after a mid-copy failure.
                destination = safe(staging, task["result_directory"])
                if destination.exists():
                    shutil.rmtree(destination)
                report["errors"].append({"task_id": tid, "error": str(error)})
        operations = {str(Path(t["result_directory"]).parent.as_posix()) for t in plan["tasks"]}
        if report["recovered_task_ids"]:
            for operation in operations:
                for receipt in safe(source, operation).glob("gpu-job-*.runtime.json"):
                    if not receipt.resolve().is_relative_to(source):
                        raise ValueError("runtime_receipt_outside_original_package")
                    target = safe(staging, operation) / receipt.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(receipt, target)
            files = [p for p in staging.rglob("*") if p.is_file()]
            transfer.create_bundle(
                staging,
                files,
                output / "recovered.results.zip",
                kind="results",
                binding={"batch": root.name[:-6], "input_sha256": checksum(root / "inputs.zip")},
            )
        else:
            (output / "recovered.results.zip").unlink(missing_ok=True)
    report["status"] = (
        "complete" if not report["missing_task_ids"] and not report["errors"] else "incomplete"
    )
    return report


def diagnostics(root, report):
    output = root.with_name(root.name[:-6] + "_output")
    output.mkdir(exist_ok=True)
    temporary = output / "diagnostics.zip.part"
    failure = (
        None
        if report["status"] == "complete"
        else "collection_incomplete:"
        + json.dumps(
            {k: report.get(k) for k in ("missing_task_ids", "errors", "error")}, ensure_ascii=False
        )
    )
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("recovery_status.json", json.dumps(report, ensure_ascii=False, indent=2))
        zipped.writestr(
            "dispatch-error.json",
            json.dumps(
                {"status": report["status"], "error": failure, "mode": "collect_only"},
                ensure_ascii=False,
            ),
        )
    temporary.replace(output / "diagnostics.zip")


def main(root=None):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use sbatch submit_all.sh; collection runs on a CPU compute node")
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required on the CPU compute node")
    root = Path(root or Path(__file__).resolve().parent).resolve()
    if not root.name.endswith("_input"):
        raise ValueError("collection_folder_must_end_in_input")
    try:
        report = collect(root)
    except Exception as error:
        report = {
            "mode": "collect_only",
            "status": "failed",
            "error": str(error),
            "science_submissions": 0,
        }
    diagnostics(root, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
