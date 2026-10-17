"""Derive collection-only deliveries from an immutable registered DFT package."""

import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile

from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.remote.result_directory_name import result_directory_name
from phase_agent.tools.remote.transfer_bundle import create_bundle

IDENTITY = (
    "task_id",
    "task_key",
    "batch_id",
    "config_version",
    "model_version",
    "task_checksum",
    "protocol_version",
    "input_file_version",
)


def recovery_plan(state, handoff):
    """Derive identity and destinations from original inputs and registered tasks."""
    archive = Path(handoff["upload_archive"])
    registered = {t["task_id"]: t for t in state.get("tasks") or []}
    tasks = []
    with zipfile.ZipFile(archive) as zipped:
        scripts = json.loads(zipped.read("submission_plan.json"))["scripts"]
        inventory = {
            r["path"]: r for r in json.loads(zipped.read("transfer_manifest.json"))["files"]
        }
        for script in scripts:
            directory = Path(script).parent
            name = (directory / "task.json").as_posix()
            payload = zipped.read(name)
            if hashlib.sha256(payload).hexdigest() != inventory[name]["sha256"]:
                raise ValueError("original_task_checksum_mismatch")
            task = json.loads(payload)
            if task.get("stage") not in {"dft_single_point", "dft_relax"}:
                raise ValueError("collection_repair_only_supports_registered_DFT")
            expected = {key: task.get(key) for key in IDENTITY}
            current = registered.get(task["task_id"])
            if not current or any(
                current.get(k) is not None and current.get(k) != v for k, v in expected.items()
            ):
                raise ValueError("original_task_registration_mismatch")
            inputs = next(p for p in directory.parents if p.name == "inputs")
            operation = inputs.parent
            destination = operation / "results" / result_directory_name(directory, task["stage"])
            tasks.append(
                {
                    "input_directory": directory.as_posix(),
                    "result_directory": destination.as_posix(),
                    "expected": expected,
                }
            )
    if len({t["expected"]["task_id"] for t in tasks}) != len(tasks) or not tasks:
        raise ValueError("ambiguous_original_task_scope")
    return {
        "schema": 1,
        "mode": "collect_only",
        "source_folder": Path(handoff["folder"]).name,
        "source_archive_sha256": file_checksum(archive),
        "tasks": tasks,
    }


def _read_plan(archive):
    with zipfile.ZipFile(archive) as zipped:
        if zipped.getinfo("recovery_plan.json").file_size > 1024**2:
            raise ValueError("oversized_recovery_plan")
        payload = zipped.read("recovery_plan.json")
        manifest = json.loads(zipped.read("transfer_manifest.json"))
        row = next(r for r in manifest["files"] if r["path"] == "recovery_plan.json")
        if manifest.get("kind") != "inputs" or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError("recovery_plan_checksum_mismatch")
        return json.loads(payload)


def prepare_recovery_handoff(state, handoff, cpu_profile):
    """Only add a sibling delivery; never update the original handoff or State."""
    from phase_agent.tools.relax_mc_batch.files import script as shell_script
    from phase_agent.tools.remote.write_unix_shell_script import write_unix_shell_script

    plan = recovery_plan(state, handoff)
    command = ': "${SLURM_JOB_ID:?Use sbatch submit_all.sh}"\ncd -- "${SLURM_SUBMIT_DIR:?}"\npython3 recover_results.py'
    content = shell_script(cpu_profile, command, name="phase-collect-only").replace("\r\n", "\n")
    source = Path(handoff["folder"])
    runner = Path(__file__).with_name("legacy_recovery_runner.py").read_bytes()
    limits = Path(__file__).with_name("transfer_limits.py").read_text(encoding="utf-8")
    transfer = Path(__file__).with_name("transfer_bundle.py").read_text(encoding="utf-8")
    transfer, count = re.subn(
        r"from phase_agent.tools.remote.transfer_limits import \([\s\S]*?\)\n",
        "",
        transfer,
        count=1,
    )
    if count != 1:
        raise ValueError("portable_transfer_import_contract_changed")
    with tempfile.TemporaryDirectory(prefix="collection-input-") as temporary:
        root = Path(temporary)
        (root / "recovery_plan.json").write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
        (root / "_transfer.py").write_text(limits + "\n" + transfer, encoding="utf-8")
        for sequence in range(1, 100):
            folder = source.with_name(
                source.name.removesuffix("_input") + f"_Recovery{sequence:02d}_input"
            )
            output = folder.with_name(folder.name[:-6] + "_output")
            archive = folder / "inputs.zip"
            if folder.exists() or output.exists():
                if not archive.is_file() or _read_plan(archive) != plan:
                    continue
                with zipfile.ZipFile(archive) as zipped:
                    if zipped.read("_transfer.py") != (root / "_transfer.py").read_bytes():
                        continue
                if (
                    not (folder / "recover_results.py").is_file()
                    or (folder / "recover_results.py").read_bytes() != runner
                ):
                    continue
                if (
                    not (folder / "submit_all.sh").is_file()
                    or (folder / "submit_all.sh").read_text(encoding="utf-8") != content
                    or not output.is_dir()
                ):
                    continue
                return _descriptor(handoff, folder, plan)
            options = cpu_profile.get("slurm_options") or {}
            if any(options.get(k) for k in ("gres", "gpus", "gpus-per-node", "gpus-per-task")):
                raise ValueError("collection_recovery_requires_CPU_profile")
            create_bundle(root, list(root.iterdir()), archive, kind="inputs")
            (folder / "recover_results.py").write_bytes(runner)
            write_unix_shell_script(folder / "submit_all.sh", content)
            output.mkdir()
            return _descriptor(handoff, folder, plan)
    raise ValueError("too_many_recovery_deliveries")


def _descriptor(original, folder, plan):
    return {
        "folder": str(folder),
        "upload_archive": str(folder / "inputs.zip"),
        "submit_script": str(folder / "submit_all.sh"),
        "layout_version": 4,
        "return_directory": str(folder.with_name(folder.name[:-6] + "_output")),
        "import_directory": original["import_directory"],
        "collect_only": True,
        "task_ids": [t["expected"]["task_id"] for t in plan["tasks"]],
        "allowed_prefixes": [t["result_directory"] + "/" for t in plan["tasks"]],
        "runtime_roots": sorted(
            {str(Path(t["result_directory"]).parent.as_posix()) for t in plan["tasks"]}
        ),
    }


def recovery_handoffs(state):
    """Discover only siblings bound to original registered identities, read-only."""
    seen = set()
    for batch in state.get("slurm_batches") or []:
        original = batch.get("handoff") or {}
        if not original.get("folder") or not original.get("import_directory"):
            continue
        source = Path(original["folder"])
        if str(source) in seen:
            continue
        seen.add(str(source))
        candidates = sorted(
            source.parent.glob(source.name.removesuffix("_input") + "_Recovery[0-9][0-9]_input"),
            reverse=True,
        )
        if not candidates:
            continue
        expected = recovery_plan(state, original)
        for folder in candidates:
            if _read_plan(folder / "inputs.zip") != expected:
                raise ValueError("recovery_delivery_outside_original_scope")
            yield _descriptor(original, folder, expected)
            # Older versions remain importable, but presentation shows the newest.
