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


def _original_source_folder(state, handoff):
    """A local rename must not change the remote label in a bound recovery plan."""
    source = Path(handoff["folder"])
    storage = (state.get("confirmed_config") or {}).get("storage") or {}
    layout = state.get("workspace_layout") or {}
    root = layout.get("workspace_root") or storage.get("workspace_root")
    if not root or not source.resolve().is_relative_to(Path(root).resolve()):
        return source.name
    from phase_agent.tools.state.workflow_control import read_control

    state_path = (
        Path(layout["state_path"])
        if layout.get("state_path")
        else Path(root) / (storage.get("paths") or {}).get("state", "workflow_state/state.json")
    )
    control = read_control(state_path)
    if (
        not control
        or not state.get("path_relocation_revision")
        or state["path_relocation_revision"] != control.get("path_relocation_revision")
    ):
        return source.name
    names = {
        Path(row["from"]).name
        for row in control.get("path_relocations") or []
        if Path(row["to"]).resolve() == source.resolve()
    }
    if len(names) > 1:
        raise ValueError("ambiguous_original_delivery_folder")
    return next(iter(names), source.name)


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
        "source_folder": _original_source_folder(state, handoff),
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


def _recovery_plan_matches(actual, expected):
    """Accept path-only changes made by a local submission layout migration."""
    if actual == expected:
        return True
    for key in ("schema", "mode", "source_archive_sha256"):
        if actual.get(key) != expected.get(key):
            return False
    actual_tasks = actual.get("tasks") or []
    expected_tasks = expected.get("tasks") or []
    if len(actual_tasks) != len(expected_tasks):
        return False
    for actual_task, expected_task in zip(actual_tasks, expected_tasks):
        for key in set(actual_task) | set(expected_task):
            if key in {"input_directory", "result_directory"}:
                continue
            if actual_task.get(key) != expected_task.get(key):
                return False
        if (
            Path(actual_task.get("input_directory", "")).name
            != Path(expected_task.get("input_directory", "")).name
        ):
            return False
        if (
            Path(actual_task.get("result_directory", "")).name
            != Path(expected_task.get("result_directory", "")).name
        ):
            return False
    return True


def prepare_recovery_handoff(state, handoff, cpu_profile):
    """Only add a sibling delivery; never update the original handoff or State."""
    from phase_agent.tools.relax_mc_batch.files import script as shell_script
    from phase_agent.tools.remote.write_unix_shell_script import write_unix_shell_script

    plan = recovery_plan(state, handoff)
    command = ': "${SLURM_JOB_ID:?Use sbatch submit_all.sh}"\ncd -- "${SLURM_SUBMIT_DIR:?}"\npython3 recover_results.py'
    content = shell_script(cpu_profile, command, name="phase-collect-only").replace("\r\n", "\n")
    source = Path(handoff["folder"])
    runner = Path(__file__).with_name("legacy_recovery_runner.py").read_bytes()
    with tempfile.TemporaryDirectory(prefix="collection-input-") as temporary:
        root = Path(temporary)
        (root / "recovery_plan.json").write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
        (root / "_transfer.py").write_text(portable_transfer_source(), encoding="utf-8")
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


def portable_transfer_source():
    """Reuse the standalone transfer module without rewriting its source."""
    return Path(__file__).with_name("transfer_bundle.py").read_text(encoding="utf-8-sig")


def _recovery_import_directory(original, recovery_key):
    directory = Path(original["import_directory"])
    legacy = directory.parent / "_recovery" / recovery_key
    if legacy.exists():
        return legacy
    parents = [directory, *directory.parents]
    submission = next((p for p in parents if p.name == "submissions"), None)
    epoch = next(
        (
            re.match(r"^(epoch\d+)(?:_.+)?$", p.name)
            for p in parents
            if re.match(r"^(epoch\d+)(?:_.+)?$", p.name)
        ),
        None,
    )
    if submission is not None and epoch is not None:
        from phase_agent.tools.remote.submission_architecture import epoch_evidence_directory

        return epoch_evidence_directory(submission / epoch[1], "recovery") / recovery_key
    return legacy


def _descriptor(original, folder, plan, *, remote_folder=None):
    with zipfile.ZipFile(folder / "inputs.zip") as archive:
        reassessment = "repair_options.json" in archive.namelist()
    recovery_key = hashlib.sha256(
        ((remote_folder or folder.name) + plan["source_archive_sha256"]).encode()
    ).hexdigest()[:12]
    import_directory = (
        _recovery_import_directory(original, recovery_key)
        if reassessment
        else Path(original["import_directory"])
    )
    return {
        "folder": str(folder),
        "upload_archive": str(folder / "inputs.zip"),
        "submit_script": str(folder / "submit_all.sh"),
        "layout_version": 4,
        "return_directory": str(folder.with_name(folder.name[:-6] + "_output")),
        "import_directory": str(import_directory),
        "collect_only": not reassessment,
        **({"dft_recompute": False, "same_frame_prediction": True} if reassessment else {}),
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
            if not _recovery_plan_matches(_read_plan(folder / "inputs.zip"), expected):
                raise ValueError("recovery_delivery_outside_original_scope")
            yield _descriptor(
                original,
                folder,
                expected,
                remote_folder=_original_source_folder(state, {"folder": str(folder)}),
            )
            # Older versions remain importable, but presentation shows the newest.


def reassessment_result_paths(state):
    """Prefer validated new recovery copies without overwriting old imported results."""
    paths = {}
    for handoff in recovery_handoffs(state):
        if not handoff.get("same_frame_prediction"):
            continue
        plan = _read_plan(handoff["upload_archive"])
        for task in plan["tasks"]:
            result = Path(handoff["import_directory"]) / task["result_directory"] / "result.json"
            if result.is_file():
                paths.setdefault(task["expected"]["task_id"], result)
    return paths
