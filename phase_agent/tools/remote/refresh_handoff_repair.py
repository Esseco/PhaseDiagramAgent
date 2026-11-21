"""Repackage a frozen, unsubmitted refresh without changing task identities."""

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import zipfile

from phase_agent.tools.remote.batch_runner import _compatibility
from phase_agent.tools.remote.input_validation import validate_handoff
from phase_agent.tools.remote.transfer_bundle import MANIFEST, _safe_name, checksum


def _scope(state, project_root, expected_sha256, expected_count):
    refresh = state.get("model_refresh") or {}
    ids = refresh.get("task_ids") or []
    if (
        refresh.get("status") != "waiting_results"
        or len(ids) != expected_count
        or len(set(ids)) != expected_count
    ):
        raise ValueError("refresh_repair_scope_changed")
    if not isinstance(refresh.get("approved_by"), str) or not refresh["approved_by"]:
        raise ValueError("refresh_repair_approval_identity_missing")
    wanted = set(ids)
    tasks = [t for t in state.get("tasks") or [] if t.get("task_id") in wanted]
    if len(tasks) != expected_count or {t.get("task_id") for t in tasks} != wanted:
        raise ValueError("refresh_repair_task_scope_changed")
    if any(
        t.get("status") != "pending" or t.get("job_id") or t.get("remote_job_id") for t in tasks
    ):
        raise ValueError("refresh_repair_tasks_not_unsubmitted")
    if any(
        t.get("model_version") != state.get("active_model_version")
        or t.get("config_version") != state.get("confirmed_config_version")
        or (refresh.get("approved_by") and t.get("parent_decision_id") != refresh["approved_by"])
        for t in tasks
    ):
        raise ValueError("refresh_repair_model_config_or_approval_changed")
    batches = [
        b for b in state.get("slurm_batches") or [] if wanted.intersection(b.get("task_ids") or [])
    ]
    counts = Counter(i for b in batches for i in b.get("task_ids") or [])
    if (
        set(counts) != wanted
        or any(n != 1 for n in counts.values())
        or any(b.get("status") != "prepared" or b.get("job_id") for b in batches)
    ):
        raise ValueError("refresh_repair_batch_scope_changed")
    old = batches[0].get("handoff") or {}
    archive = Path(old.get("upload_archive") or "").resolve()
    root = Path(project_root).resolve()
    if not archive.is_relative_to(root / "submissions") or archive.name != "inputs.zip":
        raise ValueError("refresh_repair_archive_outside_project")
    if any((b.get("handoff") or {}).get("upload_archive") != str(archive) for b in batches):
        raise ValueError("refresh_repair_multiple_handoffs")
    if checksum(archive) != expected_sha256:
        raise ValueError("refresh_repair_source_archive_changed")
    validate_handoff(archive.parent)
    return tasks, batches, old, archive


def _strip_publisher(content, *, allow_absent=False):
    lines = content.splitlines(keepends=True)
    root = [i for i, line in enumerate(lines) if line.startswith("PHASE_HANDOFF_ROOT=")]
    traps = [
        i
        for i, line in enumerate(lines)
        if line.startswith("trap ") and "_round_transfer.py" in line
    ]
    if allow_absent and not root and not traps:
        return content
    if len(root) != 1 or len(traps) != 1:
        raise ValueError("refresh_repair_ambiguous_old_publish_hook")
    return "".join(line for i, line in enumerate(lines) if i not in {root[0], traps[0]})


def _stage(old_archive, staging_root, state, tasks, batches, manager, references, profiles):
    from phase_agent.tools.remote.manual_upload_runner import _portable_transfer_worker
    from phase_agent.tools.remote.prepare_phase_analysis import prepare_phase_analysis
    from phase_agent.tools.remote.submission_handoff import prepare_handoff

    by_id = {t["task_id"]: t for t in tasks}
    with zipfile.ZipFile(old_archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("refresh_repair_duplicate_archive_path")
        for name in names:
            if name == MANIFEST:
                continue
            safe = _safe_name(name)
            target = staging_root.joinpath(*safe.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(name))
    plan = json.loads((staging_root / "submission_plan.json").read_text(encoding="utf8"))
    scripts = plan.get("scripts") or []
    if len(set(scripts)) != len(scripts):
        raise ValueError("refresh_repair_source_script_scope_changed")
    old_root = Path(batches[0]["handoff"]["import_directory"]).resolve()
    stage_batches = []
    for b in batches:
        row = deepcopy(b)
        for key in ("upload_directory", "script_path", "results_directory"):
            old_path = Path(b[key]).resolve()
            row[key] = str(staging_root / old_path.relative_to(old_root))
        stage_batches.append(row)
    gpu_scripts = {Path(b["script_path"]).relative_to(staging_root).as_posix() for b in stage_batches}
    extra_scripts = set(scripts) - gpu_scripts
    dependencies = plan.get("afterany") or {}
    if not gpu_scripts.issubset(scripts):
        raise ValueError("refresh_repair_script_batch_mismatch")
    if extra_scripts:
        cpu = [p for p in extra_scripts if Path(p).name == "CPU.sh"]
        finalizer = [p for p in extra_scripts if Path(p).name == "CPU_finalize.sh"]
        if (
            len(cpu) != 1 or len(finalizer) != 1 or len(extra_scripts) != 2
            or Path(cpu[0]).parent != Path(finalizer[0]).parent
            or not Path(cpu[0]).parent.name.startswith("zz-phase-analysis-")
            or set(dependencies) != extra_scripts
            or set(dependencies[cpu[0]]) != gpu_scripts
            or dependencies[finalizer[0]] != [cpu[0]]
        ):
            raise ValueError("refresh_repair_source_phase_dependency_changed")
        old_phase_dir = (staging_root / Path(cpu[0]).parent).resolve()
        if not old_phase_dir.is_relative_to(staging_root.resolve()):
            raise ValueError("refresh_repair_source_phase_path_invalid")
        shutil.rmtree(old_phase_dir)
    elif dependencies:
        raise ValueError("refresh_repair_source_phase_dependency_changed")
    predictions = defaultdict(list)
    helpers = defaultdict(set)
    relax_scripts = []
    observed = set()
    for b in stage_batches:
        directory = Path(b["upload_directory"])
        rows = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
        if {r.get("task_id") for r in rows} != set(b.get("task_ids") or []) or len(rows) != len(
            b.get("task_ids") or []
        ):
            raise ValueError("refresh_repair_manifest_scope_mismatch")
        operations = set()
        for row in rows:
            path = (directory / str(row["input_path"]).replace("\\", "/")).resolve()
            if not path.is_relative_to(directory) or path.name != "task.json":
                raise ValueError("refresh_repair_task_path_invalid")
            task = json.loads(path.read_text(encoding="utf8"))
            tid = row.get("task_id")
            source = by_id.get(tid) or {}
            fields = (
                "task_id",
                "task_key",
                "batch_id",
                "config_version",
                "model_version",
                "task_checksum",
                "protocol_version",
                "input_file_version",
            )
            if (
                tid in observed
                or not source
                or any(row.get(k) != task.get(k) for k in fields)
                or any(source.get(k) != task.get(k) for k in fields[:6])
            ):
                raise ValueError("refresh_repair_task_binding_mismatch")
            if checksum(path.parent / "initial.vasp") != task.get("input_structure_sha256"):
                raise ValueError("refresh_repair_structure_hash_mismatch")
            if source.get("input_structure_sha256") != task.get("input_structure_sha256"):
                raise ValueError("refresh_repair_structure_scope_changed")
            result_path = (directory / str(row["result_path"]).replace("\\", "/")).resolve()
            expected_result = Path(source.get("result_path") or "").resolve()
            if (
                not result_path.is_relative_to(staging_root)
                or staging_root / expected_result.relative_to(old_root) != result_path
            ):
                raise ValueError("refresh_repair_result_path_mismatch")
            observed.add(tid)
            op = (task.get("parameters") or {}).get("model_refresh_operation")
            operations.add(op)
            if op == "predict":
                worker = task.get("worker_job") or {}
                key = (
                    _compatibility(task),
                    json.dumps(
                        {
                            name: worker.get(name)
                            for name in (
                                "model_path",
                                "model_paths",
                                "model_file_evidence",
                                "parameters",
                            )
                        },
                        sort_keys=True,
                        default=str,
                    ),
                )
                predictions[key].append((row, path))
                for name in (
                    "_handoff_refresh_executor.py",
                    "_handoff_refresh_worker.py",
                    "_handoff_refresh_inline.py",
                ):
                    helpers[(key, name)].add(checksum(path.parent / name))
            elif op != "relax":
                raise ValueError("refresh_repair_unknown_operation")
        if len(operations) != 1:
            raise ValueError("refresh_repair_mixed_old_batch")
        if operations == {"relax"}:
            old_script = Path(b["script_path"])
            script = old_script.with_name("GPU_repaired.sh")
            script.write_text(
                _strip_publisher(old_script.read_text(encoding="utf8"), allow_absent=bool(extra_scripts)),
                encoding="utf8",
                newline="\n",
            )
            relax_scripts.append(script)
    if observed != set(by_id) or any(len(value) != 1 for value in helpers.values()):
        raise ValueError("refresh_repair_incomplete_or_mixed_executor_scope")
    gpu = list(relax_scripts)
    needed_helper_dirs = set()
    for group_number, (key, entries) in enumerate(sorted(predictions.items()), start=1):
        for offset in range(0, len(entries), 500):
            chunk = entries[offset : offset + 500]
            first = chunk[0][1]
            needed_helper_dirs.add(first.parent)
            source_batch = first.parent.parent
            parent = source_batch.parent
            if any(path.parent.parent.parent != parent for _, path in chunk):
                raise ValueError("refresh_repair_mixed_result_roots")
            folder = parent / f"regroup-{group_number:03d}-{offset // 500 + 1:03d}"
            folder.mkdir()
            manifest = []
            for row, path in chunk:
                entry = deepcopy(row)
                entry["input_path"] = "../" + path.relative_to(parent).as_posix()
                manifest.append(entry)
            (folder / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf8"
            )
            for name in ("_gpu_accounting.py", "_gpu_transfer_bundle.py"):
                shutil.copy2(source_batch / name, folder / name)
            _portable_transfer_worker(folder, "run_mlip_batch", "_handoff_mlip_batch")
            (folder / "run_mlip_batch.py").write_text(
                "from pathlib import Path\nfrom _handoff_mlip_batch import main\n"
                "if __name__ == '__main__':\n"
                "    raise SystemExit(main(['--manifest', str(Path(__file__).resolve().with_name('manifest.json')), '--executor', '_handoff_refresh_executor:execute_mlip_task']))\n",
                encoding="utf8",
                newline="\n",
            )
            original = (source_batch / "GPU.sh").read_text(encoding="utf8")
            content = _strip_publisher(original, allow_absent=bool(extra_scripts))
            content, count = re.subn(
                r"(?m)^#SBATCH --job-name=.*$",
                f"#SBATCH --job-name=refresh-group-{group_number}-{offset // 500 + 1}",
                content,
            )
            if count != 1:
                raise ValueError("refresh_repair_job_name_invalid")
            script = folder / "GPU.sh"
            script.write_text(content, encoding="utf8", newline="\n")
            gpu.append(script)
    staged_state = {"tasks": state.get("tasks") or [], "slurm_batches": stage_batches}
    metadata = prepare_phase_analysis(staged_state, stage_batches, manager, references, profiles)
    cpu = Path(metadata["script_path"])
    rel = lambda p: Path(p).resolve().relative_to(staging_root).as_posix()
    afterany = {rel(cpu): [rel(p) for p in gpu]}
    scripts = [*gpu, cpu]
    roots = sorted({rel(b["results_directory"]) for b in stage_batches})
    last = cpu
    if metadata.get("finalizer_path"):
        last = Path(metadata["finalizer_path"])
        afterany[rel(last)] = [rel(cpu)]
        scripts.append(last)
    prediction_dirs = {path.parent for entries in predictions.values() for _, path in entries}
    prediction_batches = {directory.parent for directory in prediction_dirs}
    helper_names = {
        "_handoff_refresh_executor.py",
        "_handoff_refresh_worker.py",
        "_handoff_refresh_inline.py",
    }
    files = [
        p
        for p in staging_root.rglob("*")
        if p.is_file()
        and p.name != MANIFEST
        and not (
            p.parent in prediction_dirs
            and p.parent not in needed_helper_dirs
            and p.name in helper_names
        )
        and not (
            p.parent in prediction_batches
            and p.name not in {"manifest.json", "batch.snapshot.json"}
        )
    ]
    handoff = prepare_handoff(
        staging_root,
        scripts,
        payload_files=files,
        afterany=afterany,
        extra_result_roots={rel(last): roots},
        publish_on_success=[rel(last)],
    )
    return handoff, metadata, [len(v) for v in predictions.values()], len(gpu)


def rebuild_refresh_handoff(
    state, project_root, expected_sha256, expected_count, manager, references, profiles
):
    """Build and publish a new handoff, preserving the old inputs and tasks.

    The caller must hold the project lifecycle lock and save state with a
    version guard. No remote submission or scientific computation occurs.
    """
    from phase_agent.tools.remote.handoff_naming import package_folder
    from phase_agent.tools.remote.input_validation import verification_summary

    previous = [
        row
        for row in state.get("refresh_handoff_repairs") or []
        if row.get("source_archive_sha256") == expected_sha256
    ]
    if previous:
        if len(previous) != 1 or previous[0].get("task_count") != expected_count:
            raise ValueError("refresh_repair_history_ambiguous")
        saved = previous[0].get("new_handoff") or {}
        path = Path(saved.get("upload_archive") or "").resolve()
        if not path.is_relative_to(Path(project_root).resolve() / "submissions"):
            raise ValueError("refresh_repair_history_outside_project")
        ids = set((state.get("model_refresh") or {}).get("task_ids") or [])
        tasks = [t for t in state.get("tasks") or [] if t.get("task_id") in ids]
        batches = [
            b for b in state.get("slurm_batches") or [] if ids.intersection(b.get("task_ids") or [])
        ]
        counts = Counter(i for batch in batches for i in batch.get("task_ids") or [])
        approval = (state.get("model_refresh") or {}).get("approved_by")
        if (
            len(ids) != expected_count
            or len(tasks) != expected_count
            or not approval
            or set(counts) != ids
            or any(n != 1 for n in counts.values())
            or any(
                t.get("status") != "pending"
                or t.get("job_id")
                or t.get("remote_job_id")
                or t.get("parent_decision_id") != approval
                or t.get("model_version") != state.get("active_model_version")
                or t.get("config_version") != state.get("confirmed_config_version")
                for t in tasks
            )
            or not batches
            or any(
                b.get("status") != "prepared"
                or b.get("job_id")
                or (b.get("handoff") or {}).get("upload_archive") != str(path)
                for b in batches
            )
        ):
            raise ValueError("refresh_repair_history_scope_changed")
        verification = validate_handoff(path.parent)
        if verification.get("fingerprint") != (saved.get("verification") or {}).get("fingerprint"):
            raise ValueError("refresh_repair_history_archive_changed")
        expected_output = path.parent.with_name(path.parent.name[:-6] + "_output")
        if Path(saved.get("return_directory") or "").resolve() != expected_output:
            raise ValueError("refresh_repair_history_return_path_changed")
        expected_output.mkdir(exist_ok=True)
        return {
            "status": "reused",
            "task_count": expected_count,
            "handoff": deepcopy(saved),
            "submitted": False,
        }
    tasks, batches, old, archive = _scope(state, project_root, expected_sha256, expected_count)
    match = re.fullmatch(r"(.+)_Batch\d+_input", archive.parent.name)
    if not match:
        raise ValueError("refresh_repair_invalid_handoff_name")
    old_root = Path(old["import_directory"]).resolve()
    with TemporaryDirectory(prefix="refresh-handoff-repair-") as temporary:
        stage_root = Path(temporary) / "submissions" / old_root.name
        stage_root.mkdir(parents=True)
        staged, phase, groups, gpu_count = _stage(
            archive, stage_root, state, tasks, batches, manager, references, profiles
        )
        if checksum(archive) != expected_sha256:
            raise ValueError("refresh_repair_source_changed_during_preparation")
        fingerprint = staged["package_fingerprint"]
        destination = package_folder(archive.parent.parent, match.group(1), fingerprint)
        if not destination.exists():
            with TemporaryDirectory(prefix=".refresh-repair-", dir=destination.parent) as publish:
                ready = Path(publish) / destination.name
                shutil.copytree(staged["folder"], ready)
                validate_handoff(ready)
                ready.rename(destination)
        verification = verification_summary(validate_handoff(destination))
        destination.with_name(destination.name[:-6] + "_output").mkdir(exist_ok=True)
        new = deepcopy(old)
        new.update(
            folder=str(destination),
            upload_archive=str(destination / "inputs.zip"),
            submit_script=str(destination / "submit_all.sh"),
            dispatcher=str(destination / "dispatch_tasks.py"),
            return_directory=str(destination.with_name(destination.name[:-6] + "_output")),
            import_directory=str(old_root),
            package_fingerprint=fingerprint,
            verification=verification,
            job_count=staged["job_count"],
            repaired_from_archive_sha256=expected_sha256,
            repair_profile="regroup_refresh_with_remote_phase",
            phase_script=Path(phase["script_path"]).relative_to(stage_root).as_posix(),
            phase_group_count=phase.get("group_count", 1),
        )
    for batch in batches:
        batch.setdefault("superseded_handoffs", []).append(deepcopy(batch.get("handoff")))
        batch["handoff"] = deepcopy(new)
        batch["upload_archive"] = new["upload_archive"]
    state["model_refresh"]["phase_analysis_status"] = "remote_cpu_planned"
    state.setdefault("refresh_handoff_repairs", []).append(
        {
            "source_archive_sha256": expected_sha256,
            "task_count": len(tasks),
            "old_handoff": deepcopy(old),
            "new_handoff": deepcopy(new),
            "submitted": False,
        }
    )
    return {
        "status": "prepared",
        "task_count": len(tasks),
        "gpu_job_count": gpu_count,
        "prediction_compatibility_counts": sorted(groups),
        "handoff": new,
        "submitted": False,
    }
