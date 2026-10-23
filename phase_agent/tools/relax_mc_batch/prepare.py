"""Build one approved portable batch; no upload, submission or scientific execution."""

from copy import deepcopy
import shutil
from pathlib import Path
from time import perf_counter

from phase_agent.runtime.turn_process import timed_call, record_timing, process_event

from phase_agent.tools.relax_mc_batch.plan import diagnose, relax_tasks, context_config
from phase_agent.tools.relax_mc_batch.files import (
    write,
    rebase,
    script,
    coordinator_profile,
)
from phase_agent.tools.remote.transfer_bundle import create_bundle, input_files
from phase_agent.tools.remote.integrity import file_checksum
from phase_agent.tools.budget.reserve_budget import reserve_owned_budget
from phase_agent.science.structures.boundary_utils import load_structure


def _runtime_source_checksums():
    return {
        name: file_checksum(Path(__file__).with_name(name))
        for name in ("run.py", "dependencies.py", "return_payload.py", "prepare.py")
    }


def prepare_batch(*, action, context):
    from phase_agent.tools.state.execution_progress import preparation_progress
    from phase_agent.tools.remote.input_validation import (
        InputValidationError,
        validate_handoff,
        verification_summary,
    )

    preparation_progress("started", {"protocol": 1, "mode": "relax_mc_inputs"})
    config, manager = (
        context_config(context["effective_config"], context),
        context["manager"],
    )
    state = timed_call(
        "RelaxMC：建立本批独立状态",
        deepcopy,
        context.get("event_state") or {},
        category="processing",
    )
    approved = (action.get("parameters") or {}).get("relax_mc_preview")
    if not approved:
        return {
            "status": "awaiting_approval",
            "reason": "relax_mc_preview_required",
            "state": state,
        }
    existing = (state.get("relax_mc_batches") or {}).get(approved["bundle_id"])
    if existing:
        if existing.get("status") in {"cancelled", "superseded"}:
            return {
                "status": "rejected",
                "reason": "cancelled_relax_mc_bundle_cannot_be_reused",
                "state": state,
                "submitted": False,
            }
        if existing["plan_checksum"] != approved["checksum"]:
            raise ValueError("bundle_identity_conflict")
        import json

        packaged = json.loads(
            (Path(existing["directory"]) / "batch.json").read_text(encoding="utf-8")
        )
        if packaged.get("runtime_source_checksums") != _runtime_source_checksums():
            return {
                "status": "rejected",
                "reason": "relax_mc_runtime_changed_reprepare_required",
                "state": state,
                "submitted": False,
            }
        from phase_agent.tools.relax_mc_batch.handoff import (
            compact_handoff,
            output_directory,
        )

        preparation_progress("effects_started", {"directory": existing["directory"]})
        try:
            upload = compact_handoff(existing["directory"])
            existing["verification"] = verification_summary(
                validate_handoff(upload, kind="relax_mc")
            )
        except InputValidationError as error:
            existing["verification"] = error.report
            existing["status"] = "validation_failed"
            return {
                "status": "not_configured",
                "state": state,
                "reason": str(error),
                "verification": error.report,
                "submitted": False,
            }
        existing["status"] = "prepared"
        existing.update(
            upload_folder=str(upload),
            upload_archive=str(upload / "inputs.zip"),
            script_path=str(upload / "submit_all.sh"),
            expected_results_archive=str(output_directory(existing["directory"]) / "results.zip"),
        )
        preparation_progress(
            "complete",
            {
                key: existing[key]
                for key in (
                    "bundle_id",
                    "plan_checksum",
                    "directory",
                    "upload_folder",
                    "upload_archive",
                    "script_path",
                )
            },
        )
        return {
            "status": "already_prepared",
            "state": state,
            **existing,
            "submitted": False,
        }
    plan = timed_call(
        "RelaxMC：校验已批准方案",
        diagnose,
        action.get("target_ids") or [],
        state,
        manager,
        config,
        context["config_version"],
        max_concurrent_jobs=(action.get("parameters") or {}).get("max_concurrent_jobs"),
        phase_references=context.get("phase_references"),
        category="processing",
    )
    if plan["checksum"] != approved["checksum"]:
        return {
            "status": "awaiting_approval",
            "state": state,
            "reason": "relax_mc_inputs_changed_since_approval",
        }
    from phase_agent.tools.remote.model_upload_directory import model_upload_directory

    base = (
        model_upload_directory(
            Path(config["upload_batches_directory"]).resolve(), state, plan["model_version"]
        )
        / "relax_mc"
    )
    from phase_agent.tools.remote.handoff_naming import batch_label

    sequence = 1 + sum(
        row.get("model_version") == plan["model_version"]
        for row in (state.get("relax_mc_batches") or {}).values()
    )
    label = batch_label(base, state, plan["model_version"], "RelaxMC", sequence)
    root = base / label.split("_", 1)[-1]
    preparation_progress(
        "plan_verified",
        {
            "directory": str(root),
            "input_directory": str(root.with_name(root.name + "_input")),
            "output_directory": str(root.with_name(root.name + "_output")),
            "bundle_id": plan["bundle_id"],
            "plan_checksum": plan["checksum"],
            "task_count": len(plan["structures"]),
        },
    )
    if root.exists():
        raise FileExistsError("Unregistered bundle already exists; inspect before retry")
    tasks = relax_tasks(plan, config)
    coordinator_profile(plan["stage_profiles"])
    process_event(
        "RelaxMC输入准备",
        {
            "task_count": len(tasks),
            "branch_count": len(plan["branch_ids"]),
            "reuse_approved_plan": True,
        },
    )
    reservation_started = perf_counter()
    for task in tasks:
        reservation = reserve_owned_budget(
            state,
            task_key=task["task_key"],
            stage=task["stage"],
            amount=task["planned_relative_cost"],
            limits=config["budgets"],
            config_version=context["config_version"],
            model_version=plan["model_version"],
        )
        if reservation["status"] == "rejected":
            return {
                "status": "rejected",
                "state": context.get("event_state") or {},
                "reason": str(reservation["reasons"]),
            }
        state = reservation["state"]
    record_timing(
        "RelaxMC：登记本批预算", perf_counter() - reservation_started, category="processing"
    )
    from phase_agent.graphs.cancellation import check_cancelled

    check_cancelled()
    preparation_progress("effects_started", {"directory": str(root)})
    root.mkdir(parents=True)
    mapping = {}
    for index, entry in enumerate(plan["inputs"]):
        source = Path(entry["path"])
        if file_checksum(source) != entry["sha256"]:
            raise ValueError("input_file_changed_during_packaging")
        target = root / "inputs" / f"{index:05d}.vasp"
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        mapping[str(source)] = str(target)
    structures = {}
    branches = {}
    for bid in plan["branch_ids"]:
        branch = manager.data["branches"][bid]
        branches[bid] = {
            key: deepcopy(branch[key])
            for key in ("branch_id", "P", "H", "x", "T", "composition")
            if key in branch
        }
        branches[bid]["structure_ids"] = [
            r["structure_id"] for r in plan["structures"] if r["branch_id"] == bid
        ]
        for entry in plan["inputs"]:
            if entry["kind"] == "full_na" and entry["id"] == bid:
                branches[bid]["full_na_structure_path"] = mapping[entry["path"]]
    for row in plan["structures"]:
        sid = row["structure_id"]
        structures[sid] = {
            "structure_id": sid,
            "branch_id": row["branch_id"],
            "source_path": mapping[row["source_path"]],
            "composition": row["composition"],
            "stage_history": {stage: [] for stage in manager.STAGES},
            "metadata": {"initialization_method": "electrostatic_top10_random3_layer_occupied"},
        }
        full = branches[row["branch_id"]].get("full_na_structure_path")
        if full:
            structures[sid]["full_na_structure_path"] = full
    references = {}
    for index, (name, reference) in enumerate(
        (
            context.get("phase_references")
            or (manager.data.get("system_config") or {}).get("phase_references")
            or {}
        ).items()
    ):
        path = root / "references" / f"phase-{index}.vasp"
        path.parent.mkdir(exist_ok=True)
        load_structure(reference).to(filename=path, fmt="poscar")
        references[name] = str(path)
    system = {
        "species": deepcopy(
            (manager.data.get("system_config") or {}).get("species") or {"mobile": ["Na"]}
        )
    }
    data = {
        "schema_version": 2,
        "boundary": deepcopy(manager.boundary),
        "system_config": system,
        "branches": branches,
        "structures": structures,
    }
    model = deepcopy(config["mlip"])
    model.update(plan["model_references"])
    if plan["model_references"]["model_paths"]:
        model["members"] = deepcopy(plan["model_references"]["model_file_evidence"])
    else:
        model.pop("members", None)
        model.update(plan["model_references"]["model_file_evidence"][0])
    for key in ("model_directory", "local_model_path", "local_model_paths"):
        model.pop(key, None)
    remote_config = {
        "mlip": model,
        "mc_policy": plan["mc_policy"],
        "budgets": deepcopy(config["budgets"]),
        "bohb": deepcopy(config.get("bohb") or {}),
        "supercomputer": {
            "worker": deepcopy(config["supercomputer"]["worker"]),
            "batch_sizes": plan["batch_sizes"],
        },
    }
    remote_tasks = deepcopy(tasks)
    for row in remote_tasks:
        row["structure_path"] = mapping[row["structure_path"]]
    history = deepcopy(plan["history_records"])
    for row in history:
        if row.get("structure_path") in mapping:
            row["structure_path"] = mapping[row["structure_path"]]
        elif row.get("structure_path"):
            row.pop(
                "structure_path"
            )  # Historical energy evidence remains; absent geometry is not fabricated.
    runtime = root / "runtime"
    source = Path(__file__).resolve().parents[2]
    timed_call(
        "RelaxMC：复制运行代码",
        shutil.copytree,
        source,
        runtime / "phase_agent",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        category="processing",
    )
    from phase_agent.tools.remote.manual_upload_runner import _bundle_mc_backend

    _bundle_mc_backend(runtime)
    manifest = {
        "schema_version": 1,
        "runtime_source_checksums": _runtime_source_checksums(),
        "bundle_id": plan["bundle_id"],
        "plan_checksum": plan["checksum"],
        "config_version": plan["config_version"],
        "model_version": plan["model_version"],
        "config": remote_config,
        "manager": data,
        "phase_references": references,
        "bootstrap_first_mc": plan["bootstrap_first_mc"],
        "require_hull_before_first_mc": True,
        "stage_profiles": plan["stage_profiles"],
        "workload": plan["workload"],
        "seed": plan["seed"],
        "initial_state": {
            "tasks": remote_tasks,
            "phase_records": history,
            "active_model_version": plan["model_version"],
            "confirmed_config_version": plan["config_version"],
            "budget_reservations": deepcopy(state["budget_reservations"]),
            "dedup_gate": {"status": "ready"},
        },
    }
    write(root / "batch.json", rebase(manifest, root, portable=True))
    write(root / "workload.json", plan["workload"])
    (root / "run_batch.py").write_text(
        "from phase_agent.tools.relax_mc_batch.cli import main\nif __name__ == '__main__':\n    raise SystemExit(main())\n",
        encoding="utf-8",
    )
    command = (
        'cd -- "${SLURM_SUBMIT_DIR:?Submit from the extracted batch directory}"\n'
        'export PYTHONPATH="$PWD/runtime/mc_backend:$PWD/runtime:${PYTHONPATH:-}"\n'
        'python3 run_batch.py run --root "$PWD" --submit'
    )
    (root / "submit_all.sh").write_text(
        script(coordinator_profile(plan["stage_profiles"]), command, name=plan["bundle_id"]),
        encoding="utf-8",
        newline="\n",
    )
    guide = (
        "Upload the complete <batch>_input folder (submit_all.sh, run_upload.py and inputs.zip). Run sbatch submit_all.sh ONCE; Python executes on the compute node. Download the automatically generated sibling <batch>_output folder intact.\n"
        "Short CPU entry submits grouped GPU Relax jobs and an afterany analysis job, then exits. Analysis verifies the whole phase before submitting parallel MC jobs and the next dependency analysis. No resident coordinator.\n"
        f"Workload upper GPU compute jobs:{sum(plan['workload']['gpu_jobs_upper_by_wave'])}; up to five short CPU entry/analysis jobs.\n"
        "No model weights are copied: model references must exist on HPC and are checked before child submission.\n"
        "The mace environment needs numpy/scipy/pymatgen/ase/torch/MACE; scientific code and Process_AL_MC are shipped.\n"
        "Resume with the same uploaded directory; do not delete its workflow_state.json or rerun a batch already active.\n"
    )
    (root / "README.md").write_text(guide, encoding="utf-8")
    archive = timed_call(
        "RelaxMC：压缩并校验输入",
        create_bundle,
        root,
        input_files(root),
        root / "inputs.zip",
        kind="inputs",
        category="processing",
    )
    import zipfile

    with zipfile.ZipFile(archive) as zipped:
        (root / "transfer_manifest.json").write_bytes(zipped.read("transfer_manifest.json"))
    from phase_agent.tools.relax_mc_batch.handoff import (
        compact_handoff,
        output_directory,
    )

    validation_error = None
    try:
        upload = timed_call(
            "RelaxMC：生成并验收交付目录", compact_handoff, root, category="processing"
        )
        verification = verification_summary(validate_handoff(upload, kind="relax_mc"))
    except InputValidationError as error:
        from phase_agent.tools.relax_mc_batch.handoff import input_directory

        upload = input_directory(root)
        validation_error, verification = str(error), error.report
    record = {
        "bundle_id": plan["bundle_id"],
        "plan_checksum": plan["checksum"],
        "config_version": plan["config_version"],
        "model_version": plan["model_version"],
        "branch_ids": plan["branch_ids"],
        "package_label": label,
        "batch_sequence": sequence,
        "status": "validation_failed" if validation_error else "prepared",
        "directory": str(root),
        "upload_folder": str(upload),
        "upload_archive": str(upload / "inputs.zip"),
        "script_path": str(upload / "submit_all.sh"),
        "expected_results_archive": str(output_directory(root) / "results.zip"),
        "workload": plan["workload"],
        "verification": verification,
    }
    for row in tasks:
        row.update(slurm_batch_id=plan["bundle_id"], relax_mc_bundle_id=plan["bundle_id"])
    state.setdefault("tasks", []).extend(tasks)
    state.setdefault("pending_tasks", []).extend(deepcopy(tasks))
    state.setdefault("relax_mc_batches", {})[plan["bundle_id"]] = record
    if validation_error:
        return {
            "status": "not_configured",
            "state": state,
            "reason": validation_error,
            "verification": verification,
            "submitted": False,
        }
    preparation_progress(
        "complete",
        {
            key: record[key]
            for key in (
                "bundle_id",
                "plan_checksum",
                "directory",
                "upload_folder",
                "upload_archive",
                "script_path",
            )
        },
    )
    return {"status": "prepared", "state": state, **record, "submitted": False}
