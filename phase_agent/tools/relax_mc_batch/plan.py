"""Read-only branch diagnosis, portable input inventory and conservative workload preview."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from phase_agent.tools.remote.integrity import file_checksum, payload_checksum
from phase_agent.science.mc.continuation_policy import effective_mc_policy
from phase_agent.science.mlip.slurm_executor import remote_model_references
from phase_agent.configuration.defaults.default_slurm_cluster_config import (
    default_slurm_cluster_config,
)
from phase_agent.analysis.cost.predict_runtime import predict_runtime
from phase_agent.tools.budget.estimate_stage_cost import estimate_stage_cost


def profiles(config):
    supplied = (config.get("manual_upload") or {}).get("stage_profiles") or (
        config.get("supercomputer") or {}
    ).get("stage_profiles")
    return deepcopy(supplied or default_slurm_cluster_config())


def context_config(config, context):
    result = deepcopy(config)
    overrides = context.get("relax_mc_runtime") or {}
    cluster = result.setdefault("supercomputer", {})
    if overrides.get("stage_profiles"):
        cluster["stage_profiles"] = deepcopy(overrides["stage_profiles"])
    if overrides.get("batch_sizes"):
        cluster["batch_sizes"] = deepcopy(overrides["batch_sizes"])
    if overrides.get("worker_command"):
        cluster["worker"] = {"command": list(overrides["worker_command"])}
    return result


def relax_tasks(plan, config):
    settings = deepcopy((config.get("mlip") or {}).get("relax_parameters") or {})
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    return [
        {
            "task_id": "RELAX-"
            + hashlib.sha256(
                f"{plan['model_version']}:{signature}:{row['structure_id']}".encode()
            ).hexdigest()[:12],
            "task_key": f"relax-screen:{plan['model_version']}:{signature}:{row['structure_id']}",
            "stage": "relax_and_feature",
            "status": "pending",
            "branch_id": row["branch_id"],
            "structure_id": row["structure_id"],
            "structure_path": row["source_path"],
            "model_version": plan["model_version"],
            "config_version": plan["config_version"],
            "parameters": deepcopy(settings),
            "atom_count": row["atom_count"],
            "planned_relative_cost": row["cost"],
            "relax_mc_bundle_id": plan["bundle_id"],
        }
        for row in plan["structures"]
    ]


def diagnose(
    branch_ids,
    state,
    manager,
    config,
    config_version,
    *,
    max_concurrent_jobs=None,
    phase_references=None,
):
    if not branch_ids or len(branch_ids) != len(set(branch_ids)):
        raise ValueError("Explicit unique recommended branch IDs are required")
    if (state.get("dedup_gate") or {}).get("status") != "ready":
        raise ValueError("structure_dedup_not_ready")
    model = config.get("mlip") or {}
    version = model.get("version") or model.get("name")
    if not version or state.get("active_model_version", version) != version:
        raise ValueError("active_model_version_mismatch")
    if any(
        row.get("stage") == "deep_search"
        and row.get("branch_id") in branch_ids
        and row.get("model_version") == version
        for row in state.get("tasks") or []
    ):
        raise ValueError(
            "branches_already_searched_under_current_model; use existing continuation instead of resetting MC counts"
        )
    main, models, evidence = remote_model_references(model)
    count = min(3, int((config.get("bohb") or {}).get("relax_structures_per_branch", 3)))
    if count <= 0:
        raise ValueError("relax_structures_per_branch must be positive")
    records = manager.data.get("structures") or {}
    valid_ids = set((state.get("dedup_gate") or {}).get("valid_structure_ids") or records)
    selected, structures, inputs = [], [], []
    references = (
        phase_references or (manager.data.get("system_config") or {}).get("phase_references") or {}
    )
    reference_fingerprints = {}
    for name, reference in references.items():
        if hasattr(reference, "as_dict"):
            reference_fingerprints[name] = payload_checksum(reference.as_dict())
        else:
            path = Path(reference)
            if not path.is_file():
                raise ValueError(f"missing_phase_reference:{name}")
            reference_fingerprints[name] = file_checksum(path)
    for bid in branch_ids:
        branch = (manager.data.get("branches") or {}).get(bid)
        if branch is None:
            raise ValueError(f"unknown_branch:{bid}")
        selected.append(deepcopy(branch))
        ids = [
            sid for sid in branch.get("structure_ids") or [] if sid in records and sid in valid_ids
        ][:count]
        if not ids:
            raise ValueError(f"branch_has_no_valid_structure:{bid}")
        for sid in ids:
            source = Path(records[sid].get("source_path") or "")
            if not source.is_file():
                raise ValueError(f"missing_structure:{sid}")
            atoms = sum((records[sid].get("composition") or {}).values())
            if not atoms:
                from pymatgen.core import Structure

                atoms = len(Structure.from_file(source))
            cost = estimate_stage_cost(
                "relax_and_feature", atom_count=int(atoms), budgets=config["budgets"]
            )["value"]
            structures.append(
                {
                    "branch_id": bid,
                    "structure_id": sid,
                    "source_path": str(source),
                    "atom_count": int(atoms),
                    "composition": deepcopy(records[sid].get("composition")),
                    "cost": cost,
                }
            )
            inputs.append(
                {
                    "kind": "structure",
                    "id": sid,
                    "path": str(source),
                    "sha256": file_checksum(source),
                }
            )
        if Fraction(str(branch["x"])) > 0:
            full = branch.get("full_na_structure_path") or records[ids[0]].get(
                "full_na_structure_path"
            )
            if not full or not Path(full).is_file():
                raise ValueError(
                    f"missing_full_na_template:{bid}; prepare the existing template before packaging"
                )
            inputs.append(
                {"kind": "full_na", "id": bid, "path": str(full), "sha256": file_checksum(full)}
            )
    diagram = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if diagram.get("entries") and diagram.get("model_version") != version:
        raise ValueError("existing_phase_diagram_model_mismatch")
    history = [
        deepcopy(row)
        for row in state.get("phase_records") or []
        if row.get("energy_method") == "mlip"
        and row.get("model_version", row.get("source_version")) == version
        and row.get("status") == "completed"
        and row.get("phase_identification_status") == "identified"
    ]
    if not history and diagram.get("status") == "completed":
        for row in diagram.get("entries") or []:
            history.append(
                {
                    **deepcopy(row),
                    "energy": row["normalized_total_energy"],
                    "energy_unit": "eV",
                    "energy_method": "mlip",
                    "model_version": version,
                    "status": "completed",
                }
            )
    bootstrap = not bool(history)
    for row in history:
        path = row.get("structure_path")
        if path and Path(path).is_file():
            inputs.append(
                {
                    "kind": "history_geometry",
                    "id": str(len(inputs)),
                    "path": path,
                    "sha256": file_checksum(path),
                }
            )
    mobile = [b for b in selected if Fraction(str(b["x"])) > 0]
    if len(mobile) != len(selected) and not references:
        raise ValueError("phase_references_required_for_empty_mobile_sites")
    sizes = (config.get("supercomputer") or {}).get("batch_sizes") or {}
    relax_size, mc_size = (
        int(sizes.get("relax_and_feature", 100)),
        min(10, int(sizes.get("deep_search", 10))),
    )
    if min(relax_size, mc_size) <= 0:
        raise ValueError("batch_sizes must be positive")
    settings = (config.get("supercomputer") or {}).get("relax_mc_batch") or {}
    concurrency = (
        max_concurrent_jobs
        if max_concurrent_jobs is not None
        else settings.get("max_concurrent_jobs", 1)
    )
    if type(concurrency) is not int or concurrency <= 0:
        raise ValueError("max_concurrent_jobs must be a positive integer")
    bcount = len(mobile)
    upper = [bcount, bcount if bcount >= 10 else 0, bcount if bcount >= 10 else 0]
    reference = [bcount]
    for _ in range(2):
        reference.append(min(bcount, max(10, math.ceil(reference[-1] / 2))) if bcount >= 10 else 0)
    cost_mc = sum(
        estimate_stage_cost(
            "deep_search",
            atom_count=max(r["atom_count"] for r in structures if r["branch_id"] == b["branch_id"]),
            mc_steps=100 + (400 if bcount >= 10 else 0),
            budgets=config["budgets"],
        )["value"]
        for b in mobile
    )
    timings = [
        predict_runtime("relax_and_feature", atom_count=r["atom_count"], state=state)
        for r in structures
    ]
    phase_counts = {
        p: sum(b.get("P") == p for b in selected) for p in sorted({b.get("P") for b in selected})
    }
    plan = {
        "schema_version": 1,
        "config_version": config_version,
        "model_version": version,
        "config_checksum": payload_checksum(config),
        "branches": selected,
        "boundary": deepcopy(manager.boundary),
        "branch_ids": list(branch_ids),
        "structures": structures,
        "inputs": inputs,
        "history_records": history,
        "bootstrap_first_mc": bootstrap,
        "model_references": {
            "model_path": main,
            "model_paths": models,
            "model_file_evidence": evidence,
        },
        "mc_policy": effective_mc_policy(config),
        "stage_profiles": profiles(config),
        "phase_reference_checksums": reference_fingerprints,
        "diagnosis": {
            "phases": phase_counts,
            "Na_contents": sorted({str(b["x"]) for b in selected}),
            "warnings": [
                "After bootstrap MC, fewer than two successful Na contents pauses continuation"
            ]
            if bootstrap and len({str(b["x"]) for b in selected}) < 2
            else [],
        },
        "workload": {
            "relax_structure_tasks": len(structures),
            "mc_task_reference": reference,
            "mc_task_upper": upper,
            "mc_step_upper": bcount * (500 if bcount >= 10 else 100),
            "gpu_jobs_upper_by_wave": [math.ceil(len(structures) / relax_size)]
            + [math.ceil(n / mc_size) for n in upper],
            "coordinator_jobs": 1,
            "max_concurrent_jobs": concurrency,
            "relax_serial_seconds": sum(r["elapsed_seconds"] for r in timings)
            if all(r.get("elapsed_seconds") is not None for r in timings)
            else None,
            "note": "Scientific task counts differ from Slurm jobs/ntasks; actual continuation is feedback-dependent; no queue ETA.",
        },
        "estimated_relative_cost_upper": sum(r["cost"] for r in structures) + cost_mc,
        "seed": int((config.get("run") or {}).get("seed", 42)),
        "batch_sizes": {"relax_and_feature": relax_size, "deep_search": mc_size},
    }
    plan["checksum"] = payload_checksum(plan)
    plan["bundle_id"] = "relax-mc-" + plan["checksum"].split(":")[-1][:16]
    future = relax_tasks(plan, config)
    existing = {r.get("task_key") for r in state.get("tasks") or []}
    if any(row["task_key"] in existing for row in future):
        raise ValueError(
            "recommended structures already have Relax tasks; do not duplicate the original pipeline"
        )
    return plan


def attach_preview(action, state, context, config):
    if (action.get("parameters") or {}).get("mode") != "relax_mc_inputs":
        return action
    result = deepcopy(action)
    config = context_config(config, context)
    plan = diagnose(
        action.get("target_ids") or [],
        state,
        context["manager"],
        config,
        context.get("config_version") or state.get("confirmed_config_version"),
        max_concurrent_jobs=(action.get("parameters") or {}).get("max_concurrent_jobs"),
        phase_references=context.get("phase_references"),
    )
    result.setdefault("parameters", {})["relax_mc_preview"] = plan
    result["budget"] = plan["estimated_relative_cost_upper"]
    result["expected_purpose"] = (
        f"一次打包Relax+最多三段MC；{len(plan['branch_ids'])}分支，GPU计算作业上限{sum(plan['workload']['gpu_jobs_upper_by_wave'])}个，另1个控制作业；固定模型，人工上传提交。"
    )
    return result


def batch_facts(state):
    return [
        {
            **{
                key: deepcopy(row.get(key))
                for key in (
                    "bundle_id",
                    "status",
                    "model_version",
                    "branch_ids",
                    "workload",
                    "upload_archive",
                    "script_path",
                    "expected_results_archive",
                    "return_error",
                    "result_summary",
                )
            },
            "return_available": Path(row.get("expected_results_archive") or "").is_file(),
        }
        for row in list((state.get("relax_mc_batches") or {}).values())[-5:]
    ]
