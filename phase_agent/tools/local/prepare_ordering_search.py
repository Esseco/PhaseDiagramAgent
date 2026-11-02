"""Thin approved adapter to Process_AL_MC; no duplicate scientific search."""

from copy import deepcopy
from pathlib import Path
import shutil

from phase_agent.science.structures.boundary_utils import load_structure


def resolve_request(action, state, context, config):
    params = action.get("parameters") or {}
    allowed = {
        "mode",
        "phase",
        "tm_ratios",
        "na_content",
        "max_det_H",
        "h_matrices",
        "oxidation_states",
        "options",
        "reference_path",
        "ordering_search_preview",
    }
    unknown = set(params) - allowed
    if unknown:
        raise ValueError("Unknown independent search parameters: " + ",".join(sorted(unknown)))
    phase = str(params.get("phase") or "").upper()
    reference = params.get("reference_path") or (context.get("phase_references") or {}).get(phase)
    if reference is None:
        reference = (
            (config.get("system") or config.get("system_config") or {}).get("phase_references")
            or {}
        ).get(phase)
    if reference is None:
        raise ValueError("Missing full-Na reference for phase " + phase)
    reference = load_structure(reference)
    model = {**deepcopy(config.get("mlip") or {}), **deepcopy(state.get("active_model") or {})}
    model["version"] = (
        state.get("active_model_version")
        or model.get("version")
        or (config.get("calculation") or {}).get("mlip_version")
        or model.get("name")
    )
    from Process_AL_MC.ordering.common import freeze_model

    model = freeze_model(model)
    request = {
        name: deepcopy(params[name])
        for name in (
            "phase",
            "tm_ratios",
            "na_content",
            "max_det_H",
            "h_matrices",
            "oxidation_states",
            "options",
        )
        if name in params
    }
    request.update(reference=reference, model=model)
    from Process_AL_MC.ordering.common import digest

    binding = digest(
        {**{k: v for k, v in request.items() if k != "reference"}, "reference": reference.as_dict()}
    )
    return request, binding


def attach_ordering_preview(action, state, context, config):
    if (
        action.get("tool") != "prepare_local_batch_files"
        or (action.get("parameters") or {}).get("mode") != "ordering_search_inputs"
    ):
        return action
    from Process_AL_MC import plan_ordering_search

    result = deepcopy(action)
    request, binding = resolve_request(result, state, context, config)
    plan = plan_ordering_search(**request)
    result["ordering_search_plan"] = plan
    result["ordering_source_binding"] = binding
    from phase_agent.tools.budget.estimate_stage_cost import estimate_stage_cost

    estimated = 0.0
    for matrix in plan["matrices"]:
        atoms = len(matrix["reference"]["sites"]) - len(matrix["na_sites"]) + matrix["na_count"]
        one = estimate_stage_cost(
            "relax_and_feature", atom_count=atoms, budgets=config.get("budgets") or {}
        )["value"]
        calls = min(
            matrix["space_size"],
            len(matrix["initial_occupations"]) * (plan["options"]["max_steps"] + 1),
        )
        estimated += one * calls
    result.setdefault("parameters", {})["ordering_search_preview"] = {
        "plan_id": plan["plan_id"],
        "phase": plan["phase"],
        "na_content": plan["na_content"],
        "tm_ratios": plan["tm_ratios"],
        "model_version": plan["model"]["version"],
        "workload": plan["workload"],
        "estimated_relative_cost_upper": estimated,
        "matrices": [
            {
                "matrix_id": m["matrix_id"],
                "H": m["H"],
                "det_H": m["det_H"],
                "move_kind": m["move_kind"],
                "charge_scheme": m["charge_scheme"],
                "charge_source": m["charge_source"],
                "tm_charge_discriminating": m["tm_charge_discriminating"],
                "initial_na_occupations": [o["na"] for o in m["initial_occupations"]],
                "initial_tm_occupations": [o["tm"] for o in m["initial_occupations"]],
            }
            for m in plan["matrices"]
        ],
        "skipped": plan["skipped"],
        "submitted": False,
    }
    result["budget"] = estimated
    result["expected_purpose"] = (
        f"独立搜索{plan['workload']['matrix_count']}个H、{plan['workload']['chain_count']}条链；"
        "批准仅准备输入，不提交；按TM归组输出总排名与逐H排名。"
    )
    return result


def stale_ordering_error(action, state, context, config):
    if (action.get("parameters") or {}).get("mode") != "ordering_search_inputs":
        return None
    if not action.get("ordering_search_plan") or not action.get("ordering_source_binding"):
        return "ordering_search_requires_frozen_library_preview"
    try:
        _, binding = resolve_request(action, state, context, config)
    except (ValueError, OSError) as error:
        return str(error)
    if binding != action["ordering_source_binding"]:
        return "独立搜索母结构、H范围或模型设置已变化；请重新审阅方案。"
    return None


def prepare_ordering_inputs(*, action, context):
    config = context.get("effective_config") or {}
    state = deepcopy(context.get("event_state") or {})
    error = stale_ordering_error(action, state, context, config)
    if error:
        return {"status": "rejected", "state": state, "reason": error, "submitted": False}
    from Process_AL_MC import prepare_ordering_search

    plan = action["ordering_search_plan"]
    base = config.get("upload_batches_directory") or config.get("local_action_directory")
    if not base:
        return {
            "status": "not_configured",
            "state": state,
            "reason": "ordering_search_output_root_missing",
        }
    root = Path(base).resolve() / "ordering_searches" / plan["plan_id"][:16]
    registered = (state.get("ordering_searches") or {}).get(plan["plan_id"])
    if root.exists() and not registered:
        return {
            "status": "reconciliation_required",
            "state": state,
            "reason": "Existing unregistered ordering directory; inspect receipt before retry",
            "submitted": False,
        }
    prepared = prepare_ordering_search(plan, root)
    from phase_agent.tools.remote.manual_upload_runner import _bundle_mc_backend

    import tempfile

    target_backend = root / "runtime" / "Process_AL_MC"
    with tempfile.TemporaryDirectory(prefix=".ordering-backend-", dir=root.parent) as temporary:
        _bundle_mc_backend(Path(temporary))
        backend = Path(temporary) / "mc_backend" / "Process_AL_MC"
        if not target_backend.exists():
            target_backend.parent.mkdir(exist_ok=True)
            backend.rename(target_backend)
        else:
            for candidate in backend.rglob("*"):
                if candidate.is_file():
                    target = target_backend / candidate.relative_to(backend)
                    if not target.is_file() or target.read_bytes() != candidate.read_bytes():
                        raise ValueError("ordering backend differs; reconcile the existing handoff")
    remote = Path(__file__).parents[1] / "remote"
    shutil.copy2(remote / "transfer_bundle.py", root / "transfer_bundle.py")
    (root / "export_ordering_result.py").write_text(
        (remote / "export_ordering_result.py")
        .read_text(encoding="utf-8")
        .replace(
            "from phase_agent.tools.remote.transfer_bundle import", "from transfer_bundle import"
        ),
        encoding="utf-8",
    )
    profiles = (context.get("relax_mc_runtime") or {}).get("stage_profiles") or {}
    if profiles:
        from phase_agent.tools.relax_mc_batch.files import gpu_profile, script

        command = (
            'cd -- "${SLURM_SUBMIT_DIR:?Submit from the extracted search directory}"\n'
            'export PYTHONPATH="$PWD/runtime:${PYTHONPATH:-}"\npython3 run_search.py'
        )
        (root / "submit_search.sh").write_text(
            script(gpu_profile(profiles), command, name="ordering-" + plan["plan_id"][:10]),
            encoding="utf-8",
            newline="\n",
        )
    (root / "README.md").write_text(
        "Independent fixed-composition TM-focused search. Inputs only; no job submitted.\n"
        "Use the configured MACE environment: python run_search.py\n"
        "Submit manually with the reviewed site profile if submit_search.sh is present.\n"
        "After completion: python export_ordering_result.py .\n"
        "Return only the generated .results.zip; keep models, logs, runtime and search history on the server.\n"
        "The archive retains the plan, model binding, checksummed result manifest, ranking CSVs and referenced structures.\n"
        "Each parallel worker loads a model; workers and GPU memory are part of the reviewed workload.\n"
        "Interrupted running claims require explicit reconciliation; preserve history files.\n",
        encoding="utf-8",
    )
    from phase_agent.tools.remote.transfer_bundle import create_bundle, input_files

    archive = create_bundle(
        root,
        [
            p
            for p in input_files(root)
            if p.relative_to(root).parts[:2] == ("runtime", "Process_AL_MC")
            and p.suffix == ".py"
            or p.parent == root
            and p.name
            in {
                "plan.json",
                "run_search.py",
                "workload.json",
                "README.md",
                "submit_search.sh",
                "transfer_bundle.py",
                "export_ordering_result.py",
            }
        ],
        str(root) + ".inputs.zip",
        kind="inputs",
        binding={"plan_id": plan["plan_id"]},
    )
    state.setdefault("ordering_searches", {})[plan["plan_id"]] = {
        "plan_id": plan["plan_id"],
        "directory": str(root),
        "status": "prepared",
        "model_version": plan["model"]["version"],
        "config_version": context.get("config_version"),
        "phase": plan["phase"],
        "workload": plan["workload"],
        "input_archive": archive,
    }
    return {
        **prepared,
        "status": "prepared",
        "state": state,
        "input_archive": archive,
        "actual_cost": 0.0,
        "submitted": False,
    }


def recover_ordering_results(*, action, context):
    from Process_AL_MC.ordering.api import inspect_ordering_search

    state = deepcopy(context.get("event_state") or {})
    params = action.get("parameters") or {}
    plan_id = params.get("plan_id")
    record = (state.get("ordering_searches") or {}).get(plan_id)
    if not record:
        return {"status": "rejected", "state": state, "reason": "Unknown registered ordering plan"}
    directory = params.get("results_directory") or record["directory"]
    archive = params.get("results_archive")
    if archive:
        import json
        import tempfile
        import zipfile
        from phase_agent.tools.remote.transfer_bundle import (
            import_bundle,
            validate_bundle,
            MANIFEST,
        )
        from phase_agent.tools.remote.export_ordering_result import allowed_result

        validate_bundle(archive, extra_result_suffixes=(".csv",))
        with zipfile.ZipFile(archive) as bundle:
            inventory = json.loads(bundle.read(MANIFEST))
        if inventory.get("batch_binding", {}).get("plan_id") != plan_id or any(
            not allowed_result(row["path"]) for row in inventory["files"]
        ):
            raise ValueError("ordering return archive scope or plan mismatch")
        with tempfile.TemporaryDirectory(prefix=".ordering-return-") as temporary:
            import_bundle(archive, temporary, extra_result_suffixes=(".csv",))
            inspect_ordering_search(temporary, expected_plan_id=plan_id)
        directory = str(Path(record["directory"]) / "returned")
        import_bundle(archive, directory, extra_result_suffixes=(".csv",))
    elif Path(directory).resolve() != Path(record["directory"]).resolve():
        raise ValueError("external ordering returns require a validated results_archive")
    result = inspect_ordering_search(directory, expected_plan_id=plan_id)
    state["ordering_searches"][plan_id].update(
        status=result["status"], results_directory=str(Path(directory).resolve()), result=result
    )
    return {
        "status": "completed",
        "state": state,
        "ordering_result": result,
        "submitted": False,
        "actual_cost": 0.0,
    }
