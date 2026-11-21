"""Execute DIRECT on the complete, task-bound recovered batch."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import zipfile


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_direct_receipt(receipt):
    """Require the recorded plan and manifest to remain byte-for-byte intact."""
    try:
        plan = Path(receipt["plan_path"])
        manifest_path = plan.parent / "direct_result.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        return (
            plan.name == "selection.md"
            and _hash(plan) == receipt["plan_sha256"]
            and _hash(manifest_path) == receipt["manifest_sha256"]
            and (
                not receipt.get("inventory_sha256")
                or _hash(plan.parent / "candidate_inventory.json") == receipt["inventory_sha256"]
            )
            and manifest == {k: v for k, v in receipt.items() if k != "manifest_sha256"}
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def successful_dft_references(state):
    """Return verified finished DFT geometries as DIRECT coverage references."""
    references = [r for r in state.get("dft_dataset_records") or []
                  if r.get("status") == "completed" and r.get("checks_passed") is True
                  and r.get("converged") is True and r.get("final_frame_valid") is True
                  and r.get("training_ready") is True]
    if any(not r.get("task_id") or not isinstance(r.get("structure"), dict) for r in references):
        raise ValueError("direct_reference_frame_invalid")
    return [{"task_id": r["task_id"], "structure": deepcopy(r["structure"])} for r in references]


def _whole_batch_candidates(state, tasks, scope, model):
    """Use each task's verified final frame, including non-failure frames."""
    from phase_agent.analysis.phase.phase_dft_candidates import extrapolation_dft_candidates

    failures = {}
    for row in extrapolation_dft_candidates(state, []):
        tid = row.get("source_task_id")
        if tid in scope:
            if tid in failures:
                raise ValueError("direct_multiple_final_failure_frames:" + tid)
            failures[tid] = row
    archived = {}
    supplement = state.get("endpoint_supplement") or {}
    if supplement:
        from phase_agent.evidence.endpoint_supplement import supplement_facts
        facts = supplement_facts(state)
        if not facts or facts.get("invalid"):
            raise ValueError("direct_endpoint_supplement_invalid")
        directory = Path(supplement["directory"])
        with (directory / "evidence.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        with zipfile.ZipFile(directory / "frames.zip") as archive:
            for row in rows:
                tid = row["task_id"]
                if tid not in scope:
                    continue
                info = archive.getinfo(row["archive_path"])
                if info.file_size > 10_000_000:
                    raise ValueError("direct_archive_frame_too_large:" + tid)
                data = archive.read(info)
                if hashlib.sha256(data).hexdigest() != row["final_sha256"]:
                    raise ValueError("direct_archive_frame_invalid:" + tid)
                archived[tid] = (row, data)
    candidates, materialize = [], {}
    for tid in sorted(scope):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", tid):
            raise ValueError("direct_task_id_invalid")
        task = tasks[tid]
        output = task.get("outputs") or {}
        failure = failures.get(tid)
        if not failure and (
            output.get("extrapolation_failures")
            or (task.get("extrapolation_review") or {}).get("status") == "extrapolation_failure"
        ):
            raise ValueError("direct_failure_candidate_missing:" + tid)
        if failure:
            row = deepcopy(failure)
        elif tid in archived:
            meta, data = archived[tid]
            if meta["classification"] == "extrapolation_failure":
                raise ValueError("direct_failure_candidate_missing:" + tid)
            row = {
                "candidate_id": "DIRECT-" + tid,
                "structure_id": task.get("structure_id"),
                "source_task_id": tid,
                "structure_sha256": meta["final_sha256"],
                "model_version": model,
                "extrapolation_failure": False,
                "archive_source": meta["archive_path"],
            }
            materialize[tid] = data
        else:
            path, checksum = output.get("structure_path"), output.get("structure_checksum")
            if task.get("status") != "completed" or not path or not checksum:
                raise ValueError("direct_batch_final_frame_missing:" + tid)
            row = {
                "candidate_id": "DIRECT-" + tid,
                "structure_id": task.get("structure_id"),
                "source_task_id": tid,
                "structure_path": path,
                "structure_sha256": checksum,
                "model_version": model,
                "composition": output.get("composition"),
                "extrapolation_failure": False,
            }
        if row.get("model_version") != model or not row.get("structure_sha256"):
            raise ValueError("direct_candidate_identity_invalid:" + tid)
        # Bind the feature difference to the exact candidate final frame.
        score = None
        if tid in archived and archived[tid][0].get("final_sha256") == row["structure_sha256"]:
            try:
                score = float(archived[tid][0]["score"])
            except (ValueError, KeyError, TypeError):
                pass
        elif output.get("structure_checksum") == row["structure_sha256"]:
            score = (output.get("extrapolation_monitor") or {}).get("final_feature_difference")
        elif row.get("extrapolation_evidence"):
            score = row["extrapolation_evidence"].get("final_feature_difference")
        row["final_feature_difference"] = score
        row.setdefault("branch_id", task.get("branch_id"))
        row.setdefault("candidate_origin", "direct_recovery")
        row.setdefault("qbc", {"status": "not_available_exact_current_geometry"})
        if tid not in materialize and _hash(row["structure_path"]) != row["structure_sha256"]:
            raise ValueError("direct_candidate_identity_invalid:" + tid)
        candidates.append(row)
    return candidates, materialize


def _mc_batch_candidates(tasks, scope, model):
    """Require an exact final frame for every monitored MC relaxation."""
    rows = []
    for tid in sorted(scope):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", tid):
            raise ValueError("direct_task_id_invalid")
        task = tasks[tid]
        output = task.get("outputs") or {}
        if task.get("stage") == "relax_and_feature":
            initial, _ = _whole_batch_candidates({"tasks": [task], "active_model_version": model},
                                                 tasks, {tid}, model)
            rows.extend(initial)
            continue
        if task.get("stage") != "deep_search":
            raise ValueError("direct_mc_task_stage_invalid:" + tid)
        observations = output.get("relaxation_observations") or []
        steps = output.get("actual_mc_steps")
        if (output.get("relaxation_observations_complete") is not True or type(steps) is not int
            or steps < 0 or len(observations) != steps + 1):
            raise ValueError("direct_mc_complete_frame_inventory_required:" + tid)
        for index, observation in enumerate(observations):
            path, checksum = observation.get("structure_path"), observation.get("structure_checksum")
            if (observation.get("monitor_complete") is not True
                or type(observation.get("extrapolation_failure")) is not bool
                or not path or not checksum):
                raise ValueError("direct_mc_complete_frame_inventory_required:" + tid)
            try:
                actual = _hash(path)
            except OSError as error:
                raise ValueError("direct_mc_frame_unavailable:" + tid) from error
            if actual != checksum:
                raise ValueError("direct_mc_frame_checksum_invalid:" + tid)
            rows.append({"candidate_id": f"DIRECT-{tid}-{index}", "source_task_id": tid,
                         "structure_id": task.get("structure_id"), "branch_id": task.get("branch_id"),
                         "structure_path": path, "structure_sha256": checksum, "model_version": model,
                         "extrapolation_failure": observation["extrapolation_failure"],
                         "final_feature_difference": observation.get("final_feature_difference"),
                         "candidate_origin": "direct_recovery", "qbc": {"status": "not_available_exact_current_geometry"}})
    return rows, {}


def prepare_direct_sampling(*, action, context):
    """Build auditable representatives; no DFT tasks are created or submitted."""
    started = time.perf_counter()
    source_state = context.get("event_state") or {}
    if context.get("_event_state_owned"):
        # The graph owns this state. Copy only the maps this handler changes;
        # the graph still copies the merged result at its normal isolation boundary.
        state = dict(source_state)
        for key in ("direct_sampling_results", "analysis_step_diagnostics"):
            if key in state:
                state[key] = deepcopy(state[key])
    else:
        state = deepcopy(source_state)
    parameters = action.get("parameters") or {}
    scope_id = parameters.get("scope_id")
    if isinstance(scope_id, str) and scope_id and scope_id != "epoch_refresh":
        if scope_id not in (state.get("relax_mc_batches") or {}):
            raise ValueError("direct_batch_not_verified")
        raise ValueError("direct_recovery_requires_model_refresh_scope")
    target = parameters.get("target", 100)
    if (
        type(target) is not int
        or not 1 <= target <= 100
        or not isinstance(scope_id, str)
        or not scope_id
    ):
        raise ValueError("direct_scope_or_target_invalid")
    model = state.get("active_model_version")
    if not model:
        raise ValueError("direct_active_model_missing")
    tasks = {t.get("task_id"): t for t in state.get("tasks") or []}
    if scope_id == "epoch_refresh":
        refresh = state.get("model_refresh") or {}
        if refresh.get("new_model_version") != model:
            raise ValueError("direct_refresh_model_mismatch")
        scope = set(refresh.get("task_ids") or [])
    else:
        batch = (state.get("relax_mc_batches") or {}).get(scope_id) or {}
        if batch.get("model_version") != model or batch.get("formal_recovery_status") != "accepted":
            raise ValueError("direct_batch_not_verified")
        scope = {
            t.get("task_id") for t in tasks.values() if t.get("relax_mc_bundle_id") == scope_id
        }
    if not scope:
        raise ValueError("direct_empty_scope")
    if not scope.issubset(tasks) or any(tasks[tid].get("model_version") != model for tid in scope):
        raise ValueError("direct_scope_task_or_model_invalid")
    from phase_agent.agents.reliability.extrapolation_round import assess_current_rounds

    round_facts = next(
        (row for row in assess_current_rounds(state)["rounds"] if row.get("scope_id") == scope_id),
        None,
    )
    if not round_facts or not round_facts.get("evidence_digest"):
        raise ValueError("direct_round_evidence_missing")
    if (
        parameters.get("round_evidence_digest")
        and parameters["round_evidence_digest"] != round_facts["evidence_digest"]
    ):
        raise ValueError("direct_round_evidence_changed")
    from phase_agent.evidence.extrapolation_review import extrapolation_review

    if (
        scope_id == "epoch_refresh"
        and
        state.get("endpoint_supplement")
        and extrapolation_review(state).get("failure_count") is None
    ):
        raise ValueError("direct_extrapolation_evidence_invalid")
    route = parameters.get("recovery_route", "extrapolation")
    if route == "extrapolation":
        candidates, materialize = (_whole_batch_candidates(state, tasks, scope, model)
                                   if scope_id == "epoch_refresh" else _mc_batch_candidates(tasks, scope, model))
        if {r["source_task_id"] for r in candidates} != scope:
            raise ValueError("direct_batch_candidates_missing_or_duplicated")
        perturbation_excluded = None
    elif route == "perturbation":
        from phase_agent.tools.workflows.perturbation_direct import perturbation_candidates
        candidates, perturbation_excluded = perturbation_candidates(
            state, state.get("perturbation_recovery_batch") or {}, scope_id=scope_id,
            round_digest=round_facts["evidence_digest"])
        materialize = {}
    else:
        raise ValueError("direct_recovery_route_invalid")
    if len({r["candidate_id"] for r in candidates}) != len(candidates):
        raise ValueError("direct_batch_candidates_missing_or_duplicated")
    # Only frames with a verified, finite final feature difference <= 1 are
    # eligible; missing observations do not establish that a frame is safe.
    excluded = {"above_threshold": 0, "unknown": 0}
    eligible = []
    for row in candidates:
        score = row.get("final_feature_difference")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
            excluded["unknown"] += 1
        elif score > 1:
            excluded["above_threshold"] += 1
        else:
            eligible.append(row)
    candidates = eligible
    if not candidates:
        raise ValueError("direct_no_candidates_with_verified_feature_difference_le_1")
    # Only exact duplicate source bytes are removed; DIRECT performs feature-space representation.
    unique = {}
    for row in candidates:
        unique.setdefault(row["structure_sha256"], row)
    candidates = list(unique.values())
    approved_target = None
    if route == "perturbation":
        from phase_agent.tools.workflows.perturbation_direct import approved_perturbation_direct_target
        approved_target = approved_perturbation_direct_target(state)
        if approved_target is not None:
            if target != approved_target:
                raise ValueError("direct_target_differs_from_approved_perturbation_target")
            if len(candidates) < approved_target:
                raise ValueError("direct_insufficient_verified_candidates_for_approved_target")
    materialize = {
        r["source_task_id"]: materialize[r["source_task_id"]]
        for r in candidates
        if r["source_task_id"] in materialize
    }
    references = successful_dft_references(state)
    # The bundled M3GNet checkpoint uses legacy Keras; select it before the
    # first TensorFlow import, or fail explicitly if the process loaded Keras 3.
    if os.environ.get("TF_USE_LEGACY_KERAS") != "1" and "tensorflow" in sys.modules:
        raise ValueError("direct_m3gnet_requires_fresh_legacy_keras_process")
    if os.environ.get("TF_USE_LEGACY_KERAS") not in (None, "1"):
        raise ValueError("direct_m3gnet_legacy_keras_disabled")
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    import maml

    model_path = (
        Path(maml.__file__).parent / "describers/data/m3gnet_models/matbench_mp_e_form/0/m3gnet"
    )
    files = sorted(p for p in model_path.rglob("*") if p.is_file())
    if not files:
        raise ValueError("direct_feature_model_missing")
    model_digest = hashlib.sha256()
    for file in files:
        model_digest.update(file.relative_to(model_path).as_posix().encode() + b"\0")
        model_digest.update(file.read_bytes())
    snapshot = {
        "feature_model_sha256": model_digest.hexdigest(),
        "recovery_route": route,
        "model_version": model,
        "scope_id": scope_id,
        "round_evidence_digest": round_facts["evidence_digest"],
        "target": target,
        "feature_difference_filter": {"maximum": 1.0, "excluded": excluded},
        "candidates": [(r["candidate_id"], r["structure_sha256"], r["final_feature_difference"]) for r in candidates],
        "references": [{"task_id": r["task_id"], "structure": r["structure"]} for r in references],
    }
    if parameters.get("extrapolation_policy_version") == "epoch_refresh_v1":
        snapshot["policy_version"] = "epoch_refresh_v1"
    digest = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    root = Path(
        (context.get("effective_config") or {}).get("phase_diagram_directory")
        or state.get("analysis_output_directory")
        or ""
    ).resolve()
    if not root.is_dir():
        raise ValueError("direct_analysis_directory_missing")
    from phase_agent.analysis.feedback.output_paths import analysis_epoch_directory
    from phase_agent.evidence.model_epoch import model_epoch

    epoch = model_epoch(state, model)
    legacy = root / "model_epoch_unrecorded" / "direct_sampling" / digest[:16]
    previous_receipt = (state.get("direct_sampling_results") or {}).get(scope_id) or {}
    current_cycle = state.get("extrapolation_recovery_cycle") or {}
    same_receipt = (previous_receipt.get("source_digest") == digest
                    and valid_direct_receipt(previous_receipt)
                    and (not current_cycle or current_cycle.get("direct_source_digest") == digest))
    if parameters.get("extrapolation_policy_version") == "epoch_refresh_v1" and not same_receipt:
        from phase_agent.evidence.extrapolation_recovery import epoch_recovery_policy
        policy = epoch_recovery_policy(state, round_facts, legacy=True)
        if policy != {"status": "sampling_required", "route": route}:
            raise ValueError("direct_epoch_policy_rejected:" + str(policy.get("status")))
    epoch_directory = analysis_epoch_directory(root, state, model) / "direct_sampling"
    old_directory = epoch_directory / digest[:16]
    if parameters.get("extrapolation_policy_version") == "epoch_refresh_v1":
        from phase_agent.evidence.extrapolation_recovery import recovery_label_facts
        cycle = state.get("extrapolation_recovery_cycle") or {}
        progress = recovery_label_facts(state) if cycle.get("policy_version") == "epoch_refresh_v1" else {}
        from phase_agent.evidence.extrapolation_policy import recovery_lineage
        attempt = progress.get("attempt_count", 0) + 1
        metadata = recovery_lineage(cycle, model, round_facts["evidence_digest"], epoch)
        directory = epoch_directory / f"round_{metadata['extrapolation_round']:02d}_attempt_{attempt:02d}_{route}_{digest[:16]}"
        if same_receipt:
            directory = Path(previous_receipt["plan_path"]).parent
        elif old_directory.exists():
            directory = old_directory
    else:
        directory = legacy if not epoch and legacy.exists() else old_directory
    for row in candidates:
        if row["source_task_id"] in materialize:
            row["structure_path"] = str(
                directory / "archive_frames" / (row["source_task_id"] + ".vasp")
            )
    if directory.exists():
        manifest = json.loads((directory / "direct_result.json").read_text(encoding="utf8"))
        if manifest.get("source_digest") != digest or _hash(
            directory / "selection.md"
        ) != manifest.get("plan_sha256"):
            raise ValueError("direct_existing_result_conflict")
        manifest["manifest_sha256"] = _hash(directory / "direct_result.json")
        if not valid_direct_receipt(manifest):
            raise ValueError("direct_existing_result_conflict")
        previous = (state.get("direct_sampling_results") or {}).get(scope_id)
        if previous and (not valid_direct_receipt(previous) or previous != manifest):
            if parameters.get("extrapolation_policy_version") != "epoch_refresh_v1":
                raise ValueError("direct_existing_receipt_conflict")
            from phase_agent.evidence.extrapolation_recovery import epoch_recovery_policy
            if epoch_recovery_policy(state, round_facts, legacy=True) != {"status": "sampling_required", "route": route}:
                raise ValueError("direct_existing_receipt_conflict")
        state.setdefault("direct_sampling_results", {})[scope_id] = manifest
        _register_selected_candidates(state, manifest, candidates)
        _publish_direct_step(state, root, manifest)
        return {
            "status": "completed",
            "state": state,
            "direct_sampling": manifest,
            "submitted": False,
        }
    from ase.io import read, write
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from Process_MLIP.direct_sampling.features import M3GNetEmbedding
    from Process_MLIP.direct_sampling.workflow import extract, run
    from phase_agent.runtime.turn_process import process_event
    import h5py

    process_event("DIRECT 开始", {"结构数": len(candidates), "DFT参考数": len(references), "目标数": target})
    base = directory.parent
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".direct-", dir=base) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        if materialize:
            (stage / "archive_frames").mkdir()
            for tid, data in materialize.items():
                (stage / "archive_frames" / (tid + ".vasp")).write_bytes(data)
        paths = []
        if references:
            refpath = stage / "references.extxyz"
            write(
                refpath,
                [
                    AseAtomsAdaptor.get_atoms(Structure.from_dict(r["structure"]))
                    for r in references
                ],
                format="extxyz",
            )
            paths.append(refpath)
        paths += [
            (stage / "archive_frames" / (r["source_task_id"] + ".vasp")).resolve()
            if r["source_task_id"] in materialize
            else Path(r["structure_path"]).resolve()
            for r in candidates
        ]
        model_started = time.perf_counter()
        encoder = M3GNetEmbedding(model_path)
        model_seconds = time.perf_counter() - model_started
        if encoder.metadata.get("model_sha256") != model_digest.hexdigest():
            raise ValueError("direct_feature_model_changed")
        cache = stage / "features.h5"
        extraction_started = time.perf_counter()
        total_frames = len(references) + len(candidates)
        progress_step = max(50, (total_frames + 19) // 20)

        def progress(count):
            if count % progress_step == 0 or count == total_frames:
                process_event("DIRECT 特征计算", {"已完成": count, "总帧数": total_frames,
                              "耗时秒": round(time.perf_counter() - extraction_started, 1)})

        extract(paths, encoder, cache, progress=progress)
        extraction_seconds = time.perf_counter() - extraction_started
        sampling_started = time.perf_counter()
        sampling = run(
            cache, stage / "selected", min(target, len(candidates)), reference_count=len(references)
        )
        sampling_seconds = time.perf_counter() - sampling_started
        selected = json.loads((stage / "selected/selected_indices.json").read_text(encoding="utf8"))
        bypath = {str(path): row for path, row in zip(paths[len(references) > 0 :], candidates)}
        chosen = []
        for item in selected:
            row = bypath.get(str(Path(item["source"]).resolve()))
            if (
                not row
                or item.get("frame") != 0
                or _hash(item["source"]) != row["structure_sha256"]
            ):
                raise ValueError("direct_selected_frame_invalid")
            chosen.append(
                {
                    key: row.get(key)
                    for key in (
                        "candidate_id",
                        "structure_id",
                        "source_task_id",
                        "structure_path",
                        "structure_sha256",
                        "composition",
                        "extrapolation_failure",
                        "extrapolation_evidence",
                        "final_feature_difference",
                    )
                }
            )
            atoms = read(item["source"], index=0)
            symbols = atoms.get_chemical_symbols()
            oxygen = symbols.count("O")
            chosen[-1]["na_fraction"] = 2 * symbols.count("Na") / oxygen if oxygen else None
            source_task = tasks[row["source_task_id"]]
            manager = context.get("manager")
            branch = ((getattr(manager, "data", {}) or {}).get("branches") or {}).get(
                source_task.get("branch_id"), {}
            )
            chosen[-1]["phase"] = source_task.get("actual_phase") or source_task.get("source_phase") or branch.get("P")
            chosen[-1]["phase_basis"] = ("task" if source_task.get("actual_phase") or source_task.get("source_phase")
                                          else "parent_branch" if branch.get("P") else "unknown")
            chosen[-1]["H"] = branch.get("H")
        if len(chosen) != sampling["n_samples"] or len({r["candidate_id"] for r in chosen}) != len(
            chosen
        ):
            raise ValueError("direct_selection_result_invalid")
        if approved_target is not None and len(chosen) != approved_target:
            raise ValueError("direct_selected_count_differs_from_approved_target")
        lines = [
            "# " + ("微扰结构微调" if route == "perturbation" else "外推结构微调") + " DIRECT 方案",
            "",
            "本方案仅为审核准备，不创建或提交 DFT 任务。",
            "成功 DFT 作为覆盖参考；只采样同帧特征差不超过 1 的结构，不追加 GS。",
            f"特征差大于 1 排除 {excluded['above_threshold']} 个；特征差未知排除 {excluded['unknown']} 个。",
            f"目标 {target}，实际 {len(chosen)}，本批任务 {len(scope)}，去重后结构 {len(candidates)}，历史 DFT 参考 {len(references)}。",
            "外推失败帧无可信相图能量，不能强行画在主凸包上。",
            "",
            "| 序号 | 候选 | 来源任务 | 结构 | 相 | Na x | 外推失败 | 末态 SHA256 |",
            "|---:|---|---|---|---|---:|---|---|",
        ]
        for i, row in enumerate(chosen, 1):
            lines.append(
                f"| {i} | {row['candidate_id']} | {row['source_task_id']} | {row['structure_id']} | {row.get('phase') or '未知'} | {row['na_fraction'] if row['na_fraction'] is not None else '未知'} | {'是' if row.get('extrapolation_failure') else '否'} | {row['structure_sha256']} |"
            )
        plan = stage / "selection.md"
        plan.write_text("\n".join(lines) + "\n", encoding="utf8")
        inventory = stage / "candidate_inventory.json"
        inventory.write_text(
            json.dumps(
                {"scope_task_count": len(scope), "source_digest": digest, "candidates": candidates},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf8",
        )
        with h5py.File(cache, "r+") as handle:
            sources = json.loads(handle.attrs["sources"])
            if references:
                sources[0] = str(directory / "references.extxyz")
            for index, row in enumerate(candidates, start=int(bool(references))):
                sources[index] = str(Path(row["structure_path"]).resolve())
            handle.attrs["sources"] = json.dumps(sources)
        for item in selected:
            row = bypath[str(Path(item["source"]).resolve())]
            item["source"] = str(Path(row["structure_path"]).resolve())
        (stage / "selected/selected_indices.json").write_text(json.dumps(selected), encoding="utf8")
        manifest = {
            "source_digest": digest,
            "recovery_route": route,
            **({"policy_version": "epoch_refresh_v1"} if parameters.get("extrapolation_policy_version") == "epoch_refresh_v1" else {}),
            "recovery_approval_record_id": ((state.get("perturbation_recovery_batch") or {}).get("approval_record_id")
                                              if route == "perturbation" else context.get("approval_record_id")),
            "perturbation_excluded": perturbation_excluded,
            "model_version": model,
            "scope_id": scope_id,
            "round_evidence_digest": round_facts["evidence_digest"],
            "target": target,
            "selected_count": len(chosen),
            "reference_count": len(references),
            "feature_difference_filter": {"maximum": 1.0, "excluded": excluded},
            "batch_task_count": len(scope),
            "unique_candidate_count": len(candidates),
            "selected": chosen,
            "feature_model": encoder.metadata,
            "sampling": sampling,
            "timing_seconds": {"model_initialization": model_seconds,
                               "feature_extraction": extraction_seconds,
                               "sampling": sampling_seconds,
                               "total_before_publish": time.perf_counter() - started},
            "plan_sha256": _hash(plan),
            "inventory_sha256": _hash(inventory),
            "plan_path": str(directory / "selection.md"),
            "force_jump_evidence_required_for_fraction": True,
        }
        (stage / "direct_result.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8"
        )
        stage.rename(directory)
    manifest["manifest_sha256"] = _hash(directory / "direct_result.json")
    state.setdefault("direct_sampling_results", {})[scope_id] = manifest
    _register_selected_candidates(state, manifest, candidates)
    _publish_direct_step(state, root, manifest)
    return {"status": "completed", "state": state, "direct_sampling": manifest, "submitted": False}


def _publish_direct_step(state, root, receipt):
    try:
        from phase_agent.analysis.feedback.analysis_events import direct_event
        direct_event(state, root, receipt)
        (state.get("analysis_step_diagnostics") or {}).pop("direct", None)
    except (OSError, ValueError, KeyError, TypeError) as error:
        state.setdefault("analysis_step_diagnostics", {})["direct"] = str(error)


def _register_selected_candidates(state, manifest, candidates):
    """Keep selected exact failure frames available to the existing DFT selector."""
    by_id = {row.get("candidate_id"): row for row in candidates}
    existing = {row.get("candidate_id"): row for row in state.get("qbc_candidates") or []}
    for selected in manifest.get("selected") or []:
        row = by_id.get(selected.get("candidate_id"))
        if not row or row.get("structure_sha256") != selected.get("structure_sha256"):
            raise ValueError("direct_selected_candidate_binding_invalid")
        old = existing.get(row["candidate_id"])
        if old and old.get("structure_sha256") != row["structure_sha256"]:
            raise ValueError("direct_candidate_id_conflict")
        existing[row["candidate_id"]] = deepcopy(row)
    state["qbc_candidates"] = list(existing.values())
