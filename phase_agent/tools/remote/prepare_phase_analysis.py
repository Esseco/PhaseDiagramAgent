"""Freeze one CPU phase-analysis job beside a model-refresh upload wave."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import shlex


def copy_phase_runtime(runtime):
    """Copy the minimal, independently importable phase runtime."""
    runtime = Path(runtime)
    source = Path(__file__).resolve().parents[2]
    files = (
        "__init__.py",
        "analysis/__init__.py",
        "analysis/phase/__init__.py",
        "analysis/phase/identify_result_phase.py",
        "science/__init__.py",
        "science/structures/__init__.py",
        "science/structures/load_result_structure.py",
        "science/structures/boundary_utils.py",
        "science/structures/identify_branch.py",
        "science/structures/identify_layered_phase_fast.py",
        "tools/__init__.py",
        "tools/remote/integrity.py",
        "tools/remote/phase_analysis_wave.py",
    )
    for name in files:
        target = runtime / "phase_agent" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    (runtime / "phase_agent/tools/remote/__init__.py").write_text(
        '"""Portable phase-analysis runtime."""\n', encoding="utf8"
    )


def prepare_phase_analysis(state, batches, manager, references, profiles):
    if not batches:
        return None
    from phase_agent.tools.remote.submission_handoff import handoff_root
    from phase_agent.tools.relax_mc_batch.files import coordinator_profile, script
    from phase_agent.tools.remote.transfer_bundle import local_path
    from phase_agent.science.structures.boundary_utils import json_safe, load_structure

    root = handoff_root(Path(batches[0]["upload_directory"]).parent.parent).resolve()
    if any(
        handoff_root(Path(b["upload_directory"]).parent.parent).resolve() != root for b in batches
    ):
        raise ValueError("phase_analysis_refresh_must_share_handoff")
    directory = Path(batches[0]["upload_directory"]).parent / (
        "zz-phase-analysis-" + batches[0]["batch_id"]
    )
    directory_io = local_path(directory)
    directory_io.mkdir(parents=True, exist_ok=True)
    copy_phase_runtime(directory_io / "runtime")
    refs = references or ((manager.data.get("system_config") or {}).get("phase_references") or {})
    ref_data = {name: load_structure(value).as_dict() for name, value in refs.items()}
    prior = {}
    for task in state.get("tasks") or []:
        evidence = (task.get("outputs") or {}).get("phase_identification") or {}
        digest = evidence.get("structure_sha256")
        if (
            task.get("status") == "completed"
            and digest
            and evidence.get("portable_classification_scope")
        ):
            prior.setdefault(digest, []).append(evidence)
    rows = []
    for batch in batches:
        folder = Path(batch["upload_directory"]).resolve()
        for entry in json.loads((folder / "manifest.json").read_text(encoding="utf8")):
            task = json.loads((folder / entry["input_path"]).read_text(encoding="utf8"))
            result = (folder / entry["result_path"]).resolve()
            row = {
                k: entry.get(k)
                for k in (
                    "task_id",
                    "task_key",
                    "batch_id",
                    "config_version",
                    "model_version",
                    "protocol_version",
                    "input_file_version",
                    "task_checksum",
                )
            }
            row.update(
                result_path=result.relative_to(root).as_posix(),
                branch_id=task.get("branch_id"),
                input_structure_sha256=task.get("input_structure_sha256"),
            )
            candidates = prior.get(task.get("input_structure_sha256")) or []
            if candidates and all(item == candidates[0] for item in candidates):
                row["source_phase"] = deepcopy(candidates[0])
            rows.append(row)
    data = {
        "boundary": json_safe(manager.boundary),
        "branches": {},
        "structures": {},
        "system_config": json_safe(
            {
                "species": (manager.data.get("system_config") or {}).get("species")
                or {"mobile": ["Na"]}
            }
        ),
    }
    spec = {"manager": data, "references": ref_data, "tasks": rows}
    (directory_io / "phase_wave.json").write_text(
        json.dumps(spec, ensure_ascii=False, allow_nan=False), encoding="utf8"
    )
    upward = "/".join([".."] * len(directory.relative_to(root).parts)) or "."
    from phase_agent.tools.remote.phase_analysis_wave import GROUP_SIZE

    prefix = 'cd -- "${SLURM_SUBMIT_DIR:?Missing Slurm submission directory}"\nexport PYTHONPATH="$PWD/runtime:${PYTHONPATH:-}"\n'
    from phase_agent.tools.remote.integrity import file_checksum

    spec_checksum = file_checksum(directory_io / "phase_wave.json")
    base = (
        f"python3 -m phase_agent.tools.remote.phase_analysis_wave phase_wave.json "
        f"{shlex.quote(upward)} --spec-checksum {spec_checksum}"
    )
    groups = (len(rows) + GROUP_SIZE - 1) // GROUP_SIZE
    cpu_command = (
        prefix + base + (' --group-index "$SLURM_ARRAY_TASK_ID"' if groups > 1 else "") + "\n"
    )
    path = directory / "CPU.sh"
    content = script(coordinator_profile(profiles), cpu_command, name="refresh-phase-analysis")
    if groups > 1:
        content = content.replace(
            "#!/usr/bin/env bash\n", f"#!/usr/bin/env bash\n#SBATCH --array=0-{groups - 1}\n", 1
        )
    local_path(path).write_text(content, encoding="utf8", newline="\n")
    metadata = {
        "script_path": str(path),
        "batch_ids": [b["batch_id"] for b in batches],
        "publish_on_success": True,
    }
    if groups > 1:
        finalizer = directory / "CPU_finalize.sh"
        local_path(finalizer).write_text(
            script(
                coordinator_profile(profiles),
                prefix + base + " --finalize\n",
                name="refresh-phase-finalize",
            ),
            encoding="utf8",
            newline="\n",
        )
        metadata.update(finalizer_path=str(finalizer), group_count=groups)
    for batch in state.get("slurm_batches") or []:
        if batch.get("batch_id") in metadata["batch_ids"]:
            batch["phase_analysis"] = deepcopy(metadata)
    return metadata
