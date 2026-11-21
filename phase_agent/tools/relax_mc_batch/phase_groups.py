"""Freeze grouped phase analysis for a Relax–MC GPU phase."""

import json
from pathlib import Path
import shlex

from phase_agent.tools.remote.phase_analysis_wave import GROUP_SIZE
from phase_agent.tools.relax_mc_batch.files import coordinator_profile, python_environment, script


def prepare_groups(root, state, manager, manifest, batches, phase):
    root = Path(root).resolve()
    task_by_id = {t["task_id"]: t for t in state.get("tasks") or []}
    ids = [task_id for batch in batches for task_id in batch["task_ids"]]
    if len(ids) <= GROUP_SIZE:
        return None
    if len(ids) != len(set(ids)) or not set(ids).issubset(task_by_id):
        raise ValueError("phase_analysis_invalid_task_scope")
    from phase_agent.science.structures.boundary_utils import json_safe, load_structure

    attempt = (state.get("relax_mc_remote") or {}).get("resume_attempt") or {}
    plan_id = attempt.get("plan_id")
    suffix = "-" + str(plan_id)[:16] if plan_id else ""
    if suffix and not all(c.isalnum() or c in "-_" for c in suffix):
        raise ValueError("phase_analysis_invalid_attempt_identity")
    directory = root / f"phase_analysis_{phase}{suffix}"
    directory.mkdir(parents=True, exist_ok=True)
    refs = {
        name: load_structure(value).as_dict()
        for name, value in (manifest.get("phase_references") or {}).items()
    }
    rows = []
    for task_id in ids:
        task = task_by_id[task_id]
        result = Path(task["result_path"]).resolve()
        row = {
            key: task.get(key)
            for key in (
                "task_id",
                "task_key",
                "batch_id",
                "config_version",
                "model_version",
                "protocol_version",
                "input_file_version",
                "task_checksum",
                "branch_id",
                "input_structure_sha256",
            )
        }
        row["result_path"] = result.relative_to(root).as_posix()
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
    spec = directory / "phase_wave.json"
    payload = json.dumps(
        {"manager": data, "references": refs, "tasks": rows},
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    if spec.exists() and spec.read_text(encoding="utf8") != payload:
        raise ValueError("phase_analysis_existing_scope_conflict")
    spec.write_text(payload, encoding="utf8")
    from phase_agent.tools.remote.integrity import file_checksum

    spec_checksum = file_checksum(spec)
    count = (len(rows) + GROUP_SIZE - 1) // GROUP_SIZE
    command = (
        python_environment(root)
        + f"cd -- {shlex.quote(str(directory))}\n"
        + f'python3 -m phase_agent.tools.remote.phase_analysis_wave phase_wave.json {shlex.quote(str(root))} --group-index "$SLURM_ARRAY_TASK_ID" --spec-checksum {spec_checksum}\n'
    )
    path = directory / "CPU.sh"
    text = script(
        coordinator_profile(manifest["stage_profiles"]),
        command,
        name=manifest["bundle_id"] + f"-phase-{phase}",
    )
    text = text.replace(
        "#!/usr/bin/env bash\n", f"#!/usr/bin/env bash\n#SBATCH --array=0-{count - 1}\n", 1
    )
    if path.exists() and path.read_text(encoding="utf8") != text:
        raise ValueError("phase_analysis_existing_script_conflict")
    path.write_text(text, encoding="utf8", newline="\n")
    return {
        "script_path": str(path),
        "spec_relative": spec.relative_to(root).as_posix(),
        "spec_checksum": spec_checksum,
        "group_count": count,
    }
