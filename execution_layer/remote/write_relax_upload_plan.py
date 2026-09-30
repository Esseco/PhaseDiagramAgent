"""Write the manual upload checklist grouped by branch."""

import json
from pathlib import Path


def write_relax_upload_plan(state, upload_root):
    root = Path(upload_root)
    grouped = {}
    for task in state.get("tasks") or []:
        if task.get("stage") != "relax_and_feature" or not task.get("input_path"):
            continue
        if task.get("status") not in {"pending", "running"}:
            continue
        directory = Path(task["input_path"]).parent
        branch = grouped.setdefault(task["branch_id"], {"branch_id": task["branch_id"],
                         "task_ids": [], "batch_directories": [], "results_directories": [],
                         "task_directories": []})
        branch["task_ids"].append(task["task_id"])
        for key, value in (("batch_directories", str(directory.parent)),
                           ("results_directories", str(directory.parent.parent / "results")),
                           ("task_directories", str(directory))):
            if value not in branch[key]:
                branch[key].append(value)
    plan = {"instructions": (
        "Upload every listed batch directory as a sibling under the same stage folder. "
        "Each task directory contains initial.vasp, "
        "task.json. Each batch root contains one run_mlip_batch.py. Submit GPU.sh once from the batch root; "
        "it runs the compatible Relax tasks and collects all results in the stage-level results/. "
        "The remote mace environment must import Process_PhaseDiagram and "
        "Process_AL_MC; the MACE model path in task.json must exist remotely."
    ), "branch_count": len(grouped), "task_count": sum(len(row["task_ids"]) for row in grouped.values()),
        "branches": [grouped[key] for key in sorted(grouped)]}
    path = root / "RELAX_UPLOAD_PLAN.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
