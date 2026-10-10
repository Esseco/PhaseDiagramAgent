"""Expose saved task lineage without guessing current rounds from array order."""

import json
from phase_agent.analysis.state.summarize_task_stages import summarize_task_stages

LINEAGE_FIELDS = (
    "model_version",
    "search_group_index",
    "parent_relax_round",
    "upload_operation_id",
    "segment_index",
)


def summarize_task_rounds(state):
    groups = {}
    for task in state.get("tasks") or []:
        lineage = {key: task.get(key) for key in LINEAGE_FIELDS}
        key = json.dumps(lineage, sort_keys=True)
        group = groups.setdefault(key, {"lineage": lineage, "tasks": []})
        group["tasks"].append(task)
    output = []
    for key in sorted(groups):
        group = groups[key]
        output.append(
            {
                "lineage": group["lineage"],
                "missing_lineage_fields": [
                    name for name, value in group["lineage"].items() if value is None
                ],
                "task_count": len(group["tasks"]),
                "stage_evidence": summarize_task_stages(
                    {
                        "tasks": group["tasks"],
                        "processed_task_ids": state.get("processed_task_ids") or [],
                    }
                )["stages"],
            }
        )
    return {
        "groups": output,
        "instruction": "Saved lineage only. Missing identifiers remain unknown; groups are not chronologically ranked and do not establish current-round completion.",
    }
