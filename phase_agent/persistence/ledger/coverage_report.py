"""台账汇总、覆盖统计和 CSV 输出。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from phase_agent.science.structures.boundary_utils import compact_json


def branch_table(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "branch_id": item["branch_id"],
            "P": item["P"],
            "H": item["H"],
            "det_H": item["det_H"],
            "x": item["x"],
            "T": item["T"],
            "composition": item["composition"],
            "structure_count": len(item["structure_ids"]),
        }
        for item in sorted(data["branches"].values(), key=lambda row: row["branch_id"])
    ]


def structure_table(
    data: dict[str, Any], stages: tuple[str, ...], labels: dict[str, str]
) -> list[dict[str, Any]]:
    rows = []
    for item in sorted(data["structures"].values(), key=lambda row: row["structure_id"]):
        branch = data["branches"][item["branch_id"]]
        completed = [stage for stage in stages if item["stage_history"].get(stage)]
        current = completed[-1] if completed else None
        rows.append(
            {
                "structure_id": item["structure_id"],
                "branch_id": item["branch_id"],
                "P": branch["P"],
                "H": branch["H"],
                "det_H": branch["det_H"],
                "x": branch["x"],
                "composition": item["composition"],
                "current_stage": current,
                "current_stage_label": labels.get(current),
                "source_path": item["source_path"],
            }
        )
    return rows


def coverage(
    data: dict[str, Any], stages: tuple[str, ...], labels: dict[str, str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], dict[str, set[str]]] = {}
    for structure in data["structures"].values():
        branch = data["branches"][structure["branch_id"]]
        for stage in stages:
            if not structure["stage_history"].get(stage):
                continue
            key = (
                branch["P"],
                compact_json(structure["composition"]),
                compact_json(branch["H"]),
                stage,
            )
            bucket = groups.setdefault(key, {"branches": set(), "structures": set()})
            bucket["branches"].add(branch["branch_id"])
            bucket["structures"].add(structure["structure_id"])
    return [
        {
            "P": phase,
            "composition": json.loads(composition),
            "H": json.loads(matrix),
            "stage": stage,
            "stage_label": labels[stage],
            "branch_count": len(bucket["branches"]),
            "structure_count": len(bucket["structures"]),
        }
        for (phase, composition, matrix, stage), bucket in sorted(groups.items())
    ]


def save_coverage_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "P",
        "composition",
        "H",
        "stage",
        "stage_label",
        "branch_count",
        "structure_count",
    )
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = dict(row)
            record["composition"] = compact_json(record["composition"])
            record["H"] = compact_json(record["H"])
            writer.writerow(record)
    return output
