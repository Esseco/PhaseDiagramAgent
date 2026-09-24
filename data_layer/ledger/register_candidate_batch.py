"""将候选 branch、具体构型和来源登记到 PhaseDataManager。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def register_candidate_batch(
    manager: Any,
    candidates: list[dict[str, Any]],
    *,
    structure_directory: str | Path,
    ledger_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """复用稳定编号登记候选，并将结构保存为 POSCAR 文件。"""
    directory = Path(structure_directory)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for candidate in candidates:
        missing = {"P", "H", "x", "T", "structure", "arrangement"} - candidate.keys()
        if missing:
            raise ValueError(f"候选缺少字段：{sorted(missing)}")
        branch_id = manager.add_branch(
            P=candidate["P"],
            H=candidate["H"],
            x=candidate["x"],
            T=candidate["T"],
            composition=candidate.get("composition"),
        )
        structure_id = manager.add_structure(
            branch_id=branch_id,
            arrangement=candidate["arrangement"],
            composition=candidate.get("composition"),
            metadata={"calculation_status": "pending"},
        )
        path = directory / branch_id / f"{structure_id}.vasp"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            candidate["structure"].to(filename=path, fmt="poscar")
        manager.add_structure(
            branch_id=branch_id,
            arrangement=candidate["arrangement"],
            source_path=path,
        )
        record = manager.data["structures"][structure_id]
        record.setdefault("metadata", {})
        record["metadata"].setdefault("calculation_status", "pending")
        if candidate.get("estimated_cost") is not None:
            record["metadata"]["estimated_cost"] = candidate["estimated_cost"]
        sources = record.setdefault("generation_sources", [])
        for source in _sources(candidate):
            if source not in sources:
                sources.append(source)
        outputs.append(
            {
                "branch_id": branch_id,
                "structure_id": structure_id,
                "structure_path": str(path),
                "calculation_status": record["metadata"]["calculation_status"],
                "estimated_cost": record["metadata"].get("estimated_cost"),
            }
        )
    if ledger_path is not None:
        manager.save(ledger_path)
    return outputs


def _sources(candidate):
    sources = list(candidate.get("generation_sources", []))
    current = {
        "candidate_id": candidate.get("candidate_id"),
        "strategy": candidate.get("strategy"),
        "parent_branch_id": candidate.get("parent_branch_id"),
        "seed": candidate.get("seed"),
        "initialization_seed": candidate.get("initialization_seed"),
    }
    if current not in sources:
        sources.append(current)
    return sources
