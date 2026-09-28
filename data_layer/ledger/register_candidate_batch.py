"""将候选 branch、具体构型和来源登记到 PhaseDataManager。"""

from __future__ import annotations

from pathlib import Path
import hashlib
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
        full_na = candidate.get("full_na_structure")
        if full_na is not None:
            full_na_path = path.parent / "full_na_structure.vasp"
            if not full_na_path.exists():
                full_na.to(filename=full_na_path, fmt="poscar")
            full_na_checksum = "sha256:" + hashlib.sha256(full_na_path.read_bytes()).hexdigest()
            manager.data["branches"][branch_id]["full_na_structure_path"] = str(full_na_path)
            manager.data["branches"][branch_id]["full_na_structure_checksum"] = full_na_checksum
            record["full_na_structure_path"] = str(full_na_path)
            record["full_na_structure_checksum"] = full_na_checksum
        elif str(candidate.get("x")) in {"0", "0.0"}:
            manager.data["branches"][branch_id]["mc_search_applicable"] = False
            record["mc_search_applicable"] = False
        record.setdefault("metadata", {})
        record["metadata"].setdefault("calculation_status", "pending")
        if candidate.get("initialization_method"):
            record["metadata"].setdefault("initialization_method", candidate["initialization_method"])
            record["metadata"].setdefault("initial_state_index", candidate.get("initial_state_index"))
            record["metadata"].setdefault("initialization_seed", candidate.get("initialization_seed"))
        if "electrostatic_rank" in candidate:
            record["metadata"].setdefault("electrostatic_rank", candidate["electrostatic_rank"])
            record["metadata"].setdefault("electrostatic_rank_pool_size", candidate.get("electrostatic_rank_pool_size", 10))
            record["metadata"].setdefault("electrostatic_raw_rank", candidate.get("electrostatic_raw_rank"))
        if candidate.get("electrostatic_energy") is not None:
            record["metadata"].setdefault("electrostatic_energy", candidate["electrostatic_energy"])
        if candidate.get("electrostatic_charge_scheme") is not None:
            record["metadata"].setdefault("electrostatic_charge_scheme", candidate["electrostatic_charge_scheme"])
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
                "full_na_structure_path": record.get("full_na_structure_path"),
                "full_na_structure_checksum": record.get("full_na_structure_checksum"),
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
