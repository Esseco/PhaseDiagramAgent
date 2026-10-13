"""Merge compact MC returns and prepare a branch-local history seed."""

from pathlib import Path
import hashlib
import json

from Process_AL_MC.occupation_history import decode_snapshot, pack_occupation, site_layout

from .occupation_database import OccupationHistory


def history_database(config):
    path = config.get("state_path")
    if not path:
        return None
    return Path(path).resolve().parent / "cache/mc_occupancies.sqlite"


def history_seed(config, branch_id, template_path, *, tasks=None):
    """Seed from all recovered rounds/models of this branch; no model in key."""
    from pymatgen.core import Structure

    database = history_database(config)
    if database is None:
        return None
    template = Structure.from_file(template_path)
    count, basis = site_layout(template)
    history = OccupationHistory(database, branch_id, count, basis)
    try:
        if tasks is None:
            state_path = Path(config["state_path"])
            state = (
                json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
            )
            tasks = state.get("tasks") or []
        for task in tasks:
            if task.get("stage") != "deep_search" or task.get("branch_id") != branch_id:
                continue
            if task.get("status") not in {"completed", "failed", "timeout", "cancelled"}:
                continue
            result_path = task.get("calculation_result_path") or task.get("result_path")
            if not result_path:
                continue
            result_path = Path(result_path)
            delta = result_path.parent / "mc_occupations.bin"
            if delta.is_file():
                data = delta.read_bytes()
                checksum = hashlib.sha256(data).hexdigest()
                source = str(task["task_id"]) + ":occupancy"
                if not history.imported(source, checksum):
                    header, _ = decode_snapshot(data)
                    if header["scope"] != branch_id:
                        raise ValueError("Returned MC occupancy history belongs to another branch")
                    history.merge(data)
                    history.mark_imported(source, checksum)
                continue
            # Old traces did not save trial occupations. Backfill only saved
            # input/final structures, never claim unsaved trials are known.
            if task.get("status") != "completed":
                continue
            legacy_paths = [task.get("structure_path")]
            outputs = task.get("outputs") or {}
            final = outputs.get("structure_path") or outputs.get("final_structure_path")
            if final:
                final = Path(final)
                if not final.is_absolute():
                    final = result_path.parent / final
                if not final.is_file():
                    final = result_path.parent / str(final).replace("\\", "/").split("/")[-1]
                legacy_paths.append(final)
            legacy_paths.extend(result_path.parent.glob("rank_*.vasp"))
            legacy_paths.extend((result_path.parent / "pool").glob("rank_*.vasp"))
            for value in legacy_paths:
                path = Path(value) if value else None
                if path is None or not path.is_file():
                    continue
                checksum = hashlib.sha256(path.read_bytes()).hexdigest()
                source = str(task["task_id"]) + ":legacy:" + str(path)
                if history.imported(source, checksum):
                    continue
                from Process_AL_MC.MC_sample import LayeredOxide_MCOrderingClass

                mapper = LayeredOxide_MCOrderingClass.__new__(LayeredOxide_MCOrderingClass)
                mapper.structure = Structure.from_file(path)
                mapper.full_na_structure = template
                history.add(pack_occupation(mapper._build_occ_from_input_structure()))
                history.mark_imported(source, checksum)
        return history.seed()
    finally:
        history.close()
