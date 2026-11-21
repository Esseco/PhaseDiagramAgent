"""Shared parameter-fingerprinted MP competing-phase DFT repo and pure-DFT global Ehull."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


def dft_parameter_fingerprint(dft_parameters: dict[str, Any] | None) -> str:
    """Return a canonical SHA256 digest of POTCAR (or fallback DFT) parameters."""
    params = dft_parameters or {}
    potcar_keys = (
        "potcar",
        "POTCAR",
        "potcar_symbols",
        "potcar_functional",
        "user_potcar_settings",
        "functional",
    )
    potcar_subset = {k: params[k] for k in potcar_keys if k in params}
    target = potcar_subset if potcar_subset else params
    payload = json.dumps(
        target,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            tmp_path = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


class CompetingPhaseRepo:
    """Parameter-isolated repository of MP e_above_hull==0 elementary & binary DFT entries."""

    def __init__(self, repo_path: str | Path):
        self.path = Path(repo_path).resolve()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema": "mp-competing-dft-repo-v1", "entries": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema") != "mp-competing-dft-repo-v1" or not isinstance(
            data.get("entries"), dict
        ):
            raise ValueError("mp_competing_repo_schema_invalid")
        return data

    @staticmethod
    def entry_key(geometry_sha256: str, param_hash: str) -> str:
        return f"{geometry_sha256}:{param_hash}"

    def lookup(
        self,
        elements: list[str] | set[str],
        dft_parameters: dict[str, Any] | None,
        *,
        max_arity: int = 2,
    ) -> list[dict[str, Any]]:
        """Return cached competing DFT entries matching `elements` (up to `max_arity`) and DFT params."""
        allowed = {str(el) for el in elements}
        param_hash = dft_parameter_fingerprint(dft_parameters)
        data = self._load()
        matched = []
        for row in data["entries"].values():
            if row.get("dft_param_hash") != param_hash:
                continue
            comp = row.get("composition") or {}
            row_elements = {str(k) for k, v in comp.items() if float(v) > 0}
            if (
                not row_elements
                or len(row_elements) > max_arity
                or not row_elements.issubset(allowed)
            ):
                continue
            matched.append(dict(row))
        matched.sort(
            key=lambda r: (len(r.get("composition") or {}), str(r.get("geometry_sha256") or ""))
        )
        return matched

    def missing_mp_ids(
        self,
        candidates: list[dict[str, Any]],
        dft_parameters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Filter candidate structures to those not yet computed under `dft_parameters`."""
        param_hash = dft_parameter_fingerprint(dft_parameters)
        existing = self._load()["entries"]
        missing = []
        for row in candidates:
            key = self.entry_key(str(row["geometry_sha256"]), param_hash)
            if key not in existing:
                missing.append(dict(row))
        return missing

    def register(
        self,
        entries: list[dict[str, Any]],
        dft_parameters: dict[str, Any] | None,
    ) -> int:
        """Persist only composition, structure/hash, energy, and POTCAR parameter hash."""
        param_hash = dft_parameter_fingerprint(dft_parameters)
        data = self._load()
        added = 0
        for row in entries:
            geom = str(row.get("geometry_sha256") or "")
            struct = row.get("structure")
            if not geom and isinstance(struct, dict):
                geom = hashlib.sha256(
                    json.dumps(struct, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
            comp = row.get("composition") or {}
            energy = row.get("dft_energy_eV")
            if (
                not geom
                or not comp
                or isinstance(energy, bool)
                or not isinstance(energy, (int, float))
                or not math.isfinite(float(energy))
            ):
                raise ValueError("invalid_competing_phase_dft_entry")
            key = self.entry_key(geom, param_hash)
            record = {
                "composition": {str(k): float(v) for k, v in comp.items() if float(v) > 0},
                "dft_energy_eV": float(energy),
                "geometry_sha256": geom,
                "dft_param_hash": param_hash,
            }
            if isinstance(struct, dict):
                record["structure"] = struct
            data["entries"][key] = record
            added += 1
        _atomic_write_json(self.path, data)
        return added


def fetch_mp_stable_binaries(
    elements: list[str] | set[str],
    *,
    api_key: str | None = None,
    max_arity: int = 2,
    rester_cls: Any = None,
) -> list[dict[str, Any]]:
    """Fetch e_above_hull==0 elementary and binary structures from Materials Project."""
    if rester_cls is None:
        try:
            from mp_api.client import MPRester as rester_cls
        except ImportError as exc:
            raise RuntimeError("mp_api_not_installed") from exc
    key = api_key or os.environ.get("MP_API_KEY")
    if not key and rester_cls.__name__ == "MPRester":
        raise ValueError("mp_api_key_required")
    elems = sorted({str(e) for e in elements})
    from itertools import combinations

    all_chemsys = [
        "-".join(sorted(combo))
        for r in range(1, max(1, int(max_arity)) + 1)
        for combo in combinations(elems, r)
    ]
    results: list[dict[str, Any]] = []
    with rester_cls(key) as mpr:
        docs = mpr.materials.summary.search(
            chemsys=all_chemsys,
            is_stable=True,
            fields=["structure"],
        )
        for doc in docs:
            struct = doc.structure
            comp = {str(el): float(amt) for el, amt in struct.composition.get_el_amt_dict().items()}
            if not set(comp).issubset(set(elems)):
                continue
            geom_hash = hashlib.sha256(
                json.dumps(struct.as_dict(), sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            results.append(
                {
                    "composition": comp,
                    "geometry_sha256": geom_hash,
                    "structure": struct.as_dict(),
                }
            )
    return results


def import_vasprun_results(
    repo: CompetingPhaseRepo,
    root_dir: str | Path,
    dft_parameters: dict[str, Any] | None,
    *,
    require_converged: bool = True,
) -> dict[str, Any]:
    """Scan local VASP vasprun.xml(.gz) folders and register converged entries into `repo`."""
    from pymatgen.io.vasp.outputs import Vasprun
    import re

    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dft_results_dir_not_found: {root}")
    files = sorted(list(root.rglob("vasprun.xml")) + list(root.rglob("vasprun.xml.gz")))
    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for vfile in files:
        try:
            vr = Vasprun(str(vfile), parse_dos=False, parse_eigen=False)
            if require_converged and not vr.converged:
                skipped.append({"path": str(vfile), "reason": "not_converged"})
                continue
            struct = vr.final_structure
            comp = {str(el): float(amt) for el, amt in struct.composition.get_el_amt_dict().items()}
            geom_hash = hashlib.sha256(
                json.dumps(struct.as_dict(), sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            imported.append(
                {
                    "composition": comp,
                    "structure": struct.as_dict(),
                    "dft_energy_eV": float(vr.final_energy),
                    "geometry_sha256": geom_hash,
                }
            )
        except Exception as exc:
            skipped.append({"path": str(vfile), "reason": f"{type(exc).__name__}: {exc}"})
    added = repo.register(imported, dft_parameters) if imported else 0
    return {"status": "imported", "added": added, "skipped": skipped}


def compute_gs_global_dft_ehull(
    gs_dft_records: list[dict[str, Any]],
    competing_entries: list[dict[str, Any]],
    *,
    stability_tolerance_ev_per_atom: float = 1e-4,
) -> dict[str, Any]:
    """Evaluate pure-DFT global Ehull of GS structures against MP competing DFT phases."""
    if not gs_dft_records:
        return {"status": "no_gs_dft_records", "entries": []}
    if not competing_entries:
        return {"status": "missing_competing_phases", "entries": []}

    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition

    comp_pd_entries = []
    for row in competing_entries:
        energy = row.get("dft_energy_eV")
        comp = row.get("composition") or {}
        if (
            isinstance(energy, bool)
            or not isinstance(energy, (int, float))
            or not math.isfinite(float(energy))
            or not comp
        ):
            continue
        comp_pd_entries.append(
            PDEntry(
                Composition(comp),
                float(energy),
                name=str(row.get("mp_id") or row.get("formula") or "competing"),
            )
        )
    if not comp_pd_entries:
        return {"status": "missing_competing_phases", "entries": []}

    gs_pd_entries = []
    valid_gs = []
    for rec in gs_dft_records:
        energy = rec.get("energy") if rec.get("energy") is not None else rec.get("dft_energy_eV")
        comp = rec.get("composition") or {}
        if (
            isinstance(energy, bool)
            or not isinstance(energy, (int, float))
            or not math.isfinite(float(energy))
            or not comp
        ):
            continue
        entry = PDEntry(
            Composition(comp),
            float(energy),
            name=str(rec.get("structure_id") or rec.get("task_id") or "gs"),
        )
        gs_pd_entries.append(entry)
        valid_gs.append(rec)

    if not gs_pd_entries:
        return {"status": "no_valid_gs_dft_energies", "entries": []}

    diagram = PhaseDiagram([*comp_pd_entries, *gs_pd_entries])
    results = []
    for rec, entry in zip(valid_gs, gs_pd_entries):
        decomp, ehull = diagram.get_decomp_and_e_above_hull(entry, allow_negative=True)
        ehull_val = max(0.0, float(ehull))
        results.append(
            {
                "task_id": rec.get("task_id"),
                "structure_id": rec.get("structure_id"),
                "composition": rec.get("composition"),
                "dft_energy_eV": float(entry.energy),
                "dft_global_ehull_eV_per_atom": ehull_val,
                "is_globally_stable": ehull_val <= stability_tolerance_ev_per_atom,
                "decomposition_products": {
                    str(p.name): round(float(frac), 6) for p, frac in decomp.items()
                },
            }
        )
    max_ehull = max(r["dft_global_ehull_eV_per_atom"] for r in results)
    unstable_count = sum(1 for r in results if not r["is_globally_stable"])
    return {
        "status": "evaluated",
        "energy_basis": "pure_dft_only",
        "competing_phase_count": len(comp_pd_entries),
        "gs_count": len(results),
        "max_gs_global_ehull_eV_per_atom": max_ehull,
        "unstable_gs_count": unstable_count,
        "entries": results,
    }


GS_GLOBAL_EHULL_FIELDS = (
    "task_id",
    "structure_id",
    "composition",
    "dft_energy_eV",
    "dft_global_ehull_eV_per_atom",
    "is_globally_stable",
    "decomposition_products",
)


def export_gs_global_dft_ehull_csv(ehull_result: dict[str, Any], output_path: str | Path) -> Path:
    """Write pure-DFT global Ehull evaluation rows to CSV."""
    import csv

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = (ehull_result or {}).get("entries") or []
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(GS_GLOBAL_EHULL_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "task_id": row.get("task_id"),
                    "structure_id": row.get("structure_id"),
                    "composition": json.dumps(row.get("composition") or {}, sort_keys=True),
                    "dft_energy_eV": row.get("dft_energy_eV"),
                    "dft_global_ehull_eV_per_atom": row.get("dft_global_ehull_eV_per_atom"),
                    "is_globally_stable": row.get("is_globally_stable"),
                    "decomposition_products": json.dumps(
                        row.get("decomposition_products") or {}, sort_keys=True
                    ),
                }
            )
    return target


def prepare_mp_competing_phases(
    elements: list[str] | set[str],
    repo_path: str | Path,
    *,
    dft_parameters: dict[str, Any] | None = None,
    local_entries: list[dict[str, Any]] | None = None,
    local_vasprun_dir: str | Path | None = None,
    api_key: str | None = None,
    max_arity: int = 2,
    rester_cls: Any = None,
    fetch_mp: bool = True,
) -> dict[str, Any]:
    """Fetch MP stable binaries, import local DFT results, and list remaining DFT tasks."""
    repo = CompetingPhaseRepo(repo_path)
    if local_entries:
        repo.register(local_entries, dft_parameters)
    if local_vasprun_dir:
        import_vasprun_results(repo, local_vasprun_dir, dft_parameters)

    fetched = (
        fetch_mp_stable_binaries(
            elements, api_key=api_key, max_arity=max_arity, rester_cls=rester_cls
        )
        if fetch_mp
        else []
    )
    cached = repo.lookup(elements, dft_parameters, max_arity=max_arity)
    cached_hashes = {row.get("geometry_sha256") for row in cached}
    missing = [row for row in fetched if row.get("geometry_sha256") not in cached_hashes]
    pending_tasks = [
        {
            "stage": "dft_relax",
            "composition": row["composition"],
            "structure": row["structure"],
            "geometry_sha256": row["geometry_sha256"],
            "parameter_source": "mp_defaults",
        }
        for row in missing
    ]
    return {
        "status": "ready" if not pending_tasks else "dft_required",
        "dft_param_hash": dft_parameter_fingerprint(dft_parameters),
        "cached_entries": cached,
        "missing_structures": missing,
        "pending_dft_tasks": pending_tasks,
    }
