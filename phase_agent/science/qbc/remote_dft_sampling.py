"""Remote trajectory-based weighted KMeans++ DFT sampling and local acceptance."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ENERGY_STD_CUTOFF_EV_PER_ATOM = 0.010  # 10 meV/atom
FORCE_STD_CUTOFF_EV_PER_ANG = 0.100  # 100 meV/A
EHULL_CUTOFF_EV_PER_ATOM = 0.050  # 50 meV/atom
DEFAULT_SAMPLE_TARGET = 100
SAMPLING_SCHEMA = "remote-dft-kmeans-v1"


def compute_sampling_weight(
    energy_per_atom_std: float,
    f_std_max: float,
    ehull: float,
    *,
    energy_cutoff: float = ENERGY_STD_CUTOFF_EV_PER_ATOM,
    force_cutoff: float = FORCE_STD_CUTOFF_EV_PER_ANG,
    ehull_cutoff: float = EHULL_CUTOFF_EV_PER_ATOM,
) -> dict[str, float]:
    """Compute f_E, f_F, f_H and combined Weight = f_E * f_F * f_H."""
    vals = (energy_per_atom_std, f_std_max, ehull, energy_cutoff, force_cutoff, ehull_cutoff)
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vals
    ):
        raise ValueError("sampling_weight_nonfinite_input")
    if energy_cutoff <= 0 or force_cutoff <= 0 or ehull_cutoff <= 0:
        raise ValueError("sampling_weight_cutoff_must_be_positive")
    e_val = max(0.0, float(energy_per_atom_std))
    f_val = max(0.0, float(f_std_max))
    h_val = max(0.0, float(ehull))
    f_e = 1.0 - math.exp(-e_val / float(energy_cutoff))
    f_f = 1.0 - math.exp(-f_val / float(force_cutoff))
    f_h = math.exp(-h_val / float(ehull_cutoff))
    return {
        "f_E": f_e,
        "f_F": f_f,
        "f_H": f_h,
        "weight": f_e * f_f * f_h,
    }


def _read_json_maybe_gz(path: Path) -> Any:
    gz_path = Path(f"{path}.gz")
    if gz_path.is_file():
        with gzip.open(gz_path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atoms_from_frame(frame: dict[str, Any]):
    from ase import Atoms
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    if isinstance(frame.get("structure"), dict):
        return AseAtomsAdaptor.get_atoms(Structure.from_dict(frame["structure"]))
    if isinstance(frame.get("atoms"), Atoms):
        return frame["atoms"]
    if frame.get("structure_path") and Path(frame["structure_path"]).is_file():
        from ase.io import read

        return read(frame["structure_path"], index=0)
    return None


def _poscar_bytes(atoms) -> bytes:
    from io import StringIO
    from ase.io import write

    buf = StringIO()
    write(buf, atoms, format="vasp", direct=True)
    return buf.getvalue().encode("utf-8")


def _extract_frame_qbc(frame: dict[str, Any]) -> tuple[float | None, float | None]:
    qbc = frame.get("qbc") or {}
    e_std = frame.get("energy_per_atom_std")
    if e_std is None:
        e_std = frame.get("energy_std_per_atom")
    if e_std is None:
        e_std = qbc.get("energy_per_atom_std")
    if e_std is None and isinstance(frame.get("energy_per_model_per_atom"), (list, tuple)):
        vals = [float(x) for x in frame["energy_per_model_per_atom"]]
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            e_std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))

    f_std = frame.get("f_std_max")
    if f_std is None:
        f_std = frame.get("force_uncertainty_max")
    if f_std is None:
        f_std = qbc.get("f_std_max", qbc.get("force_max_atom_disagreement"))
    return (
        float(e_std) if e_std is not None else None,
        float(f_std) if f_std is not None else None,
    )


def _cleaned_hull_snapshot(
    state: dict[str, Any], snapshot: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    """Return a hull snapshot with extreme outliers excluded for Ehull/GS only.

    Outlier points remain eligible in the trajectory sampling pool.
    """
    entries = list(snapshot.get("entries") or [])
    excluded_shas: set[str] = {
        e.get("structure_sha256")
        for e in entries
        if e.get("structure_sha256")
        and (
            e.get("extrapolation_failure") is True
            or e.get("hull_outlier") is True
            or e.get("automatic_exclusion_eligible") is True
        )
    }
    try:
        from phase_agent.analysis.phase.phase_quality_reports import get_phase_quality_report

        if state.get("phase_records") and snapshot.get("status") == "completed":
            report = get_phase_quality_report(state, write=False)
            for row in report.get("structures") or []:
                if row.get("automatic_exclusion_eligible"):
                    excluded_shas.update(s for s in (row.get("structure_sha256") or []) if s)
    except (ImportError, ValueError, KeyError, TypeError, OSError):
        pass

    if not excluded_shas:
        return snapshot, 0
    cleaned_entries = [e for e in entries if e.get("structure_sha256") not in excluded_shas]
    # Recompute is_stable on the cleaned hull so GS selection reflects the cleaned convex hull.
    hull_pts = []
    for entry in cleaned_entries:
        comp = entry.get("composition") or {}
        na = float(comp.get("Na") or 0.0)
        o = float(comp.get("O") or 0.0)
        eform = entry.get("eform_per_O2")
        if o > 0 and na >= 0 and eform is not None and math.isfinite(float(eform)):
            hull_pts.append((2.0 * na / o, float(eform)))
    if hull_pts:
        from phase_agent.analysis.phase.phase_quality import _hull

        hull_line = _hull(hull_pts)
        cleaned = []
        for entry in cleaned_entries:
            comp = entry.get("composition") or {}
            na = float(comp.get("Na") or 0.0)
            o = float(comp.get("O") or 0.0)
            eform = entry.get("eform_per_O2")
            if o > 0 and na >= 0 and eform is not None and math.isfinite(float(eform)):
                x_na = 2.0 * na / o
                # Interpolate hull energy at x_na
                hull_y = hull_line[-1][1]
                for i in range(len(hull_line) - 1):
                    x0, y0 = hull_line[i]
                    x1, y1 = hull_line[i + 1]
                    if x0 - 1e-12 <= x_na <= x1 + 1e-12:
                        t = 0.0 if x1 == x0 else (x_na - x0) / (x1 - x0)
                        hull_y = y0 + t * (y1 - y0)
                        break
                on_hull = float(eform) <= hull_y + 1e-9
                cleaned.append({**entry, "is_stable": on_hull})
            else:
                cleaned.append(entry)
        cleaned_entries = cleaned
    return {**snapshot, "entries": cleaned_entries}, len(excluded_shas)


def _hull_ehull_evaluator(snapshot: dict[str, Any]):
    """Build an Ehull (eV/atom) lookup against the final MLIP hull."""
    entries = snapshot.get("entries") or []
    hull_pts = []
    for entry in entries:
        comp = entry.get("composition") or {}
        na = float(comp.get("Na") or 0.0)
        o = float(comp.get("O") or 0.0)
        eform = entry.get("eform_per_O2")
        if o > 0 and na >= 0 and eform is not None and math.isfinite(float(eform)):
            hull_pts.append((2.0 * na / o, float(eform)))
    if not hull_pts:
        return None
    hull_pts.sort(key=lambda p: p[0])
    # Lower convex hull in (x_Na_per_O2, eform_per_O2)
    lower: list[tuple[float, float]] = []
    for pt in hull_pts:
        while len(lower) >= 2:
            x1, y1 = lower[-2]
            x2, y2 = lower[-1]
            x3, y3 = pt
            if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) <= 0:
                lower.pop()
            else:
                break
        lower.append(pt)

    def _ehull_from_eform(
        x_na: float, eform_per_o2: float, o_count: float, atom_count: int
    ) -> float:
        if atom_count <= 0 or o_count <= 0:
            return 0.0
        if x_na <= lower[0][0]:
            hull_y = lower[0][1]
        elif x_na >= lower[-1][0]:
            hull_y = lower[-1][1]
        else:
            hull_y = lower[-1][1]
            for i in range(len(lower) - 1):
                x0, y0 = lower[i]
                x1, y1 = lower[i + 1]
                if x0 <= x_na <= x1:
                    t = 0.0 if x1 == x0 else (x_na - x0) / (x1 - x0)
                    hull_y = y0 + t * (y1 - y0)
                    break
        # Convert eform_per_O2 difference (eV/O2) to eV/atom
        diff_per_o2 = max(0.0, eform_per_o2 - hull_y)
        return diff_per_o2 * (o_count / 2.0) / float(atom_count)

    return _ehull_from_eform


def collect_final_gs_frames(
    snapshot: dict[str, Any],
    existing_dft_hashes: set[str],
) -> list[dict[str, Any]]:
    """Collect all unique ground-state frames from the final MLIP phase diagram."""
    gs_frames: list[dict[str, Any]] = []
    seen = set(existing_dft_hashes)
    for entry in snapshot.get("entries") or []:
        if not entry.get("is_stable"):
            continue
        if entry.get("phase_identification_status") not in (None, "identified"):
            continue
        raw_path = entry.get("structure_path")
        digest = entry.get("structure_sha256")
        if not raw_path or not digest:
            continue
        path = Path(raw_path)
        if not path.is_file():
            continue
        actual = _sha256_bytes(path.read_bytes())
        if actual != digest or digest in seen:
            continue
        seen.add(digest)
        gs_frames.append(
            {
                "candidate_id": entry.get("structure_id") or f"GS-{digest[:12]}",
                "structure_id": entry.get("structure_id"),
                "structure_path": str(path),
                "structure_sha256": digest,
                "composition": entry.get("composition"),
                "phase": entry.get("phase"),
                "ehull": float(entry.get("ehull") or 0.0),
                "selection_mode": "phase_diagram_gs",
            }
        )
    return gs_frames


def collect_trajectory_frames(
    root: Path,
    state: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    existing_hashes: set[str],
) -> list[dict[str, Any]]:
    """Collect all unique relax-mc trajectory frames with E_std, F_std, and Ehull."""
    frames: list[dict[str, Any]] = []
    seen = set(existing_hashes)
    hull_fn = _hull_ehull_evaluator(snapshot)
    snapshot_ehull_by_sha = {
        e.get("structure_sha256"): float(e["ehull"])
        for e in snapshot.get("entries") or []
        if e.get("structure_sha256") and e.get("ehull") is not None
    }

    for task in state.get("tasks") or []:
        if task.get("status") != "completed":
            continue
        if task.get("stage") not in {"relax_and_feature", "deep_search"}:
            continue
        outputs = task.get("outputs") or {}
        result_path = Path(task.get("result_path") or "")
        if not result_path.is_absolute():
            result_path = root / result_path
        task_dir = result_path.parent if result_path.name == "result.json" else result_path

        traj_sources: list[tuple[str, list[dict[str, Any]]]] = []
        init_traj = _read_json_maybe_gz(task_dir / "initial_relax_traj.json")
        if isinstance(init_traj, list):
            traj_sources.append(("initial_relax_traj", init_traj))
        pool_summary = _read_json_maybe_gz(task_dir / "pool" / "pool_summary.json")
        if not isinstance(pool_summary, list):
            pool_summary = outputs.get("pool_summary") or []
        for row in pool_summary:
            traj_name = row.get("relax_traj_file")
            if traj_name:
                traj = _read_json_maybe_gz(task_dir / "pool" / traj_name)
                if not isinstance(traj, list):
                    traj = _read_json_maybe_gz(task_dir / traj_name)
                if isinstance(traj, list):
                    traj_sources.append((traj_name, traj))
            # Also allow pool_summary row itself as fallback frame
            if row.get("structure_file"):
                spath = task_dir / "pool" / row["structure_file"]
                if spath.is_file():
                    traj_sources.append(
                        (
                            row["structure_file"],
                            [{**row, "structure_path": str(spath)}],
                        )
                    )

        for source_name, raw_frames in traj_sources:
            for idx, frame in enumerate(raw_frames):
                if not isinstance(frame, dict):
                    continue
                digest = frame.get("structure_sha256")
                if not digest and isinstance(frame.get("structure"), dict):
                    digest = _sha256_bytes(
                        json.dumps(
                            frame["structure"], sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                    )
                if (
                    not digest
                    and frame.get("structure_path")
                    and Path(frame["structure_path"]).is_file()
                ):
                    digest = _sha256_bytes(Path(frame["structure_path"]).read_bytes())
                if not digest or digest in seen:
                    continue
                e_std, f_std = _extract_frame_qbc(frame)
                if e_std is None or f_std is None:
                    continue
                atoms = _atoms_from_frame(frame)
                if atoms is None:
                    continue
                ehull = frame.get("ehull")
                if ehull is None:
                    ehull = snapshot_ehull_by_sha.get(digest)
                if ehull is None and hull_fn is not None and frame.get("eform_per_O2") is not None:
                    syms = atoms.get_chemical_symbols()
                    ehull = hull_fn(
                        2.0 * syms.count("Na") / max(1, syms.count("O")),
                        float(frame["eform_per_O2"]),
                        float(syms.count("O")),
                        len(atoms),
                    )
                if ehull is None:
                    ehull = 0.0
                weight_info = compute_sampling_weight(e_std, f_std, float(ehull))
                seen.add(digest)
                frames.append(
                    {
                        "candidate_id": f"TRAJ-{task.get('task_id')}-{source_name}-{idx}",
                        "source_task_id": task.get("task_id"),
                        "structure_id": task.get("structure_id"),
                        "branch_id": task.get("branch_id"),
                        "structure_sha256": digest,
                        "atoms": atoms,
                        "energy_per_atom_std": e_std,
                        "f_std_max": f_std,
                        "ehull": float(ehull),
                        **weight_info,
                        "selection_mode": "weighted_kmeans_pp",
                    }
                )
    return frames


def extract_mace_omat_l1_l2_features(
    atoms_list: list[Any],
    *,
    model_path: str | None = None,
    head: str = "omat_pbe",
    device: str | None = None,
) -> np.ndarray:
    """Extract per-structure MACE-omat L1+L2 descriptors (atom-mean pooled)."""
    if not atoms_list:
        return np.zeros((0, 0), dtype=np.float64)
    import torch
    from mace.calculators import MACECalculator, mace_mp

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model_path:
        calc = MACECalculator(
            model_paths=str(model_path),
            device=resolved_device,
            head=head,
            default_dtype="float64",
        )
    else:
        calc = mace_mp(model="medium-omat-0", device=resolved_device, default_dtype="float64")
    vectors = []
    for atoms in atoms_list:
        desc = calc.get_descriptors(atoms, invariants_only=False)
        arr = np.asarray(desc, dtype=np.float64)
        if arr.ndim == 2:
            vec = arr.mean(axis=0)
        elif arr.ndim == 1:
            vec = arr
        else:
            vec = arr.reshape(-1)
        vectors.append(vec)
    return np.vstack(vectors)


def select_weighted_kmeans_pp(
    candidate_features: np.ndarray,
    candidate_weights: np.ndarray,
    *,
    reference_features: np.ndarray | None = None,
    n_select: int,
    seed: int = 0,
    fps: bool = True,
) -> list[int]:
    """Select n_select candidate indices via weighted FPS/kmeans++ preserving reference centers."""
    n_cand = len(candidate_features)
    if n_select <= 0 or n_cand == 0:
        return []
    if n_cand <= n_select:
        return list(range(n_cand))

    # Prefer Process_MLIP's weighted kmeans++ only when stochastic kmeans++ is explicitly requested.
    if not fps:
        try:
            from Process_MLIP.direct_sampling import weighted_kmeans_pp  # type: ignore

            return list(
                weighted_kmeans_pp(
                    candidate_features,
                    candidate_weights,
                    reference_features=reference_features,
                    n_samples=n_select,
                    seed=seed,
                )
            )
        except (ImportError, AttributeError):
            pass

    rng = np.random.default_rng(seed)
    weights = np.maximum(np.asarray(candidate_weights, dtype=np.float64), 0.0)
    feats = np.asarray(candidate_features, dtype=np.float64)
    selected: list[int] = []
    chosen_mask = np.zeros(n_cand, dtype=bool)

    if reference_features is not None and len(reference_features) > 0:
        refs = np.asarray(reference_features, dtype=np.float64)
        feats_norm = np.sum(feats * feats, axis=1)
        refs_norm = np.sum(refs * refs, axis=1)
        min_dist_sq = np.empty(n_cand, dtype=np.float64)
        chunk_size = 512
        for start in range(0, n_cand, chunk_size):
            end = min(start + chunk_size, n_cand)
            dists = (
                feats_norm[start:end, None] + refs_norm[None, :] - 2.0 * (feats[start:end] @ refs.T)
            )
            min_dist_sq[start:end] = np.maximum(0.0, np.min(dists, axis=1))
    else:
        if fps:
            first = int(np.argmax(weights)) if float(weights.max()) > 0.0 else 0
        else:
            probs = weights.copy()
            total = probs.sum()
            probs = probs / total if total > 0 else np.full(n_cand, 1.0 / n_cand)
            first = int(rng.choice(n_cand, p=probs))
        selected.append(first)
        chosen_mask[first] = True
        diff = feats - feats[first]
        min_dist_sq = np.sum(diff * diff, axis=1)

    while len(selected) < n_select:
        scores = np.where(chosen_mask, 0.0, weights * min_dist_sq)
        if fps:
            max_score = float(scores.max())
            if max_score <= 0.0:
                remaining = np.flatnonzero(~chosen_mask)
                if len(remaining) == 0:
                    break
                nxt = int(remaining[int(np.argmax(weights[remaining]))])
            else:
                nxt = int(np.argmax(scores))
        else:
            total = float(scores.sum())
            if total <= 0.0:
                remaining = np.flatnonzero(~chosen_mask)
                if len(remaining) == 0:
                    break
                rem_weights = weights[remaining]
                rem_total = float(rem_weights.sum())
                if rem_total > 0.0:
                    nxt = int(rng.choice(remaining, p=rem_weights / rem_total))
                else:
                    nxt = int(rng.choice(remaining))
            else:
                nxt = int(rng.choice(n_cand, p=scores / total))
        selected.append(nxt)
        chosen_mask[nxt] = True
        diff = feats - feats[nxt]
        min_dist_sq = np.minimum(min_dist_sq, np.sum(diff * diff, axis=1))
    return selected


def run_remote_dft_sampling(
    root: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    *,
    target: int = DEFAULT_SAMPLE_TARGET,
    seed: int = 0,
    feature_extractor=None,
    sampler=None,
) -> dict[str, Any]:
    """Run GS-first + weighted kmeans++ DFT sampling on remote and export frames."""
    root = Path(root).resolve()
    raw_snapshot = (state.get("phase_diagrams") or {}).get("mlip") or {}
    snapshot, excluded_outlier_count = _cleaned_hull_snapshot(state, raw_snapshot)
    dft_refs = manifest.get("dft_references") or []
    existing_dft_hashes = {row["sha256"] for row in dft_refs if row.get("sha256")}

    gs_frames = collect_final_gs_frames(snapshot, existing_dft_hashes)
    gs_selected = gs_frames[:target]
    seen_hashes = set(existing_dft_hashes) | {row["structure_sha256"] for row in gs_selected}

    traj_frames = collect_trajectory_frames(root, state, snapshot, existing_hashes=seen_hashes)
    remaining_quota = max(0, target - len(gs_selected))

    sampled_traj: list[dict[str, Any]] = []
    feature_dim = 0
    features_archive_rel = None
    features_archive_sha256 = None
    cand_feats = None
    ref_feats = None
    cand_weights = None
    chosen_indices: list[int] = []
    if remaining_quota > 0 and traj_frames:
        from ase.io import read as ase_read

        ref_atoms = [
            ase_read(Path(root / r["path"])) for r in dft_refs if (root / r["path"]).is_file()
        ]
        gs_atoms = [ase_read(Path(r["structure_path"])) for r in gs_selected]
        cand_atoms = [r["atoms"] for r in traj_frames]
        all_atoms = ref_atoms + gs_atoms + cand_atoms

        extractor = feature_extractor or extract_mace_omat_l1_l2_features
        model_cfg = (manifest.get("config") or {}).get("mlip") or {}
        feats = extractor(
            all_atoms,
            model_path=model_cfg.get("feature_model_path") or model_cfg.get("model_path"),
        )
        feats = np.asarray(feats, dtype=np.float64)
        feature_dim = int(feats.shape[1]) if feats.ndim == 2 else 0
        n_ref = len(ref_atoms) + len(gs_atoms)
        ref_feats = feats[:n_ref] if n_ref > 0 else None
        cand_feats = feats[n_ref:]
        cand_weights = np.asarray([r["weight"] for r in traj_frames], dtype=np.float64)

        sample_fn = sampler or select_weighted_kmeans_pp
        chosen_indices = list(
            sample_fn(
                cand_feats,
                cand_weights,
                reference_features=ref_feats,
                n_select=remaining_quota,
                seed=seed,
            )
        )
        sampled_traj = [traj_frames[i] for i in chosen_indices]

    out_dir = root / "sampled_dft_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    if cand_feats is not None:
        features_path = out_dir / "sampled_dft_features.npz"
        selected_feats = (
            cand_feats[chosen_indices]
            if chosen_indices
            else np.empty((0, feature_dim), dtype=np.float32)
        )
        np.savez_compressed(
            features_path,
            reference_features=(
                np.asarray(ref_feats, dtype=np.float32)
                if ref_feats is not None
                else np.empty((0, feature_dim), dtype=np.float32)
            ),
            selected_features=np.asarray(selected_feats, dtype=np.float32),
            candidate_weights=np.asarray(cand_weights, dtype=np.float32),
            chosen_indices=np.asarray(chosen_indices, dtype=np.int64),
            candidate_ids=np.asarray([str(r["candidate_id"]) for r in traj_frames]),
        )
        features_archive_rel = "sampled_dft_frames/sampled_dft_features.npz"
        features_archive_sha256 = _sha256_bytes(features_path.read_bytes())
    exported: list[dict[str, Any]] = []
    for idx, row in enumerate([*gs_selected, *sampled_traj]):
        fname = f"frame_{idx:04d}.vasp"
        fpath = out_dir / fname
        if "poscar_bytes" in row:
            fpath.write_bytes(row["poscar_bytes"])
        elif "atoms" in row:
            fpath.write_bytes(_poscar_bytes(row["atoms"]))
        else:
            fpath.write_bytes(Path(row["structure_path"]).read_bytes())
        sha = _sha256_bytes(fpath.read_bytes())
        exported.append(
            {
                "candidate_id": row["candidate_id"],
                "structure_id": row.get("structure_id"),
                "source_task_id": row.get("source_task_id"),
                "branch_id": row.get("branch_id"),
                "structure_path": f"sampled_dft_frames/{fname}",
                "structure_sha256": sha,
                "selection_mode": row["selection_mode"],
                "energy_per_atom_std": row.get("energy_per_atom_std"),
                "f_std_max": row.get("f_std_max"),
                "ehull": row.get("ehull"),
                "f_E": row.get("f_E"),
                "f_F": row.get("f_F"),
                "f_H": row.get("f_H"),
                "weight": row.get("weight"),
            }
        )

    payload = {
        "schema": SAMPLING_SCHEMA,
        "bundle_id": manifest.get("bundle_id"),
        "model_version": manifest.get("model_version"),
        "target": target,
        "cutoffs": {
            "energy_std_eV_per_atom": ENERGY_STD_CUTOFF_EV_PER_ATOM,
            "force_std_eV_per_A": FORCE_STD_CUTOFF_EV_PER_ANG,
            "ehull_eV_per_atom": EHULL_CUTOFF_EV_PER_ATOM,
        },
        "feature_model": "MACE-omat-L1+L2",
        "feature_dim": feature_dim,
        "sampling_method": "weighted_kmeans_pp_fps",
        **(
            {
                "features_archive": features_archive_rel,
                "features_archive_sha256": features_archive_sha256,
            }
            if features_archive_rel
            else {}
        ),
        "reference_dft_count": len(existing_dft_hashes),
        "excluded_hull_outlier_count": excluded_outlier_count,
        "gs_count": len(gs_selected),
        "kmeans_count": len(sampled_traj),
        "trajectory_pool_count": len(traj_frames),
        "selected": exported,
    }
    (root / "dft_sampling_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def verify_remote_dft_sampling(
    destination: Path,
    state: dict[str, Any],
    *,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Local acceptance check for remote weighted kmeans++ DFT sampling."""
    destination = Path(destination).resolve()
    result_path = destination / "dft_sampling_result.json"
    if not result_path.is_file():
        return {"status": "not_present"}
    data = json.loads(result_path.read_text(encoding="utf-8"))
    if data.get("schema") != SAMPLING_SCHEMA:
        raise ValueError("remote_dft_sampling_schema_invalid")
    selected = data.get("selected") or []
    target = int(data.get("target") or DEFAULT_SAMPLE_TARGET)
    if len(selected) > target:
        raise ValueError("remote_dft_sampling_exceeds_target")

    existing_dft_hashes = {
        r.get("structure_sha256")
        for r in state.get("dft_dataset_records") or []
        if r.get("status") == "completed"
        and r.get("training_ready") is True
        and r.get("structure_sha256")
    }
    seen_hashes: set[str] = set()
    for row in selected:
        rel = row.get("structure_path") or ""
        fpath = (destination / rel).resolve()
        if not fpath.is_relative_to(destination) or not fpath.is_file():
            raise ValueError("remote_dft_sampling_frame_missing:" + str(rel))
        actual = _sha256_bytes(fpath.read_bytes())
        if actual != row.get("structure_sha256"):
            raise ValueError("remote_dft_sampling_frame_checksum_mismatch:" + str(rel))
        if actual in existing_dft_hashes:
            raise ValueError("remote_dft_sampling_reselected_existing_dft:" + actual)
        if actual in seen_hashes:
            raise ValueError("remote_dft_sampling_duplicate_frame:" + actual)
        seen_hashes.add(actual)
        if row.get("selection_mode") == "weighted_kmeans_pp":
            expected = compute_sampling_weight(
                row["energy_per_atom_std"],
                row["f_std_max"],
                row["ehull"],
            )
            for k in ("f_E", "f_F", "f_H", "weight"):
                if not math.isclose(float(row[k]), expected[k], rel_tol=1e-7, abs_tol=1e-9):
                    raise ValueError(f"remote_dft_sampling_weight_mismatch:{k}")

    if snapshot is not None:
        cleaned_snapshot, _ = _cleaned_hull_snapshot(state, snapshot)
        required_gs = {
            e.get("structure_sha256")
            for e in cleaned_snapshot.get("entries") or []
            if e.get("is_stable")
            and e.get("structure_sha256")
            and e.get("structure_sha256") not in existing_dft_hashes
        }
        if len(required_gs) <= target and not required_gs.issubset(seen_hashes):
            raise ValueError("remote_dft_sampling_missing_final_gs_frames")

    return {
        "status": "accepted",
        "selected_count": len(selected),
        "gs_count": data.get("gs_count", 0),
        "kmeans_count": data.get("kmeans_count", 0),
        "reference_dft_count": data.get("reference_dft_count", 0),
        "feature_model": data.get("feature_model"),
        "result_path": str(result_path),
    }
