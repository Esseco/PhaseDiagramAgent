"""Batch-fill M3GNet structural embeddings in out_data CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from phase_agent.analysis.feedback.out_data import (
    DFT_FIELDS,
    STRUCT_FIELDS,
    _atomic_csv,
    _write_versioned_snapshots,
)

_ENCODER = None


def default_model_path() -> Path:
    import importlib.util

    spec = importlib.util.find_spec("maml")
    if not spec or not spec.submodule_search_locations:
        raise RuntimeError("maml_not_available")
    return Path(next(iter(spec.submodule_search_locations))) / "describers/data/m3gnet_models/matbench_mp_e_form/0/m3gnet"


def _model_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode() + b"\\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _init_worker(model_path: str):
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from Process_MLIP.direct_sampling.features import M3GNetEmbedding

    global _ENCODER
    _ENCODER = M3GNetEmbedding(model_path)


def _resolve_structure(raw: str) -> Path:
    path = Path(raw)
    if not path.exists() and raw.startswith("\\\\?\\"):
        path = Path(raw[4:])
    return path

def _compute_batch(items):
    results = []
    try:
        import numpy as np
        from ase.io import read
        from pymatgen.io.ase import AseAtomsAdaptor
        from Process_MLIP.direct_sampling.features import _windows_neighbor_compat
        from m3gnet.graph import Index, tf_compute_distance_angle
        from m3gnet.layers import polynomial
    except Exception as error:
        return [(key, "", "failed", f"{type(error).__name__}: {error}"[:500]) for key, _ in items]
    structures = []
    valid = []
    for key, raw_path in items:
        try:
            path = _resolve_structure(raw_path)
            if not path.is_file():
                raise FileNotFoundError(str(path))
            suffix = path.suffix.lower()
            atoms = read(path, format="vasp" if suffix in {".vasp", ".poscar"} else None)
            structures.append(AseAtomsAdaptor.get_structure(atoms))
            valid.append((key, raw_path))
        except Exception as error:
            results.append((key, "", "failed", f"{type(error).__name__}: {error}"[:500]))
    if not structures:
        return results
    try:
        model = _ENCODER.encoder.describer_model
        with _windows_neighbor_compat():
            graph = model.graph_converter.convert_many(structures).as_list()
            graph = tf_compute_distance_angle(graph)
            three_basis = model.basis_expansion(graph)
            three_cutoff = polynomial(graph[Index.BONDS], model.threebody_cutoff)
            hidden = model.featurizer(graph)
            hidden = model.feature_adjust(hidden)
            for index in range(model.n_blocks):
                hidden = model.three_interactions[index](hidden, three_basis, three_cutoff)
                hidden = model.graph_layers[index](hidden)
            values = np.asarray(model.layers[-2].layers[0](hidden))
        if values.ndim == 1:
            values = values[None, :]
        if len(values) != len(valid):
            raise ValueError(f"batch_feature_count_mismatch:{len(values)}!={len(valid)}")
        results.extend((key, json.dumps(value.tolist(), ensure_ascii=False, separators=(",", ":")), "available", "") for (key, _), value in zip(valid, values))
    except Exception as error:
        results.extend((key, "", "failed", f"{type(error).__name__}: {error}"[:500]) for key, _ in valid)
    return results

def _compute(item):
    key, raw_path = item
    try:
        from ase.io import read

        path = _resolve_structure(raw_path)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        suffix = path.suffix.lower()
        structure = read(path, format="vasp" if suffix in {".vasp", ".poscar"} else None)
        values = _ENCODER(structure)
        return key, json.dumps(values.tolist(), ensure_ascii=False, separators=(",", ":")), "available", ""
    except Exception as error:
        return key, "", "failed", f"{type(error).__name__}: {error}"[:500]


def _write_metadata(path: Path, value: dict):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    temporary.replace(path)


def fill_out_data(directory, *, model_path=None, workers=1, checkpoint_every=100, batch_size=16, limit=None):
    root = Path(directory).resolve()
    files = sorted(root.glob("*_structs_features.csv"))
    if not files:
        raise FileNotFoundError(f"no_struct_feature_csv:{root}")
    model = Path(model_path).resolve() if model_path else default_model_path().resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"m3gnet_model_missing:{model}")
    rows_by_file = {}
    pending = {}
    cache = {}
    total_rows = 0
    for csv_path in files:
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows_by_file[csv_path] = rows
        total_rows += len(rows)
        for index, row in enumerate(rows):
            row.setdefault("m3gnet_feature_error", "")
            key = row.get("structure_checksum") or row.get("structure_path") or f"{csv_path}:{index}"
            if row.get("m3gnet_feature_status") == "available" and row.get("m3gnet_features"):
                cache[key] = (row["m3gnet_features"], "available", row.get("m3gnet_feature_error", ""))
                continue
            if row.get("structure_path") and key not in pending:
                pending[key] = row["structure_path"]
    if limit is not None:
        pending = dict(list(pending.items())[: int(limit)])
    fields = list(STRUCT_FIELDS)
    if "m3gnet_feature_error" not in fields:
        fields.insert(fields.index("dh_extrapolation"), "m3gnet_feature_error")

    def apply_cache():
        for rows in rows_by_file.values():
            for row in rows:
                key = row.get("structure_checksum") or row.get("structure_path")
                if key not in cache:
                    continue
                features, status, error = cache[key]
                row["m3gnet_features"] = features
                row["m3gnet_feature_status"] = status
                row["m3gnet_feature_error"] = error

    def checkpoint():
        apply_cache()
        for csv_path, rows in rows_by_file.items():
            _atomic_csv(csv_path, fields, rows)

    completed = 0
    failed = 0
    if pending:
        if int(workers) <= 1:
            _init_worker(str(model))
            items = list(pending.items())
            step = max(1, int(batch_size))
            for start in range(0, len(items), step):
                for key, features, status, error in _compute_batch(items[start:start + step]):
                    cache[key] = (features, status, error)
                    completed += status == "available"
                    failed += status != "available"
                if (completed + failed) % max(1, int(checkpoint_every)) < step:
                    checkpoint()
        else:
            with ProcessPoolExecutor(max_workers=int(workers), initializer=_init_worker, initargs=(str(model),)) as pool:
                futures = [pool.submit(_compute, item) for item in pending.items()]
                for future in as_completed(futures):
                    key, features, status, error = future.result()
                    cache[key] = (features, status, error)
                    completed += status == "available"
                    failed += status != "available"
                    if (completed + failed) % max(1, int(checkpoint_every)) == 0:
                        checkpoint()
    checkpoint()
    metadata = {
        "status": "completed" if failed == 0 else "partial",
        "model_path": str(model),
        "model_sha256": _model_digest(model),
        "total_rows": total_rows,
        "unique_pending_structures": len(pending),
        "computed_successfully": completed,
        "failed": failed,
        "available_feature_rows": sum(row.get("m3gnet_feature_status") == "available" for rows in rows_by_file.values() for row in rows),
        "failed_feature_rows": sum(row.get("m3gnet_feature_status") == "failed" for rows in rows_by_file.values() for row in rows),
        "d_h_policy": "preserve_existing_only; no inference",
    }
    _write_metadata(root / "m3gnet_feature_metadata.json", metadata)
    return metadata


def merge_remote_features(directory, result_file, *, model_source="hpc_matgl_m3gnet"):
    """Merge checksum-keyed HPC results into every epoch CSV atomically."""
    root = Path(directory).resolve()
    result_path = Path(result_file).resolve()
    latest = {}
    with result_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            checksum = row.get("structure_checksum")
            if checksum:
                latest[str(checksum)] = row
    files = sorted(root.glob("*_structs_features.csv"))
    fields = list(STRUCT_FIELDS)
    changed = available = failed = 0
    structure_rows = {}
    for csv_path in files:
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row.setdefault("m3gnet_feature_error", "")
            checksum = str(row.get("structure_checksum") or "")
            result = latest.get(checksum)
            if not result:
                continue
            prior = (row.get("m3gnet_features"), row.get("m3gnet_feature_status"), row.get("m3gnet_feature_error"))
            status = result.get("status") or "failed"
            features = result.get("features")
            row["m3gnet_features"] = json.dumps(features, ensure_ascii=False, separators=(",", ":")) if features is not None else ""
            row["m3gnet_feature_status"] = "available" if status == "available" and features is not None else "failed"
            row["m3gnet_feature_error"] = result.get("error") or ""
            if prior != (row["m3gnet_features"], row["m3gnet_feature_status"], row["m3gnet_feature_error"]):
                changed += 1
            if row["m3gnet_feature_status"] == "available":
                available += 1
            else:
                failed += 1
        _atomic_csv(csv_path, fields, rows)
        epoch = csv_path.stem.removesuffix("_structs_features")
        structure_rows[epoch] = rows
    dft_rows = []
    dft_path = root / "DFT_err_data.csv"
    if dft_path.is_file():
        with dft_path.open(encoding="utf-8-sig", newline="") as handle:
            dft_rows = list(csv.DictReader(handle))
    _write_versioned_snapshots(root, dft_rows, structure_rows, action="m3gnet_features")
    return {
        "status": "completed",
        "result_file": str(result_path),
        "model_source": model_source,
        "changed_rows": changed,
        "available_rows": available,
        "failed_rows": failed,
        "epoch_files": len(files),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--merge-results")
    args = parser.parse_args()
    root = Path(args.project_root).resolve() / "analysis_outputs" / "out_data"
    if args.merge_results:
        result = merge_remote_features(root, args.merge_results)
    else:
        result = fill_out_data(root, model_path=args.model_path, workers=args.workers, checkpoint_every=args.checkpoint_every, batch_size=args.batch_size, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
