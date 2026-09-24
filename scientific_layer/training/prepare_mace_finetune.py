"""把已检查 DFT 记录整理为可复现的 MACE 委员会训练目录。"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor


def prepare_mace_finetune(records: list[dict], output_directory, *, config: dict) -> dict:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    labels = config.get("labels", {})
    accepted, rejected = [], []
    for record in records:
        try:
            accepted.append((_group_key(record, config), _to_atoms(record, labels), record))
        except Exception as error:
            rejected.append({"structure_id": record.get("structure_id"), "reason": f"{type(error).__name__}: {error}"})
    splits = _group_split(accepted, config.get("split", {}), int(config.get("seed", 2026)))
    shared = output / "_shared_data"
    shared.mkdir(exist_ok=True)
    for name, rows in splits.items():
        path = shared / f"{name}.xyz"
        write(path, [row[1] for row in rows], format="extxyz") if rows else path.write_text("", encoding="utf-8")
    committees = []
    common = dict(config.get("training", {}))
    for index, override in enumerate(config.get("committee", []), start=1):
        directory = output / f"com_{index}"
        directory.mkdir(exist_ok=True)
        parameters = dict(override)
        bootstrap_seed = parameters.pop("bootstrap_seed", None)
        train_path = shared / "train.xyz"
        if bootstrap_seed is not None and splits["train"]:
            generator = random.Random(int(bootstrap_seed))
            sampled = [generator.choice(splits["train"])[1] for _ in splits["train"]]
            train_path = directory / "train_bootstrap.xyz"
            write(train_path, sampled, format="extxyz")
        training = {**common, **parameters, "name": f"com_{index}", "train_file": str(train_path.resolve()), "valid_file": str((shared / "valid.xyz").resolve()), "test_file": str((shared / "test.xyz").resolve()), "energy_key": labels.get("energy_key", "REF_energy"), "forces_key": labels.get("forces_key", "REF_forces")}
        if labels.get("include_stress"):
            training["stress_key"] = labels.get("stress_key", "REF_stress")
        config_path = directory / "mace_train.yaml"
        config_path.write_text(yaml.safe_dump(training, sort_keys=False), encoding="utf-8")
        committees.append({"model_id": f"com_{index}", "directory": str(directory), "config_path": str(config_path), "bootstrap_seed": bootstrap_seed, "parameters": training})
    report = {"status": "prepared", "accepted": len(accepted), "rejected": rejected, "split_counts": {key: len(value) for key, value in splits.items()}, "split_groups": {key: sorted({row[0] for row in value}) for key, value in splits.items()}, "committees": committees, "config": config}
    _atomic_json(output / "finetune_report.json", report)
    return report


def _group_key(record, config):
    for key in config.get("group_keys", ["branch_id", "framework_id"]):
        if record.get(key) is not None:
            return f"{key}:{record[key]}"
    identifier = record.get("structure_id") or record.get("data_id")
    if identifier is None:
        raise ValueError("缺少 branch/framework/structure 分组标识")
    return f"structure_id:{identifier}"


def _to_atoms(record, labels):
    if record.get("status") != "completed" or record.get("converged") is not True or record.get("checks_passed", True) is not True:
        raise ValueError("DFT 记录未完成、未收敛或未通过检查")
    structure = record.get("structure")
    atoms = structure.copy() if isinstance(structure, Atoms) else AseAtomsAdaptor.get_atoms(structure)
    energy, forces = record.get("energy"), np.asarray(record.get("forces"), dtype=float)
    if energy is None or forces.shape != (len(atoms), 3):
        raise ValueError("缺少能量或 forces shape 不正确")
    atoms.info[labels.get("energy_key", "REF_energy")] = float(energy)
    atoms.arrays[labels.get("forces_key", "REF_forces")] = forces
    atoms.info["config_type"] = str(record.get("config_type", "Default"))
    atoms.info["source"] = str(record.get("source_path", record.get("structure_id", "unknown")))
    if labels.get("include_stress") and record.get("stress") is not None:
        stress = np.asarray(record["stress"], dtype=float)
        if stress.shape not in {(3, 3), (6,), (9,)}:
            raise ValueError("stress shape 必须为 3x3、6 或 9")
        atoms.info[labels.get("stress_key", "REF_stress")] = stress.reshape(-1)
    return atoms


def _group_split(rows, split, seed):
    ratios = [float(split.get(key, value)) for key, value in (("train", 0.8), ("valid", 0.1), ("test", 0.1))]
    if any(value < 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("train/valid/test 比例必须非负且总和为 1")
    groups = {}
    for row in rows:
        groups.setdefault(row[0], []).append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    first, second = int(len(keys) * ratios[0]), int(len(keys) * (ratios[0] + ratios[1]))
    assigned = {"train": keys[:first], "valid": keys[first:second], "test": keys[second:]}
    return {name: [row for key in group_keys for row in groups[key]] for name, group_keys in assigned.items()}


def _atomic_json(path, value):
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)
