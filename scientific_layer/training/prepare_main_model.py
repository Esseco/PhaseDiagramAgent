"""Grouped cross-validation and full-data main-model inputs."""
import random
from pathlib import Path

import yaml
from ase.io import write

from scientific_layer.training.prepare_mace_finetune import _group_key, _to_atoms


def grouped_folds(records, settings):
    labels = settings.get("labels", {})
    groups = {}
    for record in records:
        try:
            atoms = _to_atoms(record, labels)
        except (ValueError, TypeError):
            continue
        groups.setdefault(_group_key(record, settings), []).append(atoms)
    folds = settings.get("cross_validation", {}).get("folds", 5)
    if type(folds) is not int or folds != 5:
        raise ValueError("统一评估流程要求分组5折，不自动切换折数；修改需明确确认新方案")
    if len(groups) < folds:
        raise ValueError(f"分组K折需要至少{folds}个独立来源；当前{len(groups)}个")
    keys = sorted(groups)
    random.Random(int(settings.get("seed", 2026))).shuffle(keys)
    partitions = []
    for index in range(folds):
        valid = keys[index::folds]
        train = [key for key in keys if key not in valid]
        train_elements = {element for key in train for atoms in groups[key] for element in atoms.get_chemical_symbols()}
        valid_elements = {element for key in valid for atoms in groups[key] for element in atoms.get_chemical_symbols()}
        missing = valid_elements - train_elements
        if missing:
            raise ValueError(f"第{index+1}折训练数据缺少留出结构元素{sorted(missing)}；需补数据，不自动更换划分方法")
        partitions.append((train, valid))
    return groups, partitions


def prepare_main_model(records, output, settings):
    labels = settings.get("labels", {})
    groups, partitions = grouped_folds(records, settings)
    folds = len(partitions)
    jobs = []
    common = dict(settings.get("training", {}))
    common.pop("minimum_new_dft_records", None)
    common.update(energy_key=labels.get("energy_key", "REF_energy"),
                  forces_key=labels.get("forces_key", "REF_forces"))
    if labels.get("include_stress"):
        common["stress_key"] = labels.get("stress_key", "REF_stress")
    for index, (train, valid) in enumerate(partitions):
        name = f"main_cv_{index + 1}"
        directory = Path(output) / name
        directory.mkdir(parents=True, exist_ok=True)
        for split, assigned in (("train", train), ("valid", valid)):
            write(directory / f"{split}.xyz", [atom for key in assigned for atom in groups[key]], format="extxyz")
        parameters = {**common, "name": name, "train_file": "train.xyz", "valid_file": "valid.xyz"}
        parameters.pop("test_file", None)
        (directory / "mace_train.yaml").write_text(yaml.safe_dump(parameters, sort_keys=False), encoding="utf-8")
        jobs.append({"model_id": name, "train_groups": train, "valid_groups": valid})
    return {"folds": folds, "groups": len(groups), "fold_jobs": jobs,
            "evaluation_type": "grouped_cross_validation", "independent_test": False,
            "final_model": f"com_{int(settings.get('main_model_index', 0))+1}",
            "note": "主模型复用committee成员，不重复训练；全部数据参与正式成员训练，泛化误差只使用K折留出数据。"}
