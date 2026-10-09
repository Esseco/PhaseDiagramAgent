import pytest
from ase import Atoms
from scientific_layer.training.prepare_main_model import grouped_folds


def records(count=5, rare=False):
    return [{"branch_id": f"B{i}", "structure": Atoms("He" if rare and i == 0 else "H"),
        "status": "completed", "converged": True, "checks_passed": True,
        "energy": 0.0, "forces": [[0, 0, 0]]} for i in range(count)]


def test_fixed_five_folds_cover_each_group_once():
    groups, folds = grouped_folds(records(), {})
    held = [group for train, valid in folds for group in valid]
    assert len(folds) == 5 and len(held) == len(set(held)) == len(groups)
    assert all(not set(train) & set(valid) for train, valid in folds)


def test_insufficient_groups_and_unsupported_fold_count_block():
    with pytest.raises(ValueError, match="至少5"):
        grouped_folds(records(4), {})
    with pytest.raises(ValueError, match="统一评估流程"):
        grouped_folds(records(), {"cross_validation": {"folds": 3}})


def test_validation_elements_must_be_covered_in_training():
    with pytest.raises(ValueError, match="缺少留出结构元素"):
        grouped_folds(records(rare=True), {})
