"""Small orchestration entry point for historical-data preparation."""

from experiments.branch_surrogate.build_branch_dataset import build_branch_dataset
from experiments.branch_surrogate.save_branch_dataset import save_branch_dataset
from experiments.branch_surrogate.split_branch_dataset import split_branch_dataset
from experiments.branch_surrogate.validate_branch_dataset import validate_branch_dataset


def run_branch_dataset_preparation(source, *, config: dict, hull_reference: dict, output_directory) -> dict:
    dataset = build_branch_dataset(source, config=config, hull_reference=hull_reference)
    dataset = split_branch_dataset(dataset, config=config)
    report = validate_branch_dataset(dataset)
    return {"dataset": dataset, "report": report, "paths": save_branch_dataset(dataset, report, output_directory)}
