"""Small entry point; does not alter the production scheduler."""

from experiments.branch_surrogate.compare_branch_surrogates import compare_branch_surrogates
from experiments.branch_surrogate.save_surrogate_comparison import save_surrogate_comparison


def run_surrogate_comparison(dataset: dict, feature_table: dict | None, *, surrogate_config: dict, comparison_config: dict, output_directory, hull_metric=None) -> dict:
    report = compare_branch_surrogates(dataset, feature_table, surrogate_config=surrogate_config, comparison_config=comparison_config, hull_metric=hull_metric)
    return {"report": report, "paths": save_surrogate_comparison(report, output_directory)}
