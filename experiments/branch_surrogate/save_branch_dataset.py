"""Save dataset, split manifest and audit report as JSON."""

import json
from pathlib import Path


def save_branch_dataset(dataset: dict, report: dict, directory) -> dict:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    paths = {"dataset": root / "branch_dataset.json", "split": root / "split_manifest.json", "report": root / "data_check_report.json"}
    for key, value in (("dataset", dataset), ("split", dataset.get("split_manifest", {})), ("report", report)):
        paths[key].write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}
