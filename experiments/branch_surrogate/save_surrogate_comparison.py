"""Save the comparison report and loadable recommendation."""

import json
from pathlib import Path


def save_surrogate_comparison(report: dict, directory) -> dict:
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    outputs = {"report": root / "surrogate_comparison_report.json", "recommendation": root / "recommended_surrogate_config.json"}
    outputs["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs["recommendation"].write_text(json.dumps(report.get("recommendation", {}), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {key: str(value) for key, value in outputs.items()}
