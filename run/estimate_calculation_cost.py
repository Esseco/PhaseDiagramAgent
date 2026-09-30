"""Offline cost report: python -m run.estimate_calculation_cost --tasks spec.json."""
import argparse
import json
from pathlib import Path
from analysis_layer.cost.estimate_task_cost import estimate_batch_cost


def main():
    parser = argparse.ArgumentParser(description="Read-only task/batch relative cost estimate")
    parser.add_argument("--tasks", required=True, help="JSON array of stage/atom_count/workload specifications")
    parser.add_argument("--state", help="Optional state.json for measured calibration")
    parser.add_argument("--config", help="Optional confirmed config JSON containing budgets")
    args = parser.parse_args()
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8-sig")) if path else {}
    state = read(args.state)
    config = read(args.config) or state.get("confirmed_config") or {}
    report = estimate_batch_cost(read(args.tasks), budgets=config.get("budgets"), state=state)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
