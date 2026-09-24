"""不运行 MLIP 的 BOHB 模拟示例。"""

import random

from .compare_bohb_methods import compare_bohb_methods
from scientific_layer.bohb.default_bohb_config import default_bohb_config
from .evaluate_budget_correlation import evaluate_budget_correlation


def run_example():
    config = default_bohb_config()
    config["scope"] = {"mlip_version": "mh-1-fixed", "hull_reference_version": "hull-v1", "candidate_set_version": "pool-v1"}
    candidates = [{"branch_id": f"B{i:03d}", "P": "O3" if i % 2 else "P3", "x": str((i % 4) / 4), "T": ["Fe", "Mn"], "composition_group": f"x={(i % 4) / 4}", "bohb_features": [i % 2, i % 4], "true_loss": (i % 7) / 10, "important": i in {3, 11}} for i in range(24)]

    def simulator(*, branch, action):
        rng = random.Random(action["seed"])
        noise = rng.gauss(0, 1 / max(action["budget"], 1) ** 0.5)
        return {"status": "completed", "minimum_energy_per_atom": branch["true_loss"] + noise, "composition_group": branch["composition_group"], "group_reference_energy_per_atom": 0.0, "group_energy_scale": 1.0, "actual_cost": action["incremental_budget"], "checkpoint": f"mock/{branch['branch_id']}/{action['budget']}"}

    def metrics(*, state, candidates):
        completed = [item for item in state.get("observations", []) if item.get("status") == "completed"]
        best = {}
        for item in completed:
            best[item["group"]] = min(best.get(item["group"], float("inf")), item["loss"])
        return {"hull_quality": sum(best.values()) / len(best) if best else None, "metric_note": "simulated mean best normalized loss by composition group"}

    comparison = compare_bohb_methods(candidates, config=config, total_mc_budget=540, seed=7, simulator=simulator, metric_evaluator=metrics)
    correlation = evaluate_budget_correlation(comparison["results"]["bohb"]["state"]["observations"], low_budget=10, high_budget=30, minimum_pairs=2)
    return {"comparison": comparison, "budget_correlation": correlation}


if __name__ == "__main__":
    print(run_example())
