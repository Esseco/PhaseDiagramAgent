"""Validate explicit strategy allocations; no scientific values are inferred."""
from collections import Counter
from typing import get_args
from pydantic import ValidationError
from decision_layer.agent.generation_contracts import GenerationAllocation, GenerationStrategy

STRATEGIES = set(get_args(GenerationStrategy))


def validate_generation_plan(params):
    plan = params.get("generation_plan")
    if not isinstance(plan, list) or not 1 <= len(plan) <= 12:
        raise ValueError("生成方案需包含1–12项实际策略分配")
    quotas = params.get("quotas")
    if (not isinstance(quotas, dict) or any(k not in STRATEGIES or type(v) is not int or v < 0
                                           for k, v in quotas.items())):
        raise ValueError("quotas必须是合法策略到非负JSON整数的映射")
    if type(params.get("total_quota")) is not int or params["total_quota"] <= 0:
        raise ValueError("total_quota必须是正JSON整数")
    totals = Counter()
    for index, row in enumerate(plan):
        try:
            GenerationAllocation.model_validate(row)
        except ValidationError as error:
            messages = [f"generation_plan[{index}]." + ".".join(map(str, e["loc"])) + ": " + e["msg"]
                        for e in error.errors(include_input=False, include_url=False)]
            raise ValueError("; ".join(messages)) from None
        totals[row["strategy"]] += row["quota"]
    if dict(totals) != {k: v for k, v in quotas.items() if v}:
        raise ValueError("策略清单与quotas不一致，不能静默采用默认分配")
    if sum(totals.values()) != params.get("total_quota"):
        raise ValueError("策略分配总数与total_quota不一致")
    return plan
