"""Versioned evidence for the three scientific choices, never an action selector."""

from collections import Counter, defaultdict
from copy import deepcopy


def decision_brief(state):
    version = state.get("active_model_version")
    config = state.get("confirmed_config") or {}
    tasks = [t for t in state.get("tasks") or [] if t.get("model_version") == version]
    groups = defaultdict(Counter)
    for task in tasks:
        groups[task.get("branch_id") or "unmapped"][task.get("status") or "unknown"] += 1
    candidates = [
        r for r in state.get("qbc_candidates") or [] if r.get("model_version", version) == version
    ]
    strata = Counter(
        (str(r.get("phase") or r.get("P") or "unknown"), str(r.get("x_Na_per_O2", "unknown")))
        for r in candidates
    )
    return {
        "schema_version": 1,
        "config_version": state.get("confirmed_config_version"),
        "model_version": version,
        "policy": deepcopy(state.get("scientific_decision_policy") or {}),
        "task_status_by_branch": dict(groups),
        "candidate_strata": [
            {"phase": k[0], "x_Na_per_O2": k[1], "count": v} for k, v in sorted(strata.items())
        ],
        "coverage": deepcopy(state.get("coverage") or {}),
        "recent_generation": [
            {
                k: deepcopy(r.get(k))
                for k in ("task_key", "model_version", "generation_plan", "summary")
            }
            for r in (state.get("generation_history") or [])[-5:]
        ],
        "recent_rewards": deepcopy((state.get("rewards") or [])[-5:]),
        "recent_model_reviews": [
            {
                "candidate_model_version": key,
                "status": row.get("status"),
                "scientific_decision": deepcopy(
                    (row.get("agent_review") or {}).get("scientific_decision")
                ),
            }
            for key, row in list((state.get("candidate_models") or {}).items())[-5:]
            if (row.get("agent_review") or {}).get("scientific_decision")
        ],
        "budget_usage": deepcopy(
            {k: v for k, v in (state.get("budget_usage") or {}).items() if k != "llm"}
        ),
        "limits": deepcopy(config.get("budgets") or {}),
        "unknowns": [
            "Unrecorded costs, strategy attribution and gains remain unknown.",
            "QBC agreement is not proof of accuracy or independent validation.",
            "No search exhaustion or global convergence follows from an empty batch.",
        ],
        "instruction": "Use matching model/config evidence only. Candidate strata reflect the pool, not exhaustive physical coverage. Historical rewards are labeled by their recorded versions; never compare energies across MLIP versions. Propose a strategy, not MC steps or a sequence of tool calls.",
    }
