from phase_agent.agents.exploration.branch_pool_review import review_errors


def _action(decision, *, relax_only=None):
    return {
        "tool": "generate_branches",
        "parameters": {
            "regions": [],
            **({"relax_only_x_values": relax_only} if relax_only is not None else {}),
        },
        "branch_pool_review": {
            "choice": "reuse_existing",
            "sufficiency_for_next_search": "需要继续处理端点",
            "rationale": "当前模型端点需要单独处理",
            "alternative_reason": "其他路径证据不足",
            "missing_evidence": [],
            "revisit_conditions": ["完成 Relax 后复核"],
            "evidence_refs": ["current"],
            "existing_branch_groups": [
                {
                    "decision": decision,
                    "branch_ids": ["x0"],
                    "reason": "Na=0 端点",
                    "evidence_refs": ["current"],
                }
            ],
        },
    }


def _context():
    return {
        "branch_pool_review_required": True,
        "branch_pool_facts": {"existing_branches": [{"branch_id": "x0", "has_mobile_ions": False}]},
        "evidence_catalog": {"references": {"current": {}}},
    }


def test_x0_relax_only_group_is_valid_when_explicitly_declared():
    assert review_errors(_action("relax_only", relax_only=[0]), _context()) == []


def test_x0_continue_search_is_normalized_to_relax_only():
    assert review_errors(_action("continue_search"), _context()) == []


def test_x0_relax_only_does_not_require_model_field():
    assert review_errors(_action("relax_only", relax_only=[0.25]), _context()) == []


def test_x0_continue_group_is_normalized_to_relax_only_when_declared():
    assert review_errors(_action("continue_search", relax_only=[0]), _context()) == []
