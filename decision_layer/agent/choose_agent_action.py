"""请求可配置 LLM 给出结构化动作，无客户端时规则回退。"""

from __future__ import annotations

from typing import Any, Callable

from .choose_rule_action import choose_rule_action


def choose_agent_action(summary: dict, *, agent_client: Callable[[dict], dict] | None = None, config=None) -> dict[str, Any]:
    if agent_client is None:
        return choose_rule_action(summary, config=config)
    try:
        action = agent_client({"state": summary, "instruction": "Choose one allowed action. Do not calculate energies, Ehull, confidence, or rewards."})
        if not isinstance(action, dict):
            raise TypeError("agent output must be a dict")
        return {**action, "source": "llm_agent"}
    except Exception as error:
        fallback = choose_rule_action(summary, config=config)
        fallback["fallback_reason"] = f"agent_error: {type(error).__name__}: {error}"
        return fallback
