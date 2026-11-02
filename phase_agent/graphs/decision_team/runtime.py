"""Live clients and provider accounting stay outside graph State."""

from dataclasses import dataclass, field
from typing import Any


def merge_usage(usages):
    if not usages:
        return None
    return {
        **usages[-1],
        **{
            k: sum(u.get(k) or 0 for u in usages)
            for k in ("calls", "input_tokens", "output_tokens")
        },
        "cost": sum(u["cost"] for u in usages)
        if all(u.get("cost") is not None for u in usages)
        else None,
    }


@dataclass
class DecisionTeamRuntime:
    agent_client: Any
    payload: dict
    usages: list[dict] = field(default_factory=list)

    def call(self, client, request):
        try:
            result = client(request)
        except Exception as error:
            if getattr(error, "llm_usage", None):
                self.usages.append(error.llm_usage)
            raise
        if isinstance(result, dict) and result.get("_llm_usage"):
            self.usages.append(result["_llm_usage"])
        return result

    def expert_call(self, request):
        client = getattr(self.agent_client, "expert_client", self.agent_client)
        return self.call(client, request)

    def proposal_call(self, request):
        return self.call(self.agent_client, request)

    def team_enabled(self):
        client = self.agent_client
        return (
            getattr(client, "supports_expert_advice", False) is True
            and not getattr(client, "_validates_original_facts", False)
            and not getattr(client, "_codex_structured", False)
            and getattr(client, "_decision_source", None) != "codex"
            and self.payload.get("mode") == "autonomous_search"
            and bool(self.payload.get("allowed_tools"))
            and not self.payload.get("bounded_control")
            and (self.payload.get("evidence_reader") or {}).get("phase") != "select_topics"
        )

    def semantic_dialogue(self):
        return self.payload.get("semantic_dialogue") is True or getattr(
            self.agent_client, "_semantic_dialogue", False
        )
