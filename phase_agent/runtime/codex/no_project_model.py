"""Fail-closed project model boundary for Codex takeover."""


class NoProjectModel:
    _codex_structured = True
    _decision_source = "codex"
    _validates_original_facts = True

    def __call__(self, _payload):
        raise RuntimeError("Codex 接管期间禁止调用项目模型；请由 Codex 处理当前请求。")
