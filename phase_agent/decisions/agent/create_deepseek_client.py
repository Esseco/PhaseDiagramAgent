"""创建单次调用、失败直接回报的 DeepSeek JSON 客户端。"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from phase_agent.decisions.agent.prepare_llm_request import (
    prepare_llm_request,
    needs_deep_reasoning,
)

from phase_agent.decisions.agent.relax_mc_preference import RELAX_MC_PREFERENCE

SEMANTIC_PROMPT = (
    "You are the project's conversational collaborator. Understand the current question and its context. "
    "Answer it directly using current authoritative facts. Read needed evidence with inspect_evidence; "
    "use advance only for requested scientific planning. Explain known failures, concrete remedies and "
    "what happens after repair, without repeating unresolved approval/continue questions. "
    "Prior dialogue resolves references but grants no permission. Respect the typed dialogue contract; "
    "return JSON only. Never claim an unperformed action, invent data or approve a new action."
)


class DeepSeekResponseError(RuntimeError):
    """可安全展示的 DeepSeek 错误；不包含 API 响应正文或密钥。"""

    def __init__(self, code: str, safe_message: str):
        super().__init__(safe_message)
        self.code = code
        if code in {"timeout", "network_error", "service_unavailable"}:
            from phase_agent.runtime.api_deadline import local_analysis

            safe_message += "\n" + local_analysis({})
            self.args = (safe_message,)
        self.safe_message = safe_message


def create_deepseek_client(
    *,
    api_key=None,
    model="deepseek-v4-pro",
    base_url="https://api.deepseek.com",
    max_tokens=800,
    timeout=60,
    total_timeout=120,
    system_prompt=None,
    thinking=None,
    routine_max_tokens=1600,
    reasoning_max_tokens=8192,
    proposal_harness="legacy",
    deepagents_max_tokens=4096,
    deepagents_recursion_limit=8,
):
    import math

    if not math.isfinite(float(total_timeout)) or float(total_timeout) <= 0:
        raise ValueError("total_timeout must be positive and finite")
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("需要 api_key 或 DEEPSEEK_API_KEY")
    if thinking not in {None, "enabled", "disabled"}:
        raise ValueError("thinking 必须是 enabled、disabled 或 None")
    if proposal_harness not in {"legacy", "deepagents", "hybrid"}:
        raise ValueError("proposal_harness must be legacy or deepagents")
    if proposal_harness == "hybrid":
        from phase_agent.decisions.agent.hybrid_proposal import create_hybrid_client

        legacy = create_deepseek_client(
            api_key=key,
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            timeout=timeout,
            total_timeout=total_timeout,
            system_prompt=system_prompt,
            thinking=thinking,
            routine_max_tokens=routine_max_tokens,
            reasoning_max_tokens=reasoning_max_tokens,
        )

        def deep_factory():
            from phase_agent.decisions.agent.deepagents_proposal import (
                create_deepseek_proposal_client,
            )

            return create_deepseek_proposal_client(
                api_key=key,
                model=model,
                base_url=base_url,
                timeout=timeout,
                max_tokens=int(deepagents_max_tokens),
                recursion_limit=int(deepagents_recursion_limit),
            )

        return create_hybrid_client(legacy, deep_factory)
    if proposal_harness == "deepagents":
        from phase_agent.decisions.agent.deepagents_proposal import (
            create_deepseek_proposal_client,
        )

        return create_deepseek_proposal_client(
            api_key=key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max(int(max_tokens), int(reasoning_max_tokens)),
        )

    from uuid import uuid4

    cache_namespace = ("legacy_client", uuid4().hex)

    def uncached_call(payload: dict) -> dict:
        from time import monotonic

        bounded_control = payload.get("bounded_control") is True
        effective_timeout = min(float(total_timeout), 12.0) if bounded_control else float(total_timeout)
        deadline = monotonic() + max(0.001, effective_timeout)
        planning = (payload.get("evidence_reader") or {}).get("phase") == "select_topics"
        important = not bounded_control and needs_deep_reasoning(payload)
        payload = prepare_llm_request(payload)
        from phase_agent.decisions.agent.transport_tables import evidence_row_count

        context = payload.get("decision_context") or {}
        count = len((context.get("dft_sampling_recommendation") or {}).get("candidate_ids") or [])
        if not count and "select_dft_candidates" in (payload.get("allowed_tools") or []):
            count = evidence_row_count(context.get("qbc_candidates"))
        # Scientific input preparation may select DFT through an operation alias;
        # its candidate evidence still needs enough output headroom on the first call.
        if not count and important:
            count = evidence_row_count(context.get("qbc_candidates"))
        if planning or payload.get("semantic_dialogue") is True:
            count = 0
        contracts = payload.get("output_contracts") or {}
        scientific_json = any(
            name in contracts
            for name in (
                "scientific_decision",
                "post_dft_review",
                "generation_plan_item",
            )
        )
        # Scientific reasoning plus per-candidate decisions/purposes need more
        # space than a short dialogue. Respect the configured limit as well.
        json_floor = 4096 if scientific_json else 1024
        per_candidate = 160 if scientific_json else 100
        json_ceiling = max(
            int(max_tokens),
            int(routine_max_tokens),
            min(16384, json_floor + per_candidate * count),
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Select read-only evidence topics. Follow the evidence_request contract only. JSON only; no scientific action or conclusion."
                    if planning
                    else RELAX_MC_PREFERENCE
                    + (
                        system_prompt
                        or (
                            SEMANTIC_PROMPT
                            if payload.get("semantic_dialogue")
                            else (
                                "Follow the supplied output contract. For unified_dialogue, interpret the CURRENT user_instruction with saved facts and conversation memory. The current instruction takes precedence over earlier dialogue. Return an answer for discussion, inspect_configuration for viewing, configure for editing, or a legal tool action for a requested scientific proposal. A tool action only proposes work for subsequent validation and approval; waiting for approval does not prevent proposing it. Choose proposal details within confirmed limits yourself when supported by facts; do not request approval before forming the proposal. Return JSON only. Never invent energy, "
                                "Ehull, reward, confidence, or completed calculations."
                            )
                        )
                    )
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        thinking_this_try = "enabled" if important and thinking != "disabled" else "disabled"
        token_ceiling = (
            max(int(max_tokens), int(reasoning_max_tokens))
            if thinking_this_try == "enabled"
            else json_ceiling
        )
        token_ceiling = max(token_ceiling, json_ceiling)
        max_tokens_this_try = (
            token_ceiling
            if thinking_this_try == "enabled"
            else min(
                max(int(max_tokens), 1024 + 100 * count) if count else int(max_tokens),
                token_ceiling,
            )
        )
        if scientific_json:
            max_tokens_this_try = max(max_tokens_this_try, json_ceiling)
        if planning:
            max_tokens_this_try = max(1024, int(routine_max_tokens))
        if bounded_control:
            max_tokens_this_try = 384
            thinking_this_try = "disabled"
        input_tokens = output_tokens = 0
        last_error = None

        # A failed response is surfaced immediately; user decides when to retry.
        from phase_agent.graphs.cancellation import check_cancelled

        for attempt in range(1):
            check_cancelled()
            body = {
                "model": model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "max_tokens": max_tokens_this_try,
                "temperature": 0.0,
            }
            if thinking_this_try is not None:
                body["thinking"] = {"type": thinking_this_try}
            request = urllib.request.Request(
                f"{base_url.rstrip('/')}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            from phase_agent.runtime.token_usage import begin_api_call, finish_api_call

            usage_entry = begin_api_call(
                model,
                payload.get("mode"),
                attempt,
                sum(len(m["content"]) for m in messages),
                phase="evidence_topics" if planning else "analysis",
                max_output_tokens=max_tokens_this_try,
                reasoning_enabled=thinking_this_try == "enabled",
            )
            try:
                from phase_agent.runtime.api_deadline import (
                    read_response,
                    APIDeadlineExceeded,
                    APITransportBusy,
                )

                raw_response = read_response(
                    request,
                    timeout if attempt == 0 else min(120, max(timeout, timeout * 1.5)),
                    deadline,
                )
                check_cancelled()
            except APITransportBusy:
                from phase_agent.runtime.api_deadline import local_analysis

                raise DeepSeekResponseError(
                    "transport_busy",
                    "之前的模型连接尚未释放，已停止发起请求。\n" + local_analysis(payload),
                ) from None
            except APIDeadlineExceeded:
                from phase_agent.runtime.api_deadline import local_analysis

                failure = DeepSeekResponseError(
                    "total_timeout",
                    f"模型调用已达到 {effective_timeout:g} 秒总时限，已停止等待，本轮未取得有效方案。\n"
                    + local_analysis(payload),
                )
                failure.llm_usage = {
                    "calls": attempt + 1,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "unknown_usage_calls": attempt + 1,
                    "cost": None,
                }
                raise failure from None
            except TimeoutError:
                check_cancelled()
                raise DeepSeekResponseError(
                    "timeout",
                    "模型服务读取超时，本次未得到回答，未自动重试；本次未生成方案或执行任务。",
                ) from None
            except urllib.error.HTTPError as error:
                if error.code in {502, 503, 504}:
                    failure = DeepSeekResponseError(
                        "service_unavailable",
                        f"DeepSeek 模型服务暂时不可用（HTTP {error.code}）；本轮未取得有效模型答复，未执行新任务。现有计算不因此失败，已回传文件请保留；稍后可重试分析，无需重新提交计算。",
                    )
                    failure.llm_usage = {
                        "calls": attempt + 1,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "unknown_usage_calls": attempt + 1,
                        "cost": None,
                        "model": model,
                        "mode": payload.get("mode", "unknown"),
                    }
                    raise failure from None
                if error.code in {401, 403}:
                    message = "DeepSeek API Key 鉴权失败，请检查密钥是否有效。"
                elif error.code in {402, 429}:
                    message = f"DeepSeek API 返回 HTTP {error.code}，请检查账户额度或请求频率。"
                else:
                    message = f"DeepSeek API 返回 HTTP {error.code}。"
                raise DeepSeekResponseError("http_error", message) from None
            except urllib.error.URLError as error:
                if isinstance(error.reason, TimeoutError):
                    check_cancelled()
                    message = "连接 DeepSeek API 超时，请检查网络后重试。"
                else:
                    message = "无法连接 DeepSeek API，请检查本机网络和 API 地址。"
                raise DeepSeekResponseError("network_error", message) from None

            try:
                result = json.loads(raw_response.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                last_error = DeepSeekResponseError(
                    "invalid_api_response", "DeepSeek API 返回内容格式异常。"
                )
                continue

            usage = result.get("usage") or {}
            finish_api_call(usage_entry, usage)
            input_tokens += int(usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("completion_tokens") or 0)
            choices = result.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                last_error = DeepSeekResponseError(
                    "missing_choice", "DeepSeek API 没有返回可用回答。"
                )
                continue

            choice = choices[0]
            finish_reason = choice.get("finish_reason")
            if usage_entry is not None:
                usage_entry["finish_reason"] = finish_reason
            message = choice.get("message") or {}
            content = message.get("content")
            if usage_entry is not None:
                usage_entry["response_chars"] = len(content) if isinstance(content, str) else 0
            if not isinstance(content, str) or not content.strip():
                reasoning_content = message.get("reasoning_content")
                last_error = DeepSeekResponseError(
                    "empty_content",
                    "DeepSeek 返回了空内容 "
                    f"（结束原因={finish_reason or '未知'}，"
                    f"正文长度=0，思考字段长度={len(reasoning_content or '')}，"
                    f"输出 token={int(usage.get('completion_tokens') or 0)}）。",
                )
                if finish_reason in {
                    "content_filter",
                    "insufficient_system_resource",
                    "aborted",
                }:
                    break
                continue

            try:
                if finish_reason == "length":
                    raise ValueError("incomplete response")
                action = _parse_json_object(content)
            except (json.JSONDecodeError, ValueError):
                last_error = DeepSeekResponseError(
                    "invalid_json",
                    "DeepSeek 返回的内容无法解析为 JSON 对象 "
                    f"（结束原因={finish_reason or '未知'}，字符数={len(content)}）。",
                )
                if finish_reason in {
                    "content_filter",
                    "insufficient_system_resource",
                    "aborted",
                }:
                    break
                continue

            if not isinstance(action, dict):
                last_error = DeepSeekResponseError(
                    "invalid_json_object", "DeepSeek 返回的 JSON 不是对象。"
                )
                continue

            action["_llm_usage"] = {
                "calls": attempt + 1,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": None,
                "model": result.get("model", model),
                "mode": payload.get("mode", "unknown"),
                "transport_metrics": payload.get("transport_metrics"),
                "reasoning_enabled": thinking_this_try == "enabled",
            }
            return action

        error = last_error or DeepSeekResponseError(
            "invalid_response", "DeepSeek 未能返回可解析的 JSON。"
        )
        error.llm_usage = {
            "calls": attempt + 1,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": None,
            "model": model,
            "mode": payload.get("mode", "unknown"),
            "transport_metrics": payload.get("transport_metrics"),
        }
        raise error

    def call(payload):
        from phase_agent.runtime.token_usage import cached_request
        from phase_agent.runtime.turn_process import timed_call

        return timed_call(
            "API 总时限 " + str(min(float(total_timeout), 12.0) if payload.get("bounded_control") is True else total_timeout) + " 秒",
            cached_request,
            payload,
            uncached_call,
            namespace=cache_namespace,
            category="model",
        )

    return call


def _parse_json_object(content: str) -> dict:
    """Parse plain JSON or a common fenced JSON response, without guessing fields."""
    text = str(content).strip().lstrip("\ufeff").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if (
            len(lines) < 3
            or lines[0].strip().lower() not in {"```", "```json"}
            or lines[-1].strip() != "```"
        ):
            raise ValueError("incomplete fenced JSON")
        text = "\n".join(lines[1:-1]).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("JSON response must be an object")
    return value
