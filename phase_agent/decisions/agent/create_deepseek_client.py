"""创建具备空响应重试和安全诊断的 DeepSeek JSON 客户端。"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from phase_agent.decisions.agent.prepare_llm_request import (
    prepare_llm_request,
    needs_deep_reasoning,
)


class DeepSeekResponseError(RuntimeError):
    """可安全展示的 DeepSeek 错误；不包含 API 响应正文或密钥。"""

    def __init__(self, code: str, safe_message: str):
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message


def create_deepseek_client(
    *,
    api_key=None,
    model="deepseek-v4-pro",
    base_url="https://api.deepseek.com",
    max_tokens=800,
    timeout=60,
    system_prompt=None,
    thinking=None,
    routine_max_tokens=1600,
    reasoning_max_tokens=8192,
    proposal_harness="legacy",
    deepagents_max_tokens=4096,
    deepagents_recursion_limit=8,
):
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
        from phase_agent.decisions.agent.deepagents_proposal import create_deepseek_proposal_client

        return create_deepseek_proposal_client(
            api_key=key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max(int(max_tokens), int(reasoning_max_tokens)),
        )

    def call(payload: dict) -> dict:
        important = needs_deep_reasoning(payload)
        payload = prepare_llm_request(payload)
        context = payload.get("decision_context") or {}
        count = len((context.get("dft_sampling_recommendation") or {}).get("candidate_ids") or [])
        if not count and "select_dft_candidates" in (payload.get("allowed_tools") or []):
            count = len(context.get("qbc_candidates") or [])
        json_ceiling = max(int(routine_max_tokens), min(16384, 1024 + 100 * count))
        messages = [
            {
                "role": "system",
                "content": system_prompt
                or (
                    "Follow the supplied output contract. For unified_dialogue, interpret the CURRENT user_instruction with saved facts and conversation memory. The current instruction takes precedence over earlier dialogue. Return an answer for discussion, inspect_configuration for viewing, configure for editing, or a legal tool action for a requested scientific proposal. A tool action only proposes work for subsequent validation and approval; waiting for approval does not prevent proposing it. Choose proposal details within confirmed limits yourself when supported by facts; do not request approval before forming the proposal. Return JSON only. Never invent energy, "
                    "Ehull, reward, confidence, or completed calculations."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        if payload.get("unified_dialogue") and payload.get("user_instruction"):
            messages.append(
                {
                    "role": "user",
                    "content": "当前请求（历史对话只是上下文；按所给契约返回 JSON）：\n"
                    + str(payload["user_instruction"]),
                }
            )
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
        input_tokens = output_tokens = 0
        last_error = None

        # DeepSeek JSON mode may occasionally return empty content. Retry one
        # malformed/empty model response, while accounting for both API calls.
        from phase_agent.graphs.cancellation import check_cancelled

        for attempt in range(2):
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
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw_response = response.read()
                check_cancelled()
            except urllib.error.HTTPError as error:
                if error.code in {401, 403}:
                    message = "DeepSeek API Key 鉴权失败，请检查密钥是否有效。"
                elif error.code in {402, 429}:
                    message = f"DeepSeek API 返回 HTTP {error.code}，请检查账户额度或请求频率。"
                else:
                    message = f"DeepSeek API 返回 HTTP {error.code}。"
                raise DeepSeekResponseError("http_error", message) from None
            except urllib.error.URLError as error:
                if isinstance(error.reason, TimeoutError):
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
            message = choice.get("message") or {}
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                reasoning_content = message.get("reasoning_content")
                last_error = DeepSeekResponseError(
                    "empty_content",
                    "DeepSeek 返回了空内容 "
                    f"（结束原因={finish_reason or '未知'}，"
                    f"正文长度=0，思考字段长度={len(reasoning_content or '')}，"
                    f"输出 token={int(usage.get('completion_tokens') or 0)}）。",
                )
                if finish_reason == "length":
                    if reasoning_content:
                        thinking_this_try = "disabled"
                    token_ceiling = max(token_ceiling, json_ceiling)
                    max_tokens_this_try = min(max(max_tokens_this_try * 2, 1200), token_ceiling)
                elif finish_reason in {"content_filter", "insufficient_system_resource", "aborted"}:
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
                if finish_reason == "length":
                    thinking_this_try = "disabled"
                    max_tokens_this_try = min(
                        max(max_tokens_this_try * 2, json_ceiling), token_ceiling
                    )
                elif finish_reason in {"content_filter", "insufficient_system_resource", "aborted"}:
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
        }
        raise error

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
