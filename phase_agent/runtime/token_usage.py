"""Per-turn provider-reported usage; missing usage is never treated as zero."""

from phase_agent.runtime.turn_process import _current


def begin_api_call(model, mode, attempt, input_chars, **metadata):
    record = _current.get()
    if record is None:
        return None
    entry = {
        "model": model,
        "mode": mode,
        "attempt": attempt + 1,
        "input_chars": input_chars,
        "usage_known": False,
    }
    entry.update(
        {
            k: v
            for k, v in metadata.items()
            if k in {"phase", "max_output_tokens", "reasoning_enabled"}
        }
    )
    record.setdefault("model_requests", []).append(entry)
    return entry


def finish_api_call(entry, usage):
    if entry is None or not isinstance(usage, dict):
        return
    if usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
        return
    entry.update(
        usage_known=True,
        input_tokens=int(usage["prompt_tokens"]),
        output_tokens=int(usage["completion_tokens"]),
    )
    if usage.get("prompt_cache_hit_tokens") is not None:
        entry["cached_input_tokens"] = int(usage["prompt_cache_hit_tokens"])


def usage_summary(record):
    calls = record.get("model_requests") or []
    known = [entry for entry in calls if entry["usage_known"]]
    inputs = sum(entry["input_tokens"] for entry in known)
    outputs = sum(entry["output_tokens"] for entry in known)
    return {
        "phase_calls": {
            phase: sum(row.get("phase", "unclassified") == phase for row in calls)
            for phase in sorted({row.get("phase", "unclassified") for row in calls})
        },
        "truncated_calls": sum(row.get("finish_reason") == "length" for row in calls),
        "api_calls": len(calls),
        "retries": sum(entry["attempt"] > 1 for entry in calls),
        "input_tokens": inputs,
        "output_tokens": outputs,
        "total_tokens": inputs + outputs,
        "unknown_usage_calls": len(calls) - len(known),
        "complete": len(calls) == len(known),
        "reused_requests": record.get("reused_model_requests", 0),
    }


def cached_request(payload, callback, *, namespace=None):
    """Exact identical requests only, scoped to one turn and never across changes."""
    import hashlib, json
    from copy import deepcopy

    record = _current.get()
    if record is None:
        return callback(payload)
    # Cache outside persisted diagnostics: no payloads or credentials are saved.
    cache = record.setdefault("_model_request_cache", {})
    digest = hashlib.sha256(
        json.dumps(
            {"namespace": namespace, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    if digest in cache:
        record["reused_model_requests"] = record.get("reused_model_requests", 0) + 1
        result = deepcopy(cache[digest])
        result.pop("_llm_usage", None)  # Provider spend has already been recorded once.
        return result
    result = callback(payload)
    cache[digest] = deepcopy(result)
    return result


def provider_usage_callback(mode, *, phase="complex_analysis"):
    """Observe each deep-agent provider call, including unsuccessful requests."""
    from langchain_core.callbacks import BaseCallbackHandler

    class UsageCallback(BaseCallbackHandler):
        def __init__(self):
            self.entries = {}

        def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
            model = (serialized.get("kwargs") or {}).get("model_name", "deepagents")
            chars = sum(len(str(message.content)) for batch in messages for message in batch)
            self.entries[run_id] = begin_api_call(model, mode, 0, chars, phase=phase)

        def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
            self.entries[run_id] = begin_api_call(
                "deepagents", mode, 0, sum(map(len, prompts)), phase=phase
            )

        def on_llm_end(self, response, *, run_id, **kwargs):
            usage = (response.llm_output or {}).get("token_usage")
            if not usage:
                generations = response.generations or []
                metadata = (
                    getattr(getattr(generations[0][0], "message", None), "usage_metadata", None)
                    if generations and generations[0]
                    else None
                )
                if metadata:
                    usage = {
                        "prompt_tokens": metadata.get("input_tokens"),
                        "completion_tokens": metadata.get("output_tokens"),
                    }
            finish_api_call(self.entries.pop(run_id, None), usage)

    return UsageCallback()
