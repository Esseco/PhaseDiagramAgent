# Cr reasoning truncation, 2026-10-20

The 19:05 continue turn returned `finish_reason=length`, empty answer content,
29283 reasoning characters, and 8192 completion tokens. The workflow correctly
rejected the missing proposal; the service itself remained available.

The shared legacy DeepSeek client now allows one recovery only for a reasoning
response ending with `length`. It doubles the output budget up to 32768 tokens,
keeps reasoning and the original input/contract, and shares the original total
deadline and cancellation checks. Configured larger budgets are never reduced.
Other failures still stop immediately. Truncated JSON is never accepted. Both
responses count toward usage. Repeated truncation has code `output_truncated`
and reports the budget and remediation instead of a generic empty answer.

Regression: `tests/test_reasoning_truncation.py`; all responses are synthetic.
No real model call, computation, or service restart is used for validation.
The existing running service loads the change on the user's next manual restart.

Limits: a second response may also truncate, or the shared deadline may expire;
this change does not guarantee provider availability or a valid scientific plan.
The two existing failures in `test_model_failure_gate.py` (history resume and
approval invitation formatting) also reproduce with the original client source.
