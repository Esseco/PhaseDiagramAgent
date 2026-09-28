"""Reject a chat model label accidentally pasted into a workspace path."""

from __future__ import annotations

import re


def validate_workspace_root(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("工作区根路径不能为空")
    candidate = value.strip()
    if any(ord(char) < 32 or ord(char) == 127 for char in candidate):
        raise ValueError("工作区根路径不能包含换行或控制字符")
    if re.search(r"(?:^|\s)(?:agent|deepseek|模型)(?:\s*模型)?(?:\s*版本)?\s*[:：]", candidate, re.I):
        raise ValueError("工作区路径混入了 Agent 模型文字；请只填写目录路径")
    return candidate
