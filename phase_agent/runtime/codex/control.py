"""Codex-facing client for the existing local Agent control service.

This module is an adapter only: it talks to the already running local HTTP
service and never edits project state or invokes scientific tools directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any


OPERATOR_MODES = frozenset({"agent", "codex"})
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class CodexControlError(RuntimeError):
    """A local Agent control request failed."""

    def __init__(self, message: str, *, status: int | None = None, payload: Any = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


@dataclass(frozen=True)
class CodexControlClient:
    """Small client for the existing local Agent HTTP surface."""

    base_url: str = "http://127.0.0.1:8765"
    control_token: str | None = None
    api_token: str | None = None
    timeout: float = 30.0
    operator: str = "agent"

    def __post_init__(self):
        if self.operator not in OPERATOR_MODES:
            raise ValueError(f"operator must be one of {sorted(OPERATOR_MODES)}")

    def _ensure_agent_mode(self):
        if self.operator == "codex":
            raise CodexControlError(
                "Codex 接管模式不会调用 Agent/DeepSeek 生成方案；请由当前 Codex 读取状态、"
                "形成结构化方案后再提交已有审批决定。"
            )

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> Any:
        url = self.base_url.rstrip("/") + path
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as error:
            raw = error.read()
            raise self._error_from_response(error.code, raw) from error
        except URLError as error:
            raise CodexControlError(f"无法连接本地 Agent 服务：{error.reason}") from error
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CodexControlError("本地 Agent 返回了非 JSON 响应") from error

    @staticmethod
    def _error_from_response(status: int, raw: bytes) -> CodexControlError:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        message = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
            message = message or payload.get("message")
        return CodexControlError(
            message or f"本地 Agent 请求失败（HTTP {status}）", status=status, payload=payload
        )

    def _control(self, path: str, *, method: str = "GET", payload=None) -> Any:
        if not self.control_token:
            raise CodexControlError("缺少控制接口令牌；请设置 PHASE_CONTROL_API_KEY")
        return self._request(path, method=method, payload=payload, token=self.control_token)

    def health(self) -> Any:
        return self._request("/health")

    def status(self) -> Any:
        return self._control("/phase/status")

    def pending(self) -> Any:
        return self._control("/phase/pending")

    def tasks(self) -> Any:
        return self._control("/phase/tasks")

    def config(self) -> Any:
        return self._control("/phase/config")

    def charts(self) -> Any:
        return self._control("/phase/charts")

    def diagnostics(self) -> Any:
        return self._control("/phase/diagnostics")

    def context(self) -> Any:
        if self.operator != "codex":
            raise CodexControlError("Codex 上下文入口必须使用 operator=codex")
        return self._control("/phase/codex/context")

    def propose(self, instruction: str, *, conversation_id: str = "codex-control") -> Any:
        self._ensure_agent_mode()
        return self._control(
            "/phase/propose",
            method="POST",
            payload={"instruction": instruction, "conversation_id": conversation_id},
        )

    def continue_project(self, *, conversation_id: str = "codex-control") -> Any:
        return self.propose("继续", conversation_id=conversation_id)

    def codex_propose(
        self, action: dict[str, Any], *, conversation_id: str = "codex-control"
    ) -> Any:
        if self.operator != "codex":
            raise CodexControlError("结构化 Codex 提案必须使用 operator=codex")
        context = self.context()
        return self._control(
            "/phase/codex/propose",
            method="POST",
            payload={
                "action": action,
                "context_version": context.get("context_version"),
                "context_digest": context.get("context_digest"),
                "conversation_id": conversation_id,
            },
        )

    def codex_report(self, plan_id: str) -> Any:
        if self.operator != "codex":
            raise CodexControlError("Codex 记录入口必须使用 operator=codex")
        return self._control("/phase/codex/report", method="POST", payload={"plan_id": plan_id})

    def codex_plot(self) -> Any:
        if self.operator != "codex":
            raise CodexControlError("Codex 绘图入口必须使用 operator=codex")
        return self._control("/phase/codex/plot", method="POST", payload={})

    def decide(
        self,
        decision: str,
        *,
        plan_id: str,
        state_version: str,
        proposal_hash: str,
        comment: str = "",
        conversation_id: str = "codex-control",
    ) -> Any:
        return self._control(
            "/phase/decision",
            method="POST",
            payload={
                "decision": decision,
                "plan_id": plan_id,
                "state_version": state_version,
                "proposal_hash": proposal_hash,
                "comment": comment,
                "conversation_id": conversation_id,
            },
        )

    def chat(self, message: str, *, conversation_id: str = "codex-control") -> Any:
        self._ensure_agent_mode()
        if not self.api_token:
            raise CodexControlError("缺少对话接口令牌；请设置 PHASE_API_KEY")
        return self._request(
            "/v1/chat/completions",
            method="POST",
            payload={
                "messages": [{"role": "user", "content": message}],
                "metadata": {"chat_id": conversation_id},
            },
            token=self.api_token,
        )


def _client_from_args(args: argparse.Namespace) -> CodexControlClient:
    return CodexControlClient(
        base_url=args.base_url or os.environ.get("PHASE_CONTROL_URL", "http://127.0.0.1:8765"),
        control_token=args.control_token or os.environ.get("PHASE_CONTROL_API_KEY"),
        api_token=args.api_token or os.environ.get("PHASE_API_KEY"),
        operator=args.operator,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通过现有本地 Agent 控制接口推进项目")
    parser.add_argument("--base-url", help="本地 Agent 服务地址")
    parser.add_argument("--control-token", help="控制接口令牌；也可用 PHASE_CONTROL_API_KEY")
    parser.add_argument("--api-token", help="对话接口令牌；也可用 PHASE_API_KEY")
    parser.add_argument(
        "--operator",
        choices=sorted(OPERATOR_MODES),
        default="agent",
        help="agent 调用原有模型；codex 禁止模型生成请求，由当前 Codex 接管决策",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "health",
        "status",
        "pending",
        "tasks",
        "config",
        "charts",
        "diagnostics",
        "context",
    ):
        sub.add_parser(name)
    propose = sub.add_parser("propose", help="向现有 Agent 发送一条项目指令")
    propose.add_argument("instruction")
    propose.add_argument("--conversation-id", default="codex-control")
    cont = sub.add_parser("continue", help="继续当前项目")
    cont.add_argument("--conversation-id", default="codex-control")
    decision = sub.add_parser("decide", help="提交已有提案的审批决定")
    decision.add_argument("decision", choices=("approve", "reject"))
    decision.add_argument("--plan-id", required=True)
    decision.add_argument("--state-version", required=True)
    decision.add_argument("--proposal-hash", required=True)
    decision.add_argument("--comment", default="")
    decision.add_argument("--conversation-id", default="codex-control")
    codex_propose = sub.add_parser("codex-propose", help="把 Codex 结构化动作送入原提案验证链")
    codex_propose.add_argument("action_json", help="结构化动作 JSON")
    codex_propose.add_argument("--conversation-id", default="codex-control")
    codex_report = sub.add_parser("codex-report", help="为已有待审方案生成交互记录")
    codex_report.add_argument("plan_id")
    chat = sub.add_parser("chat", help="通过现有 OpenAI-compatible 对话接口发送消息")
    chat.add_argument("message")
    chat.add_argument("--conversation-id", default="codex-control")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    client = _client_from_args(args)
    try:
        if args.command == "health":
            result = client.health()
        elif args.command == "status":
            result = client.status()
        elif args.command == "pending":
            result = client.pending()
        elif args.command == "tasks":
            result = client.tasks()
        elif args.command == "config":
            result = client.config()
        elif args.command == "charts":
            result = client.charts()
        elif args.command == "diagnostics":
            result = client.diagnostics()
        elif args.command == "context":
            result = client.context()
        elif args.command == "propose":
            result = client.propose(args.instruction, conversation_id=args.conversation_id)
        elif args.command == "continue":
            result = client.continue_project(conversation_id=args.conversation_id)
        elif args.command == "codex-propose":
            try:
                action = json.loads(args.action_json)
            except json.JSONDecodeError as error:
                raise CodexControlError(f"结构化动作不是有效 JSON：{error}") from error
            result = client.codex_propose(action, conversation_id=args.conversation_id)
        elif args.command == "codex-report":
            result = client.codex_report(args.plan_id)
        elif args.command == "decide":
            result = client.decide(
                args.decision,
                plan_id=args.plan_id,
                state_version=args.state_version,
                proposal_hash=args.proposal_hash,
                comment=args.comment,
                conversation_id=args.conversation_id,
            )
        else:
            result = client.chat(args.message, conversation_id=args.conversation_id)
    except CodexControlError as error:
        print(
            json.dumps({"status": "error", "message": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
