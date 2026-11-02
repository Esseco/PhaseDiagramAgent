"""Small stdio MCP adapter for Codex takeover.

The adapter is deliberately thin: it exposes the existing Codex control
surface and never edits project state or calls a scientific tool directly.
Every proposal is sent through ``CodexOffline`` and therefore keeps the
Agent validation, approval, execution and memory lifecycle.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

from phase_agent.runtime.codex.offline import CodexOffline
from phase_agent.runtime.codex.team import TOOLS as TEAM_TOOLS, TOOL_NAMES, call_team_tool


_TOOLS = (
    {
        "name": "phase_recover",
        "description": "通过原入口回收和核验已登记结果，完成训练交接；不调用项目模型、不激活、不提交。新咨询前有回传结果时先调用。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_status",
        "description": "读取已保存项目状态；不回收、不迁移、不启动模型或计算。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_hpc_recovery",
        "description": "识别已登记的HPC启动/I/O故障，读取仅缺失项补算范围及原resume入口预检命令；只读，不提交。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_review_relax_mc_return",
        "description": "核对已登记Relax/MC包是否满足整批正式回收条件；quarantine=true按用户明确指令隔离旧版提前接纳的不完整批次，保留原结果和台账，不计算不提交。",
        "inputSchema": {
            "type": "object",
            "required": ["bundle_id"],
            "properties": {"bundle_id": {"type": "string"}, "quarantine": {"type": "boolean"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_set_remote_workspace",
        "description": "登记或修改当前项目的Linux远端工作根目录；仅本地元数据，不搬移、上传或提交。",
        "inputSchema": {
            "type": "object",
            "required": ["remote_root"],
            "properties": {"remote_root": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_propose_recovery",
        "description": "核对中断动作的登记产出并生成恢复审核方案，不重跑科学动作。",
        "inputSchema": {
            "type": "object",
            "required": [
                "invocation_id",
                "resolution",
                "evidence_note",
                "external_jobs",
                "output_inventory",
            ],
            "properties": {
                "invocation_id": {"type": "string"},
                "resolution": {"type": "string"},
                "evidence_note": {"type": "string"},
                "external_jobs": {"type": "string"},
                "output_inventory": {"type": "string"},
                "external_cost": {"type": ["number", "null"]},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_refresh_dft_candidates",
        "description": "通过项目Agent刷新当前MLIP相图的DFT候选帧；不选择、准备输入或执行任务。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_context",
        "description": "读取当前项目的 Codex 决策上下文、合法 action 和审批要求。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_pending",
        "description": "读取当前待审批方案；不会批准或执行。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_memory",
        "description": "读取 Agent 已保存的长短期记忆、经验候选和记忆审批队列；只读。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_propose",
        "description": "提交一个结构化 Codex action 到原有验证和审批流程。",
        "inputSchema": {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {"type": "object"},
                "conversation_id": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_decide",
        "description": "对已有待审批方案作出批准、拒绝或修改决定。",
        "inputSchema": {
            "type": "object",
            "required": ["decision", "plan_id", "state_version", "proposal_hash"],
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["approve", "reject", "confirm_sensitive", "modify"],
                },
                "plan_id": {"type": "string"},
                "state_version": {"type": "string"},
                "proposal_hash": {"type": "string"},
                "comment": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_report",
        "description": "为已有待审批方案生成可读报告；不会执行。",
        "inputSchema": {
            "type": "object",
            "required": ["plan_id"],
            "properties": {"plan_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_plot",
        "description": "使用已有数据绘制 MLIP；refresh_analysis=true 刷新各 epoch 分析归档、双图和 latest，不改变科学状态。",
        "inputSchema": {
            "type": "object",
            "properties": {"refresh_analysis": {"type": "boolean"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_list_projects",
        "description": "列出配置根目录下可供 Codex 接管的项目；只读，不修改项目。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_select_project",
        "description": "选择一个 agent_runtime.json 作为当前对话的项目；不会执行科学动作。",
        "inputSchema": {
            "type": "object",
            "required": ["runtime_config"],
            "properties": {"runtime_config": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_current_project",
        "description": "读取当前对话已选择的项目；只读。",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "phase_create_project",
        "description": "在指定路径创建新的隔离项目配置并自动选中；只创建配置，不启动计算或提交作业。",
        "inputSchema": {
            "type": "object",
            "required": ["workspace_root"],
            "properties": {
                "workspace_root": {"type": "string"},
                "execution_mode": {"type": "string", "enum": ["debug", "automatic"]},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "phase_takeover_repo",
        "description": "确认当前 Codex MCP 已接管指定源码 repo；只读，不选择或修改科学项目。",
        "inputSchema": {
            "type": "object",
            "properties": {"repo_root": {"type": "string"}},
            "additionalProperties": False,
        },
    },
)


_TOOLS = (*_TOOLS, *TEAM_TOOLS)


_ANALYSIS_TOOLS = frozenset(
    {
        "phase_plot",
        "phase_list_projects",
        "phase_select_project",
        "phase_current_project",
        "phase_takeover_repo",
    }
)


_READ_ONLY_TOOLS = frozenset(
    {
        "phase_status",
        "phase_pending",
        "phase_memory",
        "phase_list_projects",
        "phase_select_project",
        "phase_current_project",
    }
)


class CodexMcpServer:
    """JSON-RPC dispatcher used by the stdio MCP entry point and tests."""

    def __init__(
        self,
        codex: CodexOffline | None = None,
        *,
        runtime_config: str | os.PathLike[str] | None = None,
        project_roots: Iterable[str | os.PathLike[str]] = (),
        codex_factory: Callable[[str], CodexOffline] = CodexOffline,
        project_creator: Callable[[str, str], str | os.PathLike[str]] | None = None,
        repo_root: str | os.PathLike[str] | None = None,
        analysis_only: bool = False,
        read_only: bool = False,
    ):
        self.codex = codex
        self.analysis_only = analysis_only
        self.read_only = read_only
        if read_only and analysis_only:
            raise ValueError("read_only and analysis_only are separate access modes")
        if read_only:
            from phase_agent.runtime.saved_readonly_project import SavedReadOnlyProject

            codex_factory = SavedReadOnlyProject
        if analysis_only:
            from phase_agent.runtime.saved_analysis_project import SavedAnalysisProject

            codex_factory = SavedAnalysisProject
        self._codex_factory = codex_factory
        self._project_creator = project_creator or self._default_project_creator
        self.repo_root = Path(repo_root).expanduser().resolve() if repo_root else None
        self.runtime_config: Path | None = None
        self.project_roots = tuple(
            Path(root).expanduser().resolve() for root in project_roots if str(root).strip()
        )
        if runtime_config is not None:
            self._select_project(runtime_config)

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        request_id = message.get("id")
        if request_id is None and method not in {
            "notifications/initialized",
            "notifications/cancelled",
        }:
            return None
        if method == "initialize":
            return self._result(
                request_id,
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "phase-agent-codex", "version": "1.0"},
                },
            )
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            return self._result(
                request_id,
                {
                    "tools": [
                        t
                        for t in _TOOLS
                        if (not self.analysis_only or t["name"] in _ANALYSIS_TOOLS)
                        and (not self.read_only or t["name"] in _READ_ONLY_TOOLS)
                    ]
                },
            )
        if method == "tools/call":
            try:
                params = message.get("params") or {}
                result = self._call_tool(
                    str(params.get("name") or ""), params.get("arguments") or {}
                )
                return self._result(
                    request_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, ensure_ascii=False, default=str),
                            }
                        ]
                    },
                )
            except LookupError as error:
                return self._error(request_id, -32601, str(error))
            except Exception as error:
                return self._error(request_id, -32000, str(error))
        return self._error(request_id, -32601, f"method not found: {method}")

    def _call_tool(self, name: str, args: dict[str, Any]) -> Any:
        if self.read_only and name not in _READ_ONLY_TOOLS:
            raise ValueError("read_only mode permits only saved status, memory and pending views")
        if self.analysis_only and name not in _ANALYSIS_TOOLS:
            raise ValueError("analysis_only mode does not permit workflow or configuration actions")
        if name == "phase_list_projects":
            return self._list_projects()
        if name == "phase_select_project":
            return self._select_project(args.get("runtime_config", ""))
        if name == "phase_current_project":
            return self._current_project()
        if name == "phase_create_project":
            return self._create_project(
                args.get("workspace_root", ""), args.get("execution_mode", "debug")
            )
        if name == "phase_takeover_repo":
            return self._takeover_repo(args.get("repo_root"))
        self._require_project()
        if name == "phase_recover":
            return self.codex.recover()
        if name == "phase_status":
            return self.codex.control.status()
        if name in TOOL_NAMES:
            return call_team_tool(self.codex.control, name, args)
        if name == "phase_set_remote_workspace":
            return self.codex.set_remote_workspace(args["remote_root"])
        if name == "phase_propose_recovery":
            return self.codex.control.propose_recovery(args)
        if name == "phase_hpc_recovery":
            return self.codex.control.hpc_startup_recovery()
        if name == "phase_review_relax_mc_return":
            quarantine = args.get("quarantine", False)
            if not isinstance(quarantine, bool):
                raise ValueError("quarantine must be boolean")
            return self.codex.control.review_relax_mc_return(
                args["bundle_id"], quarantine=quarantine
            )
        if name == "phase_refresh_dft_candidates":
            return self.codex.control.refresh_dft_candidates()
        if name == "phase_context":
            return self.codex.context()
        if name == "phase_pending":
            return self.codex.pending()
        if name == "phase_memory":
            return self.codex.control.memory()
        if name == "phase_propose":
            return self.codex.propose(
                args["action"], conversation_id=args.get("conversation_id", "codex-mcp")
            )
        if name == "phase_decide":
            return self.codex.decide(
                args["decision"],
                plan_id=args["plan_id"],
                state_version=args["state_version"],
                proposal_hash=args["proposal_hash"],
                comment=args.get("comment", ""),
            )
        if name == "phase_report":
            return self.codex.control.codex_report(args["plan_id"])
        if name == "phase_plot":
            refresh = args.get("refresh_analysis", False)
            if not isinstance(refresh, bool):
                raise ValueError("refresh_analysis must be boolean")
            if refresh:
                return self.codex.control.codex_plot(refresh_analysis=True)
            return self.codex.control.codex_plot()
        raise LookupError(f"tool not found: {name}")

    def _require_project(self) -> None:
        if self.codex is None:
            raise RuntimeError("尚未选择项目；请先调用 phase_list_projects 或 phase_select_project")

    def _select_project(self, runtime_config: str | os.PathLike[str]) -> dict[str, Any]:
        value = str(runtime_config or "").strip()
        if not value:
            raise ValueError("runtime_config 不能为空")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"项目运行配置不存在：{path}")
        if path.name.lower() != "agent_runtime.json":
            raise ValueError("只能选择名为 agent_runtime.json 的项目配置")
        codex = self._codex_factory(str(path))
        self.codex = codex
        self.runtime_config = path
        return self._current_project()

    def _current_project(self) -> dict[str, Any]:
        if self.runtime_config is None:
            return {"selected": False, "runtime_config": None, "name": None}
        return {
            "selected": True,
            "runtime_config": str(self.runtime_config),
            "name": self.runtime_config.parent.name,
        }

    @staticmethod
    def _default_project_creator(workspace_root: str, execution_mode: str) -> str:
        from phase_agent.runtime.local_project_launcher import create_project

        return str(create_project(Path(workspace_root), execution_mode=execution_mode))

    def _create_project(self, workspace_root: str, execution_mode: str) -> dict[str, Any]:
        value = str(workspace_root or "").strip()
        if not value:
            raise ValueError("workspace_root 不能为空")
        if execution_mode not in {"debug", "automatic"}:
            raise ValueError("execution_mode 必须是 debug 或 automatic")
        runtime_config = self._project_creator(value, execution_mode)
        selected = self._select_project(runtime_config)
        return {"created": True, "execution_mode": execution_mode, **selected}

    def _takeover_repo(self, requested_repo: str | None) -> dict[str, Any]:
        path = Path(requested_repo).expanduser().resolve() if requested_repo else self.repo_root
        if path is None:
            raise ValueError("MCP 未绑定 repo；请在启动参数中传入 --repo-root")
        if not path.is_dir():
            raise FileNotFoundError(f"repo 不存在：{path}")
        required = (path / "phase_agent", path / "codex-agent.cmd")
        missing = [str(item) for item in required if not item.exists()]
        if missing:
            raise ValueError(f"不是可接管的 phase_agent repo，缺少：{', '.join(missing)}")
        from copy import deepcopy
        from phase_agent.runtime.codex.context import HPC_APPROVAL_POLICY

        self.repo_root = path
        from phase_agent.runtime.codex.team import native_team_workflow

        return {
            "codex_team_workflow": native_team_workflow(),
            "hpc_approval_policy": deepcopy(HPC_APPROVAL_POLICY),
            "status": "ready",
            "repo_root": str(path),
            "entrypoint": str(path / "codex-mcp.cmd"),
            "next": ["phase_list_projects", "phase_select_project", "phase_create_project"],
        }

    def _list_projects(self) -> dict[str, Any]:
        candidates: set[Path] = set()
        for root in self.project_roots:
            if root.is_file() and root.name.lower() == "agent_runtime.json":
                candidates.add(root)
                continue
            if not root.is_dir():
                continue
            direct = root / "agent_runtime.json"
            if direct.is_file():
                candidates.add(direct)
            try:
                children = tuple(root.iterdir())
            except OSError:
                continue
            for child in children:
                if child.is_dir():
                    candidate = child / "agent_runtime.json"
                    if candidate.is_file():
                        candidates.add(candidate)
        projects = []
        for path in sorted(candidates, key=lambda item: str(item).lower()):
            projects.append(
                {
                    "name": path.parent.name,
                    "runtime_config": str(path.resolve()),
                    "selected": path.resolve() == self.runtime_config,
                }
            )
        return {"projects": projects, "roots": [str(root) for root in self.project_roots]}

    @staticmethod
    def _result(request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通过 stdio MCP 暴露项目 Codex 接管入口")
    parser.add_argument("--runtime-config")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--analysis-only",
        action="store_true",
        help="只读加载已有数据；仅开放分析绘图与归档，不初始化或迁移科研运行时",
    )
    modes.add_argument(
        "--read-only",
        action="store_true",
        help="只读取已保存状态、记忆和待审方案，不初始化/迁移运行时或生成文件",
    )
    parser.add_argument(
        "--project-root",
        action="append",
        default=[],
        help="可发现项目的根目录；可重复传入多个目录",
    )
    parser.add_argument("--repo-root", help="Codex 接管的 phase_agent 源码 repo 根目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # Windows consoles commonly default to GBK; MCP payloads contain Chinese
    # project evidence and must always use UTF-8 JSON in both directions.
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    project_roots = list(args.project_root)
    if not project_roots:
        project_roots = [
            value for value in os.environ.get("PHASE_PROJECT_ROOTS", "").split(os.pathsep) if value
        ]
    # Scientific libraries may print progress; stdout is reserved for JSON-RPC.
    from contextlib import redirect_stdout

    protocol_stdout = sys.stdout
    with redirect_stdout(sys.stderr):
        server = CodexMcpServer(
            runtime_config=args.runtime_config,
            project_roots=project_roots,
            repo_root=args.repo_root,
            analysis_only=args.analysis_only,
            read_only=args.read_only,
        )
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            with redirect_stdout(sys.stderr):
                response = server.handle(message)
            if response is not None:
                protocol_stdout.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
                protocol_stdout.flush()
        except Exception as error:
            response = CodexMcpServer._error(None, -32700, str(error))
            protocol_stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            protocol_stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
