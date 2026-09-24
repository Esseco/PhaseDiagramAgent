"""OpenAI-compatible local chat endpoint for Open WebUI.

The endpoint passes the actual user message to the project's interactive
workflow. It does not expose model-callable tools for approving actions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
import os
from pathlib import Path
import secrets
import threading
import uuid

from copy import deepcopy

from execution_layer.step_runner.build_status_summary import build_status_summary
from execution_layer.step_runner.file_protocol import read_json


MODEL_ID = "phase-search-agent"
MAX_REQUEST_BYTES = 1_000_000
MAX_RESPONSE_CHARS = 24_000


class OpenWebUIRequestError(ValueError):
    """Invalid or ambiguous chat request."""


class RunWorkflowChatHandler:
    """Adapt actual user turns to one-step calls of the existing run_workflow.

    ``workflow_kwargs`` is the existing runtime composition: manager,
    phase_references, confirmed config, Agent client and configured adapters.
    """

    def __init__(self, workflow_kwargs: dict, *, workflow=None, history_prompt=False,
                 new_run_factory=None, deepseek_model_switcher=None):
        if not isinstance(workflow_kwargs, dict) or not workflow_kwargs.get("state_path"):
            raise ValueError("workflow_kwargs must include a persistent state_path")
        self.workflow_kwargs = dict(workflow_kwargs)
        self.state_path = Path(workflow_kwargs["state_path"])
        self.workflow = workflow
        self.lock = threading.RLock()
        self.history_prompt = bool(history_prompt)
        self.history_decision = None if history_prompt else "continue"
        self.new_run_factory = new_run_factory
        self.deepseek_model_switcher = deepseek_model_switcher
        self.conversation_id = None

    def __call__(self, messages, *, conversation_id=None):
        user_message = _latest_user_message(messages)
        with self.lock:
            state = read_json(self.state_path, {}) or {}
            if self.conversation_id not in {None, conversation_id}:
                raise OpenWebUIRequestError(
                    "此本地运行时已绑定另一个 Open WebUI 会话。当前审批状态是进程级单用户状态；"
                    "请使用原会话，或为另一用户启动独立服务和 state。"
                )
            from run.resolve_deepseek_model_request import resolve_deepseek_model_request
            requested_model = resolve_deepseek_model_request(user_message)
            if requested_model:
                if not callable(self.deepseek_model_switcher):
                    return "此运行时未配置 DeepSeek 模型切换接口；未修改任何设置。"
                try:
                    selected_model, client = self.deepseek_model_switcher(requested_model)
                except (OSError, TypeError, ValueError) as error:
                    return f"DeepSeek 模型切换失败：{type(error).__name__}: {error}。运行时设置未更改。"
                self.workflow_kwargs["agent_client"] = client
                return _deepseek_switch_reply(selected_model)
            if self.history_decision is None:
                self.conversation_id = conversation_id
                decision = _classify_history_decision(user_message)
                if decision is None:
                    return format_history_prompt(state, self.workflow_kwargs.get("manager"))
                self.conversation_id = conversation_id
                if decision == "new":
                    if not callable(self.new_run_factory):
                        raise OpenWebUIRequestError("运行时未配置安全的新建运行目录工厂。")
                    replacement = self.new_run_factory()
                    if "agent_client" in self.workflow_kwargs:
                        replacement["agent_client"] = self.workflow_kwargs["agent_client"]
                    self.workflow_kwargs = dict(replacement)
                    self.state_path = Path(replacement["state_path"])
                    state = {}
                    self.history_decision = "new"
                    return f"已新建独立运行：`{self.state_path.parent}`。旧 state/ledger 未修改。请发送下一条搜索指令。"
                self.history_decision = "continue"
                pending = state.get("pending_execution_policies") or {}
                if len(pending) > 1:
                    raise OpenWebUIRequestError(
                        "历史 state 含多个待审批 action，无法安全确定审批目标；"
                        "请先通过离线状态工具恢复为唯一待审批状态。"
                    )
                if pending:
                    proposal = next(iter(pending.values())).get("agent_proposal")
                    return format_workflow_reply(
                        {"status": "awaiting_approval", "agent_proposal": proposal,
                         "config_version": state.get("confirmed_config_version")}, self.state_path
                    )
                return "已继续原运行并加载 state/ledger。请发送下一条搜索指令；本次确认不会批准任何 action。"
            pending = state.get("pending_execution_policies") or {}
            if _is_status_command(user_message):
                return format_status_reply(state)
            if pending:
                if len(pending) != 1:
                    raise OpenWebUIRequestError("存在多个待审批 action；请先恢复到唯一待审批状态。")
                invocation_id = next(iter(pending))
                stored_proposal = pending[invocation_id].get("agent_proposal") or {}
                decision = classify_user_decision(user_message)
                if decision in {"approve", "reject"}:
                    if decision == "approve" and _is_sensitive_proposal(stored_proposal):
                        return (
                            "该建议属于敏感操作。请在本机审批页确认具体影响后批准："
                            "http://127.0.0.1:8765/phase/approval"
                        )
                    from execution_layer.policy.file_approval import proposal_hash
                    state_version = build_status_summary(
                        state, config_version=state.get("confirmed_config_version")
                    )["summary_id"]
                    outcome = self.review_pending(
                        invocation_id,
                        decision,
                        expected_state_version=state_version,
                        expected_proposal_hash=proposal_hash(stored_proposal),
                        comment=user_message,
                    )
                    return format_workflow_reply(outcome.get("result") or {}, self.state_path)
                if _is_sensitive_confirmation(user_message):
                    return ("聊天消息不能批准或拒绝动作。请打开本地审批页核对计划、路径、版本和影响："
                            "http://127.0.0.1:8765/phase/approval")
                feedback = {"decision": decision, "comment": user_message}
            else:
                invocation_id, feedback = f"webui-{uuid.uuid4().hex}", None
            self.conversation_id = conversation_id
            result = self._run(invocation_id, feedback, user_message)
            return format_workflow_reply(result, self.state_path)

    def review_pending(self, plan_id, decision, *, expected_state_version,
                       expected_proposal_hash, comment=""):
        """Accept a decision only from the authenticated local approval surface."""
        from execution_layer.policy.file_approval import proposal_hash
        with self.lock:
            state = read_json(self.state_path, {}) or {}
            completed = (state.get("invocations") or {}).get(plan_id)
            pending = (state.get("pending_execution_policies") or {}).get(plan_id)
            if pending is None and completed is not None:
                return {"status": "already_processed", "result": completed}
            if pending is None:
                raise OpenWebUIRequestError("plan is not pending")
            current_version = build_status_summary(
                state, config_version=state.get("confirmed_config_version"))["summary_id"]
            if expected_state_version != current_version:
                raise OpenWebUIRequestError("state_version_stale; refresh the approval page")
            proposal = pending.get("agent_proposal") or {}
            if expected_proposal_hash != proposal_hash(proposal):
                raise OpenWebUIRequestError("proposal_hash_mismatch; refresh the approval page")
            if decision not in {"approve", "reject", "confirm_sensitive"}:
                raise OpenWebUIRequestError("invalid approval-page decision")
            if _is_sensitive_proposal(proposal) and decision != "confirm_sensitive":
                raise OpenWebUIRequestError("sensitive action requires confirm_sensitive")
            approved = decision in {"approve", "confirm_sensitive"}
            audit_comment = str(comment or "").strip()
            if approved:
                audit_comment = (audit_comment + "\napprove").strip()
            feedback = {"decision": "approve" if approved else "reject", "comment": audit_comment}
            result = self._run(plan_id, feedback, "local approval page")
            return {"status": result.get("status"), "result": result}

    def _run(self, invocation_id, human_feedback, user_message):
        workflow = self.workflow
        if workflow is None:
            from run.main import run_workflow
            workflow = run_workflow
        kwargs = {
            key: value for key, value in self.workflow_kwargs.items()
            if key not in {"state", "execution_mode", "human_feedback", "replay_record",
                           "max_steps", "invocation_id", "state_path", "config_session_path"}
        }
        base_agent_client = kwargs.get("agent_client")
        if callable(base_agent_client):
            def user_contextualized_agent(payload):
                return base_agent_client({**payload, "user_instruction": user_message})
            kwargs["agent_client"] = user_contextualized_agent
        return workflow(
            **kwargs, execution_mode="interactive", human_feedback=human_feedback,
            max_steps=1, invocation_id=invocation_id, state_path=str(self.state_path),
        )


def classify_user_decision(message: str) -> str:
    """Only a final, exact user line can approve or reject; all else is feedback."""
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    final = lines[-1].lower() if lines else ""
    if final in {"approve", "同意"}:
        return "approve"
    if final in {"reject", "拒绝"}:
        return "reject"
    return "comment"


def _is_sensitive_confirmation(message):
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    return bool(lines and lines[-1] in {"确认敏感操作", "confirm sensitive action"})


def _is_sensitive_proposal(proposal):
    action = proposal.get("raw_action") or {}
    tool = action.get("tool") or action.get("stage")
    if tool in {"update_mlip", "activate_model", "change_hard_constraint", "dft_relax"}:
        return True
    decisions = ((action.get("parameters") or {}).get("decisions") or [])
    return any(row.get("action") == "DFT_RELAX" for row in decisions if isinstance(row, dict))


def handle_chat_request(payload: dict, chat_handler, *, model_id=MODEL_ID) -> dict:
    if not isinstance(payload, dict):
        raise OpenWebUIRequestError("request body must be a JSON object")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise OpenWebUIRequestError("messages must be a non-empty list")
    _latest_user_message(messages)
    metadata = payload.get("metadata") or {}
    conversation_id = str(metadata.get("chat_id") or payload.get("user") or "openwebui")[:128]
    content = chat_handler(messages, conversation_id=conversation_id)
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, indent=2, default=str)
    if len(content) > MAX_RESPONSE_CHARS:
        content = content[:MAX_RESPONSE_CHARS] + "\n…（输出已截断；完整记录保存在项目状态文件中）"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()), "model": model_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
    }


def create_server(chat_handler, *, api_key: str, host="127.0.0.1", port=8765, model_id=MODEL_ID,
                  local_control=None, control_api_key=None):
    """Create a dependency-free OpenAI-compatible server, bound locally by default."""
    if not callable(chat_handler):
        raise TypeError("chat_handler must be callable")
    if not isinstance(api_key, str) or len(api_key) < 16:
        raise ValueError("local API bearer key must contain at least 16 characters")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/v1/models" and self._authorized():
                self._json(200, {"object": "list", "data": [
                    {"id": model_id, "object": "model", "created": 0, "owned_by": "local-project"}
                ]})
            elif self.path == "/v1/models":
                self._json(401, _error("unauthorized", "invalid local API key"))
            elif self.path == "/phase/approval":
                self._html(200, _approval_page())
            elif self.path in {"/phase/status", "/phase/pending", "/phase/tasks", "/phase/charts", "/phase/config", "/phase/memory"} and self._control_authorized() and local_control is not None:
                name = self.path.rsplit("/", 1)[-1]
                self._json(200, getattr(local_control, name)())
            elif self.path.startswith("/phase/"):
                self._json(401 if not self._control_authorized() else 404,
                           _error("unauthorized" if not self._control_authorized() else "not_found",
                                  "invalid control API key" if not self._control_authorized() else "endpoint not found"))
            else:
                self._json(404, _error("not_found", "endpoint not found"))

        def do_POST(self):
            if self.path in {"/phase/propose", "/phase/decision", "/phase/pause", "/phase/config/patch", "/phase/config/confirm", "/phase/memory/review"}:
                if not self._control_authorized():
                    self._json(401, _error("unauthorized", "invalid control API key")); return
                if local_control is None:
                    self._json(404, _error("not_found", "local control is disabled")); return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > MAX_REQUEST_BYTES: raise OpenWebUIRequestError("invalid body size")
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    conversation = str(body.get("conversation_id") or "local-control")[:128]
                    if self.path == "/phase/propose":
                        response = local_control.propose(body.get("instruction", ""), conversation_id=conversation)
                    elif self.path == "/phase/decision":
                        response = local_control.decide(
                            body.get("decision"), plan_id=body.get("plan_id"),
                            expected_state_version=body.get("state_version"),
                            expected_proposal_hash=body.get("proposal_hash"),
                            comment=body.get("comment", ""), conversation_id=conversation)
                    elif self.path == "/phase/config/patch":
                        response = local_control.patch_config(body.get("patch"), reasons=body.get("reasons"), impacts=body.get("impacts"))
                    elif self.path == "/phase/config/confirm":
                        response = local_control.confirm_config(explicit=body.get("explicit") is True)
                    elif self.path == "/phase/memory/review":
                        response = local_control.review_memory(body.get("proposal_id"), approved=body.get("approved") is True)
                    else:
                        response = local_control.pause(body.get("reason", ""), conversation_id=conversation)
                    self._json(200, response)
                except (ValueError, KeyError, OpenWebUIRequestError) as error:
                    self._json(400, _error("invalid_request", str(error)))
                return
            if self.path != "/v1/chat/completions":
                self._json(404, _error("not_found", "endpoint not found")); return
            if not self._authorized():
                self._json(401, _error("unauthorized", "invalid local API key")); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise OpenWebUIRequestError("request body is empty or too large")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                response = handle_chat_request(payload, chat_handler, model_id=model_id)
            except (UnicodeDecodeError, json.JSONDecodeError, OpenWebUIRequestError) as error:
                self._json(400, _error("invalid_request", str(error))); return
            except Exception as error:
                self.log_error("local workflow failed (%s)", type(error).__name__)
                self._json(500, _error("workflow_error", f"local Agent workflow failed ({type(error).__name__}); inspect server stderr")); return
            if payload.get("stream"):
                self._stream(response)
            else:
                self._json(200, response)

        def _authorized(self):
            value = self.headers.get("Authorization", "")
            return value.startswith("Bearer ") and secrets.compare_digest(value[7:], api_key)

        def _control_authorized(self):
            value = self.headers.get("Authorization", "")
            return (isinstance(control_api_key, str) and len(control_api_key) >= 16 and
                    value.startswith("Bearer ") and secrets.compare_digest(value[7:], control_api_key))

        def _json(self, status, value):
            body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)

        def _html(self, status, value):
            body = value.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'")
            self.end_headers(); self.wfile.write(body)

        def _stream(self, response):
            base = {key: response[key] for key in ("id", "created", "model")}
            content = response["choices"][0]["message"]["content"]
            chunks = [
                {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]},
                {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            body = "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers(); self.wfile.write(encoded)

    return ThreadingHTTPServer((host, int(port)), Handler)


def serve_open_webui(chat_handler, *, api_key, host="127.0.0.1", port=8765, model_id=MODEL_ID,
                     local_control=None, control_api_key=None):
    server = create_server(chat_handler, api_key=api_key, host=host, port=port, model_id=model_id,
                           local_control=local_control, control_api_key=control_api_key)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def format_workflow_reply(result: dict, state_path) -> str:
    if not isinstance(result, dict):
        return str(result)
    events = result.get("events") or []
    proposal = result.get("agent_proposal") or next(
        (event.get("agent_proposal") for event in reversed(events) if event.get("agent_proposal")), None
    )
    if result.get("status") == "awaiting_approval" and proposal:
        cost = proposal.get("estimated_cost") or {}
        lines = [
            "## Agent action proposal",
            f"状态分析：{_display(proposal.get('current_state_analysis'))}",
            f"建议 action：`{proposal.get('recommended_action')}`",
            f"参数：`{_display(proposal.get('action_parameters'))}`",
            f"选择原因：{proposal.get('reason') or '未提供'}",
            f"下一轮计算量：`{_display(proposal.get('calculation_plan'))}`",
            f"预计成本：`{_display(cost)}`",
            f"预期目的：{proposal.get('expected_purpose') or '未提供'}",
            "", "核对无误后回复“同意”批准本轮；回复“拒绝”拒绝；直接写修改意见可要求 Agent 重新分析。",
        ]
        if _is_sensitive_proposal(proposal):
            lines.extend([
                "该建议涉及敏感操作，仍需在本机审批页确认具体影响：",
                "http://127.0.0.1:8765/phase/approval",
            ])
        return "\n".join(lines)
    state = read_json(state_path, {}) or {}
    summary = build_status_summary(state, config_version=result.get("config_version"))
    fields = ("status", "config_version", "human_feedback", "final_action", "execution_result",
              "scientific_feedback", "reconciled", "batch")
    compact = {key: result[key] for key in fields if key in result}
    compact["agent_proposal"] = proposal
    compact["state_summary"] = summary
    return "工作流返回：\n```json\n" + json.dumps(compact, ensure_ascii=False, indent=2, default=str) + "\n```"


def format_status_reply(state: dict) -> str:
    summary = build_status_summary(state, config_version=state.get("confirmed_config_version"))
    return "当前项目状态：\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n```"


def format_history_prompt(state: dict, manager=None) -> str:
    summary = build_status_summary(state, config_version=state.get("confirmed_config_version"))
    ledger = getattr(manager, "data", {}) or {}
    actions = state.get("action_records") or state.get("decisions") or []
    latest = actions[-1] if actions else None
    compact = {
        "branches": len(ledger.get("branches") or {}),
        "structures": len(ledger.get("structures") or {}),
        "actions": len(actions),
        "latest_action": ({key: latest.get(key) for key in ("status", "record_id", "final_action")
                           if latest.get(key) is not None} if isinstance(latest, dict) else None),
        "tasks": len(state.get("tasks") or state.get("pending_tasks") or []),
        "pending_approvals": len(state.get("pending_execution_policies") or {}),
        "run_status": state.get("run_status") or state.get("status"),
        "config_version": summary.get("config_version"),
    }
    return ("检测到已配置的本地历史：\n```json\n" +
            json.dumps(compact, ensure_ascii=False, indent=2) +
            "\n```\n请明确回复 `继续` 或 `新建`。确认前不会提出或执行 action；`继续` 也不会批准待审批 action。")


def _approval_page():
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>相图项目本地审批</title>
<style>body{font-family:system-ui;margin:2rem;max-width:1050px;color:#18212b}input,textarea,button,select{font:inherit;padding:.55rem;margin:.25rem}input{min-width:28rem}.card{border:1px solid #ccd4dd;border-radius:8px;padding:1rem;margin:1rem 0;background:#fff}pre{white-space:pre-wrap;background:#f5f7f9;padding:.8rem;overflow:auto}.charts svg{max-width:100%;height:auto;border:1px solid #eee;margin:.4rem 0}.warn{color:#a33}</style></head><body>
<h1>相图项目本地审批</h1><p>此页面只连接本机受限接口。聊天中的“同意/继续”不会执行动作。</p>
<label>本地控制令牌 <input id="token" type="password" autocomplete="off"></label><button onclick="loadAll()">读取</button>
<p id="message" class="warn"></p><section id="plans"></section><h2>项目生成图表</h2><section id="charts" class="charts"></section>
<script>
const el=id=>document.getElementById(id); const auth=()=>({'Authorization':'Bearer '+el('token').value,'Content-Type':'application/json'});
async function api(path, options={}){const r=await fetch(path,{...options,headers:auth()});const j=await r.json();if(!r.ok)throw Error(j.error?.message||r.status);return j}
function pretty(x){return JSON.stringify(x,null,2)}
async function loadAll(){try{el('message').textContent='';const p=await api('/phase/pending');const c=await api('/phase/charts');renderPlans(p);el('charts').innerHTML='';Object.values(c.charts).forEach(x=>{const d=document.createElement('div');d.className='card';d.innerHTML=x.svg;const v=document.createElement('code');v.textContent='数据版本 '+x.data_version;d.appendChild(v);el('charts').appendChild(d)})}catch(e){el('message').textContent=e.message}}
function renderPlans(value){el('plans').innerHTML='<h2>待批计划（'+value.count+'）</h2>';value.pending.forEach(p=>{const d=document.createElement('div');d.className='card';const pre=document.createElement('pre');pre.textContent=pretty(p);d.appendChild(pre);const note=document.createElement('textarea');note.placeholder='审核意见（可选）';d.appendChild(note);['approve','reject','confirm_sensitive'].forEach(decision=>{const b=document.createElement('button');b.textContent={approve:'批准',reject:'拒绝',confirm_sensitive:'确认敏感操作'}[decision];b.onclick=()=>decide(p,decision,note.value);d.appendChild(b)});el('plans').appendChild(d)})}
async function decide(p,decision,comment){try{const result=await api('/phase/decision',{method:'POST',body:JSON.stringify({plan_id:p.plan_id,decision,state_version:p.state_version,proposal_hash:p.proposal_hash,comment})});el('message').textContent='处理结果：'+result.status;await loadAll()}catch(e){el('message').textContent=e.message}}
</script></body></html>"""


def _latest_user_message(messages):
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text = "\n".join(item.get("text", "") for item in content
                              if isinstance(item, dict) and item.get("type") in {"text", "input_text"})
            if text.strip():
                return text.strip()
    raise OpenWebUIRequestError("messages must contain a non-empty user message")


def _display(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _is_status_command(message):
    value = " ".join(str(message).strip().lower().split())
    return value in {
        "/status", "/状态", "status", "progress", "状态", "进度", "当前状态",
        "当前进度", "查看状态", "查看进度", "查看预算", "查看相图",
    }


def _classify_history_decision(message):
    value = " ".join(str(message).strip().lower().split())
    if value in {"继续", "continue"}:
        return "continue"
    if value in {"新建", "new", "new run"}:
        return "new"
    return None


def _has_history(state, manager) -> bool:
    ledger = getattr(manager, "data", {}) or {}
    if ledger.get("branches") or ledger.get("structures"):
        return True
    keys = ("action_records", "decisions", "tasks", "pending_tasks", "slurm_batches",
            "event_history", "pending_execution_policies")
    return any(state.get(key) for key in keys)


def _resolve_path(value, base):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _load_json_object(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"{label}不存在：{path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label}必须是 JSON object：{path}")
    return value


def _import_reference(reference, label):
    module_name, separator, name = str(reference or "").partition(":")
    if not separator:
        raise ValueError(f"{label}必须使用 package.module:function")
    value = getattr(importlib.import_module(module_name), name)
    return value() if callable(value) else value


def _reject_secrets(config):
    forbidden = {"api_key", "token", "password", "secret", "private_key"}
    found = []

    def walk(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                field = str(key).lower()
                path = f"{prefix}.{key}" if prefix else str(key)
                if field in forbidden or field.endswith("_password") or field.endswith("_secret"):
                    found.append(path)
                walk(child, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{prefix}[{index}]")

    walk(config)
    if found:
        raise ValueError("运行时 JSON 不得包含密钥字段：" + ", ".join(found))


def _session_from_state(state):
    config = state.get("confirmed_config")
    version = state.get("confirmed_config_version") or state.get("config_version")
    if not isinstance(config, dict) or not version:
        return None
    return {
        "status": "confirmed", "config": deepcopy(config), "dialogue": [],
        "confirmed_snapshot": {
            "config_version": str(version), "config_hash": state.get("confirmed_config_hash"),
            "config": deepcopy(config), "audit": {"valid": True, "errors": []},
        },
    }


def create_open_webui_runtime(config_path=None):
    """Compose the built-in runtime from a local, non-secret JSON file."""
    configured = config_path or os.environ.get("PHASE_SEARCH_OPENWEBUI_CONFIG") or "run/open_webui_runtime.json"
    config_file = Path(configured).resolve()
    settings = _load_json_object(config_file, "Open WebUI 运行时配置")
    _reject_secrets(settings)
    base = config_file.parent
    missing = [key for key in ("state_path", "ledger_path") if not settings.get(key)]
    if missing:
        raise ValueError(f"运行时配置缺少 {missing}；请编辑 {config_file}")
    from config_layer.session.load_config_session import load_config_session
    from decision_layer.agent.create_deepseek_client import create_deepseek_client

    state_path = _resolve_path(settings["state_path"], base)
    ledger_path = _resolve_path(settings["ledger_path"], base)
    phase_path = (_resolve_path(settings["phase_references_path"], base)
                  if settings.get("phase_references_path") else None)
    state = _load_json_object(state_path, "state") if state_path.is_file() else {}

    resolved_session = _resolve_path(
        settings.get("config_session_path", "local/config_session.json"), base
    )
    from config_layer.session.resolve_workspace_paths import default_workspace_storage
    path_keys = {
        "state": ("state_path", True),
        "ledger": ("ledger_path", True),
        "branch_energy_pool_ledger": ("branch_energy_pool_ledger_path", True),
        "phase_diagrams": ("phase_diagram_directory", False),
        "approvals": ("approval_directory", False),
        "approved_batches": ("local_action_directory", False),
        "upload_batches": ("manual_upload.batches_directory", False),
        "new_runs": ("new_runs_directory", False),
    }
    configured_paths = {}
    workspace_candidates = [resolved_session.parent.resolve()]
    for key, (setting_key, is_file) in path_keys.items():
        value = settings.get(setting_key)
        if setting_key == "manual_upload.batches_directory":
            value = (settings.get("manual_upload") or {}).get("batches_directory")
        if value:
            resolved_value = _resolve_path(value, base)
            configured_paths[key] = resolved_value
            workspace_candidates.append(resolved_value.parent if is_file else resolved_value)
    roots = workspace_candidates
    try:
        workspace_root = Path(os.path.commonpath([str(path) for path in roots]))
    except ValueError:
        workspace_root = resolved_session.parent
    workspace_defaults = default_workspace_storage(
        workspace_root, path_overrides=configured_paths,
    )
    if not resolved_session.is_file():
        session = _session_from_state(state)
        if session is None:
            if state:
                raise ValueError(
                    f"{state_path} 有运行数据但没有可恢复的 confirmed config；为避免覆盖，先手动恢复配置会话"
                )
            from config_layer.defaults.default_layered_search_config import default_layered_search_config
            from config_layer.session.create_config_draft import create_config_draft
            from config_layer.session.save_config_session import save_config_session
            from run.configuration_chat import BOOTSTRAP_HINTS
            session = create_config_draft(
                default_layered_search_config(), require_workspace_path=True
            )
            session["bootstrap_hints"] = deepcopy(BOOTSTRAP_HINTS)
            save_config_session(session, resolved_session)
    else:
        session = load_config_session(resolved_session)

    if session.get("status") != "confirmed" and not session.get("setup_stage"):
        # Upgrade pre-path-first drafts without discarding their edits or files.
        session["setup_stage"] = "awaiting_storage_path"
        session.setdefault("default_parameter_prompt", {})["status"] = "skipped"
        session.setdefault("dialogue", []).append({
            "type": "workspace_path_question",
            "role": "assistant",
            "message": (
                "请先发送本地工作区根目录路径；我会展示保存位置，"
                "并在你确认后才生成设置 JSON。"
            ),
        })
        from config_layer.session.save_config_session import save_config_session
        save_config_session(session, resolved_session)

    if (session.get("status") != "confirmed"
            and "storage" not in (session.get("config") or {})
            and not session.get("setup_stage")):
        from config_layer.session.apply_config_revision import apply_config_revision
        from config_layer.session.save_config_session import save_config_session
        session = apply_config_revision(
            session, {"storage": workspace_defaults},
            reasons={"storage": "沿用当前本地配置会话目录作为统一工作区默认值；用户可在确认前调整。"},
            author="workspace_storage_default",
        )
        save_config_session(session, resolved_session)

    if session.get("status") != "confirmed":
        from run.configuration_chat import (
            BOOTSTRAP_HINTS, CONFIG_AGENT_SYSTEM_PROMPT, ConfigurationChatHandler,
            normalize_workspace_path, safe_config_filename,
        )
        editable_path_setting = settings.get("editable_config_draft_path") or "search_config.draft.json"
        editable_config_filename = safe_config_filename(Path(editable_path_setting).name)
        setup_stage = session.get("setup_stage")
        if setup_stage == "json_ready":
            storage = (session.get("config") or {}).get("storage") or {}
            try:
                workspace_root = normalize_workspace_path(
                    storage.get("workspace_root"), base_directory=base,
                )
                saved_path = session.get("editable_config_json_path")
                if saved_path:
                    draft_candidate = normalize_workspace_path(saved_path, base_directory=base)
                else:
                    draft_candidate = workspace_root / editable_config_filename
                if (not draft_candidate.is_relative_to(workspace_root)
                        or draft_candidate.suffix.lower() != ".json"):
                    raise ValueError("设置 JSON 必须位于已确认工作区内")
            except (TypeError, ValueError, OSError):
                # Recover sessions written by older versions that accepted arbitrary
                # multi-line input as a path. Keep unrelated draft fields and files.
                session.pop("editable_config_json_path", None)
                session.pop("pending_workspace_root", None)
                session["setup_stage"] = "awaiting_storage_path"
                session.setdefault("config", {}).pop("storage", None)
                session.setdefault("default_parameter_prompt", {})["status"] = "skipped"
                session.setdefault("dialogue", []).append({
                    "type": "workspace_path_recovery",
                    "role": "assistant",
                    "message": (
                        "检测到上次保存的工作区路径格式无效，已清除该路径并恢复到路径选择。"
                        "没有删除或覆盖任何文件；请重新发送一行本地工作区目录路径。"
                    ),
                })
                from config_layer.session.save_config_session import save_config_session
                save_config_session(session, resolved_session)
                setup_stage = "awaiting_storage_path"

        elif setup_stage == "awaiting_storage_confirmation":
            pending_root = session.get("pending_workspace_root")
            try:
                if not pending_root:
                    raise ValueError("缺少待确认工作区")
                normalize_workspace_path(pending_root, base_directory=base)
            except (TypeError, ValueError, OSError):
                session.pop("pending_workspace_root", None)
                session.pop("editable_config_json_path", None)
                session["setup_stage"] = "awaiting_storage_path"
                from config_layer.session.save_config_session import save_config_session
                save_config_session(session, resolved_session)
                setup_stage = "awaiting_storage_path"

        saved_draft_path = session.get("editable_config_json_path")
        if setup_stage in {"awaiting_storage_path", "awaiting_storage_confirmation"}:
            editable_config_path = None
        elif setup_stage == "json_ready":
            editable_config_path = draft_candidate
        elif saved_draft_path:
            try:
                editable_config_path = normalize_workspace_path(saved_draft_path, base_directory=base)
            except (TypeError, ValueError, OSError):
                editable_config_path = None
        elif not setup_stage:
            if "storage" in (session.get("config") or {}) and settings.get("editable_config_draft_path"):
                editable_config_path = _resolve_path(settings["editable_config_draft_path"], base)
            else:
                session_setting = Path(settings.get("config_session_path", "local/config_session.json"))
                editable_config_path = _resolve_path(
                    session_setting.with_name(editable_config_filename), base
                )
        else:
            editable_config_path = None
        if session.get("status") == "draft" and editable_config_path is not None:
            from config_layer.session.create_editable_config_json import create_editable_config_json
            # Create the template once. A user-edited file is never overwritten on restart.
            try:
                create_editable_config_json(
                    editable_config_path, session.get("config") or {},
                    bootstrap_hints=session.get("bootstrap_hints") or BOOTSTRAP_HINTS,
                    workspace_defaults=workspace_defaults,
                )
            except (OSError, TypeError, ValueError):
                session.pop("editable_config_json_path", None)
                session.pop("pending_workspace_root", None)
                session["setup_stage"] = "awaiting_storage_path"
                session.setdefault("config", {}).pop("storage", None)
                session.setdefault("default_parameter_prompt", {})["status"] = "skipped"
                session.setdefault("dialogue", []).append({
                    "type": "workspace_path_recovery",
                    "role": "assistant",
                    "message": (
                        "上次工作区不可写，已恢复到路径选择。没有删除已有文件；"
                        "请重新发送一行可写的本地工作区目录路径。"
                    ),
                })
                from config_layer.session.save_config_session import save_config_session
                save_config_session(session, resolved_session)
                editable_config_path = None
        deepseek = settings.get("deepseek") or {}
        try:
            agent_client = create_deepseek_client(
                model=deepseek.get("model", "deepseek-v4-pro"),
                base_url=deepseek.get("base_url", "https://api.deepseek.com"),
                max_tokens=int(deepseek.get("max_tokens", 800)),
                timeout=int(deepseek.get("timeout", 60)),
                system_prompt=CONFIG_AGENT_SYSTEM_PROMPT,
                thinking=deepseek.get("configuration_thinking", "disabled"),
            )
        except ValueError:
            agent_client = None
        model_switcher = _make_deepseek_model_switcher(
            config_file, deepseek, system_prompt=CONFIG_AGENT_SYSTEM_PROMPT,
            thinking=deepseek.get("configuration_thinking", "disabled"),
        )
        workflow_kwargs = {
            "state_path": str(state_path), "config_session": session,
            "config_session_path": str(resolved_session),
        }
        return ConfigurationChatHandler(
            workflow_kwargs, config_session_path=resolved_session, base_directory=base,
            phase_references_path=phase_path, editable_config_path=editable_config_path,
                editable_config_filename=editable_config_filename,
                workspace_root_default=workspace_root,
            agent_client=agent_client,
            runtime_factory=lambda: create_open_webui_runtime(config_file),
            current_deepseek_model=deepseek.get("model", "deepseek-v4-pro"),
            deepseek_model_switcher=model_switcher,
        )

    snapshot = session.get("confirmed_snapshot") or {}
    if not snapshot.get("config") or not snapshot.get("config_version"):
        raise ValueError("配置会话不是完整的 confirmed snapshot；不会自动确认配置")

    from config_layer.session.resolve_workspace_paths import resolve_workspace_paths
    effective_config = deepcopy(snapshot["config"])
    if "storage" not in effective_config:
        # Older confirmed snapshots retain their original local workspace by default.
        effective_config["storage"] = workspace_defaults
    resolved_storage = resolve_workspace_paths(effective_config, base_directory=base)
    selected_state_path = resolved_storage["state"]
    selected_ledger_path = resolved_storage["ledger"]
    for label, previous, selected in (
        ("state", state_path, selected_state_path),
        ("ledger", ledger_path, selected_ledger_path),
    ):
        if previous.resolve() != selected.resolve() and previous.exists():
            raise ValueError(
                f"工作区路径变更检测到已有 {label} 文件：{previous}。"
                f"不会自动移动或忽略它；请在配置 JSON 中将 storage.paths.{label} 指回原位置，"
                "或先由用户手动迁移并核对后再继续。"
            )
    state_path, ledger_path = selected_state_path, selected_ledger_path
    state = _load_json_object(state_path, "state") if state_path.is_file() else {}

    configured_references = ((effective_config.get("system") or {}).get("phase_references") or {})
    if phase_path and phase_path.is_file():
        phase_references = _load_json_object(phase_path, "相图参考")
    else:
        phase_references = deepcopy(configured_references)
    if not isinstance(phase_references, dict):
        raise ValueError("相图参考必须是映射")
    phase_references = {
        key: (str(_resolve_path(value, base)) if isinstance(value, str) else value)
        for key, value in phase_references.items()
    }

    from data_layer.ledger.phase_data_manager import PhaseDataManager
    from run.default_run_config import default_run_config
    if ledger_path.is_file():
        manager = PhaseDataManager.load(ledger_path)
    else:
        boundary = ((snapshot["config"].get("system") or {}).get("boundary"))
        if not isinstance(boundary, dict):
            raise ValueError(f"台账不存在且 confirmed config 没有 system.boundary；请配置 {ledger_path}")
        manager = PhaseDataManager(boundary, system_config=effective_config.get("system"))
        manager.save(ledger_path)

    runtime_config = default_run_config()
    runtime_config.update({"state_path": str(state_path), "ledger_path": str(ledger_path)})
    for key in ("structure_directory", "phase_diagram_directory", "work_directory",
                "branch_energy_pool_ledger_path", "approval_directory",
                "local_action_directory", "mlip"):
        if key in settings:
            value = deepcopy(settings[key])
            if key != "mlip" and isinstance(value, str):
                value = str(_resolve_path(value, base))
            runtime_config[key] = value
    runtime_config.update({
        "structure_directory": str(resolved_storage["structures"]),
        "state_path": str(state_path),
        "ledger_path": str(ledger_path),
        "branch_energy_pool_ledger_path": str(resolved_storage["branch_energy_pool_ledger"]),
        "phase_diagram_directory": str(resolved_storage["phase_diagrams"]),
        "work_directory": str(resolved_storage["work"]),
        "approval_directory": str(resolved_storage["approvals"]),
        "local_action_directory": str(resolved_storage["approved_batches"]),
    })
    runtime_config.setdefault("qbc", {})["output_path"] = str(resolved_storage["qbc_results"])
    deepseek = settings.get("deepseek") or {}
    agent_client = create_deepseek_client(
        model=deepseek.get("model", "deepseek-v4-pro"),
        base_url=deepseek.get("base_url", "https://api.deepseek.com"),
        max_tokens=int(deepseek.get("max_tokens", 800)), timeout=int(deepseek.get("timeout", 60)),
        thinking=deepseek.get("thinking"),
    )
    model_switcher = _make_deepseek_model_switcher(
        config_file, deepseek, system_prompt=None, thinking=deepseek.get("thinking"),
    )
    kwargs = {"manager": manager, "phase_references": phase_references,
              "run_config": runtime_config, "config_session": session,
              "state_path": str(state_path), "agent_client": agent_client}
    kwargs["config_session_path"] = str(resolved_session)
    for key in ("dispatcher", "task_runner"):
        reference = settings.get(f"{key}_factory")
        if reference:
            kwargs[key] = _import_reference(reference, f"{key}_factory")
    for key, reference in (settings.get("runtime_adapters") or {}).items():
        kwargs[key] = _import_reference(reference, f"runtime_adapters.{key}")
    manual = settings.get("manual_upload") or {}
    if manual.get("enabled"):
        if kwargs.get("task_runner") is not None:
            raise ValueError("manual_upload 与 task_runner_factory 不能同时配置")
        required = [key for key in ("batches_directory", "worker_command") if not manual.get(key)]
        if required:
            raise ValueError(f"manual_upload 缺少 {required}；不会猜测超算路径或命令")
        if manual.get("submit"):
            raise ValueError("manual_upload 只允许本地生成，submit 必须为 false")
        command = manual["worker_command"]
        if not isinstance(command, list) or not all(isinstance(item, str) and item for item in command):
            raise ValueError("manual_upload.worker_command 必须是非空字符串数组")
        from execution_layer.remote.manual_upload_runner import ManualUploadBatchRunner
        task_preparer = None
        if manual.get("task_preparer_factory"):
            task_preparer = _import_reference(manual["task_preparer_factory"], "manual_upload.task_preparer_factory")
        kwargs["task_runner"] = ManualUploadBatchRunner(
            resolved_storage["upload_batches"], worker_command=command,
            dispatcher=kwargs.get("dispatcher"),
            stage_batch_sizes=manual.get("stage_batch_sizes"),
            stage_profiles=manual.get("stage_profiles"), task_preparer=task_preparer,
        )

    def new_run():
        root = resolved_storage["new_runs"]
        run_dir = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
        new_state = run_dir / "state.json"
        new_ledger = run_dir / "phase_data.json"
        fresh = PhaseDataManager(manager.boundary, system_config=manager.data.get("system_config"))
        run_dir.mkdir(parents=True, exist_ok=False)
        fresh.save(new_ledger)
        initial_state = {
            "confirmed_config": deepcopy(snapshot["config"]),
            "confirmed_config_version": snapshot["config_version"],
        }
        temporary = new_state.with_name(new_state.name + ".tmp")
        temporary.write_text(json.dumps(initial_state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        temporary.replace(new_state)
        fresh_kwargs = dict(kwargs)
        fresh_config = deepcopy(runtime_config)
        fresh_config.update({"state_path": str(new_state), "ledger_path": str(new_ledger)})
        fresh_kwargs.update(manager=fresh, run_config=fresh_config, state_path=str(new_state))
        return fresh_kwargs

    return RunWorkflowChatHandler(
        kwargs, history_prompt=_has_history(state, manager),
        new_run_factory=new_run, deepseek_model_switcher=model_switcher,
    )


def _make_deepseek_model_switcher(runtime_config_path, settings, *, system_prompt, thinking):
    """Create a local-only switcher that persists the selected model and replaces the client."""
    from decision_layer.agent.create_deepseek_client import create_deepseek_client
    from run.set_deepseek_runtime_model import set_deepseek_runtime_model

    def switch(model):
        client = create_deepseek_client(
            model=model,
            base_url=settings.get("base_url", "https://api.deepseek.com"),
            max_tokens=int(settings.get("max_tokens", 800)),
            timeout=int(settings.get("timeout", 60)),
            system_prompt=system_prompt,
            thinking=thinking,
        )
        selected = set_deepseek_runtime_model(runtime_config_path, model)
        settings["model"] = selected
        return selected, client

    return switch


def _deepseek_switch_reply(model):
    label = "DeepSeek V4.1 Flash" if model == "deepseek-flash" else "DeepSeek V4 Pro"
    return (
        f"已将本地 Agent 切换为 {label}（`{model}`），并保存到本地 Open WebUI 运行时配置；"
        "从下一条消息起生效。搜索配置、API Key 和计算任务未修改；未调用计算后端。"
    )


def _error(error_type, message):
    return {"error": {"message": message, "type": error_type}}


def _load_factory(reference):
    module_name, separator, name = str(reference or "").partition(":")
    if not separator:
        raise ValueError("handler factory must use package.module:function")
    handler = getattr(importlib.import_module(module_name), name)()
    if isinstance(handler, dict):
        return RunWorkflowChatHandler(handler)
    if not callable(handler):
        raise TypeError("factory must return a callable or run_workflow kwargs dict")
    return handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the local Agent to Open WebUI")
    parser.add_argument("--handler-factory", default=os.environ.get("PHASE_SEARCH_OPENWEBUI_FACTORY"),
                        help="optional module:function override returning a chat handler or run_workflow kwargs")
    parser.add_argument("--runtime-config", default=os.environ.get("PHASE_SEARCH_OPENWEBUI_CONFIG", "run/open_webui_runtime.json"),
                        help="built-in runtime JSON (default: run/open_webui_runtime.json)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token-env", default="OPENWEBUI_TOOL_TOKEN")
    parser.add_argument("--control-token-env", default="OPENWEBUI_CONTROL_TOKEN")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env)
    if not token or len(token) < 16:
        parser.error(f"set a 16+ character local bearer key in {args.token_env}")
    control_token = os.environ.get(args.control_token_env)
    if not control_token or len(control_token) < 16:
        parser.error(f"set a separate 16+ character control bearer key in {args.control_token_env}")
    try:
        handler = (_load_factory(args.handler_factory) if args.handler_factory
                   else create_open_webui_runtime(args.runtime_config))
    except (ValueError, TypeError, FileNotFoundError, ImportError, AttributeError) as error:
        parser.error(str(error))
    print(f"Open WebUI endpoint: http://{args.host}:{args.port}/v1")
    control = None
    if isinstance(handler, RunWorkflowChatHandler):
        from run.local_agent_control import LocalAgentControl
        control = LocalAgentControl(handler)
    serve_open_webui(handler, api_key=token, host=args.host, port=args.port,
                     local_control=control, control_api_key=control_token)


if __name__ == "__main__":
    main()
