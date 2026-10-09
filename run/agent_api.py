"""Local Agent HTTP endpoint and runtime factory.

Chat application logic lives in chat_application; workflow call assembly lives
in chat_execution. This transport never approves model tools by itself.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import threading
import uuid

from copy import deepcopy

from run.chat_application import (
    OpenWebUIRequestError,
    RunWorkflowChatHandler,
    classify_user_decision,
    _is_config_migration_approval,
    _is_pending_relax_input_proposal,
    _is_pending_mc_proposal,
    _unsafe_debug_calculation,
    _workspace_path_clarification_reply,
    _mentions_explicit_branch_generation,
    _drop_finished_pending,
    _is_sensitive_confirmation,
    format_status_reply,
    format_history_prompt,
    _latest_user_message,
    _open_webui_metadata_reply,
    _display,
    _is_status_command,
    _phase_csv_request,
    _classify_history_decision,
    _is_navigation_command,
    _deepseek_switch_reply,
)
from execution_layer.step_runner.build_status_summary import build_status_summary
from execution_layer.step_runner.file_protocol import read_json, write_json


MODEL_ID = "phase-search-agent"
MAX_REQUEST_BYTES = 1_000_000
MAX_RESPONSE_CHARS = 3_200











from run.chat_approval_rules import is_sensitive_proposal as _is_sensitive_proposal


from run.chat_state_presentation import brief_chat_state  # Compatible public import.


def handle_chat_request(payload: dict, chat_handler, *, model_id=MODEL_ID) -> dict:
    from run.ui_contracts import validate_chat_request, ChatResponse
    try:
        validate_chat_request(payload)
    except ValueError as error:
        raise OpenWebUIRequestError(str(error)) from None
    if not isinstance(payload, dict):
        raise OpenWebUIRequestError("request body must be a JSON object")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise OpenWebUIRequestError("messages must be a non-empty list")
    _latest_user_message(messages)
    metadata = payload.get("metadata") or {}
    conversation_id = str(metadata.get("chat_id") or payload.get("user") or "openwebui")[:128]
    content = _chat_content(chat_handler(messages, conversation_id=conversation_id))
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()), "model": model_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
    }
    return ChatResponse.model_validate(response).model_dump()


def _chat_content(value):
    """Never dump full scientific state or candidate arrays into the chat pane."""
    from run.response_preferences import detailed_response
    max_chars = 16000 if detailed_response() else MAX_RESPONSE_CHARS
    if isinstance(value, dict):
        fields = ("status", "tool", "action", "reason", "record_id")
        parts = [f"{key}: {str(value[key])[:180]}" for key in fields
                 if value.get(key) is not None and not isinstance(value[key], (dict, list))]
        return "结果摘要：" + ("；".join(parts) if parts else "详细数据请查看项目记录。")
    content = value if isinstance(value, str) else str(value)
    if "入选结构（Na/O₂；相；Ehull eV/atom）" in content and len(content) <= 16000:
        return content  # Bounded, intentional list of at most 100 approved candidates.
    if len(content) <= max_chars:
        return content
    stripped = content.lstrip()
    if stripped.startswith(("{", "[", "```json")) or '"composition"' in content:
        return "本轮回复含大量内部候选明细，已从聊天窗口省略；请查看项目状态或审批文件。"
    return content[:max_chars].rsplit("\n", 1)[0] + "\n…（详细内容请查看项目记录）"


def create_server(chat_handler, *, api_key: str, host="127.0.0.1", port=8765, model_id=MODEL_ID,
                  local_control=None, control_api_key=None, deepseek_key_setup=None,
                  deepseek_model="deepseek-flash"):
    """Compatibility adapter: bind chat functions to the independent HTTP server."""
    from run.local_http_server import create_local_http_server
    return create_local_http_server(
        chat_handler, api_key=api_key, host=host, port=port, model_id=model_id,
        local_control=local_control, control_api_key=control_api_key,
        deepseek_key_setup=deepseek_key_setup, deepseek_model=deepseek_model,
        request_handler=handle_chat_request, request_error=OpenWebUIRequestError,
        approval_page=_approval_page, setup_page=_deepseek_setup_page,
        error_formatter=_error, max_request_bytes=MAX_REQUEST_BYTES,
    )


def serve_open_webui(chat_handler, *, api_key, host="127.0.0.1", port=8765, model_id=MODEL_ID,
                     local_control=None, control_api_key=None, deepseek_key_setup=None,
                     deepseek_model="deepseek-flash", parent_pid=None):
    server = create_server(chat_handler, api_key=api_key, host=host, port=port, model_id=model_id,
                           local_control=local_control, control_api_key=control_api_key,
                           deepseek_key_setup=deepseek_key_setup, deepseek_model=deepseek_model)
    if parent_pid is not None:
        threading.Thread(
            target=_stop_server_when_parent_exits,
            args=(server, int(parent_pid)), daemon=True,
        ).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _stop_server_when_parent_exits(server, parent_pid):
    """Bind a launcher-owned Agent to that launcher's lifetime."""
    if os.name == "nt":
        import ctypes
        synchronize = 0x00100000
        infinite = 0xFFFFFFFF
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, parent_pid)
        if not handle:
            server.shutdown()
            return
        try:
            ctypes.windll.kernel32.WaitForSingleObject(handle, infinite)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
        server.shutdown()
        return
    while True:
        try:
            os.kill(parent_pid, 0)
        except OSError:
            server.shutdown()
            return
        threading.Event().wait(1.0)


from run.workflow_reply_presentation import (
    format_workflow_reply,
    _format_workflow_reply_verbose,
    _is_model_failure_proposal,
    _friendly_validation_errors,
    _friendly_workflow_rejection,
    _proposal_directory,
)  # Keep existing imports compatible while presentation owns these functions.





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





from run.runtime_config_io import (
    _has_history, _resolve_path, _load_json_object, _import_reference,
    _reject_secrets, _session_from_state,
)


def create_agent_runtime(config_path=None):
    """Compatibility entry binding chat factories to runtime composition."""
    from run.runtime_composition import compose_runtime
    return compose_runtime(
        config_path, chat_handler_factory=RunWorkflowChatHandler,
        model_switcher_factory=_make_deepseek_model_switcher,
        runtime_factory=create_agent_runtime,
    )


def _make_deepseek_model_switcher(runtime_config_path, settings, *, system_prompt, thinking):
    """Create a local-only switcher that persists the selected model and replaces the client."""
    from decision_layer.agent.create_deepseek_client import create_deepseek_client
    from run.deepseek_credentials import load_deepseek_api_key
    from run.set_deepseek_runtime_model import set_deepseek_runtime_model

    def switch(model):
        from run.runtime_client_settings import search_client_settings, configuration_client_settings
        options = {**settings, "model": model}
        parameters = (configuration_client_settings(options, system_prompt=system_prompt)
                      if system_prompt is not None else search_client_settings(options))
        parameters["thinking"] = thinking
        client = create_deepseek_client(
            api_key=load_deepseek_api_key(),
            **parameters,
        )
        selected = set_deepseek_runtime_model(runtime_config_path, model)
        settings["model"] = selected
        return selected, client

    return switch


def _make_deepseek_key_setup(chat_handler, runtime_config_path):
    """Build a local key tester that securely saves and activates a successful key."""
    from decision_layer.agent.create_deepseek_client import create_deepseek_client
    from run.deepseek_credentials import load_deepseek_api_key
    from run.configuration_chat import CONFIG_AGENT_SYSTEM_PROMPT, ConfigurationChatHandler
    from run.deepseek_setup import test_and_save_api_key

    def activate(api_key, deepseek, target):
        from run.runtime_client_settings import configuration_client_settings, search_client_settings, intent_client_settings
        is_configuration = isinstance(target, ConfigurationChatHandler)
        if not is_configuration and not isinstance(target, RunWorkflowChatHandler):
            raise ValueError("当前自定义 handler 不支持本地 API Key 设置")
        key = api_key or load_deepseek_api_key()
        parameters = (configuration_client_settings(deepseek, system_prompt=CONFIG_AGENT_SYSTEM_PROMPT)
                      if is_configuration else search_client_settings(deepseek))
        client = create_deepseek_client(api_key=key, **parameters)
        if is_configuration:
            target.agent_client = client
        else:
            from run.runtime_client_settings import CONFIG_INTENT_PROMPT
            intent_client = create_deepseek_client(api_key=key,
                **intent_client_settings(deepseek, system_prompt=CONFIG_INTENT_PROMPT))
            target.workflow_kwargs["agent_client"] = client
            target.config_intent_client = intent_client

    def setup(api_key):
        from contextlib import nullcontext
        with getattr(chat_handler, "lock", nullcontext()):
            target = getattr(chat_handler, "config_delegate", None) or chat_handler
            current_path = getattr(target, "runtime_config_path",
                                   getattr(chat_handler, "runtime_config_path", runtime_config_path))
            settings = _load_json_object(current_path, "Agent 运行时配置")
            deepseek = settings.get("deepseek") or {}
            return test_and_save_api_key(api_key, settings=deepseek,
                                        activate_client=lambda key: activate(key, deepseek, target))

    return setup


def _deepseek_setup_page(model, *, configured):
    from run.deepseek_setup import setup_page
    return setup_page(model=model, configured=configured)





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
    parser.add_argument("--runtime-config", default=os.environ.get("PHASE_AGENT_RUNTIME_CONFIG", "run/agent_runtime.json"),
                        help="built-in runtime JSON (default: run/agent_runtime.json)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--decision-backend", choices=["langgraph"], default="langgraph",
                        help="optional proposal validation backend; no autonomous execution")
    parser.add_argument("--token-env", default="OPENWEBUI_TOOL_TOKEN")
    parser.add_argument("--control-token-env", default="OPENWEBUI_CONTROL_TOKEN")
    parser.add_argument("--parent-pid", type=int,
                        help="stop this local Agent automatically when its launcher exits")
    args = parser.parse_args(argv)
    from decision_layer.agent.decision_backend import check_decision_backend
    try:
        check_decision_backend(args.decision_backend)
    except ValueError as error:
        parser.error(str(error))
    token = os.environ.get(args.token_env)
    if not token or len(token) < 16:
        parser.error(f"set a 16+ character local bearer key in {args.token_env}")
    control_token = os.environ.get(args.control_token_env)
    if not control_token or len(control_token) < 16:
        parser.error(f"set a separate 16+ character control bearer key in {args.control_token_env}")
    try:
        handler = (_load_factory(args.handler_factory) if args.handler_factory
                   else create_agent_runtime(args.runtime_config))
    except (ValueError, TypeError, FileNotFoundError, ImportError, AttributeError) as error:
        parser.error(str(error))
    print(f"Local Agent endpoint: http://{args.host}:{args.port}/v1")
    if isinstance(handler, RunWorkflowChatHandler):
        from run.agent_web_service import configure_decision_backend
        configure_decision_backend(handler, args.decision_backend)
    elif args.decision_backend != "legacy":
        parser.error("custom handler does not support decision backend selection")
    print(f"Decision backend: {args.decision_backend}")
    deepseek_setup = None
    deepseek_model = "deepseek-flash"
    if not args.handler_factory:
        runtime_settings = _load_json_object(args.runtime_config, "Agent 运行时配置")
        deepseek_model = (runtime_settings.get("deepseek") or {}).get("model", deepseek_model)
        deepseek_setup = _make_deepseek_key_setup(handler, args.runtime_config)
        print(f"本机 DeepSeek 设置页: http://127.0.0.1:{args.port}/phase/setup")
    control = None
    if isinstance(handler, RunWorkflowChatHandler):
        from run.local_agent_control import LocalAgentControl
        control = LocalAgentControl(handler)
    serve_open_webui(handler, api_key=token, host=args.host, port=args.port,
                     local_control=control, control_api_key=control_token,
                     deepseek_key_setup=deepseek_setup, deepseek_model=deepseek_model,
                     parent_pid=args.parent_pid)


if __name__ == "__main__":
    main()
