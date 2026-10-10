"""Local Agent HTTP endpoint and runtime factory.

Chat application logic lives in chat_application; workflow call assembly lives
in chat_execution. This transport never approves model tools by itself.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from phase_agent.module_references import import_saved_module
import json
import os
from pathlib import Path
import threading
import uuid

from copy import deepcopy

from phase_agent.runtime.chat_application import (
    OpenWebUIRequestError,
    RunWorkflowChatHandler,
    classify_user_decision,
    _is_config_migration_approval,
    _mentions_explicit_branch_generation,
    _drop_finished_pending,
    format_status_reply,
    format_history_prompt,
    _latest_user_message,
    _open_webui_metadata_reply,
    _is_status_command,
    _phase_csv_request,
    _classify_history_decision,
    _deepseek_switch_reply,
)
from phase_agent.tools.step_runner.build_status_summary import build_status_summary
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


MODEL_ID = "phase-search-agent"
MAX_REQUEST_BYTES = 1_000_000
MAX_RESPONSE_CHARS = 3_200


from phase_agent.runtime.chat_approval_rules import is_sensitive_proposal as _is_sensitive_proposal


from phase_agent.runtime.chat_state_presentation import (
    brief_chat_state,
)  # Compatible public import.


def handle_chat_request(payload: dict, chat_handler, *, model_id=MODEL_ID) -> dict:
    from phase_agent.runtime.ui_contracts import validate_chat_request, ChatResponse

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
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    return ChatResponse.model_validate(response).model_dump()


def _chat_content(value):
    """Never dump full scientific state or candidate arrays into the chat pane."""
    from phase_agent.runtime.response_preferences import detailed_response

    max_chars = 16000 if detailed_response() else MAX_RESPONSE_CHARS
    if isinstance(value, dict):
        fields = ("status", "tool", "action", "reason", "record_id")
        parts = [
            f"{key}: {str(value[key])[:180]}"
            for key in fields
            if value.get(key) is not None and not isinstance(value[key], (dict, list))
        ]
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


def create_server(
    chat_handler,
    *,
    api_key: str,
    host="127.0.0.1",
    port=8765,
    model_id=MODEL_ID,
    local_control=None,
    control_api_key=None,
    deepseek_key_setup=None,
    deepseek_model="deepseek-flash",
):
    """Compatibility adapter: bind chat functions to the independent HTTP server."""
    from phase_agent.runtime.local_http_server import create_local_http_server

    return create_local_http_server(
        chat_handler,
        api_key=api_key,
        host=host,
        port=port,
        model_id=model_id,
        local_control=local_control,
        control_api_key=control_api_key,
        deepseek_key_setup=deepseek_key_setup,
        deepseek_model=deepseek_model,
        request_handler=handle_chat_request,
        request_error=OpenWebUIRequestError,
        approval_page=_approval_page,
        setup_page=_deepseek_setup_page,
        error_formatter=_error,
        max_request_bytes=MAX_REQUEST_BYTES,
    )


def serve_open_webui(
    chat_handler,
    *,
    api_key,
    host="127.0.0.1",
    port=8765,
    model_id=MODEL_ID,
    local_control=None,
    control_api_key=None,
    deepseek_key_setup=None,
    deepseek_model="deepseek-flash",
    parent_pid=None,
):
    server = create_server(
        chat_handler,
        api_key=api_key,
        host=host,
        port=port,
        model_id=model_id,
        local_control=local_control,
        control_api_key=control_api_key,
        deepseek_key_setup=deepseek_key_setup,
        deepseek_model=deepseek_model,
    )
    if parent_pid is not None:
        threading.Thread(
            target=_stop_server_when_parent_exits,
            args=(server, int(parent_pid)),
            daemon=True,
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


from phase_agent.runtime.workflow_reply_presentation import (
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
<style>body{font-family:system-ui;margin:2rem;max-width:1050px;color:#18212b}input,textarea,button,select{font:inherit;padding:.55rem;margin:.25rem}input{min-width:28rem}.card{border:1px solid #ccd4dd;border-radius:8px;padding:1rem;margin:1rem 0;background:#fff}pre{white-space:pre-wrap;background:#f5f7f9;padding:.8rem;overflow:auto}.charts svg{max-width:100%;height:auto;border:1px solid #eee;margin:.4rem 0}.warn{color:#a33}table{border-collapse:collapse;width:100%;table-layout:fixed}th,td{border:1px solid #ccd4dd;padding:.5rem;text-align:left;overflow-wrap:anywhere}textarea{display:block;width:90%;min-height:4rem}input{max-width:90%;min-width:0}</style></head><body>
<h1>相图项目本地审批</h1><p>审阅具体方案后批准、修改或拒绝。“继续”只推进分析，不代表批准。</p>
<label>本地控制令牌 <input id="token" type="password" autocomplete="off"></label><button onclick="loadAll()">读取</button>
<p id="message" class="warn"></p><section id="recovery"></section><section id="plans"></section><h2>项目生成图表</h2><section id="charts" class="charts"></section>
<script>
const el=id=>document.getElementById(id); const auth=()=>({'Authorization':'Bearer '+el('token').value,'Content-Type':'application/json'});
async function api(path, options={}){const r=await fetch(path,{...options,headers:auth()});const j=await r.json();if(!r.ok)throw Error(j.error?.message||r.status);return j}
function pretty(x){return JSON.stringify(x,null,2)}
async function loadAll(){try{el('message').textContent='';const p=await api('/phase/pending');const c=await api('/phase/charts');const status=await api('/phase/status');renderRecovery(status);renderPlans(p);el('charts').innerHTML='';Object.values(c.charts).forEach(x=>{const d=document.createElement('div');d.className='card';d.innerHTML=x.svg;const v=document.createElement('code');v.textContent='数据版本 '+x.data_version;d.appendChild(v);el('charts').appendChild(d)})}catch(e){el('message').textContent=e.message}}
function renderRecovery(status){
 const root=el('recovery');root.replaceChildren();const heading=document.createElement('h2');heading.textContent=status.waiting_state?.label||'项目状态';root.appendChild(heading);const next=document.createElement('p');next.textContent=status.waiting_state?.next_action||'';root.appendChild(next);
 (status.recovery_report?.checks||[]).forEach(x=>{
 const card=document.createElement('div');card.className='card';const text=document.createElement('p');text.textContent='中断动作 '+(x.identity?.invocation_id||'未记录')+'：已登记完成 '+x.completed_task_ids.length+' 个任务，等待 '+x.pending_task_ids.length+' 个任务。'+x.next_action;card.appendChild(text);
 const detail=document.createElement('details');const summary=document.createElement('summary');summary.textContent='查看文件、任务、执行日志和预算核对证据';detail.appendChild(summary);const pre=document.createElement('pre');pre.textContent=pretty(x);detail.appendChild(pre);card.appendChild(detail);
 if(x.identity?.invocation_id){
 function select(label,choices){const wrap=document.createElement('label');wrap.textContent=label;const input=document.createElement('select');[['','请选择'],...choices].forEach(([value,name])=>{const option=document.createElement('option');option.value=value;option.textContent=name;input.appendChild(option)});wrap.appendChild(input);card.appendChild(wrap);return input}
 const resolution=select('处理方式 ',[['verified_registered_effects','复用已登记任务与结果'],['verified_no_effect','结束已核对无副作用的旧动作']]);
 const jobs=select('外部作业核对 ',[['registered_only','仅有已登记作业'],['none_found','已核对，没有作业'],['not_applicable','此动作不涉及外部作业']]);
 const inventory=select('产出核对 ',[['registered_outputs','已核对登记产出'],['checked_no_outputs','已核对，没有产出']]);
 const cost=document.createElement('input');cost.type='number';cost.min='0';cost.step='any';cost.placeholder='无副作用时填写 0';const costLabel=document.createElement('label');costLabel.textContent='科学计算实际成本 ';costLabel.appendChild(cost);card.appendChild(costLabel);
 const note=document.createElement('textarea');note.placeholder='填写实际核对的目录、任务来源、作业和台账依据；文件缺失不能证明没有副作用。';card.appendChild(note);
 const button=document.createElement('button');button.textContent='生成核对处理方案';button.onclick=async()=>{try{await api('/phase/recovery/propose',{method:'POST',body:JSON.stringify({invocation_id:x.identity.invocation_id,resolution:resolution.value,evidence_note:note.value,external_jobs:jobs.value,output_inventory:inventory.value,external_cost:cost.value===''?null:Number(cost.value)})});await loadAll();el('message').textContent='处理方案已生成，请审阅后确认；尚未解除阻塞。'}catch(e){el('message').textContent=e.message}};card.appendChild(button);
 }root.appendChild(card);
 })
}
function renderPlans(value){
 el('plans').replaceChildren();const heading=document.createElement('h2');heading.textContent='待审方案（'+value.count+'）';el('plans').appendChild(heading);
 value.pending.forEach(p=>{const c=p.review_card||{};const d=document.createElement('div');d.className='card';
 const title=document.createElement('h3');title.textContent=(c.title||p.recommended_action)+' · 修订 '+p.revision;d.appendChild(title);
 function line(label,value){const x=document.createElement('p');x.textContent=label+'：'+value;d.appendChild(x)}
 line('方案',p.plan_id+' · '+p.proposal_hash.slice(0,12));line('目的',c.purpose||p.reason||'未记录');line('理由',c.reason||p.reason||'未记录');
 line('执行范围',c.scope||((c.target_count??p.target_ids.length)+' 个目标'));line('预算',c.budget??'未记录');line('审批边界',c.approval_boundary||'仅本次展示的动作');if(c.blocked)line('暂不可批准',c.blocked);
 if(c.changes?.length){const table=document.createElement('table');const head=table.insertRow();['修改项','原值','新值'].forEach(v=>{const cell=document.createElement('th');cell.textContent=v;head.appendChild(cell)});c.changes.forEach(x=>{const row=table.insertRow();[x.field,pretty(x.before),pretty(x.after)].forEach(v=>{const cell=row.insertCell();cell.textContent=v})});d.appendChild(table);line('修订确认','旧版批准不适用于新版方案')}
 const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='查看目标、成本预估与完整参数依据';details.appendChild(summary);const pre=document.createElement('pre');pre.textContent=pretty(p);details.appendChild(pre);d.appendChild(details);
 const note=document.createElement('textarea');note.placeholder=p.review_kind==='configuration'?'配置修改填写 JSON 路径与值，或在聊天中描述要求':p.review_kind==='execution_recovery'?'审核意见；修改核对依据请重新生成方案':'修改要求或审核意见';note.setAttribute('aria-label','修改要求或审核意见');d.appendChild(note);
 [c.sensitive?'confirm_sensitive':'approve',...(p.review_kind==='execution_recovery'?[]:['modify']),'reject'].forEach(decision=>{const b=document.createElement('button');b.textContent={approve:'批准本方案',modify:'修改后重新审阅',reject:'拒绝',confirm_sensitive:p.review_kind==='execution_recovery'?'确认恢复处理':p.review_kind==='model_activation'?'确认切换模型':'确认敏感操作'}[decision];b.disabled=(decision==='approve'&&c.sensitive)||(!!c.blocked&&['approve','confirm_sensitive'].includes(decision));b.onclick=()=>decide(p,decision,note.value);d.appendChild(b)});el('plans').appendChild(d)})
}
async function decide(p,decision,comment){try{const result=await api('/phase/decision',{method:'POST',body:JSON.stringify({plan_id:p.plan_id,decision,state_version:p.state_version,proposal_hash:p.proposal_hash,comment})});el('message').textContent='处理结果：'+result.status;await loadAll()}catch(e){el('message').textContent=e.message}}
</script></body></html>"""


from phase_agent.runtime.runtime_config_io import (
    _has_history,
    _resolve_path,
    _load_json_object,
    _import_reference,
    _reject_secrets,
    _session_from_state,
)


def create_agent_runtime(config_path=None):
    """Compatibility entry binding chat factories to runtime composition."""
    from phase_agent.runtime.runtime_composition import compose_runtime

    return compose_runtime(
        config_path,
        chat_handler_factory=RunWorkflowChatHandler,
        model_switcher_factory=_make_deepseek_model_switcher,
        runtime_factory=create_agent_runtime,
    )


def _make_deepseek_model_switcher(runtime_config_path, settings, *, system_prompt, thinking):
    """Create a local-only switcher that persists the selected model and replaces the client."""
    from phase_agent.decisions.agent.create_deepseek_client import create_deepseek_client
    from phase_agent.runtime.deepseek_credentials import load_deepseek_api_key
    from phase_agent.runtime.set_deepseek_runtime_model import set_deepseek_runtime_model

    def switch(model):
        from phase_agent.runtime.runtime_client_settings import (
            search_client_settings,
            configuration_client_settings,
        )

        options = {**settings, "model": model}
        parameters = (
            configuration_client_settings(options, system_prompt=system_prompt)
            if system_prompt is not None
            else search_client_settings(options)
        )
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
    from phase_agent.decisions.agent.create_deepseek_client import create_deepseek_client
    from phase_agent.runtime.deepseek_credentials import load_deepseek_api_key
    from phase_agent.runtime.configuration_chat import (
        CONFIG_AGENT_SYSTEM_PROMPT,
        ConfigurationChatHandler,
    )
    from phase_agent.runtime.deepseek_setup import test_and_save_api_key

    def activate(api_key, deepseek, target):
        from phase_agent.runtime.runtime_client_settings import (
            configuration_client_settings,
            search_client_settings,
        )

        is_configuration = isinstance(target, ConfigurationChatHandler)
        if not is_configuration and not isinstance(target, RunWorkflowChatHandler):
            raise ValueError("当前自定义 handler 不支持本地 API Key 设置")
        key = api_key or load_deepseek_api_key()
        parameters = (
            configuration_client_settings(deepseek, system_prompt=CONFIG_AGENT_SYSTEM_PROMPT)
            if is_configuration
            else search_client_settings(deepseek)
        )
        client = create_deepseek_client(api_key=key, **parameters)
        if is_configuration:
            target.agent_client = client
        else:
            target.workflow_kwargs["agent_client"] = client

    def setup(api_key):
        from contextlib import nullcontext

        with getattr(chat_handler, "lock", nullcontext()):
            target = getattr(chat_handler, "config_delegate", None) or chat_handler
            current_path = getattr(
                target,
                "runtime_config_path",
                getattr(chat_handler, "runtime_config_path", runtime_config_path),
            )
            settings = _load_json_object(current_path, "Agent 运行时配置")
            deepseek = settings.get("deepseek") or {}
            return test_and_save_api_key(
                api_key,
                settings=deepseek,
                activate_client=lambda key: activate(key, deepseek, target),
            )

    return setup


def _deepseek_setup_page(model, *, configured):
    from phase_agent.runtime.deepseek_setup import setup_page

    return setup_page(model=model, configured=configured)


def _error(error_type, message):
    return {"error": {"message": message, "type": error_type}}


def _load_factory(reference):
    module_name, separator, name = str(reference or "").partition(":")
    if not separator:
        raise ValueError("handler factory must use package.module:function")
    handler = getattr(import_saved_module(module_name), name)()
    if isinstance(handler, dict):
        return RunWorkflowChatHandler(handler)
    if not callable(handler):
        raise TypeError("factory must return a callable or run_workflow kwargs dict")
    return handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the local Agent to Open WebUI")
    parser.add_argument(
        "--handler-factory",
        default=os.environ.get("PHASE_SEARCH_OPENWEBUI_FACTORY"),
        help="optional module:function override returning a chat handler or run_workflow kwargs",
    )
    parser.add_argument(
        "--runtime-config",
        default=os.environ.get("PHASE_AGENT_RUNTIME_CONFIG", "settings/agent_runtime.json"),
        help="built-in runtime JSON (default: settings/agent_runtime.json)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--decision-backend",
        choices=["langgraph"],
        default="langgraph",
        help="optional proposal validation backend; no autonomous execution",
    )
    parser.add_argument("--token-env", default="OPENWEBUI_TOOL_TOKEN")
    parser.add_argument("--control-token-env", default="OPENWEBUI_CONTROL_TOKEN")
    parser.add_argument(
        "--parent-pid", type=int, help="stop this local Agent automatically when its launcher exits"
    )
    args = parser.parse_args(argv)
    from phase_agent.decisions.agent.decision_backend import check_decision_backend

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
        handler = (
            _load_factory(args.handler_factory)
            if args.handler_factory
            else create_agent_runtime(args.runtime_config)
        )
    except (ValueError, TypeError, FileNotFoundError, ImportError, AttributeError) as error:
        parser.error(str(error))
    print(f"Local Agent endpoint: http://{args.host}:{args.port}/v1")
    # Initial configuration handlers acquire their search handler after confirmation.
    # Bind the backend to those factories as well as already-confirmed handlers.
    from phase_agent.runtime.agent_web_service import configure_decision_backend

    configure_decision_backend(handler, args.decision_backend)
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
        from phase_agent.runtime.local_agent_control import LocalAgentControl

        control = LocalAgentControl(handler)
    serve_open_webui(
        handler,
        api_key=token,
        host=args.host,
        port=args.port,
        local_control=control,
        control_api_key=control_token,
        deepseek_key_setup=deepseek_setup,
        deepseek_model=deepseek_model,
        parent_pid=args.parent_pid,
    )


if __name__ == "__main__":
    main()
