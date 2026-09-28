"""Small browser chat for times when the optional Open WebUI service is offline."""


def local_chat_page() -> str:
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>相图搜索 Agent</title>
<style>
body{font:16px/1.55 system-ui,sans-serif;margin:0;background:#f6f8fb;color:#17212b}
main{max-width:860px;margin:auto;padding:20px}h1{font-size:1.35rem;margin:0 0 4px}
.hint{color:#526273;font-size:.9rem}.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:14px 0}
input,textarea,button{font:inherit}input{flex:1;min-width:210px;padding:9px;border:1px solid #bdc8d3;border-radius:7px}
button{border:0;background:#1463ad;color:white;border-radius:7px;padding:9px 14px;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}a{color:#1463ad}
#messages{background:white;border:1px solid #d9e1e8;border-radius:10px;min-height:46vh;max-height:62vh;overflow:auto;padding:16px}
.message{white-space:pre-wrap;margin:0 0 14px;padding:10px 12px;border-radius:8px;max-width:92%}
.user{background:#e7f1fb;margin-left:auto}.assistant{background:#f0f2f5}
form{display:flex;gap:8px;margin-top:12px}textarea{flex:1;resize:vertical;min-height:58px;padding:10px;border:1px solid #bdc8d3;border-radius:7px}
</style></head><body><main>
<h1>相图搜索 Agent</h1>
<div class="hint">本地简洁对话页。Open WebUI 未启动时仍可继续当前项目；计算审批规则保持不变。</div>
<div class="bar"><input id="token" type="password" placeholder="本地连接密钥（启动器已复制到剪贴板）" autocomplete="off">
<button id="paste" type="button">从剪贴板连接</button><span id="status" class="hint"></span></div>
<div id="messages" aria-live="polite"></div>
<form id="form"><textarea id="input" placeholder="和 Agent 对话；Enter 发送，Shift+Enter 换行" required></textarea>
<button id="send" type="submit">发送</button></form>
<p class="hint"><a href="/phase/setup" target="_blank" rel="noopener">设置 DeepSeek Key</a> ·
<a href="/phase/approval" target="_blank" rel="noopener">查看审批</a></p>
</main><script>
const token=document.getElementById('token'), status=document.getElementById('status');
const messages=document.getElementById('messages'), input=document.getElementById('input');
const send=document.getElementById('send');
token.value=sessionStorage.getItem('phaseAgentToolToken')||'';
const chatId=crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
function add(role,content){const box=document.createElement('div');box.className='message '+role;
  box.textContent=content;messages.appendChild(box);messages.scrollTop=messages.scrollHeight;return box;}
document.getElementById('paste').onclick=async()=>{try{token.value=await navigator.clipboard.readText();
  sessionStorage.setItem('phaseAgentToolToken',token.value);status.textContent='连接密钥已就绪';}
  catch(e){status.textContent='请手动粘贴启动器复制的连接密钥';token.focus();}};
token.onchange=()=>sessionStorage.setItem('phaseAgentToolToken',token.value.trim());
document.getElementById('form').onsubmit=async event=>{event.preventDefault();const value=input.value.trim();if(!value)return;
  const key=token.value.trim();if(!key){status.textContent='请先粘贴本地连接密钥';token.focus();return;}
  sessionStorage.setItem('phaseAgentToolToken',key);add('user',value);input.value='';send.disabled=true;
  const reply=add('assistant','正在分析…');
  try{const response=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+key},
    body:JSON.stringify({model:'phase-search-agent',messages:[{role:'user',content:value}],metadata:{chat_id:chatId},stream:false})});
    const result=await response.json();if(!response.ok)throw Error(result.error?.message||'请求失败 '+response.status);
    reply.textContent=result.choices?.[0]?.message?.content||'Agent 未返回文字';status.textContent='';}
  catch(error){reply.textContent='发送失败：'+error.message;status.textContent='可检查本地 Agent 日志后重试';}
  finally{send.disabled=false;input.focus();}};
input.onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('form').requestSubmit();}};
</script></body></html>"""
