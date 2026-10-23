"""Same-origin controls for Studio threads; scientific data is not deleted."""


def conversation_url(port):
    return f"http://127.0.0.1:{int(port)}/phase/conversations"


PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>选择／继续项目对话</title>
<style>body{font:16px/1.6 system-ui;max-width:1000px;margin:32px auto;padding:0 20px;color:#203040;background:#f4f6f8}article{background:white;border:1px solid #dce3ea;border-radius:12px;padding:18px;margin:14px 0}button{font:inherit;padding:6px 14px;margin:4px;cursor:pointer}small{display:block;overflow-wrap:anywhere}.error{color:#a22}pre{white-space:pre-wrap}</style></head><body>
<h1>选择／继续项目对话</h1><p>打开已有对话会保留该对话的上下文；页面本身不创建对话或执行计算。</p><p>只管理当前项目的本地 Studio 对话。暂停保留项目状态；删除对话不会删除计算结果，也不会取消超算作业。</p>
<p><a href="/phase/flow">科学流程</a> · <a href="/phase/process">本轮过程</a></p><button id="continue" disabled>继续上次对话</button> <a id="new">新对话</a> <button id="refresh">刷新列表</button><p><button id="select-all">全选已加载</button><button id="clear-selection" disabled>取消选择</button><button id="delete-selected" disabled>删除所选对话</button> <span id="selection-count" role="status">已选择 0 条</span></p><p id="notice" role="status" style="white-space:pre-wrap"></p><main id="list"></main><button id="more" hidden>加载更早对话</button>
<script>
const list=document.getElementById('list'),notice=document.getElementById('notice');
let busy=false,offset=0,selected=null;
const pageSize=100,seen=new Map(),checked=new Set();
const storageKey=`phase-studio-thread:${location.origin}:phase_chat`;
const fields=['thread_id','metadata','config','status','created_at','updated_at'];
let preferred=null;
try{preferred=localStorage.getItem(storageKey);}catch{say('浏览器未允许保存选择；仍可打开已有对话。');}
if(preferred&&!/^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(preferred))preferred=null;
function studioUrl(id){
 const url=new URL('https://smith.langchain.com/studio/');
 // Studio selects an existing thread through the query, not a path segment.
 if(id){url.pathname='/studio/thread';url.searchParams.set('threadId',id);}
 url.searchParams.set('baseUrl',location.origin);return url.toString();
}
document.getElementById('new').href=studioUrl();
document.getElementById('new').onclick=()=>{preferred=null;try{localStorage.removeItem(storageKey);}catch{}};
function rootChat(thread){
 const metadata=thread.metadata||{},configurable=thread.config?.configurable||{};
 return metadata.graph_id==='phase_chat'&&!metadata.checkpoint_ns&&!configurable.checkpoint_ns&&!metadata.parent_thread_id;
}
async function search(options){
 return api('/threads/search','POST',{
  metadata:{graph_id:'phase_chat'},select:fields,limit:pageSize,offset:0,
  sort_by:'updated_at',sort_order:'desc',...options
 });
}
async function existing(id){
 return (await search({ids:[id],limit:1})).find(rootChat)||null;
}
function choose(thread){
 selected=thread;
 const button=document.getElementById('continue');button.disabled=!thread;
 button.textContent=thread?'继续上次对话':'暂无可继续的对话';
 button.title=thread?thread.thread_id:'';
}
async function openThread(thread){
 const current=await existing(thread.thread_id);
 if(!current){say('这条对话已不存在或不是项目聊天，请刷新后选择。',true);return;}
 preferred=current.thread_id;choose(current);
 try{localStorage.setItem(storageKey,preferred);}catch{say('浏览器未允许保存选择；本次仍打开该对话。');}
 location.assign(studioUrl(preferred));
}
async function api(path,method='GET',data){
 const options={method,headers:{'Content-Type':'application/json'}};
 if(data!==undefined)options.body=JSON.stringify(data);
 const response=await fetch(path,options);
 if(!response.ok)throw Error(`请求失败 ${response.status}: ${(await response.text()).slice(0,160)}`);
 const text=await response.text();return text?JSON.parse(text):null;
}
function say(text,error=false){notice.textContent=text;notice.className=error?'error':'';}
const active=run=>['running','pending'].includes(run.status);
async function pause(thread){
 const runs=await api(`/threads/${thread.thread_id}/runs?limit=100`);
 for(const run of runs.filter(active))await api(`/threads/${thread.thread_id}/runs/${run.run_id}/cancel?wait=false&action=interrupt`,'POST');
 if(!runs.some(active)){say('没有活动请求。若仍显示busy，这是残留状态，可删除该对话。');return;}
 say('暂停已请求，等待后台退出；当前模型请求可能需要等待返回或超时。');
 for(let n=0;n<30;n++){
  await new Promise(resolve=>setTimeout(resolve,1000));
  const current=await api(`/threads/${thread.thread_id}`);
  if(current.status!=='busy'){say('已暂停。可在Studio中发送新的“继续”恢复项目。');return;}
 }
 say('暂停信号已发出，后台尚未退出。稍后刷新；不要重复提交计算。');
}
async function remove(thread,confirmed=false){
 if(!confirmed&&!confirm('删除这条Studio对话及其聊天历史？项目计算结果会保留。'))return false;
 for(const status of ['running','pending']){
  const runs=await api(`/threads/${thread.thread_id}/runs?limit=1&status=${status}`);
  if(runs.some(active)){say('对话仍有活动请求，请先暂停，等待后台退出再删除。',true);return false;}
 }
 await api(`/threads/${thread.thread_id}`,'DELETE');say('对话已删除，项目计算结果保留。');return true;
}
function updateSelection(){
 document.querySelectorAll('input[type="checkbox"]').forEach(box=>{box.checked=checked.has(box.value);box.disabled=busy;});
 document.getElementById('selection-count').textContent=`已选择 ${checked.size} 条（已加载 ${seen.size} 条）`;
 document.getElementById('select-all').disabled=busy||seen.size===0||checked.size===seen.size;
 document.getElementById('clear-selection').disabled=busy||checked.size===0;
 document.getElementById('delete-selected').disabled=busy||checked.size===0;
}
async function removeSelected(){
 const threads=[...checked].map(id=>seen.get(id));
 if(!threads.length||!confirm(`删除选中的 ${threads.length} 条Studio对话及其聊天历史？项目计算结果会保留；有活动请求的对话会跳过。`))return false;
 let removed=0;const skipped=[],failed=[];
 for(const thread of threads){
  try{
   if(await remove(thread,true)){removed++;checked.delete(thread.thread_id);}
   else skipped.push(thread.thread_id);
  }catch(error){failed.push(`${thread.thread_id}：${error.message}`);}
 }
 say(`批量删除完成：已删除 ${removed} 条，活动对话跳过 ${skipped.length} 条，失败 ${failed.length} 条。`+
  (skipped.length?`\n活动对话：${skipped.join('、')}`:'')+(failed.length?`\n删除失败：\n${failed.join('\n')}`:''),failed.length>0);
}
async function operation(fn,thread){
 if(busy)return;busy=true;document.querySelectorAll('button').forEach(b=>b.disabled=true);updateSelection();
 try{if(await fn(thread)!==false)await refresh();}catch(error){say(error.message,true);}finally{busy=false;document.querySelectorAll('button').forEach(b=>b.disabled=false);choose(selected);updateSelection();}
}
async function refresh(append=false){
 const threads=await search({offset:append?offset:0});
 if(!append){list.replaceChildren();seen.clear();checked.clear();offset=0;}
 offset+=threads.length;document.getElementById('more').hidden=threads.length<pageSize;
 const chats=threads.filter(rootChat);
 for(const thread of chats){
  if(seen.has(thread.thread_id))continue;
  if(!seen.size)list.replaceChildren();seen.set(thread.thread_id,thread);
  const card=document.createElement('article');const label=document.createElement('label'),box=document.createElement('input');
  box.type='checkbox';box.value=thread.thread_id;
  box.onchange=()=>{if(busy)return;box.checked?checked.add(thread.thread_id):checked.delete(thread.thread_id);updateSelection();};
  label.append(box,'选择此对话');card.append(label);const title=document.createElement('h2');
  title.textContent=thread.metadata?.title||thread.metadata?.name||`对话 · ${thread.created_at||thread.thread_id}`;card.append(title);
  const info=document.createElement('small');info.textContent=`${thread.thread_id} · ${thread.status} · ${thread.updated_at||''}`;card.append(info);
  for(const [label,fn] of [['打开／继续',openThread],['暂停',pause],['删除对话',remove]]){const button=document.createElement('button');button.textContent=label;button.onclick=()=>operation(fn,thread);card.append(button);}
  list.append(card);
 }
 if(!seen.size)list.textContent='暂无项目聊天记录；原有科研记录不会改变。';
 if(!append){
  let previous=preferred?(chats.find(thread=>thread.thread_id===preferred)||await existing(preferred)):null;
  if(preferred&&!previous){preferred=null;try{localStorage.removeItem(storageKey);}catch{}}
  choose(previous||chats[0]||null);
 }else if(!selected){choose(chats[0]||null);}
 updateSelection();
}
document.getElementById('select-all').onclick=()=>{if(busy)return;for(const id of seen.keys())checked.add(id);updateSelection();};
document.getElementById('clear-selection').onclick=()=>{if(busy)return;checked.clear();updateSelection();};
document.getElementById('delete-selected').onclick=()=>operation(removeSelected);
document.getElementById('continue').onclick=()=>{if(selected)operation(openThread,selected);};
async function reload(){await refresh();return false;}
document.getElementById('refresh').onclick=()=>operation(reload);
document.getElementById('more').onclick=async()=>{if(busy)return;busy=true;try{await refresh(true);}catch(error){say(error.message,true);}finally{busy=false;updateSelection();}};
operation(reload);
</script></body></html>"""
