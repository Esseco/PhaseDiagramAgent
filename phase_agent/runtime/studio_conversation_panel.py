"""Same-origin controls for Studio threads; scientific data is not deleted."""

PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Studio 对话管理</title>
<style>body{font:16px/1.6 system-ui;max-width:1000px;margin:32px auto;padding:0 20px;color:#203040;background:#f4f6f8}article{background:white;border:1px solid #dce3ea;border-radius:12px;padding:18px;margin:14px 0}button{font:inherit;padding:6px 14px;margin:4px;cursor:pointer}small{display:block;overflow-wrap:anywhere}.error{color:#a22}pre{white-space:pre-wrap}</style></head><body>
<h1>Studio 对话管理</h1><p>只管理当前项目的本地 Studio 对话。暂停保留项目状态；删除对话不会删除计算结果，也不会取消超算作业。</p>
<p><a href="/phase/flow">科学流程</a> · <a href="/phase/process">本轮过程</a></p><button id="refresh">刷新列表</button><p id="notice" role="status"></p><main id="list"></main>
<script>
const list=document.getElementById('list'),notice=document.getElementById('notice');
let busy=false;
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
async function remove(thread){
 if(!confirm('删除这条Studio对话及其聊天历史？项目计算结果会保留。'))return;
 const runs=await api(`/threads/${thread.thread_id}/runs?limit=100`);
 if(runs.some(active)){say('对话仍有活动请求，请先暂停，等待后台退出再删除。',true);return;}
 await api(`/threads/${thread.thread_id}`,'DELETE');say('对话已删除，项目计算结果保留。');
}
async function operation(fn,thread){
 if(busy)return;busy=true;document.querySelectorAll('button').forEach(b=>b.disabled=true);
 try{await fn(thread);await refresh();}catch(error){say(error.message,true);}finally{busy=false;document.querySelectorAll('button').forEach(b=>b.disabled=false);}
}
async function refresh(){
 const threads=await api('/threads/search','POST',{limit:100});list.replaceChildren();
 if(!threads.length){list.textContent='暂无Studio对话。';return;}
 for(const thread of threads){
  const card=document.createElement('article');const title=document.createElement('h2');
  title.textContent=thread.metadata?.title||thread.metadata?.name||'Studio 对话';card.append(title);
  const info=document.createElement('small');info.textContent=`${thread.thread_id} · ${thread.status} · ${thread.updated_at||''}`;card.append(info);
  for(const [label,fn] of [['暂停',pause],['删除对话',remove]]){const button=document.createElement('button');button.textContent=label;button.onclick=()=>operation(fn,thread);card.append(button);}
  list.append(card);
 }
}
document.getElementById('refresh').onclick=()=>{if(!busy)refresh().catch(error=>say(error.message,true));};
refresh().catch(error=>say(error.message,true));
</script></body></html>"""
