"""Exercise the actual page script against a bounded fake native API."""

import json
import re
import shutil
import subprocess

import pytest

from phase_agent.runtime.studio_conversation_panel import PAGE


NODE = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const {script,scenario,origin}=JSON.parse(fs.readFileSync(0,'utf8'));
class Element {
 constructor(tag='div'){this.tag=tag;this.children=[];this.disabled=false;this.hidden=false;this.textContent='';}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(){this.children=[];this.textContent='';}
}
const elements=new Map(['list','notice','continue','new','refresh','more','select-all','clear-selection','delete-selected','selection-count'].map(id=>[id,new Element(['list','notice','selection-count'].includes(id)?'div':id==='new'?'a':'button')]));
const calls=[],opened=[],confirmations=[],memory=new Map();
const key=`phase-studio-thread:${origin}:phase_chat`;
const id=n=>`00000000-0000-0000-0000-${String(n).padStart(12,'0')}`;
const row=(n,extra={})=>({thread_id:id(n),metadata:{graph_id:'phase_chat'},config:{},status:'idle',created_at:'2026-10-20',updated_at:'2026-10-21',...extra});
let rows=[row(2),row(1,{status:'error'})],deleted=new Set();
let release=null,blockedOnce=false;
if(['saved','row','tabs','open_deleted','new'].includes(scenario))memory.set(key,id(1));
if(scenario==='malformed')memory.set(key,'not-a-thread-id');
if(scenario==='deleted'){memory.set(key,id(9));deleted.add(id(9));}
if(scenario==='cross_project')memory.set('phase-studio-thread:http://127.0.0.1:2024:phase_chat',id(1));
if(scenario==='internal')rows=[row(7,{metadata:{graph_id:'phase_chat',checkpoint_ns:'agent:child'}}),row(8,{config:{configurable:{checkpoint_ns:'child:1'}}}),row(2),row(3,{metadata:{graph_id:'project_status'}})];
if(scenario==='pages'||scenario==='batch_pages')rows=Array.from({length:102},(_,i)=>row(102-i));
if(scenario==='batch_mixed')rows=[row(5),row(4),row(3),row(2),row(1)];
if(scenario==='batch_delete')memory.set(key,id(1));
const query=selector=>{
 const found=[];
 const walk=node=>{if(!node||typeof node!=='object')return;
  if((selector==='button'&&node.tag==='button')||(selector==='input[type="checkbox"]'&&node.tag==='input'&&node.type==='checkbox'))found.push(node);
  node.children.forEach(walk);
 };
 for(const node of elements.values())walk(node);return found;
};
const context={URL,console,setTimeout,AbortSignal,confirm:text=>{confirmations.push(text);return scenario.startsWith('batch_')&&scenario!=='batch_cancel';},location:{origin,assign:url=>opened.push(url)},
 document:{getElementById:id=>elements.get(id),createElement:tag=>new Element(tag),querySelectorAll:query},
 localStorage:{getItem:k=>{if(scenario==='storage_denied')throw Error('denied');return memory.get(k)||null;},setItem:(k,v)=>memory.set(k,v),removeItem:k=>memory.delete(k)},
 fetch:async(path,options={})=>{
  const payload=options.body?JSON.parse(options.body):null;calls.push({path,method:options.method,payload});
  if(path!=='/threads/search'){
   assert(scenario.startsWith('batch_'),'navigation must only search existing records');
   const match=path.match(/^\/threads\/([0-9a-f-]+)(\/runs\?limit=(?:100|1&status=(running|pending)))?$/);assert(match,path);
   const tid=match[1],runs=!!match[2];
   assert.equal(options.method,runs?'GET':'DELETE');
   if(scenario==='batch_busy'&&runs&&!blockedOnce){blockedOnce=true;await new Promise(resolve=>release=resolve);}
   if(scenario==='batch_mixed'&&((runs&&tid===id(4))||(!runs&&tid===id(3))))
    return {ok:false,status:500,text:async()=> 'synthetic failure'};
   if(scenario==='batch_old_active'&&runs){
    const status=tid===id(2)?'running':'pending';
    const history=Array.from({length:100},()=>({status:'success'})).concat([{status}]);
    const data=match[3]?history.filter(run=>run.status===match[3]).slice(0,1):history.slice(0,100);
    return {ok:true,text:async()=>JSON.stringify(data)};
   }
   if(!runs)deleted.add(tid);
   return {ok:true,text:async()=>JSON.stringify(runs?(scenario==='batch_mixed'&&tid===id(5)?[{status:'running'}]:[]):null)};
  }
  assert.equal(path,'/threads/search','navigation must only search existing records');
  assert.equal(options.method,'POST');
  if(scenario==='batch_refresh'&&calls.length===2)await new Promise(resolve=>release=resolve);
  assert.deepEqual(payload.metadata,{graph_id:'phase_chat'});
  assert(!payload.select.includes('values'),'do not read raw messages/scientific state');
  assert(payload.limit<=100,'bounded page size');
  let data=rows.filter(r=>!deleted.has(r.thread_id));
  if(payload.ids)data=data.filter(r=>payload.ids.includes(r.thread_id));
  else data=data.slice(payload.offset,payload.offset+payload.limit);
  return {ok:true,text:async()=>JSON.stringify(data)};
 }
};
vm.createContext(context);
const settle=()=>new Promise(resolve=>setImmediate(resolve));
async function main(){
 vm.runInContext(script,context);await settle();
 assert(!elements.get('notice').className,'initial script must not fail');
 const selected=()=>elements.get('continue').title;
 assert.equal(opened.length,0,'opening history never creates/runs/navigates');
 if(scenario.startsWith('batch_')){
  const boxes=()=>query('input[type="checkbox"]');
  const click=id=>elements.get(id).onclick();
  assert.equal(boxes().length,scenario==='batch_pages'?100:rows.length);
  assert(elements.get('delete-selected').disabled);
  const previous=selected();boxes()[0].checked=true;boxes()[0].onchange();
  assert.equal(selected(),previous,'bulk selection must not change continue target');
  assert(!elements.get('delete-selected').disabled);
  if(scenario==='batch_pages'){
   await click('more');assert.equal(boxes().length,102);
   assert.equal(boxes().filter(box=>box.checked).length,1,'loading does not select more');
   assert.equal(calls.length,2);click('select-all');assert(boxes().every(box=>box.checked));
   assert.match(elements.get('selection-count').textContent,/已选择 102 条/);
   click('clear-selection');assert(boxes().every(box=>!box.checked));
   assert(elements.get('delete-selected').disabled);
   boxes()[101].checked=true;boxes()[101].onchange();
   await click('delete-selected');assert(deleted.has(id(1)));assert.equal(deleted.size,1);
   assert.equal(elements.get('list').children.length,100);
   await click('more');assert.equal(elements.get('list').children.length,101,'reset offset after delete');
  }else if(scenario==='batch_select'){
   click('select-all');assert.equal(boxes().filter(box=>box.checked).length,2);
   boxes()[0].checked=false;boxes()[0].onchange();
   assert.match(elements.get('selection-count').textContent,/已选择 1 条/);
   click('clear-selection');assert(elements.get('delete-selected').disabled);
   assert.equal(calls.length,1);
  }else if(scenario==='batch_old_active'){
   click('select-all');await click('delete-selected');
   assert.equal(deleted.size,0,'older running/pending requests must block deletion');
   assert.match(elements.get('notice').textContent,/已删除 0 条，活动对话跳过 2 条，失败 0 条/);
   assert(calls.some(call=>call.path.endsWith('status=running')));
   assert(calls.some(call=>call.path.endsWith('status=pending')));
  }else if(scenario==='batch_refresh'){
   click('select-all');const refreshing=click('refresh');await settle();
   assert.equal(typeof release,'function');assert(elements.get('delete-selected').disabled);
   const count=calls.length;await click('delete-selected');await click('more');
   assert.equal(calls.length,count);assert.equal(confirmations.length,0);
   release();await refreshing;assert.equal(deleted.size,0);
   assert.equal(elements.get('list').children.length,2);
   click('select-all');await click('delete-selected');assert.equal(deleted.size,2);
  }else if(scenario==='batch_busy'){
   click('select-all');const pending=click('delete-selected');await settle();
   assert.equal(typeof release,'function');assert(elements.get('delete-selected').disabled);
   assert(boxes().every(box=>box.disabled));const count=calls.length;
   await click('delete-selected');await click('more');click('clear-selection');
   assert.equal(calls.length,count);assert.equal(confirmations.length,1);
   assert.equal(boxes().filter(box=>box.checked).length,2);
   release();await pending;assert.equal(deleted.size,2);
  }else{
   click('select-all');const before=calls.length;
   await click('delete-selected');assert.equal(confirmations.length,1);
   if(scenario==='batch_cancel'){
    assert.equal(calls.length,before);assert.equal(deleted.size,0);
    assert.equal(boxes().filter(box=>box.checked).length,2,'cancel preserves selection');
   }else if(scenario==='batch_mixed'){
    assert.deepEqual([...deleted].sort(),[id(1),id(2)]);
    assert.match(elements.get('notice').textContent,/已删除 2 条，活动对话跳过 1 条，失败 2 条/);
    assert(elements.get('notice').textContent.includes(id(3)));
    assert(elements.get('notice').textContent.includes(id(4)));
    assert(!calls.some(c=>c.method==='DELETE'&&[id(4),id(5)].includes(c.path.split('/')[2])));
    assert.equal(elements.get('list').children.length,3);
   }else if(scenario==='batch_delete'){
    assert.equal(deleted.size,2);assert(!memory.has(key));
    assert.match(elements.get('notice').textContent,/已删除 2 条/);
    assert(elements.get('continue').disabled);assert(elements.get('delete-selected').disabled);
   }
  }
 }else if(scenario==='row'){
  assert.equal(selected(),id(1));
  await query('button').find(button=>button.textContent==='打开／继续').onclick();
  const target=new URL(opened[0]);
  assert.equal(target.pathname,'/studio/thread');
  assert.equal(target.searchParams.get('threadId'),id(2));
  assert.equal(target.searchParams.get('baseUrl'),origin);
  assert.equal(memory.get(key),id(2));
 }else if(scenario==='saved'){
  assert.equal(selected(),id(1));await vm.runInContext('openThread(selected)',context);
  const target=new URL(opened[0]);
  assert.equal(target.pathname,'/studio/thread');
  assert.equal(target.searchParams.get('threadId'),id(1));
  assert.equal(target.searchParams.get('baseUrl'),origin);
  assert.equal(memory.get(key),id(1));
 }else if(scenario==='deleted'){
  assert.equal(selected(),id(2));assert(!memory.has(key));
 }else if(scenario==='internal'){
  assert.equal(selected(),id(2));assert.equal(elements.get('list').children.length,1);
 }else if(scenario==='tabs'){
  memory.set(key,id(2));await vm.runInContext('refresh()',context);assert.equal(selected(),id(1));
 }else if(scenario==='pages'){
  assert.equal(calls.length,1);assert.equal(elements.get('list').children.length,100);
  await vm.runInContext('refresh(true)',context);
  assert.equal(calls.length,2);assert.equal(calls[1].payload.offset,100);
  assert.equal(elements.get('list').children.length,102);assert(elements.get('more').hidden);
 }else if(scenario==='open_deleted'){
  deleted.add(id(1));await vm.runInContext('openThread(selected)',context);
  assert.equal(opened.length,0);assert.equal(elements.get('notice').className,'error');
 }else if(scenario==='cross_project'||scenario==='storage_denied'){
  assert.equal(selected(),id(2));
 }else if(scenario==='new'){
  elements.get('new').onclick();assert(!memory.has(key));
 }else if(scenario==='malformed'){
  assert.equal(calls.length,1);assert.equal(selected(),id(2));
 }
 const newUrl=new URL(elements.get('new').href);
 assert.equal(newUrl.pathname,'/studio/');assert.equal(newUrl.searchParams.get('baseUrl'),origin);
 assert.equal(rows.length,['pages','batch_pages'].includes(scenario)?102:scenario==='internal'?4:scenario==='batch_mixed'?5:2);
 console.log(JSON.stringify({scenario,calls:calls.length,opened:opened.length}));
}
main().catch(error=>{console.error(error.stack);process.exitCode=1;});
"""


@pytest.mark.parametrize(
    "scenario",
    [
        "saved",
        "row",
        "deleted",
        "internal",
        "tabs",
        "pages",
        "open_deleted",
        "cross_project",
        "storage_denied",
        "new",
        "malformed",
        "batch_select",
        "batch_pages",
        "batch_cancel",
        "batch_delete",
        "batch_mixed",
        "batch_busy",
        "batch_refresh",
        "batch_old_active",
    ],
)
@pytest.mark.parametrize("origin", ["http://127.0.0.1:2025", "http://127.0.0.1:2026"])
def test_page_navigation_uses_existing_threads_only(scenario, origin):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the page-script fixture")
    script = re.search(r"<script>(.*?)</script>", PAGE, re.S).group(1)
    result = subprocess.run(
        [node, "-e", NODE],
        input=json.dumps({"script": script, "scenario": scenario, "origin": origin}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dashboard_opens_local_conversation_selector_without_starting_service(
    tmp_path, monkeypatch
):
    from phase_agent.runtime import project_dashboard as module

    opened = []
    monkeypatch.setattr(module, "read_service", lambda _: {"studio_port": 2026})
    monkeypatch.setattr(module, "start_agent", lambda *_: pytest.fail("must not start a service"))
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    module.open_project_conversations(tmp_path / "runtime.json")
    assert opened == ["http://127.0.0.1:2026/phase/conversations"]
    monkeypatch.setattr(module, "read_service", lambda _: None)
    with pytest.raises(ValueError, match="先启动"):
        module.open_project_conversations(tmp_path / "runtime.json")
    assert len(opened) == 1
