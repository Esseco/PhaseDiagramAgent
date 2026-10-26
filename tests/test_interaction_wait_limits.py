"""Exercise real page scripts with stalled network/clock and no project services."""

from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from phase_agent.runtime.agent_api import _approval_page
from phase_agent.runtime.interaction_reports import save_interaction
from phase_agent.runtime.studio_conversation_panel import PAGE


NODE = r"""
const vm=require('vm'),assert=require('assert/strict'),fs=require('fs');
const {script,scenario}=JSON.parse(fs.readFileSync(0,'utf8'));
class Element{
 constructor(){this.disabled=false;this.textContent='';this.children=[];this.value='fixture-token';}
 append(...values){this.children.push(...values);}
 appendChild(value){this.children.push(value);}
 replaceChildren(){this.children=[];}
}
const ids=['list','notice','continue','new','refresh','more','select-all','clear-selection','delete-selected','selection-count','token','message','plans'];
const elements=new Map(ids.map(id=>[id,new Element()])),buttons=[new Element(),new Element()];
const calls=[];let clock=0;
const DateView=class extends Date{static now(){clock+=scenario==='approval_total_deadline'?60000:scenario==='pause_total_deadline'?10000:0;return clock;}};
const stall=signal=>new Promise((resolve,reject)=>{
 if(signal.aborted){reject(signal.reason);return;}
 signal.addEventListener('abort',()=>reject(signal.reason),{once:true});
});
const answer=(value,status=200)=>({ok:status<400,status,text:async()=>JSON.stringify(value),json:async()=>value});
const context={URL,Date:DateView,console,crypto:{randomUUID:()=> 'single-request-id'},
 AbortSignal:{timeout:()=>AbortSignal.timeout(5)},
 setTimeout:(callback,ms)=>setTimeout(callback,Math.min(ms,1)),
 location:{origin:'http://127.0.0.1:2025',assign:()=>assert.fail('must not navigate')},
 localStorage:{getItem:()=>null,setItem(){},removeItem(){}},confirm:()=>true,
 document:{getElementById:id=>elements.get(id),createElement:()=>new Element(),querySelectorAll:()=>buttons},
 fetch:async(path,options={})=>{
  calls.push({path,method:options.method||'GET'});
  assert(options.signal,'every request must have a deadline');
  if(path==='/threads/search'){
   if(scenario==='conversation_initial_timeout')return stall(options.signal);
   return answer([]);
  }
  if(path.includes('/runs?'))return answer([{run_id:'existing',status:'running'}]);
  if(path.includes('/cancel?'))return answer(null);
  if(path==='/threads/owned'){
   if(scenario==='pause_request_timeout')return stall(options.signal);
   return answer({status:'busy'});
  }
  if(path==='/phase/decision'){
   if(scenario==='approval_post_timeout')return stall(options.signal);
   return answer({status:'pending',status_url:'/phase/operations/original'},202);
  }
  if(path==='/phase/operations/original'){
   if(scenario==='approval_poll_timeout')return stall(options.signal);
   return answer({status:'running',status_url:'/phase/operations/original'});
  }
  throw Error('unexpected request '+path);
 }
};
vm.createContext(context);
const settle=()=>new Promise(resolve=>setTimeout(resolve,30));
async function main(){
 vm.runInContext(script,context);await settle();
 if(scenario==='conversation_initial_timeout'){
  assert.equal(vm.runInContext('busy',context),false);
  assert.match(elements.get('notice').textContent,/等待超时.*待核对/);
  assert.equal(calls.length,1,'no automatic network retry');
  assert(!elements.get('refresh').disabled,'refresh must become available');
 }else if(scenario.startsWith('pause_')){
  await vm.runInContext("operation(()=>pause({thread_id:'owned'}))",context);
  assert.equal(vm.runInContext('busy',context),false);
  assert.match(elements.get('notice').textContent,/(待核对|尚未退出)/);
  assert(!elements.get('notice').textContent.includes('已暂停。'),'do not claim cancellation is complete');
  assert(!elements.get('refresh').disabled);
  assert(calls.filter(c=>c.path.includes('/cancel?')).length<=1);
  assert(calls.length<10,'overall pause deadline must bound polling');
 }else{
  const holding=setTimeout(()=>{},200);
  try{
   await assert.rejects(vm.runInContext("api('/phase/decision',{method:'POST',body:'{}'})",context),/待核对.*读取/);
  }finally{clearTimeout(holding);}
  assert.equal(calls.filter(c=>c.method==='POST').length,1,'never resend approval');
  assert(buttons.every(button=>button.disabled),'unknown result must require a fresh read');
  if(scenario==='approval_total_deadline')assert(calls.length<=2,'stop polling pending operations');
 }
 console.log('verified '+scenario);
}
main().catch(error=>{console.error(error);process.exitCode=1;});
"""


@pytest.mark.parametrize(
    "scenario",
    [
        "conversation_initial_timeout",
        "pause_request_timeout",
        "pause_total_deadline",
        "approval_post_timeout",
        "approval_poll_timeout",
        "approval_total_deadline",
    ],
)
def test_page_wait_finishes_without_retrying_operations(scenario):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for real page-script verification")
    page = _approval_page() if scenario.startswith("approval_") else PAGE
    script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
    result = subprocess.run(
        [node, "-e", NODE],
        input=json.dumps({"script": script, "scenario": scenario}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_inline_code_does_not_expand_receipt_and_full_markdown_is_preserved(tmp_path):
    full = "结果：已核对 `model-v1`。" + "完整结果与依据。" * 400
    state = {"tasks": [{"task_id": "kept", "status": "pending"}]}
    before = deepcopy(state)
    reply = save_interaction(tmp_path, "查看结果", full, full, state)
    short, location = reply.split("\n交互记录地址：", 1)
    assert len(short.strip()) <= 220
    path = Path(location)
    assert path.is_absolute() and path.is_file()
    assert full in path.read_text(encoding="utf-8")
    assert state == before
