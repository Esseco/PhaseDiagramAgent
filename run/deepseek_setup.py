"""Local browser setup for testing and securely saving the DeepSeek API key."""

from __future__ import annotations

import html


def setup_page(*, model: str, configured: bool) -> str:
    """Return a self-contained local-only key setup page; never echo a key."""
    status = "检测到已配置密钥，可输入新密钥测试替换。" if configured else "尚未设置。"
    model_label = html.escape(str(model))
    return f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>启用相图 Agent</title>
<style>body{{font:16px system-ui,sans-serif;margin:3rem auto;padding:0 1rem;max-width:620px;color:#18212b;background:#f5f7fa}}main{{background:white;border:1px solid #d9e0e8;border-radius:14px;padding:1.5rem;box-shadow:0 8px 30px #1b27330d}}input,button{{font:inherit;padding:.7rem;border-radius:8px}}input{{box-sizing:border-box;width:100%;border:1px solid #abb8c5;margin:.5rem 0 1rem}}button{{border:0;background:#2459d3;color:white;cursor:pointer}}button:disabled{{opacity:.55}}small,.muted{{color:#5b6876}}#message{{min-height:1.5rem;margin-top:1rem}}</style></head>
<body><main><h1>启用相图 Agent</h1><p>输入 DeepSeek API Key 并测试连接。密钥只保存在这台电脑的 Windows 凭据管理器中，不写入配置文件或上传到超算。</p>
<p class=\"muted\">当前 Agent 模型：<code>{model_label}</code></p><p id=\"state\">{status}</p>
<form id=\"setup\" autocomplete=\"off\"><label for=\"key\">DeepSeek API Key</label><input id=\"key\" type=\"password\" autocomplete=\"off\" required minlength=\"16\" placeholder=\"粘贴 API Key\"><button id=\"submit\" type=\"submit\">测试并启用</button></form>
<p id=\"message\" role=\"status\"></p><p class=\"muted\">页面不会主动把密钥写入浏览器存储；浏览器密码管理器行为取决于其设置。连接成功后返回 Open WebUI 即可继续对话。此页面仅允许本机访问。</p></main>
<script>const form=document.getElementById('setup'), key=document.getElementById('key'), button=document.getElementById('submit'), message=document.getElementById('message');
form.addEventListener('submit',async event=>{{event.preventDefault();button.disabled=true;message.textContent='正在测试连接…';try{{const response=await fetch('/phase/setup/key',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{api_key:key.value}})}});const result=await response.json();if(!response.ok)throw new Error(result.error?.message||'设置失败');key.value='';message.textContent='连接成功，Agent 已启用。可返回 Open WebUI 对话。';document.getElementById('state').textContent='已保存；密钥不会回显。';}}catch(error){{message.textContent=error.message;}}finally{{button.disabled=false;}}}});</script></body></html>"""


def test_and_save_api_key(api_key: str, *, settings: dict, activate_client) -> dict:
    """Make a tiny JSON-mode call, then persist and activate the key on success."""
    from decision_layer.agent.create_deepseek_client import create_deepseek_client
    from run.deepseek_credentials import save_deepseek_api_key

    model = str(settings.get("model") or "deepseek-flash")
    base_url = str(settings.get("base_url") or "https://api.deepseek.com")
    timeout = int(settings.get("timeout", 30))
    test_client = create_deepseek_client(
        api_key=api_key,
        model=model,
        base_url=base_url,
        max_tokens=32,
        timeout=timeout,
        system_prompt='Return only a JSON object: {"connected": true}.',
        thinking="disabled",
    )
    result = test_client({"mode": "connection_test", "instruction": "Return connected=true."})
    if not isinstance(result, dict) or not result.get("_llm_usage"):
        raise RuntimeError("DeepSeek 没有返回有效的 JSON 测试结果；密钥未保存。")

    save_deepseek_api_key(api_key)
    activate_client(api_key)
    return {"status": "connected", "model": model}
