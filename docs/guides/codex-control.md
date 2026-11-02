# Codex 本地控制入口

这个模块是现有本地 Agent HTTP 服务的薄客户端，不会直接修改项目状态，也不会绕过 LangGraph、审批、预算或恢复逻辑。

先按原方式启动某个项目的本地服务，然后在 Codex 的终端中使用：

```powershell
$env:PHASE_CONTROL_URL = 'http://127.0.0.1:8765'
$env:PHASE_CONTROL_API_KEY = '<控制接口令牌>'

python -m phase_agent.runtime.codex.control status
python -m phase_agent.runtime.codex.control pending
python -m phase_agent.runtime.codex.control continue
python -m phase_agent.runtime.codex.control propose '检查当前项目状态并给出下一步建议'
```

审批已有提案时必须带上服务返回的并发校验字段：

```powershell
python -m phase_agent.runtime.codex.control decide approve `
  --plan-id '<plan_id>' `
  --state-version '<state_version>' `
  --proposal-hash '<proposal_hash>' `
  --comment '已在 Codex 中确认'
```

可用命令包括：`health`、`status`、`pending`、`tasks`、`config`、`charts`、`diagnostics`、`propose`、`continue`、`decide` 和 `chat`。

`propose`、`continue` 和 `decide` 仍然进入现有 Agent 控制接口；项目的状态文件、审批记录和科学动作由原服务负责。`chat` 需要另外设置 `PHASE_API_KEY`，用于调用原有的 OpenAI-compatible `/v1/chat/completions` 接口。

服务地址、控制令牌和对话令牌也可以通过命令行参数传入。令牌不会写入项目文件或输出到 JSON 结果中。

## Repo 命令

在仓库根目录直接运行：

```powershell
.\codex-agent.cmd health
.\codex-agent.cmd status
.\codex-agent.cmd pending
.\codex-agent.cmd continue
.\codex-agent.cmd propose '检查当前项目并继续下一步'
```

脚本默认使用项目规定的 `py1` Python，也可以用 `PHASE_PYTHON` 指定其他 Python 可执行文件。

## Agent 服务生命周期

对指定项目使用同一个命令管理 Agent 服务：

```powershell
$runtime = 'E:\0-PhaseDiagram\O3-1ele-NaCrO2\agent_runtime.json'
.\codex-agent.cmd start --runtime-config $runtime
.\codex-agent.cmd takeover --runtime-config $runtime
.\codex-agent.cmd stop --runtime-config $runtime
```

`takeover` 会先停止与该 runtime config 精确匹配的旧 Agent，再启动新服务，使 Codex 获得新的本地会话。停止操作不会按端口扫描或终止其他项目进程；服务身份不匹配时会拒绝停止。


## 明确的 Codex 接管模式

默认 `agent` 模式保持原来的 Agent+API 行为。只有用户明确要求“由 Codex 接管”时，才使用 `codex` 模式：

```powershell
python -m phase_agent.runtime.codex.control --operator codex status
python -m phase_agent.runtime.codex.control --operator codex pending
```

`codex` 模式会阻止 `continue`、`propose` 和 `chat`，因此不会误调用项目配置的 DeepSeek。它仍允许 Codex 查看状态、审批已有冻结方案和读取任务。新科学动作必须先由当前 Codex 形成结构化方案，再接入受验证的提案写入入口；禁止直接修改项目状态文件。


### Codex 结构化提案

Codex 接管时，Codex 可以提交结构化动作；服务只把它送入原有工作流，不直接执行：

```python
client = CodexControlClient(control_token=token, operator="codex")
client.codex_propose({
    "tool": "...",
    "parameters": {...},
    "reason": "...",
    "evidence_refs": [...],
})
```

该入口仍经过原来的动作规范化、证据和参数验证、预算检查、项目身份绑定、审批卡、并发校验、执行收据和恢复流程。它是独立适配器，不改变 `/phase/propose` 和现有 UI/API。

形成待审批方案时，服务复用 Agent 的展示和交互记录，响应中的 `report_path` 是项目内 Markdown 地址。记录保存失败时，`report_error` 会提示方案仍保留；不要重复提交方案。对于已经存在的待审方案，可以通过 `python -m phase_agent.runtime.codex.control --operator codex codex-report <plan_id>` 补写记录；该命令只展示已保存方案，不会批准或执行。


## Codex 接管的数据传输限制

- Codex 接管、MCP、离线入口与项目 Agent 遵守同一数据管理规则：每次提交和回收只传必须的数据文件，并以压缩包传输；必须复用仓库已有的裁剪、清单、打包、校验和回收入口，不另建整目录复制流程。
- 传输前检查包内清单、每类文件用途、文件数、压缩前后大小与 SHA256。压缩包存在或名为 results.zip 不代表已满足最小范围；旧冻结脚本产生的包也必须检查。
- 必须文件按现有消费者的真实依赖确定：输入准备、任务身份与完成校验、本地分析、训练数据及已验证续跑协议需要的数据必须保留。未引用的 pool 结构、原始日志、轨迹、诊断副本及本地不需要的大模型/检查点留在远端，登记路径与校验和；不擅自删除科学数据或仅用路径替代本地必须读取的数据。
- 已批准提交或回收不豁免最小传输范围。旧包不满足规则时先使用现有导出/裁剪入口；入口不支持则报告缺口，不直接下载整个工作目录或扩大回收范围。回收不得重算已完成结果，不改变原审批和项目 Agent 入账边界。

具体阶段的必需文件沿用 [最小结果回传](workspace.md#doc-lightweight_result_transfer)。接管握手 `phase_takeover_repo` 和项目 `phase_context` 均返回同一 `hpc_approval_policy.data_transfer`，避免接管后丢失约束。这是 Codex 的操作合同；它不声称能拦截外部 SFTP 客户端。
