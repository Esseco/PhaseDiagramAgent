# Agent 驱动的材料相图搜索

Agent 在已确认的结构边界与预算内推荐 branch、MC、DFT 和模型更新方案。QBC、凸包、覆盖率与误差提供决策证据；执行层检查权限、参数、版本及预算。科学计算在超算计算节点运行，本地负责对话、输入准备、结果分析和流程恢复。

## 源码目录

```text
Process_PhaseDiagram/
├── phase_agent/                 # 唯一应用包
│   ├── agents/                  # 四个 Agent 与公共契约
│   │   ├── exploration/         # branch与Relax/MC探索充分性
│   │   ├── reliability/         # DFT、模型训练与相稳定性可信度
│   │   ├── strategic/           # 科研策略、专家咨询与人工意见
│   │   └── simulation/          # 执行规划、工具调用检查与诊断
│   ├── graphs/                  # dialogue/project/decision_team/actions/recovery
│   ├── evidence/                # 事实快照、科学证据、引用与按需读取
│   ├── models/                  # 模型客户端、传输、缓存与能力声明
│   ├── policies/                # 权限、预算、参数与版本约束
│   ├── tools/                   # 动作执行、输入准备及远端交接
│   ├── configuration/           # 配置草稿、校验、确认版本
│   ├── persistence/             # 科学账本、业务状态与记忆
│   ├── analysis/                # 凸包、误差、反馈、成本和可视化
│   ├── science/                 # 结构、MLIP、MC、DFT与训练算法
│   ├── compatibility/           # 历史模块描述符的集中映射
│   └── runtime/                 # 项目启动器、Studio/HTTP 及依赖组装
│       └── codex/               # Codex 协议、客户端、原生团队与生命周期适配
├── settings/                    # 本地运行时配置及示例
├── tests/                       # 单元、集成与审批安全回归
├── docs/                        # guides 操作 / architecture 维护
├── experiments/                 # 独立研究实验，非聊天控制入口
├── local/                       # 保留的本地数据，不属于源码
├── langgraph.json               # 注册 phase_chat / project_status / HTTP
└── requirements.txt
```

目录参考 [LangGraph 官方应用结构](https://docs.langchain.com/oss/python/langgraph/application-structure)，按本项目规模细分职责。先读 `docs/architecture/graph.md`，再进入对应 `graph.py`；业务算法供节点调用，避免把算法改写成图调度器。科学项目的数据目录与这里的源码目录独立。

原顶层分层包与旧入口已移出源码目录；备份位于同级 `_Process_PhaseDiagram_source_backups`。不要把备份加入 PYTHONPATH。保存的旧适配器模块引用由 `phase_agent/compatibility/module_references.py` 明确转换；配置字段、科学状态及审批内容保持原格式。

## 四 Agent决策

探索专家贯通 branch 与 Relax/MC，可靠性专家贯通 DFT、微调及验证，Strategic比较路线并形成唯一待审方案。探索与可靠性专家使用同一事实快照；Simulation 在策略提案后规划执行、检查工具结果并诊断异常；专家意见不替代审批。普通问答、既有待审方案和历史任务继续沿用原入口。详见 [架构阅读指南](docs/architecture/four-agents.md)。

## 项目文件

项目数据统一按 `epoch0/`、`epoch1/` 组织：交互记录在 `interaction_records/epochN/`，提交与回传在 `submissions/epochN/`，分析结果在 `analysis_outputs/epochN/`。完整模型版本写在各轮说明和机器记录中。目录树、上传/回传入口及代码维护点见 [项目文件管理](docs/guides/project-files.md)。

## 快速开始

默认本地环境为 `py1`。在源码目录启动，项目数据目录可以独立存放：

```powershell
conda activate py1
Set-Location 'E:\jupyter notebook\1-AL_for_PhaseDiagram\Process_PhaseDiagram'
python -m phase_agent.runtime.local_project_launcher
```

启动器中选择已有 `agent_runtime.json`，或新建独立项目并选择包含初始结构的目录。启动器是项目选择窗口，Studio 提供后续对话和图调试。

启动前或遇到故障时，可用 `python -m phase_agent.runtime.project_diagnostics --runtime '项目目录/agent_runtime.json'` 独立检查。它不启动服务、不修改项目，逐项返回文件/字段问题和恢复步骤；见 [检查与恢复](docs/guides/recovery.md)。

已有项目也可直接启动：

```powershell
python -m phase_agent.runtime.studio_service --runtime-config 'E:\0-PhaseDiagram\O3-2ele-NaFeMn\agent_runtime.json'
```

依赖见 [requirements.txt](requirements.txt)；本地安装不改变超算科学环境。各项目使用独立数据目录；同时运行多个服务还需要各自独立端口，当前控制服务默认端口为 8765。

## 配置与日常对话

| 文件 | 用途 |
| --- | --- |
| `agent_runtime.json` | 项目启动绑定、服务与路径设置 |
| `parameters/search_config.project.json` | 初始配置：两端环境、边界、体系组成、母结构、模型与计算设置 |
| `parameters/run_config.project.json` | 运行配置：预算、采样、MC、微调、验证、收敛与策略 |
| `parameters/snapshots/` | 每次确认后的完整有效配置版本 |

初始必要信息齐全即可进入搜索建议。缺项、冲突或歧义由 Agent 再次询问；后续阶段缺少的阈值到使用时检查。新项目自动生成两份科学配置；旧单文件仍兼容。

可手动编辑配置，也可直接告诉 Agent“修改运行配置：把入选上限改为……”。手动保存后说“读取配置 JSON”进行检查，或“读取配置 JSON 并继续”在检查通过后推进。任一文件变化都会使旧审核失效，历史作业仍保留原配置版本。

- **查看状态**：只查看，不推进。
- **继续**：核对回传并推进到下一处需要介入的位置。
- **激活候选 `<版本>` 原因：`<理由>`**：独立验证通过后单独批准模型。
- **拒绝候选 `<版本>` 原因：`<理由>`**：记录审阅拒绝。

日常对话支持上下文语义识别，表中的命令是便捷入口，并非唯一可接受措辞。普通对话保持一次语义理解；新的科学提案由Strategic按需咨询探索与可靠性专家，再形成一个动作；程序检查权限、参数和预算。默认四个角色复用当前配置模型，Codex接管仍不调用项目配置模型。见 [Graph 架构与维护入口](docs/architecture/graph.md#doc-graph_architecture)。

本地普通环境默认 py1；本地/远端科学环境分别配置，远端名称由用户提供，不从本地猜测。超算模型路径只作为本地元数据，不在本地加载。详情见 [初始与运行配置](docs/guides/configuration.md#doc-initial_and_run_configuration)。

## 搜索与模型更新

```text
Agent推荐branch → 合法性/覆盖/成本检查 → Relax与凸包预筛 → 分档MC
→ Agent挑选DFT → 计算节点执行 → 回传与分析
→ 比较补DFT、继续搜索或模型更新的预期收益与成本
```

覆盖率和误差是证据，不直接决定进入下一轮。Agent 比较边际收益、成本和不确定性，再提出方案；已选择动作的实际结果与成本进入事实记忆，不虚构未执行方案的收益。见 [轮次预算比较](docs/guides/search.md#doc-round_budget_review)。

初始单模型无法提供委员会QBC。微调采用分组K折评估和全数据训练的最终committee；K折误差不等于最终主模型的独立测试。只有模型独立验证通过、用户单独批准激活后才进入新 epoch；同模型多轮MC不增加epoch。

## 超算作业与结果回传

Agent 生成输入和Slurm脚本，你在超算提交。登录节点负责提交/查询；计算、模型评估及模型文件SHA256均在计算节点执行。模型和大日志留在超算，只回传本地分析必需文件。

| 作业 | 提交入口 | 回传 |
| --- | --- | --- |
| Relax / MC | 登记的完整 `_input` 内 `submit_all.sh`，一次 | 配对 `_output` |
| DFT | 登记的完整 `_input` 内 `submit_all.sh`，一次 | 配对 `_output` |
| 微调 | 登记的完整 `_input` 内 `submit_all.sh`，一次 | 配对 `_output`，含指标/模型清单/完成标记 |
| 补清单 | 原训练轮 `inputs/GPU_manifest.sh` | `results/models.json` |
| 独立验证 | 原训练轮 `inputs/GPU_validation.sh` | 指定的验证JSON |

回传后说“继续”。默认每个 GPU 作业内串行分组：Relax 至多100个 task、MC 至多20个 task，末组可少于此数；DFT每组1个结构。组容量由 `supercomputer.batch_sizes` 配置，与累计预算分别管理。外层交付脚本派发内部计算脚本，不需另行提交内部 GPU.sh；历史任务沿原登记脚本和路径回收，已提交任务不重复提交。

详细说明：[训练交接](docs/guides/training.md#doc-training_handoff)、[轻量回传](docs/guides/workspace.md#doc-lightweight_result_transfer)、[每轮传输布局](docs/guides/workspace.md#doc-round_transfer_layout)。无常驻服务的分步模式见 [phase_agent/runtime/README.md](phase_agent/runtime/README.md)。

## LangGraph 与恢复

Studio 的 `phase_chat` 进入科学主图，主图包含决策动作、审批执行和 `training_lifecycle` 子图。训练交接有独立的回收、清单检查、作业准备、验证登记与等待节点。

等待清单、验证结果、配置补齐和激活决定使用原生 `interrupt`；项目 `workflow_state/langgraph_training.sqlite` 保存检查点。重启后“继续”携带最新业务状态重新核对文件，不直接执行旧检查点中的建议。作业文件生成保持幂等，模型激活仍使用独立审批。

`workflow_state/state.json` 是业务状态事实源；检查点记录恢复位置。其他搜索阶段沿用现有状态与审批恢复机制，本项目并未把全部科学计算都改成一个全局检查点流程。

Relax、MC与DFT现有回收器完成核对后，回收子图并行汇总部分回传、等待和失败；项目保存 `langgraph_batches.sqlite` 与节点追踪 `node_trace.jsonl`。重试仍需现有审批，不自动追加超算作业。见 [阶段回收与追踪](docs/architecture/graph.md#doc-batch_recovery_and_trace)。

查看 [真实科学图总览](docs/architecture/flow.md)、[Graph 架构](docs/architecture/graph.md#doc-graph_architecture) 和 [Studio启动说明](docs/guides/start.md#doc-studio_local)。实际HPC推理需通过提交作业验证。

## 数据与记忆

项目数据与源码分开保存：

```text
parameters/       初始/运行配置与版本快照
structures/       母结构和候选结构
submissions/      epoch下的提交inputs与原始results
analysis_outputs/ 相图、误差、训练数据与诊断
workflow_state/   状态、台账、审批和检查点
agent_memory/     项目记忆视图
logs/             服务日志
history_backups/  历史备份
```

优先查看项目的 `analysis_outputs/output_index.md` 和 `round_summary.csv`。事实、交接和动作收益进入记忆候选；长期经验经过审阅，不自动把一次建议晋升为长期规则。详见 [工作区布局](docs/guides/workspace.md#doc-workspace_layout) 与 [决策记忆](docs/guides/memory.md#doc-decision_memory)。

## 开发导航

| 目录 | 职责 |
| --- | --- |
| phase_agent/graphs | LangGraph图、节点与运行上下文 |
| phase_agent/configuration | 默认配置、校验、会话与有效配置 |
| phase_agent/agents | Agent建议、策略与评分 |
| phase_agent/tools | 审批、预算、任务、回收与作业生成 |
| phase_agent/science | 结构、MLIP、MC、DFT、QBC与训练 |
| phase_agent/persistence | 台账、模型状态与记忆 |
| phase_agent/analysis | 相图、误差、覆盖、成本与反馈 |
| phase_agent/runtime | 正式启动及对话入口 |
| tests / experiments | 回归测试与独立方法比较 |

程序接口从 `from phase_agent.runtime import run_workflow` 进入。新增后端通过adapter/handler接入，不绕过权限、预算和模型版本校验。

文档索引见 [docs/README.md](docs/README.md)，源码定位与修改约定见 [源码指南](docs/architecture/development.md#doc-source_guide)。开发默认使用 py1；开发工具依赖在 requirements-dev.txt，格式和静态检查使用 ruff.toml。运行与修改遵循 [AGENTS.md](AGENTS.md)。
# Codex 接管入口

Codex 接管只替换决策生成器，仍复用 Agent 的上下文、合法 action 校验、审批、执行隔离和记忆写回。

本地命令入口：

```bat
codex-agent.cmd mcp --runtime-config E:\path\to\agent_runtime.json
```

该命令通过 stdio 提供 MCP 工具。固定项目模式使用 `--runtime-config`；通用模式省略它并传入一个或多个 `--project-root`，然后在对话内调用 `phase_list_projects` 和 `phase_select_project`。`phase_propose` 只进入待审批状态；只有显式调用 `phase_decide` 才会进入原审批处理。

通用入口命令：

```powershell
& "E:\\jupyter notebook\\1-AL_for_PhaseDiagram\\Process_PhaseDiagram\\codex-mcp.cmd" `
  --repo-root "E:\\jupyter notebook\\1-AL_for_PhaseDiagram\\Process_PhaseDiagram" `
  --project-root "E:\\0-PhaseDiagram"
```

新对话中的使用顺序：

1. 先调用 `phase_takeover_repo`，确认当前 MCP 已接管源码 repo。
2. 调用 `phase_list_projects`，查看可接管项目和对应的 `agent_runtime.json`。
3. 调用 `phase_select_project`，传入要接管的 `runtime_config` 完整路径；如果要新建项目，调用 `phase_create_project` 并传入 `workspace_root`。
4. 调用 `phase_current_project` 和 `phase_context` 核对绑定，再按原审批流程推进。

新项目只要已经生成标准的 `agent_runtime.json`，并位于 `--project-root` 下，就会自动出现在列表中。项目在其他目录时，可以把该目录加入多个 `--project-root`，或直接把它的 `agent_runtime.json` 传给 `phase_select_project`。每个 MCP 对话进程只绑定一个当前项目，不会混用不同项目的账本。

在新 Codex 对话中可以直接使用下面的自然语言入口：

```text
接管 repo：E:\jupyter notebook\1-AL_for_PhaseDiagram\Process_PhaseDiagram。
然后接管项目：E:\0-PhaseDiagram\NaNiO2-phase。
先做 phase_context 核验，不提交作业。
```

新项目：

```text
接管 repo：E:\jupyter notebook\1-AL_for_PhaseDiagram\Process_PhaseDiagram。
在 E:\0-PhaseDiagram\NewProject 下新建项目，然后选中它并运行初始化流程。
```

Codex 客户端的 MCP 配置可指向这个命令（路径按实际项目目录替换）：

```json
{
  "mcpServers": {
    "phase-agent-codex": {
      "command": "E:\\jupyter notebook\\1-AL_for_PhaseDiagram\\Process_PhaseDiagram\\codex-agent.cmd",
      "args": ["mcp", "--runtime-config", "E:\\0-PhaseDiagram\\O3-1ele-NaCrO2\\agent_runtime.json"]
    }
  }
}
```


Codex 原生四 Agent接管支持在策略决策中补充人的约束，并读取已有审核记忆；方向接受与执行审批独立，详见 [Codex 专家协作与人工介入](docs/guides/codex-team.md)。

当前目录与文件归属见 [仓库架构](docs/architecture/repository-layout.md)，新材料与跨 epoch 流程见 [科研生命周期](docs/architecture/workflow.md)；重构和旧项目只读检查见 [四 Agent 验收](docs/architecture/four-agents-validation.md)。
