# Agent-driven materials search

项目以 Agent 为流程调度者：QBC、凸包和覆盖率只提供可验证的科学观测；Agent 在已确认的边界、合法操作、冻结参数和预算内提出 action；所有 action 经过 Execution Policy 和执行层校验后才会派发。

## 目录

```text
config_layer/       defaults/ schema/ session/ runtime/
decision_layer/     agent/ strategy/ calculation/ scoring/ qbc_selection/
execution_layer/    policy/ state/ budget/ dispatch/ workflows/ slurm/
scientific_layer/   structures/ mlip/ mc/ dft/ qbc/ bohb/ features/ training/
data_layer/         ledger/ memory/ state/ models/
analysis_layer/     phase/ convergence/ feedback/ state/ cost/
run/                唯一正式启动入口
experiments/        不写正式状态的独立方法比较
tests/              单元测试和跨层集成测试
```

## 超算分步模式

计算节点无公网时使用 `python -m run.step_runner`。登录节点只执行 `advise`、
`confirm`、`submit` 和只读 `status`；计算节点只执行 `recover`、`prepare` 以及正式科学
作业。双方通过原子写入的 JSON 摘要、动作计划、manifest 和结果文件通信，不要求任何
常驻服务。

```text
compute: recover → login: advise → login: confirm → compute: prepare
→ login: submit → compute: scientific jobs → compute: recover
```

默认分组为 Relax 100 个 task/job、MC 20 个 task/job、DFT 1 个 task/job。配置位于已确认
快照的 `supercomputer.batch_sizes`，因此修改会产生新的配置版本。同组 task 必须具有
相同 stage、模型版本、参数和资源 profile。详情见 [分步运行说明](run/README.md)。

## 调用流程

```text
confirmed config + persisted state
→ state_manager 生成只读 state_t snapshot
→ decision_layer 仅依据 snapshot 提出统一 action
→ execution_layer Execution Policy
→ 版本/冻结参数/权限/预算校验
→ scientific_layer 计算后端
→ data_layer 持久化任务与结果
→ analysis_layer 生成反馈摘要和收敛证据
→ state_manager 生成 state_{t+1}
→ Agent 进入下一轮
```

QBC 只计算委员会分歧，不选择 DFT action。DFT、重训练、继续、暂停和停止均由 Agent 提议。

默认闭环使用“Agent 推荐 branch → 合法/覆盖/成本预筛 → Relax/Hull 预筛 → 分档 MC
→ 近 hull DFT 单点优先验证 → 有效数据达到门槛后模型更新 → 下一模型版本相图反馈”。
这一路径准确称为 Relax/Hull 预筛加分档 MC，不称为 BOHB。每批决策冻结 MLIP 版本和
凸包版本；模型相关能量池隔离保存。Relax 三初态能量标准差只是弱探索项，不是 QBC。
相图参考采用明确的 eV/O2 口径，模型误差采用 eV/atom；用户提供的金属电压参考只记录
数值和来源，不由程序替用户判断可靠性。

一次 epoch 只表示一次通过独立验证并激活的 MLIP 更新。同一模型下多个 MC 搜索段不增加
epoch。连续两个模型更新 epoch 的 MAE、允许相稳定区间和 hull 变化达到数值阈值后，
状态为“数值达标、等待用户接受”；相×SOC 覆盖作为透明证据报告，不是自动硬收敛条件。
预算耗尽、证据不足、数值达标待确认和用户确认后的结束是不同状态。

Agent 上下文区分人工长期知识和短期运行记忆。长期层只由人工显式修订；短期层从
当前相图、覆盖、QBC、预算、收益、失败任务和近期 action 自动重建，不会自动晋升为
长期知识。使用方法见 [决策记忆](data_layer/DECISION_MEMORY.md)。

## 运行

### Open WebUI 对话模式

首次启动进入配置专用对话：Agent 修改的只是草稿，用户发送“确认配置”后才保存配置版本快照；确认本身不启动任何计算。母结构路径和超算端 mh-1 模型路径作为待审核候选，超算模型不会在本地加载。

可用 Open WebUI 作为本地 Agent 的浏览器界面。项目实现了 OpenAI-compatible
服务入口 `run.open_webui_api`；聊天只负责呈现和传递真实用户消息，proposal、
审批、预算和计算仍由现有工作流控制。启动和配置步骤见
[Open WebUI 集成说明](run/OPEN_WEBUI.md)。

默认解释器为：

```text
C:\ProgramData\anaconda3\envs\py1\python.exe
```

Linux/WSL 对应路径通常为：

```text
/mnt/c/ProgramData/anaconda3/envs/py1/python.exe
```

正式入口：

```python
from run import default_run_config, run_workflow

result = run_workflow(
    manager,
    phase_references,
    default_run_config(),
    confirmed_config_session,
    dispatcher=dispatcher,
    agent_client=agent_client,
    execution_mode="interactive",  # 或 autonomous/dry_run/replay
    invocation_id="cycle-0001",
    max_steps=20,
    handlers={
        "update_mlip": model_update_handler,
    },
    task_runner=slurm_runner,  # 可选：回收上一批并按预算生成/提交下一批
)
```

首次配置先询问本地工作区根目录并展示派生保存路径；只有用户回复“确认存储路径”后，
才在该目录生成带注释的设置 JSON。用户编辑并导入后，Agent 检查草稿；只有用户发送“确认配置”后
才保存版本快照。这两个确认都不会启动计算。通常机器侧只需传入 `mlip.model_path` 指向 `mace-mh-1`。未填写 DFT 参数时，
atomate 使用自身默认 static/structure-optimization 配置。初始单模型不报告 QBC
不确定性；后续四成员 committee 固定第 0 个模型执行 Relax/MC，四个成员共同计算 QBC。

`generate_branches`、`allocate_mc_bohb`、`select_dft_candidates` 和
`run_calculation_stage` 已由正式入口提供默认 handler。Agent 选择本轮 Branch 批次并
给出 MC 总预算；Hyperband 分配 fidelity。BOHB 选择模型仅保留为关闭的实验接口。QBC 只生成不确定性
指标；Agent 提交分类式 DFT 决策，默认 handler 复用统一校验和预算事务生成 DFT
子任务。`dispatcher` 或 `stage_context_factory` 提供实际计算适配器。模型更新仍可
通过显式 handler 注入。

`run_active_learning_cycle.py` 仅用于旧状态/旧调用方兼容，不再是公开入口；新代码从 `from run import run_workflow` 启动。

`run_workflow` 从 confirmed snapshot 构建唯一 effective config。调用方传入的
DFT 参数、预算、随机种子等受控字段会被忽略；运行时只接受路径、后端环境、模型
文件位置和 API adapter 等机器相关设置。事件循环会在批准等待、异步任务、暂停、
预算耗尽、失败或收敛时返回，并可从 `state_path` 恢复。

真实后端环境保持原配置：主控使用 `py1`；MACE Relax、MC 和训练作业使用 `mace` 环境；atomate/DFT 通过原 adapter 或 dispatcher 注入。入口不会自行启动昂贵计算。

## 扩展位置

- 新体系或边界：`config_layer/`
- 新 Agent/规则策略：`decision_layer/`
- 新工具、handler、预算事务或调度方式：`execution_layer/`
- 新结构生成策略：`scientific_layer/structures/`
- 层氧候选超胞枚举：`scientific_layer/structures/enumerate_layered_oxide_supercells.py`；仅输出建议 H，不自动更改边界
- 新特征或代理模型：`scientific_layer/features/` 或 `scientific_layer/surrogate_models/`
- 新计算后端：`scientific_layer/mlip/`、`mc/`、`dft/`
- 新 QBC/BOHB/训练算法：对应 scientific 子目录
- 新分析指标：`analysis_layer/`
- 新离线比较或示例：`experiments/`

测试：

```bash
C:\ProgramData\anaconda3\envs\py1\python.exe -m pytest -q
```
