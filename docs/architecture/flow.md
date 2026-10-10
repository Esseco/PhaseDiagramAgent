# 科学流程总览

本文件由 `python -m phase_agent.graphs.graph_overview` 从真实编译图生成；仅用于阅读，不是第二套执行流程。实线为固定连线，虚线为条件路由，并非所有路径都会执行。

## 顶层请求流程

```mermaid
flowchart TD
  n0["开始"]
  n1["接收并检查消息"]
  n2["对话与决策流程"]
  n3["输出回复"]
  n4["返回本次结果"]
  n0 --> n1
  n2 --> n3
  n1 -.-> n2
  n1 -.-> n3
  n3 --> n4
```

展开 agent 查看真实对话分支；Chat 是交互界面，不是自动回跳节点。

```mermaid
flowchart TD
  n0["开始"]
  n1["读取项目"]
  n2["修改配置"]
  n3["查看配置"]
  n4["处理人工审批"]
  n5["Agent理解与决策"]
  n6["整理答复"]
  n7["科研流程"]
  n8["准备上下文与控制请求"]
  n9["返回本次结果"]
  n0 --> n1
  n5 -.-> n9
  n5 -.-> n2
  n5 -.-> n3
  n5 -.-> n6
  n5 -.-> n7
  n8 -.-> n9
  n8 -.-> n2
  n8 -.-> n5
  n8 -.-> n4
  n1 -.-> n9
  n1 -.-> n8
  n7 --> n6
  n2 --> n9
  n3 --> n9
  n6 --> n9
  n4 --> n9
```

## 1. 本轮生命周期

```mermaid
flowchart TD
  n0["开始"]
  n1["加载已确认项目"]
  n2["回收与核对事实"]
  n3["判断是否等待"]
  n4["分析本轮事实"]
  n5["单次模型决策、审批与执行"]
  n6["保存结果"]
  n7["等待审批或结果回传（原生暂停）"]
  n8["返回本次结果"]
  n0 --> n1
  n4 --> n5
  n6 -.-> n8
  n6 -.-> n7
  n1 -.-> n8
  n1 -.-> n2
  n2 --> n3
  n5 --> n6
  n3 -.-> n8
  n3 -.-> n4
  n3 -.-> n7
  n7 --> n8
```

## 计算阶段回收子图

```mermaid
flowchart TD
  n0["开始"]
  n1["reconcile_latest_task_status"]
  n2["relax_and_feature"]
  n3["deep_search"]
  n4["dft_single_point"]
  n5["dft_relax"]
  n6["join_stage_recovery"]
  n7["wait_for_partial_or_remaining_results"]
  n8["返回本次结果"]
  n0 --> n1
  n3 --> n6
  n5 --> n6
  n4 --> n6
  n6 -.-> n8
  n6 -.-> n7
  n1 --> n3
  n1 --> n5
  n1 --> n4
  n1 --> n2
  n2 --> n6
  n7 --> n1
```

各阶段并行检查已登记任务状态并汇合；部分回传暂停保存检查点，失败由现有审批重试入口处理。

## 训练回收与验证子图

```mermaid
flowchart TD
  n0["开始"]
  n1["collect_training_results"]
  n2["check_training_manifest"]
  n3["prepare_manifest_job"]
  n4["check_validation_prerequisites"]
  n5["prepare_validation_job"]
  n6["validate_and_register_candidate"]
  n7["persist_training_transition"]
  n8["wait_for_manifest_return"]
  n9["wait_for_validation_return"]
  n10["wait_for_activation_decision"]
  n11["wait_for_configuration_or_repair"]
  n12["返回本次结果"]
  n0 --> n1
  n2 -.-> n4
  n2 -.-> n7
  n2 -.-> n3
  n4 -.-> n7
  n4 -.-> n5
  n4 -.-> n6
  n1 -.-> n12
  n1 -.-> n2
  n7 -.-> n12
  n7 -.-> n10
  n7 -.-> n11
  n7 -.-> n8
  n7 -.-> n9
  n3 --> n7
  n5 --> n7
  n6 --> n7
  n10 --> n1
  n11 --> n1
  n8 --> n1
  n9 --> n1
```

等待节点使用原生interrupt，检查点保存在项目workflow_state/langgraph_training.sqlite；继续时携带最新业务状态重新核对。

## 2. 单次决策、审批与动作执行

```mermaid
flowchart TD
  n0["开始"]
  n1["初始化动作与防重检查"]
  n2["检查模型刷新前置条件"]
  n3["形成决策与方案"]
  n4["执行审批策略：待确认则返回"]
  n5["校验权限、参数与预算"]
  n6["审计动作结果"]
  n7["生成与去重branch"]
  n8["筛选结构与准备批次输入（Relax/MC/DFT）"]
  n9["收敛、暂停与维护"]
  n10["分配MC搜索"]
  n11["筛选DFT并准备输入"]
  n12["准备MLIP微调训练输入"]
  n13["新模型刷新结构"]
  n14["返回本次结果"]
  n0 --> n1
  n4 -.-> n14
  n4 -.-> n5
  n7 --> n6
  n9 --> n6
  n11 --> n6
  n1 -.-> n14
  n1 -.-> n2
  n10 --> n6
  n12 --> n6
  n13 --> n6
  n2 -.-> n14
  n2 -.-> n3
  n3 -.-> n14
  n3 -.-> n4
  n8 --> n6
  n5 -.-> n6
  n5 -.-> n7
  n5 -.-> n9
  n5 -.-> n11
  n5 -.-> n10
  n5 -.-> n12
  n5 -.-> n13
  n5 -.-> n8
  n6 --> n14
```

工具叶节点只在本说明图中合并。真实工具如下（每次只选一个，不是顺序流水线）：

- `adjust_strategy`
- `allocate_mc_bohb`
- `check_convergence`
- `generate_branches`
- `pause_search`
- `prepare_dedup_batch`
- `prepare_local_batch_files`
- `reevaluate_candidates`
- `restart_failed_task`
- `run_calculation_stage`
- `select_candidates`
- `select_dft_candidates`
- `update_mlip`

## Studio 阅读方式

聊天入口为 `phase_chat`。查看子图时按以下路径逐层定位；不同 Studio 版本的展开控件可能不同：

1. `agent` → `scientific_workflow` → `project_react_workflow`：回收、分析、等待和评估。
训练交接：`training_lifecycle` → `training_lifecycle`，展开检查、准备、验证与等待节点。
2. `react` → `approved_scientific_action`：直接进入提案、审批、校验和具体工具；执行后保存事实并结束本轮。

本总览不会替换 Studio 画布。实际审批与执行规则未修改；交互模式批准后通常只准备超算输入，由用户提交，再回传结果。等待、拒绝和失败均可能提前返回。
