# 科学流程总览

本文件由 `python -m orchestration.graph_overview` 从真实编译图生成；仅用于阅读，不是第二套执行流程。实线为固定连线，虚线为条件路由，并非所有路径都会执行。

## 1. 本轮生命周期

```mermaid
flowchart TD
  n0["开始"]
  n1["加载已确认运行"]
  n2["回收与核对结果"]
  n3["分析科学反馈"]
  n4["判断是否等待回传"]
  n5["评估本轮并更新输出"]
  n6["进入决策与动作子图"]
  n7["汇总本次结果"]
  n8["返回本次结果"]
  n0 --> n1
  n5 --> n6
  n6 --> n7
  n2 --> n3
  n1 -.-> n8
  n1 -.-> n2
  n4 -.-> n8
  n4 -.-> n5
  n3 --> n4
  n7 --> n8
```

## 2. 决策动作循环

```mermaid
flowchart TD
  n0["开始"]
  n1["进入审批与执行子图"]
  n2["记录结果与成本并判断是否继续"]
  n3["返回本次结果"]
  n0 --> n1
  n1 --> n2
  n2 -.-> n3
  n2 -.-> n1
```

## 3. 审批与动作执行

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

1. `confirmed_local_chat` → `scientific_lifecycle`：回收、分析、等待和评估。
2. `bounded_action_graph` → `bounded_action_iteration`：动作循环与持久化。
3. `execute_validated_action` → `approved_scientific_action`：提案、审批、校验及具体工具。

本总览不会替换 Studio 画布。实际审批与执行规则未修改；交互模式批准后通常只准备超算输入，由用户提交，再回传结果。等待、拒绝和失败均可能提前返回。
