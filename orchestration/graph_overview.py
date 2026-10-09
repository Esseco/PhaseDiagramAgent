"""Read-only documentation projection of compiled graphs; never invokes them."""

LABELS = {
    "branch_search": "生成与去重branch",
    "structure_and_batch_inputs": "筛选结构与准备批次输入（Relax/MC/DFT）",
    "mc_search": "分配MC搜索",
    "dft_inputs": "筛选DFT并准备输入",
    "mlip_finetune": "准备MLIP微调训练输入",
    "model_refresh": "新模型刷新结构",
    "convergence_and_control": "收敛、暂停与维护",
    "__start__": "开始", "__end__": "返回本次结果",
    "initialize_confirmed_run": "加载已确认运行",
    "collect_and_reconcile": "回收与核对结果",
    "batch_recovery": "Relax/MC/DFT回收汇合子图",
    "training_lifecycle": "训练回收与原生暂停子图",
    "scientific_feedback": "分析科学反馈",
    "results_wait_gate": "判断是否等待回传",
    "assess_and_export_round": "评估本轮并更新输出",
    "bounded_action_graph": "进入决策与动作子图",
    "prepare_inputs_and_finalize": "汇总本次结果",
    "execute_validated_action": "进入审批与执行子图",
    "persist_action_outcome": "记录结果与成本并判断是否继续",
    "initialize_action": "初始化动作与防重检查",
    "model_refresh_barrier": "检查模型刷新前置条件",
    "prepare_proposal": "形成决策与方案",
    "approval_gate": "执行审批策略：待确认则返回",
    "validate_action": "校验权限、参数与预算",
    "audit_outcome": "审计动作结果",
    "selected_scientific_action": "执行唯一获准动作：准备输入或其他操作",
}


def mermaid_overview(compiled_graph, *, group_tools=False):
    """Collapse only tool leaves in the documentation, not the executable graph."""
    topology = compiled_graph.get_graph()

    def project(name):
        return "selected_scientific_action" if group_tools and name.startswith("tool__") else name

    names = list(dict.fromkeys(project(name) for name in topology.nodes))
    identifiers = {name: f"n{index}" for index, name in enumerate(names)}
    lines = ["```mermaid", "flowchart TD"]
    for name in names:
        lines.append(f'  {identifiers[name]}["{LABELS.get(name, name)}"]')
    edges = set()
    for edge in topology.edges:
        source, target = project(edge.source), project(edge.target)
        key = (source, target, edge.conditional)
        if key not in edges:
            edges.add(key)
            arrow = "-.->" if edge.conditional else "-->"
            lines.append(f"  {identifiers[source]} {arrow} {identifiers[target]}")
    lines.append("```")
    return "\n".join(lines)


def render_overview():
    from orchestration.scientific_graph import scientific_graph

    lifecycle = scientific_graph()
    children = dict(lifecycle.get_subgraphs(recurse=True))
    iteration = children["bounded_action_graph"]
    actions = children["bounded_action_graph|execute_validated_action"]
    all_names = set(actions.get_graph().nodes)
    for _, child in actions.get_subgraphs(recurse=True):
        all_names.update(child.get_graph().nodes)
    tools = sorted(name.removeprefix("tool__") for name in all_names if name.startswith("tool__"))
    sections = [
        "# 科学流程总览",
        "本文件由 `python -m orchestration.graph_overview` 从真实编译图生成；仅用于阅读，不是第二套执行流程。实线为固定连线，虚线为条件路由，并非所有路径都会执行。",
        "## 1. 本轮生命周期",
        mermaid_overview(lifecycle),
        "## 计算阶段回收子图",
        mermaid_overview(children["batch_recovery"]),
        "各阶段并行检查已登记任务状态并汇合；部分回传暂停保存检查点，失败由现有审批重试入口处理。",
        "## 训练回收与验证子图",
        mermaid_overview(children["training_lifecycle"]),
        "等待节点使用原生interrupt，检查点保存在项目workflow_state/langgraph_training.sqlite；继续时携带最新业务状态重新核对。",
        "## 2. 决策动作循环",
        mermaid_overview(iteration),
        "## 3. 审批与动作执行",
        mermaid_overview(actions, group_tools=True),
        "工具叶节点只在本说明图中合并。真实工具如下（每次只选一个，不是顺序流水线）：",
        "\n".join(f"- `{name}`" for name in tools),
        "## Studio 阅读方式",
        "聊天入口为 `phase_chat`。查看子图时按以下路径逐层定位；不同 Studio 版本的展开控件可能不同：",
        "1. `confirmed_local_chat` → `scientific_lifecycle`：回收、分析、等待和评估。\n"
        "训练交接：`training_lifecycle` → `training_lifecycle`，展开检查、准备、验证与等待节点。\n"
        "2. `bounded_action_graph` → `bounded_action_iteration`：动作循环与持久化。\n"
        "3. `execute_validated_action` → `approved_scientific_action`：提案、审批、校验及具体工具。",
        "本总览不会替换 Studio 画布。实际审批与执行规则未修改；交互模式批准后通常只准备超算输入，由用户提交，再回传结果。等待、拒绝和失败均可能提前返回。",
    ]
    return "\n\n".join(sections) + "\n"


if __name__ == "__main__":
    print(render_overview(), end="")
