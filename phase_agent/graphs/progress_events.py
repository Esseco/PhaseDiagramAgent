"""Native LangGraph streaming events, with concise stage labels."""

LABELS = {
    "initialize_confirmed_run": "核对项目配置",
    "collect_and_reconcile": "回收结果",
    "batch_recovery": "核对计算任务",
    "training_lifecycle": "核对微调结果",
    "edge_direction_review": "Agent判断下一步方向",
    "scientific_feedback": "分析结果",
    "results_wait_gate": "检查审批与等待状态",
    "assess_and_export_round": "评估本轮收益",
    "bounded_action_graph": "准备或执行已批准方案",
    "prepare_inputs_and_finalize": "保存本轮结果",
}


def emit_progress(node, status):
    from langgraph.config import get_stream_writer

    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer(
        {
            "kind": "scientific_progress",
            "node": node,
            "label": LABELS.get(node, node),
            "status": status,
        }
    )
