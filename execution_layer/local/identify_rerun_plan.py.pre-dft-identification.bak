"""Read-only identification of the round/action named by a redo request."""

from pathlib import Path
import re


_ROUND_WORDS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_GENERATE_WORDS = ("重新生成", "重新准备", "重建", "重做", "重跑")
_READ_WORDS = ("重新读取", "重读", "重新回收", "重新扫描")


def is_rerun_plan_request(message):
    text = str(message or "").lower().replace(" ", "")
    return any(word in text for word in (*_GENERATE_WORDS, *_READ_WORDS))


def identify_rerun_plan(message, state):
    """Return a factual plan or ambiguity; never mutate state or files."""
    text = str(message or "").lower().replace(" ", "")
    if not is_rerun_plan_request(text):
        return None
    intent = "read" if any(word in text for word in _READ_WORDS) else "generate"
    records = list(state.get("action_records") or [])
    anchors = [index for index, row in enumerate(records)
               if row.get("status") == "completed"
               and ((row.get("final_action") or {}).get("tool") == "generate_branches")]
    if not anchors:
        return {"status": "round_unknown", "intent": intent,
                "reason": "状态中没有可用的已完成 branch 生成 action，无法确定搜索轮次"}
    requested_round = _round_number(text)
    if requested_round is None:
        if "上一轮" in text or "前一轮" in text:
            round_number = len(anchors) - 1
        elif len(anchors) == 1 or "当前轮" in text or "本轮" in text:
            round_number = len(anchors)
        else:
            return {"status": "round_ambiguous", "intent": intent,
                    "candidate_rounds": [
                        {"round_number": number, "anchor_record_id": records[index].get("record_id")}
                        for number, index in enumerate(anchors, start=1)]}
    else:
        round_number = requested_round
    if round_number < 1 or round_number > len(anchors):
        return {"status": "round_unknown", "intent": intent,
                "reason": f"只找到 {len(anchors)} 个有明确生成锚点的轮次，无法定位第 {round_number} 轮"}
    start = anchors[round_number - 1]
    end = anchors[round_number] if round_number < len(anchors) else len(records)
    anchor = records[start]
    base = {"intent": intent, "round_number": round_number,
            "round_anchor_record_id": anchor.get("record_id"),
            "round_anchor_task_key": (anchor.get("final_action") or {}).get("task_key")}
    if intent == "read":
        return _identify_read(text, state, base, latest=round_number == len(anchors))
    viable = [row for row in records[start:end]
              if row.get("status") in {"completed", "prepared"}
              and (row.get("final_action") or {}).get("tool")]
    named_record = next((row for row in viable if row.get("record_id")
                         and row["record_id"].lower() in text), None)
    matches = [named_record] if named_record else [row for row in viable
        if _matches_generation_target(text, row.get("final_action") or {})]
    if not matches:
        return {**base, "status": "action_unknown", "reason": "本轮没有与请求匹配的已执行生成 action",
                "candidates": [_action_summary(row) for row in viable]}
    if len(matches) > 1:
        return {**base, "status": "ambiguous", "reason": "本轮有多个可能的生成 action，请指定 record_id 或阶段",
                "candidates": [_action_summary(row) for row in matches]}
    row = matches[0]
    action = row["final_action"]
    branch_ids = set((state.get("branch_batch") or {}).get("branch_ids") or [])
    related = _current_round_tasks(state, branch_ids) if round_number == len(anchors) else []
    affected = _affected_tasks(action, related)
    return {**base, "status": "identified", "action": _action_summary(row),
            "affected_task_count": len(affected),
            "affected_by_stage": _count_by_stage(affected),
            "existing_input_count": sum(Path(task.get("input_path") or "").is_file()
                                        for task in affected),
            "scope_note": ("已按当前 branch_batch 关联任务；执行前仍须单独核对依赖和清理范围"
                           if round_number == len(anchors) else
                           "历史轮次缺少可靠的任务归属标记；不能据此自动清理或重做")}


def format_rerun_plan(plan):
    if plan["status"] == "round_ambiguous":
        choices = "、".join(
            f"第 {row['round_number']} 轮（`{row['anchor_record_id']}`）"
            for row in plan["candidate_rounds"])
        return f"无法确定你指哪一轮：{choices}。请指定轮次；没有执行任何操作。"
    if plan["status"] == "round_unknown":
        return f"无法确定要重做哪一轮：{plan['reason']}。没有执行任何操作。"
    prefix = (f"定位到第 {plan['round_number']} 轮（起点 action "
              f"`{plan['round_anchor_record_id']}`）。")
    if plan["status"] in {"ambiguous", "action_unknown"}:
        choices = "、".join(f"`{row['record_id']}` {row['tool']}（{row['status']}）"
                           for row in plan.get("candidates") or []) or "无"
        return f"{prefix}{plan['reason']}。候选：{choices}。没有执行任何操作。"
    if plan["intent"] == "read":
        if plan["read_operation"] != "collect_results_with_report":
            return (f"{prefix}对应读取环节：{plan['read_operation']}。"
                    "当前只识别目标，没有执行额外的重读或修改状态。")
        return (f"{prefix}对应读取环节：{plan['read_operation']}；"
                f"关联任务 {plan['task_count']} 个，现有结果文件 {plan['result_file_count']} 个。"
                "这是工作流读取环节，不是独立的 action_record。"
                "当前只识别方案，没有重新读取或回收结果。")
    action = plan["action"]
    stages = "、".join(f"{key} {value} 个" for key, value in plan["affected_by_stage"].items()) or "无"
    return (f"{prefix}匹配 action `{action['record_id']}`：{action['tool']} "
            f"（task_key=`{action['task_key']}`，状态 {action['status']}）。"
            f"当前可关联下游任务 {plan['affected_task_count']} 个（{stages}），"
            f"已有输入文件 {plan['existing_input_count']} 个。{plan['scope_note']}。"
            "当前只识别方案；尚未删除文件、撤销旧任务或重新执行。")


def _round_number(text):
    match = re.search(r"第(\d+|[一二三四五六七八九十])轮", text)
    if not match:
        return None
    token = match.group(1)
    return int(token) if token.isdigit() else _ROUND_WORDS[token]


def _matches_generation_target(text, action):
    tool = action.get("tool")
    mode = (action.get("parameters") or {}).get("mode")
    if any(word in text for word in ("branch", "分支", "初态")):
        return tool == "generate_branches"
    if "mc" in text or "蒙特卡洛" in text:
        return tool == "allocate_mc_bohb" or mode == "mc_inputs"
    if any(word in text for word in ("relax", "弛豫")):
        return tool == "prepare_local_batch_files" and mode == "relax_inputs"
    if "dft" in text:
        return tool == "select_dft_candidates"
    return tool in {"generate_branches", "prepare_local_batch_files",
                    "allocate_mc_bohb", "select_dft_candidates"}


def _action_summary(row):
    action = row.get("final_action") or {}
    return {"record_id": row.get("record_id"), "tool": action.get("tool"),
            "task_key": action.get("task_key"), "status": row.get("status"),
            "mode": (action.get("parameters") or {}).get("mode"),
            "target_count": len(action.get("target_ids") or [])}


def _current_round_tasks(state, branch_ids):
    if not branch_ids:
        return []
    return [row for row in state.get("tasks") or [] if row.get("branch_id") in branch_ids]


def _affected_tasks(action, tasks):
    tool = action.get("tool")
    mode = (action.get("parameters") or {}).get("mode")
    if tool == "generate_branches":
        return tasks
    stage = ("relax_and_feature" if mode == "relax_inputs" else
             "deep_search" if tool == "allocate_mc_bohb" or mode == "mc_inputs" else None)
    return [row for row in tasks if row.get("stage") == stage] if stage else []


def _count_by_stage(tasks):
    counts = {}
    for row in tasks:
        stage = row.get("stage") or "unknown"
        counts[stage] = counts.get(stage, 0) + 1
    return dict(sorted(counts.items()))


def _identify_read(text, state, base, *, latest):
    if "state" in text or "状态" in text:
        return {**base, "status": "identified", "read_operation": "read_state"}
    if "ledger" in text or "台账" in text:
        return {**base, "status": "identified", "read_operation": "read_ledger"}
    if not any(word in text for word in ("结果", "回收", "扫描", "任务")):
        return {**base, "status": "action_unknown",
                "reason": "请指出要重新读取的是任务结果、state 还是台账",
                "candidates": []}
    branch_ids = set((state.get("branch_batch") or {}).get("branch_ids") or [])
    tasks = _current_round_tasks(state, branch_ids) if latest else []
    stage = ("deep_search" if "mc" in text or "蒙特卡洛" in text else
             "relax_and_feature" if "relax" in text or "弛豫" in text else None)
    if stage:
        tasks = [row for row in tasks if row.get("stage") == stage]
    return {**base, "status": "identified", "read_operation": "collect_results_with_report",
            "task_count": len(tasks),
            "result_file_count": sum(Path(row.get("result_path") or "").is_file() for row in tasks),
            "scope_note": "仅当前轮任务可可靠关联" if latest else "历史轮任务归属不明，不能自动回收"}
