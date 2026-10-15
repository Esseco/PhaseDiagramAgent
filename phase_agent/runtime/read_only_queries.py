"""Deterministic queries never collect results or enter scientific execution."""

import json
from pathlib import Path


def read_only_reply(handler, message, facts, conversation_id=None):
    from phase_agent.graphs.dialogue.commands import _is_status_command, format_status_reply
    from phase_agent.runtime.plan_queries import is_plan_query, pending_plan_reply
    from phase_agent.runtime.artifact_queries import finetune_location_reply

    if handler.config_delegate is not None:
        return None
    if _is_status_command(message):
        return format_status_reply(facts)
    if is_plan_query(message):
        from phase_agent.graphs.dialogue.support import bind_presented_proposal
        bind_presented_proposal(handler, facts)
        return pending_plan_reply(facts)
    reply = finetune_location_reply(message, facts)
    if reply is not None:
        return reply
    command = "".join(str(message).lower().split()).strip("？?。.!！")
    if command in {"上传到哪里", "回传到哪里", "交哪个文件夹", "查看上传目录", "查看回传目录", "查看提交文件夹"}:
        bundles = [row for row in (facts.get("relax_mc_batches") or {}).values()
                   if row.get("status") not in {"completed", "cancelled", "failed"}
                   and row.get("upload_folder")]
        if bundles:
            latest = bundles[-1]
            uploads = [str(latest["upload_folder"])]
            folders = [str(Path(latest["expected_results_archive"]).parent)] if latest.get("expected_results_archive") else []
        else:
            active_ids = {row.get("slurm_batch_id") for row in facts.get("tasks") or []
                          if row.get("status") not in {"completed", "cancelled", "failed", "timeout"}}
            batches = [row for row in facts.get("slurm_batches") or []
                       if row.get("batch_id") in active_ids and row.get("status") != "cancelled"]
            handoffs = [row.get("handoff") or {} for row in batches]
            uploads = sorted({str(row["folder"]) for row in handoffs if row.get("folder")})
            folders = sorted({str(row["return_directory"]) for row in handoffs if row.get("return_directory")})
        lines = []
        if uploads:
            lines.append("已登记提交目录：\n" + "\n".join("- `" + p + "`" for p in uploads))
        if folders:
            lines.append("已登记回传目录：\n" + "\n".join("- `" + p + "`" for p in folders))
        return "\n".join(lines) + "\n以上为已登记目录，本次未生成或提交任务。" if lines else "当前没有可核实的提交或回传目录。本次未生成任务，请查看已有批次的交付说明。"
    if command in {"查看上次回复", "查看上次结果", "上次回复是什么"}:
        directory = Path(handler.state_path).parent / "dialogue_process"
        for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
            record = json.loads(path.read_text(encoding="utf-8"))
            if str(record.get("thread_id")) != str(conversation_id):
                continue
            replies = [event.get("data", {}).get("reply") for event in record.get("events") or [] if event.get("title") == "本轮答复"]
            if replies and replies[-1]:
                return "上次保存的回复（" + str(record.get("time", "")) + "）：\n" + str(replies[-1])
        return "当前会话没有可读取的已保存回复。本次未重跑分析或生成任务。"
    return None
