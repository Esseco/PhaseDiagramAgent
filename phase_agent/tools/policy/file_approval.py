"""Linux/Windows 通用的非阻塞文件审批适配器。"""

import hashlib
import json
from pathlib import Path
import re


def write_approval_request(directory, invocation_id, response):
    revision = int(response.get("revision", 0))
    proposal = response["agent_proposal"]
    digest = proposal_hash(proposal)
    target = Path(directory) / _safe_name(invocation_id)
    target.mkdir(parents=True, exist_ok=True)
    envelope = {
        "proposal_id": invocation_id,
        "revision": revision,
        "proposal_hash": digest,
        "status": "awaiting_approval",
        "agent_proposal": proposal,
        "feedback_history": response.get("feedback_history") or [],
    }
    _atomic_json(target / f"proposal-r{revision:03d}.json", envelope)
    _atomic_text(target / f"proposal-r{revision:03d}.md", _render_markdown(envelope))
    decision_path = target / f"decision-r{revision:03d}.json"
    if not decision_path.exists():
        _atomic_json(
            decision_path,
            {
                "proposal_id": invocation_id,
                "revision": revision,
                "proposal_hash": digest,
                "decision": "comment",
                "comment": "",
                "long_term_advice": None,
            },
        )
    return {"directory": str(target), "proposal_hash": digest, "revision": revision}


def read_approval_decision(directory, invocation_id, pending):
    revision = int(pending.get("revision", 0))
    path = Path(directory) / _safe_name(invocation_id) / f"decision-r{revision:03d}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not str(data.get("comment") or "").strip() and data.get("long_term_advice") is None:
        return None
    if data.get("proposal_id") != invocation_id or int(data.get("revision", -1)) != revision:
        raise ValueError("decision 与当前 proposal_id/revision 不一致")
    if data.get("proposal_hash") != proposal_hash(pending["agent_proposal"]):
        raise ValueError("decision 的 proposal_hash 与当前 proposal 不一致")
    return data


def proposal_hash(proposal):
    payload = json.dumps(proposal, ensure_ascii=False, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_markdown(envelope):
    proposal = envelope["agent_proposal"]
    text = "\n".join(
        [
            f"# Action proposal — revision {envelope['revision']}",
            "",
            f"- Proposal ID: `{envelope['proposal_id']}`",
            f"- Proposal hash: `{envelope['proposal_hash']}`",
            f"- 当前状态分析: {proposal.get('current_state_analysis')}",
            f"- 推荐 action: `{proposal.get('recommended_action')}`",
            f"- action 参数: `{json.dumps(proposal.get('action_parameters'), ensure_ascii=False, default=str)}`",
            f"- 选择原因: {proposal.get('reason')}",
            f"- 下一轮计算量: `{json.dumps(proposal.get('calculation_plan'), ensure_ascii=False, default=str)}`",
            f"- 校准后预计成本: `{json.dumps(proposal.get('estimated_cost'), ensure_ascii=False, default=str)}`",
            f"- 预期目的: {proposal.get('expected_purpose')}",
            "",
            "在对应 decision JSON 的 comment 中写意见会生成下一版 proposal；只有最后一条非空内容为‘同意’或 `approve` 才会执行。",
            "",
        ]
    )
    context = proposal.get("decision_context") or {}
    for title, key in (
        ("人工长期建议", "long_term_human_advice"),
        ("当前相图", "current_phase_diagram"),
        ("近期经验（非长期指令）", "recent_experience"),
    ):
        text += f"\n## {title}\n\n```json\n{json.dumps(context.get(key, {}), ensure_ascii=False, indent=2, default=str)}\n```\n"
    return text


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)) or "approval"


def _atomic_json(path, value):
    _atomic_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    )


def _atomic_text(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)
