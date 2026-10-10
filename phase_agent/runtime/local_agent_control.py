"""Small authenticated control surface; no shell, Python, or arbitrary file tools."""

from copy import deepcopy
import os
from pathlib import Path

from phase_agent.analysis.visualization.build_webui_charts import build_webui_charts
from phase_agent.configuration.runtime.path_mapping import validate_path_mappings
from phase_agent.configuration.session.apply_config_revision import apply_config_revision
from phase_agent.configuration.session.confirm_config_snapshot import confirm_config_snapshot
from phase_agent.configuration.session.save_config_session import save_config_session
from phase_agent.persistence.memory.review_queue import (
    propose_knowledge_record,
    review_memory_update,
)
from phase_agent.persistence.memory.build_system_signature import build_system_signature
from phase_agent.persistence.memory.load_matching_domain_skills import load_matching_domain_skills
from phase_agent.persistence.memory.propose_domain_skill_import import propose_domain_skill_import
from phase_agent.persistence.memory.publish_domain_skill import publish_domain_skill
from phase_agent.tools.policy.file_approval import proposal_hash
from phase_agent.tools.step_runner.build_status_summary import build_status_summary
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from phase_agent.runtime.control_ui_contracts import (
    StatusResponse,
    PendingResponse,
    TasksResponse,
    ConfigResponse,
    MemoryResponse,
    validate_control_response,
)
from phase_agent.runtime.control_request_contracts import validate_control_request


class LocalAgentControl:
    def __init__(self, chat_handler):
        self.chat_handler = chat_handler
        self.state_path = chat_handler.state_path

    def status(self):
        state = read_json(self.state_path, {}) or {}
        from phase_agent.runtime.scientific_progress import project_progress

        summary = build_status_summary(state, config_version=state.get("confirmed_config_version"))
        session = (getattr(self.chat_handler, "workflow_kwargs", None) or {}).get("config_session")
        progress = project_progress(state, self.state_path, session=session)
        summary["pending_approvals"] = progress["pending_review_count"]
        summary["waiting_state"] = progress["waiting_state"]
        summary["recovery_report"] = progress["recovery_report"]
        return validate_control_response(StatusResponse, summary)

    def pending(self):
        state = read_json(self.state_path, {}) or {}
        state_version = build_status_summary(
            state, config_version=state.get("confirmed_config_version")
        )["summary_id"]
        from phase_agent.runtime.review_requests import pending_reviews

        session = (getattr(self.chat_handler, "workflow_kwargs", None) or {}).get("config_session")
        rows = pending_reviews(state, session)
        from phase_agent.tools.state.execution_receipts import recovery_report

        if recovery_report(self.state_path, state)["unsettled"]:
            for row in rows:
                if row.get("review_kind") not in {"configuration", "execution_recovery"}:
                    row["review_card"]["blocked"] = "先核对中断执行，再批准此科学动作"
        return validate_control_response(
            PendingResponse,
            {
                "pending": rows,
                "count": len(rows),
                "state_version": state_version,
                "approval_url": f"http://127.0.0.1:{os.environ.get('PHASE_CONTROL_PORT', '8765')}/phase/approval",
            },
        )

    def tasks(self):
        state = read_json(self.state_path, {}) or {}
        return validate_control_response(
            TasksResponse,
            {
                "tasks": [
                    {
                        key: row.get(key)
                        for key in (
                            "task_id",
                            "task_key",
                            "batch_id",
                            "stage",
                            "status",
                            "model_version",
                            "config_version",
                        )
                    }
                    for row in state.get("tasks") or []
                ]
            },
        )

    def charts(self):
        state = read_json(self.state_path, {}) or {}
        return build_webui_charts(state)

    def config(self):
        handler = getattr(self.chat_handler, "config_delegate", None) or self.chat_handler
        session = handler.workflow_kwargs.get("config_session") or {}
        return validate_control_response(
            ConfigResponse,
            {
                "status": session.get("status"),
                "draft_revision": session.get("draft_revision"),
                "config": deepcopy(
                    session.get("config")
                    or (session.get("confirmed_snapshot") or {}).get("config")
                    or {}
                ),
                "confirmed_snapshot": deepcopy(session.get("confirmed_snapshot")),
                "dialogue": deepcopy((session.get("dialogue") or [])[-20:]),
                "readiness": (
                    handler.configuration_readiness(session)
                    if callable(getattr(handler, "configuration_readiness", None))
                    else None
                ),
            },
        )

    def memory(self):
        state = read_json(self.state_path, {}) or {}
        return validate_control_response(
            MemoryResponse,
            {
                "active": deepcopy((state.get("decision_memory") or {}).get("long_term") or {}),
                "records": deepcopy((state.get("decision_memory") or {}).get("records") or []),
                "candidate_count": len(state.get("memory_candidates") or []),
                "review_queue": deepcopy(state.get("memory_review_queue") or []),
            },
        )

    def propose_memory(self, record):
        validate_control_request("/phase/memory/propose", {"record": record})
        state = read_json(self.state_path, {}) or {}
        result = propose_knowledge_record(state, record, source="agent_proposal")
        write_json(self.state_path, result["state"])
        from phase_agent.persistence.memory.publish_memory_views import publish_memory_views

        publish_memory_views(result["state"], self.state_path)
        return {
            "status": result["proposal"]["status"],
            "proposal_id": result["proposal"]["proposal_id"],
        }

    def _knowledge_root(self):
        value = getattr(self.chat_handler, "knowledge_library_root", None) or os.environ.get(
            "PHASE_SEARCH_KNOWLEDGE_ROOT"
        )
        if not value:
            raise ValueError("knowledge library root is not configured")
        return Path(value).resolve()

    def domain_skill_matches(self):
        config = (self.config().get("confirmed_snapshot") or {}).get("config") or {}
        signature = build_system_signature(config.get("system") or config.get("system_config"))
        return {"matches": load_matching_domain_skills(self._knowledge_root(), signature)}

    def propose_skill_import(self):
        config = (self.config().get("confirmed_snapshot") or {}).get("config") or {}
        signature = build_system_signature(config.get("system") or config.get("system_config"))
        state = read_json(self.state_path, {}) or {}
        result = propose_domain_skill_import(state, self._knowledge_root(), signature)
        if result["proposals"]:
            write_json(self.state_path, result["state"])
            from phase_agent.persistence.memory.publish_memory_views import publish_memory_views

            publish_memory_views(result["state"], self.state_path)
        return {
            "status": "pending_review",
            "proposal_ids": [row["proposal_id"] for row in result["proposals"]],
        }

    def publish_skill(self, draft_directory, *, approved, version="1.0.0"):
        validate_control_request(
            "/phase/memory/skills/publish",
            {"draft_directory": draft_directory, "approved": approved, "version": version},
        )
        state = read_json(self.state_path, {}) or {}
        root = Path(
            self.config().get("config", {}).get("storage", {}).get("workspace_root")
            or self.state_path
        ).resolve()
        if root.is_file() or root.suffix == ".json":
            root = root.parent.parent
        draft = Path(draft_directory).resolve()
        if root not in draft.parents or draft.parent.name != "knowledge_export":
            raise ValueError("draft must be inside this project's knowledge_export directory")
        if not (
            state.get("user_accepted_convergence") is True
            and (state.get("convergence_result") or state.get("convergence") or {}).get("converged")
            is True
        ):
            raise ValueError("accepted convergence is required")
        return publish_domain_skill(
            draft, self._knowledge_root(), approved=approved, version=version
        )

    def propose(self, instruction, *, conversation_id="local-control"):
        validate_control_request(
            "/phase/propose", {"instruction": instruction, "conversation_id": conversation_id}
        )
        return {
            "reply": self.chat_handler(
                [{"role": "user", "content": str(instruction)}], conversation_id=conversation_id
            )
        }

    def decide(
        self,
        decision,
        *,
        plan_id,
        expected_state_version,
        expected_proposal_hash,
        comment="",
        conversation_id="local-control",
    ):
        validate_control_request(
            "/phase/decision",
            {
                "decision": decision,
                "plan_id": plan_id,
                "state_version": expected_state_version,
                "proposal_hash": expected_proposal_hash,
                "comment": comment,
                "conversation_id": conversation_id,
            },
        )
        if decision not in {"approve", "reject", "confirm_sensitive", "modify"}:
            raise ValueError("decision must be approve/reject/confirm_sensitive/modify")
        return self.chat_handler.review_pending(
            str(plan_id or ""),
            decision,
            expected_state_version=expected_state_version,
            expected_proposal_hash=expected_proposal_hash,
            comment=comment,
        )

    def propose_recovery(self, body):
        validate_control_request("/phase/recovery/propose", body)
        from filelock import FileLock
        from phase_agent.tools.state.execution_reconciliation import propose_reconciliation

        with self.chat_handler.lock:
            with FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ):
                state = read_json(self.state_path, {}) or {}
                attestation = {
                    key: body.get(key)
                    for key in (
                        "evidence_note",
                        "external_jobs",
                        "output_inventory",
                        "external_cost",
                    )
                }
                result = propose_reconciliation(
                    self.state_path, state, body["invocation_id"], body["resolution"], attestation
                )
                write_json(self.state_path, result["state"])
                return {"status": result["status"], "plan_id": result["plan_id"]}

    def patch_config(self, patch, *, reasons=None, impacts=None):
        validate_control_request(
            "/phase/config/patch", {"patch": patch, "reasons": reasons, "impacts": impacts}
        )
        session = self.chat_handler.workflow_kwargs.get("config_session") or {}
        if session.get("status") != "draft":
            raise ValueError("confirmed config cannot be edited; create a new draft")
        patch = deepcopy(patch)
        if not isinstance(patch, dict):
            raise ValueError("patch must be an object of dotted paths")
        for path, value in patch.items():
            if path.endswith("path_mappings"):
                patch[path] = validate_path_mappings(value)
        updated = apply_config_revision(
            session, patch, reasons=reasons or {}, author="user_via_local_ui"
        )
        if updated.get("dialogue"):
            updated["dialogue"][-1]["impacts"] = deepcopy(impacts or {})
        self._save_session(updated)
        return {
            "status": "draft_updated",
            "changes": updated["dialogue"][-1]["changes"],
            "impacts": deepcopy(impacts or {}),
        }

    def confirm_config(self, *, explicit=False):
        validate_control_request("/phase/config/confirm", {"explicit": explicit})
        session = self.chat_handler.workflow_kwargs.get("config_session") or {}
        readiness_fn = getattr(self.chat_handler, "configuration_readiness", None)
        if explicit and callable(readiness_fn):
            readiness = readiness_fn(session)
            if not readiness.get("ready"):
                return {
                    "status": "draft",
                    "confirmation_error": "configuration_not_ready",
                    "missing": readiness.get("missing"),
                    "conflicts": readiness.get("conflicts"),
                    "ambiguities": readiness.get("ambiguities"),
                }
        updated = confirm_config_snapshot(session, user_confirmed=explicit)
        self._save_session(updated)
        return {
            "status": updated.get("status"),
            "confirmation_error": updated.get("confirmation_error"),
            "confirmed_snapshot": deepcopy(updated.get("confirmed_snapshot")),
            "readiness": (readiness_fn(updated) if callable(readiness_fn) else None),
        }

    def review_memory(self, proposal_id, *, approved):
        validate_control_request(
            "/phase/memory/review", {"proposal_id": proposal_id, "approved": approved}
        )
        state = read_json(self.state_path, {}) or {}
        result = review_memory_update(
            state, proposal_id, approved=approved, reviewer="local_approval_page"
        )
        write_json(self.state_path, result["state"])
        from phase_agent.persistence.memory.publish_memory_views import publish_memory_views

        publish_memory_views(result["state"], self.state_path)
        return {"status": result["status"], "proposal_id": proposal_id}

    def _save_session(self, session):
        path = self.chat_handler.workflow_kwargs.get("config_session_path")
        if not path:
            raise ValueError("config_session_path is not configured")
        save_config_session(session, path)
        self.chat_handler.workflow_kwargs["config_session"] = session

    def pause(self, reason="user requested pause", *, conversation_id="local-control"):
        validate_control_request(
            "/phase/pause", {"reason": reason, "conversation_id": conversation_id}
        )
        return self.propose(
            f"请提出 pause_search 批次计划。原因：{reason}", conversation_id=conversation_id
        )
