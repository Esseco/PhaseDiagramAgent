"""Native Codex advisory discussions; scientific execution stays in the existing graph."""

from contextlib import nullcontext
from copy import deepcopy
import re
from uuid import uuid4

from filelock import FileLock
from phase_agent.agents.contracts import ConsultationPlan, advice_schema, validate_advice
from phase_agent.evidence.snapshot import evidence_snapshot
from phase_agent.runtime.codex.context import _json_digest
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


def team_lock(state_path):
    parent = state_path.parent
    return (
        FileLock(str(parent / "codex_team.lock"), timeout=10) if parent.is_dir() else nullcontext()
    )


class CodexTeam:
    def __init__(self, control):
        self.control = control
        self.root = control.state_path.parent
        self.current_path = self.root / "codex_team.json"

    def _path(self, team_id):
        if not isinstance(team_id, str) or not re.fullmatch(r"team-[0-9a-f]{32}", team_id):
            raise ValueError("invalid_team_id")
        return self.root / "codex_teams" / (team_id + ".json")

    def _load(self, team_id=None):
        current = read_json(self.current_path, {}) or {}
        team_id = team_id or current.get("team_id")
        if not team_id:
            raise ValueError("no_codex_team; start a discussion first")
        session = read_json(self._path(team_id))
        if not isinstance(session, dict) or session.get("team_id") != team_id:
            raise ValueError("codex_team_not_found")
        if session.get("state_path") != str(self.control.state_path.resolve()):
            raise ValueError("codex_team_project_mismatch")
        return session

    def _save(self, session):
        write_json(self._path(session["team_id"]), session)

    def _context(self):
        self.control._codex_context_cache = None
        return self.control.codex_context()

    @staticmethod
    def _binding(context):
        return _json_digest(
            {
                key: context.get(key)
                for key in (
                    "context_version",
                    "context_digest",
                    "config_version",
                    "model_version",
                    "project",
                )
            }
        )

    def _fresh(self, session):
        context = self._context()
        if session["binding"] != self._binding(context):
            raise ValueError("codex_team_stale; revise with current facts and reconsult experts")
        return context

    def _active(self, team_id, revision):
        session = self._load(team_id)
        if (read_json(self.current_path, {}) or {}).get("team_id") != team_id:
            raise ValueError("codex_team_is_archived")
        if type(revision) is not int or session["revision"] != revision:
            raise ValueError("codex_team_revision_mismatch")
        return session

    @staticmethod
    def _snapshot(context, session):
        payload = {
            "state": context.get("state") or {},
            "decision_context": {
                **(context.get("decision_context") or {}),
                "codex_consultation": {
                    "team_id": session["team_id"],
                    "revision": session["revision"],
                    "question": session["question"],
                    "guidance": session["guidance"],
                },
            },
            "allowed_tools": context["allowed_tools"],
            "scientific_operations": context.get("scientific_operations") or [],
            "module_operations": context.get("module_operations") or [],
            "current_project": context.get("project"),
            "user_instruction": session["question"] + "\n" + "\n".join(session["guidance"]),
        }
        return evidence_snapshot(payload)

    def _view(self, session, *, context=None):
        result = deepcopy(session)
        if context is None:
            context = self._context()
        result["stale"] = session["binding"] != self._binding(context)
        result["execution_approved"] = False
        result["requests"] = {}
        if not result["stale"]:
            snapshot = self._snapshot(context, session)
            result["snapshot"] = snapshot
            from phase_agent.agents.exploration.prompts import INSTRUCTION as exploration
            from phase_agent.agents.reliability.prompts import INSTRUCTION as reliability

            answered = {opinion["role"] for opinion in session["advice"]}
            for role in session["consult"] if session["status"] == "consulting" else []:
                if role in answered:
                    continue
                result["requests"][role] = {
                    "agent_role": role,
                    "allowed_tools": [],
                    "evidence_snapshot": deepcopy(snapshot),
                    "instruction": (exploration if role == "exploration" else reliability)
                    + "\nQuestion: "
                    + session["question"]
                    + "\nHuman guidance: "
                    + "\n".join(session["guidance"])
                    + "\nReturn one JSON opinion only. Do not call project mutation tools or create subagents.",
                    "output_contract": advice_schema(role),
                }
                from phase_agent.agents.permissions import role_permissions

                result["requests"][role]["permissions"] = role_permissions(role)
            if session["status"] == "planning_simulation" and session.get("draft"):
                from phase_agent.agents.simulation.agent import planning_request

                result["requests"]["simulation"] = planning_request(
                    session["draft"]["action"],
                    facts={"snapshot_id": snapshot["snapshot_id"]},
                    operations=context.get("scientific_operations"),
                )
        if session["status"] in {"submitting", "submission_uncertain"}:
            result["recorded_submission"] = self._recorded_submission(session)
        return result

    def _recorded_submission(self, session):
        state = read_json(self.control.state_path, {}) or {}
        expected = self._action(session)
        rows = list((state.get("pending_execution_policies") or {}).values())
        rows.extend(reversed(state.get("action_records") or []))
        for row in rows:
            raw = (row.get("agent_proposal") or {}).get("raw_action") or {}
            matches = all(raw.get(key) == value for key, value in expected.items())
            if not matches and expected.get("_simulation_intent"):
                from phase_agent.agents.simulation.agent import validate_frozen_plan

                # Program-owned previews may extend parameters after native
                # planning. Require the exact original intent/team/plan and a
                # valid seal of the derived execution, not a response claim.
                matches = all(
                    raw.get(key) == value for key, value in expected.items() if key != "parameters"
                )
                if matches:
                    try:
                        # History removes the raw proposal's large decision
                        # context; the final action retains the sealed payload.
                        final = row.get("final_action") or {}
                        if final.get("_simulation_intent") != expected[
                            "_simulation_intent"
                        ] or final.get("_decision_team") != expected.get("_decision_team"):
                            raise ValueError("recorded_team_intent_changed")
                        validate_frozen_plan(final)
                    except (ValueError, TypeError, KeyError):
                        matches = False
            if matches:
                recorded = {
                    "record_id": row.get("record_id"),
                    "status": row.get("status"),
                    "team_id": session["team_id"],
                }
                execution = row.get("execution_result") or {}
                execution = execution if isinstance(execution, dict) else {}
                if (execution.get("execution_authorization") or {}).get("source"):
                    recorded["authorization_source"] = execution["execution_authorization"][
                        "source"
                    ]
                    recorded["execution_status"] = execution.get("status")
                return recorded
        return _recorded_training_activation(state, expected, session["team_id"])

    def start(self, *, question, consult=None):
        question = _text(question, "question")
        consult = ["exploration", "reliability"] if consult is None else consult
        ConsultationPlan(snapshot_id="pending", consult=consult, question=question)
        with self.control.chat_handler.lock, team_lock(self.control.state_path):
            if self.current_path.exists():
                current = self._load()
                if current["status"] in {
                    "submitting",
                    "submission_uncertain",
                } and self._recorded_submission(current):
                    current["status"] = "submitted"
                    self._save(current)
                if current["status"] not in {"submitted", "paused"}:
                    raise ValueError("codex_team_active; read or revise the existing discussion")
            state = read_json(self.control.state_path, {}) or {}
            if state.get("pending_execution_policies"):
                raise ValueError("existing_pending_plan; review it before starting a new decision")
            _require_training_handoffs(state)
            context = self._context()
            session = {
                "team_id": "team-" + uuid4().hex,
                "state_path": str(self.control.state_path.resolve()),
                "revision": 1,
                "status": "consulting",
                "question": question,
                "consult": consult,
                "binding": self._binding(context),
                "context_version": context["context_version"],
                "context_digest": context["context_digest"],
                "guidance": [],
                "advice": [],
                "draft": None,
                "events": [],
            }
            self._save(session)
            write_json(self.current_path, {"team_id": session["team_id"]})
            return self._view(session, context=context)

    def read(self, team_id=None):
        with self.control.chat_handler.lock, team_lock(self.control.state_path):
            return self._view(self._load(team_id))

    def advise(self, team_id, revision, value):
        with self.control.chat_handler.lock, team_lock(self.control.state_path):
            session = self._active(team_id, revision)
            if session["status"] != "consulting":
                raise ValueError("codex_team_not_consulting")
            role = value.get("role") if isinstance(value, dict) else None
            if role not in session["consult"]:
                raise ValueError("expert_was_not_requested")
            context = self._fresh(session)
            snapshot = self._snapshot(context, session)
            opinion = validate_advice(value, role=role, snapshot=snapshot)
            previous = next((a for a in session["advice"] if a["role"] == role), None)
            if previous and previous != opinion:
                raise ValueError("expert_already_answered; human revision required")
            if not previous:
                session["advice"].append(opinion)
                session["events"].append(
                    {"revision": revision, "kind": "expert_advice", "advice": deepcopy(opinion)}
                )
                self._save(session)
            return self._view(session, context=context)

    def draft(self, team_id, revision, action, *, rationale, memory_ids=None, simulation_plan=None):
        from phase_agent.agents.common.action_contracts import action_contract_errors

        rationale = _text(rationale, "rationale")
        with self.control.chat_handler.lock, team_lock(self.control.state_path):
            session = self._active(team_id, revision)
            if session["status"] != "consulting":
                raise ValueError("codex_team_review_required; revise before replacing a draft")
            context = self._fresh(session)
            if set(session["consult"]) != {a["role"] for a in session["advice"]}:
                raise ValueError("codex_team_missing_expert_opinion")
            if not isinstance(action, dict) or action.get("_decision_team") is not None:
                raise ValueError("coordinator_action_must_not_supply_team_metadata")
            errors = action_contract_errors(action)
            if errors:
                raise ValueError("invalid_coordinator_action: " + "; ".join(errors))
            if (action.get("tool") or action.get("action_type")) not in context["allowed_tools"]:
                raise ValueError("coordinator_tool_outside_scope")
            memory_ids = list(memory_ids or [])
            available = {
                row["knowledge_id"]
                for row in (context.get("decision_context", {}).get("memory_context") or {}).get(
                    "approved_knowledge", []
                )
            }
            if set(memory_ids) - available:
                raise ValueError("unavailable_memory_reference")
            session["draft"] = {
                "action": deepcopy(action),
                "rationale": rationale,
                "memory_ids": memory_ids,
            }
            if simulation_plan is not None:
                from phase_agent.agents.simulation.agent import validate_plan

                session["draft"]["action"]["_simulation_plan"] = validate_plan(
                    simulation_plan, action
                )
                session["draft"]["action"]["_simulation_intent"] = deepcopy(action)
            session["status"] = (
                "planning_simulation"
                if action.get("tool", action.get("action_type"))
                not in {"pause_search", "check_convergence"}
                and simulation_plan is None
                else "awaiting_human_input"
            )
            if session["status"] == "awaiting_human_input":
                self._apply_confirmed_policy(session)
            session["events"].append(
                {"revision": revision, "kind": "draft", **deepcopy(session["draft"])}
            )
            self._save(session)
            return self._view(session, context=context)

    def simulate(self, team_id, revision, plan):
        from phase_agent.agents.simulation.agent import validate_plan

        with self.control.chat_handler.lock, team_lock(self.control.state_path):
            session = self._active(team_id, revision)
            context = self._fresh(session)
            if session["status"] != "planning_simulation":
                raise ValueError("simulation_requires_current_strategic_draft")
            validated = validate_plan(plan, session["draft"]["action"])
            session["draft"]["action"]["_simulation_plan"] = validated
            session["draft"]["action"]["_simulation_intent"] = {
                key: deepcopy(value)
                for key, value in session["draft"]["action"].items()
                if not key.startswith("_")
            }
            session["status"] = "awaiting_human_input"
            self._apply_confirmed_policy(session)
            session["events"].append(
                {"revision": revision, "kind": "simulation_plan", "plan": validated}
            )
            self._save(session)
            return self._view(session, context=context)

    def _apply_confirmed_policy(self, session):
        from phase_agent.agents.strategic.human_review import policy_auto_approved

        from phase_agent.runtime.project_context import current_config_session

        config_session = current_config_session(self.control.chat_handler)
        if config_session.get("status") != "confirmed":
            return
        snapshot = config_session.get("confirmed_snapshot") or {}
        if policy_auto_approved(session["draft"]["action"], snapshot.get("config") or {}):
            session["status"] = "ready_to_propose"
            session["direction_review"] = {
                "status": "policy_authorized",
                "execution_approved": False,
                "config_version": snapshot.get("config_version"),
            }
            session["events"].append(
                {
                    "revision": session["revision"],
                    "kind": "confirmed_policy_direction",
                    "operation": session["draft"]["action"].get("operation"),
                }
            )

    def reconcile(
        self, team_id, revision, *, checkpoint_id, request_id, checkpoint_sha256, evidence_note
    ):
        """End a proven proposal-validation failure, without replay or approval."""
        from phase_agent.agents.strategic.codex_submission_rejection import rejection_checkpoint

        evidence_note = _text(evidence_note, "evidence_note")
        with (
            self.control.chat_handler.lock,
            team_lock(self.control.state_path),
            FileLock(str(self.root / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
        ):
            session = self._active(team_id, revision)
            if session["status"] != "submission_uncertain":
                raise ValueError("reconcile_requires_uncertain_submission")
            if self._recorded_submission(session) is not None:
                raise ValueError("submission_has_durable_scientific_record")
            proof = rejection_checkpoint(
                self.control.state_path,
                session,
                checkpoint_id=checkpoint_id,
                request_id=request_id,
                checkpoint_sha256=checkpoint_sha256,
            )
            session["status"] = "validation_rejected"
            session["events"].append(
                {
                    "revision": revision,
                    "kind": "verified_validation_rejection",
                    "proof": proof,
                    "evidence_note": evidence_note,
                }
            )
            self._save(session)
            return self._view(session)

    def feedback(self, team_id, revision, decision, feedback):
        feedback = _text(feedback, "feedback")
        if decision not in {"accept", "revise", "pause"}:
            raise ValueError("human_decision_must_be_accept_revise_or_pause")
        with self.control.chat_handler.lock, team_lock(self.control.state_path):
            session = self._active(team_id, revision)
            if session["status"] in {"submitting", "submission_uncertain", "submitted"}:
                raise ValueError(
                    "submission_already_started; inspect existing pending plan and receipts"
                )
            context = None
            if decision == "accept":
                context = self._fresh(session)
                if session["status"] != "awaiting_human_input" or not session["draft"]:
                    raise ValueError("human_review_requires_coordinator_draft")
                session["status"] = "ready_to_propose"
                from phase_agent.agents.strategic.human_review import direction_review

                session["direction_review"] = direction_review(
                    session["draft"]["action"], feedback={"decision": decision, "comment": feedback}
                )
            elif decision == "revise":
                context = self._context()
                session["revision"] += 1
                session["guidance"].append(feedback)
                session["binding"] = self._binding(context)
                session["context_version"] = context["context_version"]
                session["context_digest"] = context["context_digest"]
                session["advice"] = []
                session["draft"] = None
                session["status"] = "consulting"
            else:
                session["status"] = "paused"
            session["events"].append(
                {
                    "revision": revision,
                    "kind": "human_" + decision,
                    "feedback": feedback,
                }
            )
            self._save(session)
            return self._view(session, context=context)

    @staticmethod
    def _action(session):
        draft = session["draft"]
        action = deepcopy(draft["action"])
        parameters = action.get("parameters") or {}
        if (
            action.get("tool") == "update_mlip"
            and parameters.get("action") == "ACTIVATE_CANDIDATE_MODEL"
            and not parameters.get("user_approval_reason")
        ):
            accepted = next(
                (
                    event
                    for event in reversed(session["events"])
                    if event.get("kind") == "human_accept"
                    and event.get("revision") == session["revision"]
                ),
                {},
            )
            if accepted.get("feedback"):
                action["parameters"] = {**parameters, "user_approval_reason": accepted["feedback"]}
        action["_decision_team"] = {
            "backend": "codex_native",
            "team_id": session["team_id"],
            "revision": session["revision"],
            "consult": session["consult"],
            "question": session["question"],
            "advice": session["advice"],
            "guidance": session["guidance"],
            "human_review": session["events"],
            "rationale": draft["rationale"],
            "memory_ids": draft["memory_ids"],
            "binding": session["binding"],
        }
        return action

    def propose(self, team_id, revision):
        # The normal control entry takes the file lock and atomically checks the
        # frozen draft, records submission intent, and enters existing approval.
        session = self._active(team_id, revision)
        if session["status"] != "ready_to_propose":
            raise ValueError("codex_team_human_review_required")
        result = self.control.propose_codex(
            self._action(session),
            context_version=session["context_version"],
            context_digest=session["context_digest"],
            conversation_id="codex-team-" + team_id,
        )
        if (result.get("reply") or {}).get("status") == "awaiting_approval":
            try:
                result["pending"] = self.control.pending()
            except Exception as error:
                result["pending_error"] = type(error).__name__
        return result


def _recorded_training_activation(state, action, team_id):
    """The original training review records activation outside action_records."""
    parameters = action.get("parameters") or {}
    if (
        action.get("tool") != "update_mlip"
        or parameters.get("action") != "ACTIVATE_CANDIDATE_MODEL"
        or set(parameters) - {"action", "candidate_model_version", "user_approval_reason"}
        or action.get("target_ids")
        or action.get("budget", 0) != 0
    ):
        return None
    version = parameters.get("candidate_model_version")
    candidate = (state.get("candidate_models") or {}).get(version) or {}
    if (
        not version
        or state.get("active_model_version") != version
        or candidate.get("status") != "activated"
        or not candidate.get("model")
        or candidate["model"] != state.get("active_model")
    ):
        return None
    jobs = [
        key
        for key, job in (state.get("remote_finetune_jobs") or {}).items()
        if job.get("activated")
        and (job.get("training_handoff") or {}).get("candidate_model_version") == version
    ]
    if len(jobs) != 1:
        return None
    for receipt in reversed(state.get("model_activation_history") or []):
        if (
            receipt.get("new_model_version") == version
            and receipt.get("old_model_version") == candidate.get("old_model_version")
            and receipt.get("dft_data_version") == candidate.get("dft_data_version")
            and receipt.get("validation_data_version") == candidate.get("validation_data_version")
            and str(receipt.get("approval_reason") or "").strip()
        ):
            return {
                "record_id": "review:model_activation:" + jobs[0],
                "status": "activated",
                "team_id": team_id,
                "authorization_source": "canonical_model_activation_history",
                "execution_status": "activated",
            }
    return None


def _require_training_handoffs(state):
    """Do not bind expert opinions before returned training evidence settles."""
    from phase_agent.tools.workflows.recover_saved_results import _training_return_facts

    jobs = state.get("remote_finetune_jobs") or {}
    for row in _training_return_facts(state):
        job = jobs[row["job_id"]]
        if job.get("activated") or job.get("status") in {"abandoned", "cancelled", "superseded"}:
            continue
        handoff = job.get("training_handoff") or {}
        fingerprint = (job.get("returned_results") or {}).get("fingerprint")
        if not handoff or (fingerprint and handoff.get("training_fingerprint") != fingerprint):
            raise ValueError(
                "training_recovery_required; call phase_recover before phase_team_start"
            )


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise ValueError(label + "_must_be_nonempty_text_up_to_4000_characters")
    return value.strip()


def guarded_submission(control, action, submit):
    """A discussion cannot be bypassed, changed after review, or blindly replayed."""
    team = CodexTeam(control)
    with control.chat_handler.lock, team_lock(control.state_path):
        if not team.current_path.exists():
            if action.get("_decision_team") is not None:
                raise ValueError("codex_team_metadata_without_discussion")
            return submit(None)
        session = team._load()
        if session["status"] != "ready_to_propose":
            raise ValueError(
                "codex_team_human_review_required; inspect current discussion and pending plan"
            )
        team._fresh(session)
        if action != team._action(session):
            raise ValueError("codex_team_frozen_action_mismatch")
        dispatched = False

        def before_dispatch():
            nonlocal dispatched
            session["status"] = "submitting"
            team._save(session)
            dispatched = True

        try:
            result = submit(before_dispatch)
        except BaseException:
            if dispatched:
                session["status"] = "submission_uncertain"
                team._save(session)
            raise
        reply = result.get("reply") or {}
        recorded = team._recorded_submission(session)
        proven_policy_execution = bool(
            recorded
            and recorded.get("authorization_source") == "confirmed_policy"
            and recorded.get("execution_status") == "completed"
        )
        proven_rejection = (
            reply.get("submitted") is False
            and not reply.get("execution")
            and not reply.get("execution_result")
            and (
                (reply.get("validation") or {}).get("valid") is False
                or (
                    reply.get("steps_executed") == 0
                    and reply.get("status")
                    in {
                        "rejected",
                        "not_configured",
                        "configuration_version_mismatch",
                        "proposal_context_changed",
                        "saved_selection_changed",
                    }
                )
            )
        )
        session["status"] = (
            "submitted"
            if reply.get("status") == "awaiting_approval" or proven_policy_execution
            else "ready_to_propose"
            if proven_rejection
            else "submission_uncertain"
        )
        session["submission"] = {
            key: deepcopy(reply[key]) for key in ("status", "record_id", "reason") if key in reply
        }
        team._save(session)
        return result
