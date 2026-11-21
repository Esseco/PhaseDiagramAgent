"""Native Codex advisory discussions; scientific execution stays in the existing graph."""

from contextlib import nullcontext
import json
from time import monotonic
from collections.abc import Mapping
from pathlib import Path
from copy import deepcopy
import re
import zlib
from uuid import uuid4

from filelock import FileLock
from phase_agent.agents.contracts import (
    ConsultationPlan,
    advice_schema,
    validate_advice,
    specialist_task,
)
from phase_agent.evidence.snapshot import evidence_snapshot, role_context, simulation_facts
from phase_agent.runtime.codex.context import _json_digest
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


# Two immutable serialized slots, replaced on source change and expired on access.
# Large contexts can be compressed without raising the stored per-slot limit.
_READ_CACHE = {}
_READ_CACHE_LIMIT = 96 * 1024 * 1024
_CONTEXT_UNCOMPRESSED_LIMIT = 256 * 1024 * 1024


def _read_cached(kind, key):
    from phase_agent.runtime.turn_process import process_event, record_read_metric

    saved = _READ_CACHE.get(kind)
    if saved and saved[0] == key and monotonic() - saved[2] <= 300:
        record_read_metric("derived_cache_hit", cache=kind, bytes_read=len(saved[1]))
        process_event("team_derived_cache", {"kind": kind, "status": "hit", "bytes": len(saved[1])})
        return json.loads(zlib.decompress(saved[1]) if saved[3] else saved[1])
    _READ_CACHE.pop(kind, None)
    record_read_metric("derived_cache_miss", cache=kind)
    process_event("team_derived_cache", {"kind": kind, "status": "miss"})
    return None


def _save_cached(kind, key, value):
    if kind == "context" and _READ_CACHE.get("context") and _READ_CACHE["context"][0][0] != key[0]:
        _READ_CACHE.pop("role", None)
    # JSON changes numeric dictionary keys to strings; bypass such historical
    # payloads so caching cannot change the existing snapshot hash semantics.
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                _READ_CACHE.pop(kind, None)
                return
            pending.extend(child for child in item.values() if type(child) in (dict, list, tuple))
        elif type(item) in (list, tuple):
            pending.extend(child for child in item if type(child) in (dict, list, tuple))
    try:
        data = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError):
        return
    _READ_CACHE.pop(kind, None)
    compressed = kind == "context" and _READ_CACHE_LIMIT < len(data) <= _CONTEXT_UNCOMPRESSED_LIMIT
    if compressed:
        data = zlib.compress(data, level=1)
    if len(data) <= _READ_CACHE_LIMIT:
        _READ_CACHE[kind] = (key, data, monotonic(), compressed)


def _cached_role_context(snapshot, role):
    key = (snapshot["snapshot_id"], role)
    value = _read_cached("role", key)
    if value is None:
        value = role_context(snapshot, role)
        if role in {"exploration", "reliability"}:
            from phase_agent.models.transport_tables import compact_evidence_tables

            value["evidence_snapshot"] = compact_evidence_tables(value["evidence_snapshot"])
        _save_cached("role", key, value)
    return value


def team_lock(state_path):
    parent = state_path.parent
    return (
        FileLock(str(parent / "codex_team.lock"), timeout=10) if parent.is_dir() else nullcontext()
    )


class _TeamFields(Mapping):
    """Memoize fields only until this operation's final strong verification."""

    def __init__(self, source):
        self.source = source
        self.values_by_key = {}
        self.missing = set()

    @property
    def __class__(self):
        # Preserve the index's full-read fallback for dict-specific consumers.
        return self.source.__class__

    def __str__(self):
        return str(self.source)

    def __getitem__(self, key):
        if key in self.values_by_key:
            return self.values_by_key[key]
        if key in self.missing:
            raise KeyError(key)
        try:
            value = self.source[key]
        except KeyError:
            self.missing.add(key)
            raise
        self.values_by_key[key] = value
        return value

    def __iter__(self):
        return iter(self.source)

    def __len__(self):
        return len(self.source)


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
        if team_id is None:
            team_id = (read_json(self.current_path, {}) or {}).get("team_id")
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

    def _context(self, *, preflight=False, question=None, role_tasks=None):
        from phase_agent.persistence.memory.decision_context import memory_query_scope

        with memory_query_scope(question, role_tasks):
            return self._read_context(preflight=preflight)

    def _read_context(self, *, preflight=False):
        from phase_agent.persistence.memory.decision_context import current_memory_query
        from phase_agent.persistence.state.read_versions import read_identity
        from phase_agent.persistence.state.field_index import IndexUnavailable, open_indexed_state
        from phase_agent.evidence.build_decision_context import build_decision_context
        from phase_agent.runtime.turn_process import has_reusable_initial_state, timed_call

        from phase_agent.tools.workflows.recover_saved_results import recovery_candidate_fingerprint

        path = self.control.state_path
        resolved = str(path.resolve())
        reusable_initial = has_reusable_initial_state(path)
        memory_query = current_memory_query()
        cached = _READ_CACHE.get("context")
        # Metadata selects a candidate; the final content digest verifies it.
        if (
            not preflight
            and not reusable_initial
            and cached
            and cached[0][0] == resolved
            and cached[0][3] == memory_query
            and monotonic() - cached[2] <= 300
            and read_identity(path) == cached[0][1][:2]
        ):
            from phase_agent.runtime.codex.context import codex_read_scope

            key = cached[0]
            with codex_read_scope(None, identity=key[1]):
                current = self.control.codex_context()
            signature = getattr(self.control, "_codex_context_cache", None)
            if _json_digest(signature[0] if signature else current) == key[2]:
                context = _read_cached("context", key)
                if context is not None:
                    returns = recovery_candidate_fingerprint(
                        context["team_return_sources"], observe_relax_archives=True
                    )
                    if returns == context["team_return_identity"]:
                        actual = timed_call(
                            "team_source_cached", read_identity, path, strong=True, category="io"
                        )
                        if actual != key[1]:
                            _READ_CACHE.pop("context", None)
                            raise ValueError("codex_team_sources_changed_during_read; reload")
                        context["team_source_identity"] = actual
                        return context
            _READ_CACHE.pop("context", None)

        before = timed_call("team_source_before", read_identity, path, strong=True, category="io")

        def build(state):
            if preflight:
                if state.get("pending_execution_policies"):
                    raise ValueError(
                        "existing_pending_plan; review it before starting a new decision"
                    )
                _require_training_handoffs(state)
            from phase_agent.runtime.codex.context import codex_read_scope

            from phase_agent.analysis.phase.phase_quality_reports import quality_report_read_scope

            with codex_read_scope(state, identity=before), quality_report_read_scope():
                context = timed_call(
                    "team_control_context", self.control.codex_context, category="processing"
                )
                context["team_decision_context"] = (
                    timed_call(
                        "team_decision_evidence",
                        build_decision_context,
                        state,
                        category="processing",
                    )
                    if state
                    else {}
                )
            context["team_state"] = {
                key: state[key]
                for key in (
                    "tasks",
                    "human_constraints",
                    "budget_remaining",
                    "budget_reservations",
                    "budget_usage",
                    "pending_execution_policies",
                    "active_model",
                    "active_model_version",
                    "confirmed_config_version",
                    "confirmed_config",
                    "current_status",
                    "relax_mc_batches",
                    "remote_finetune_jobs",
                )
                if key in state
            }
            from phase_agent.tools.state.task_waiting import awaiting_task_result

            sources = {
                "tasks": [
                    {
                        key: row[key]
                        for key in ("task_id", "status", "relax_mc_bundle_id", "result_path")
                        if key in row
                    }
                    for row in context["team_state"].get("tasks") or []
                    if awaiting_task_result(row)
                ],
                "slurm_batches": state.get("slurm_batches") or [],
                "relax_mc_batches": context["team_state"].get("relax_mc_batches") or {},
                "remote_finetune_jobs": context["team_state"].get("remote_finetune_jobs") or {},
            }
            context["team_return_sources"] = sources
            context["team_return_identity"] = recovery_candidate_fingerprint(
                sources, observe_relax_archives=True
            )
            return context

        state = (
            None
            if reusable_initial
            else timed_call(
                "team_open_index", open_indexed_state, self.control.state_path, category="io"
            )
        )
        if state is not None:
            try:
                context = build(_TeamFields(state))
                timed_call("team_verify_index", state.verify, strong=True, category="io")
            except IndexUnavailable:
                context = build(read_json(self.control.state_path, {}) or {})
        else:
            context = build(read_json(self.control.state_path, {}) or {})
        # Neither the index nor a cached control view replaces strong source binding.
        if (
            timed_call(
                "team_source_after",
                read_identity,
                self.control.state_path,
                strong=True,
                category="io",
            )
            != before
        ):
            raise ValueError("codex_team_sources_changed_during_read; reload")
        context["team_source_identity"] = before
        signature = getattr(self.control, "_codex_context_cache", None)
        base = {
            key: value
            for key, value in context.items()
            if key
            not in {
                "team_state",
                "team_decision_context",
                "team_source_identity",
                "team_return_sources",
                "team_return_identity",
            }
        }
        cache_key = (
            resolved,
            before,
            _json_digest(signature[0] if signature else base),
            memory_query,
        )
        _save_cached("context", cache_key, context)
        return context

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
                    "team_source_identity",
                )
            }
        )

    def _session_stale(self, session, context):
        return session["binding"] != self._binding(context) or (
            "return_identity" in session
            and session["return_identity"] != context["team_return_identity"]
        )

    def _fresh(self, session):
        context = self._context(question=session["question"], role_tasks=session.get("role_tasks"))
        if self._session_stale(session, context):
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
    def _snapshot_payload(context, session):
        decision_context = context.get("decision_context") or {}
        base_refs = (decision_context.get("evidence_catalog") or {}).get("references", {})
        team_refs = (
            (context.get("team_decision_context") or {}).get("evidence_catalog") or {}
        ).get("references", {})
        if "branch_pool_facts" in decision_context:
            team_refs = {
                key: value
                for key, value in team_refs.items()
                if not key.startswith("branch_pool:") or key in base_refs
            }
        payload = {
            "state": {
                "confirmed_config_version": context.get("config_version"),
                "active_model_version": context.get("model_version"),
                **(context.get("state") or {}),
                **(context.get("team_state") or {}),
            },
            "decision_context": {
                **(context.get("decision_context") or {}),
                **(context.get("team_decision_context") or {}),
                # The control view resolves this pool against the current ledger.
                **(
                    {"branch_pool_facts": context["decision_context"]["branch_pool_facts"]}
                    if "branch_pool_facts" in (context.get("decision_context") or {})
                    else {}
                ),
                "evidence_catalog": {
                    "references": {
                        **(
                            (context.get("decision_context") or {}).get("evidence_catalog") or {}
                        ).get("references", {}),
                        **team_refs,
                    }
                },
                "hpc_approval_policy": context.get("hpc_approval_policy"),
                "fact_versions": {
                    key: context.get(key)
                    for key in (
                        "context_version",
                        "context_digest",
                        "state_version",
                        "config_version",
                        "model_version",
                    )
                },
                "codex_consultation": {
                    "team_id": session["team_id"],
                    "revision": session["revision"],
                    "question": session["question"],
                    **(
                        {
                            "effective_guidance": _effective_guidance(session),
                            "role_tasks": session.get("role_tasks", {}),
                        }
                        if session.get("decision_contract_version") == 2
                        else {"guidance": session["guidance"]}
                    ),
                },
            },
            "allowed_tools": context["allowed_tools"],
            "scientific_operations": context.get("scientific_operations") or [],
            "module_operations": context.get("module_operations") or [],
            "current_project": context.get("project"),
            "user_instruction": session["question"]
            + "\n"
            + "\n".join(row["text"] for row in _effective_guidance(session)),
        }
        return payload

    @staticmethod
    def _snapshot(context, session):
        from phase_agent.evidence.advisory_projection import project_advisory_payload

        payload = CodexTeam._snapshot_payload(context, session)
        return evidence_snapshot(
            project_advisory_payload(payload, source_identity=context.get("team_source_identity"))
        )

    def _view(
        self, session, *, context=None, request_role=None, include_history=False, snapshot=None
    ):
        if request_role not in {None, "strategic", "exploration", "reliability", "simulation"}:
            raise ValueError("unknown_agent_role")
        result = deepcopy({key: value for key, value in session.items() if key != "events"})
        result["guidance"] = [row["text"] for row in _effective_guidance(session)]
        result["effective_guidance"] = _effective_guidance(session)
        result.pop("guidance_records", None)
        if include_history:
            result["guidance_history"] = _guidance_records(session)
        result["event_count"] = len(session["events"])
        result["events"] = deepcopy(
            session["events"]
            if include_history
            else [
                {key: value for key, value in event.items() if key not in {"advice", "plan"}}
                for event in session["events"]
                if event.get("revision") == session["revision"]
            ]
        )
        if context is None:
            context = self._context(
                question=session["question"], role_tasks=session.get("role_tasks")
            )
        result.pop("return_identity", None)
        result["stale"] = self._session_stale(session, context)
        result["execution_approved"] = False
        if session.get("draft"):
            result["simulation_check_status"] = {
                "plan_structure": "passed"
                if session["draft"]["action"].get("_simulation_plan")
                else "not_checked",
                "planned_checks": deepcopy(
                    (session["draft"]["action"].get("_simulation_plan") or {}).get("checks", [])
                ),
                "context_validity": "stale" if result["stale"] else "current",
                "execution_checks": "unknown"
                if session["status"] in {"submitting", "submission_uncertain", "submitted"}
                else "not_executed",
                "scientific_completion": "unknown",
                "source": "simulation.validate_plan; structural/order/action/context only",
            }
        if session.get("draft") and session["draft"]["action"].get("_simulation_program"):
            result["simulation_check_status"].update(
                handling="program",
                program_preflight=deepcopy(session["draft"]["action"]["_simulation_program"]),
                source="registered_dft_inputs/v1; no model plan; execution/authorization still pending",
            )
        result["requests"] = {}
        if not result["stale"]:
            snapshot = snapshot if snapshot is not None else self._snapshot(context, session)
            from phase_agent.tools.dispatch.scientific_operations import operation_readiness

            result["operation_readiness"] = operation_readiness(snapshot.get("facts") or {})
            result["snapshot"] = {
                key: deepcopy(value) for key, value in snapshot.items() if key != "facts"
            }
            result["context_access"] = (
                "Each request is self-contained. phase_team_read(request_role='strategic') returns full current facts; include_history=true returns persisted events."
            )
            binding = {
                "team_id": session["team_id"],
                "revision": session["revision"],
                "snapshot_id": snapshot["snapshot_id"],
            }
            from phase_agent.agents.exploration.prompts import INSTRUCTION as exploration
            from phase_agent.agents.reliability.prompts import INSTRUCTION as reliability

            answered = {opinion["role"] for opinion in session["advice"]} - set(
                session.get("pending_clarifications", {})
            )
            for role in session["consult"] if session["status"] == "consulting" else []:
                if role in answered or request_role not in {None, role}:
                    continue
                result["requests"][role] = {
                    "agent_role": role,
                    "allowed_tools": [],
                    **specialist_task(session, role),
                    **_cached_role_context(snapshot, role),
                    "consultation_binding": deepcopy(binding),
                    "instruction": (exploration if role == "exploration" else reliability)
                    + "\nDecision goal: "
                    + session["question"]
                    + "\nHuman guidance: "
                    + "\n".join(row["text"] for row in _effective_guidance(session))
                    + "\nReturn one JSON opinion only. Do not call project mutation tools or create subagents.",
                    "output_contract": advice_schema(role),
                }
                if session.get("human_directed"):
                    result["requests"][role]["instruction"] += (
                        "\nThe user has chosen the direction. Check only the required evidence, "
                        "feasibility and concrete blockers within that scope."
                    )
                if role in session.get("pending_clarifications", {}):
                    result["requests"][role]["clarification"] = deepcopy(
                        session["pending_clarifications"][role]
                    )
                    result["requests"][role]["previous_opinion"] = deepcopy(
                        next(a for a in session["advice"] if a["role"] == role)
                    )
                    result["requests"][role]["instruction"] += (
                        "\nResolve only the declared gap from these same facts; otherwise retain insufficient_evidence. No new execution or automatic retry."
                    )
            if (
                session["status"] == "planning_simulation"
                and session.get("draft")
                and request_role in {None, "simulation"}
            ):
                from phase_agent.agents.simulation.agent import planning_request

                result["requests"]["simulation"] = planning_request(
                    session["draft"]["action"],
                    facts=simulation_facts(
                        session["draft"]["action"],
                        state=context.get("team_state") or {},
                        context=snapshot["facts"]["decision_context"],
                        snapshot=snapshot,
                    ),
                    operations=[
                        *(context.get("scientific_operations") or []),
                        *(context.get("module_operations") or []),
                    ],
                )
                result["requests"]["simulation"]["consultation_binding"] = deepcopy(binding)
            if request_role == "strategic" or (
                not result["requests"] and not session.get("human_directed")
            ):
                result["snapshot"] = _cached_role_context(snapshot, "strategic")[
                    "evidence_snapshot"
                ]
        from phase_agent.models.transport_tables import compact_evidence_tables

        for role, request in result["requests"].items():
            field = "facts" if role == "simulation" else "evidence_snapshot"
            if role == "simulation":
                request[field] = compact_evidence_tables(request[field])
            request["evidence_encoding"] = (
                "Inline lossless tables: common values apply to every row; columns match row cells; "
                "dictionaries contain indexed cell values; prefixes prepend path suffixes; "
                "schema groups restore original indices. Fields marked deferred_evidence_v1 "
                "remain available from phase_team_evidence using this snapshot binding."
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

    @staticmethod
    def _is_deterministic_local_prep(action):
        if not isinstance(action, Mapping):
            return False
        tool = action.get("tool") or action.get("action_type") or ""
        op = action.get("operation") or ""
        return tool in {
            "prepare_local_batch_files",
            "prepare_parent_first_recovery",
            "initial_direct",
        } or op in {
            "prepare_local_batch_files",
            "prepare_parent_first_recovery",
            "initial_direct",
        }

    def start(
        self, *, question, consult=None, role_tasks=None, human_directed=False, compact=False
    ):
        question = _text(question, "question")
        if consult is None:
            consult = [] if human_directed else ["exploration", "reliability"]
        plan = ConsultationPlan(
            snapshot_id="pending", consult=consult, question=question, role_tasks=role_tasks or {}
        )
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
            normalized_tasks = {role: task.model_dump() for role, task in plan.role_tasks.items()}
            context = self._context(preflight=True, question=question, role_tasks=normalized_tasks)
            session = {
                "team_id": "team-" + uuid4().hex,
                "state_path": str(self.control.state_path.resolve()),
                "revision": 1,
                "decision_contract_version": 2,
                "status": "consulting",
                "question": question,
                "human_directed": human_directed,
                "consult": consult,
                "role_tasks": normalized_tasks,
                "binding": self._binding(context),
                "return_identity": context["team_return_identity"],
                "context_version": context["context_version"],
                "context_digest": context["context_digest"],
                "guidance": [],
                "advice": [],
                "draft": None,
                "events": [],
            }
            self._save(session)
            write_json(self.current_path, {"team_id": session["team_id"]})
            if compact:
                snapshot = self._snapshot(context, session)
                summary = self._session_summary(
                    session, checked=True, snapshot_id=snapshot["snapshot_id"]
                )
                summary["snapshot"] = {"snapshot_id": snapshot["snapshot_id"]}
                summary["requests"] = (
                    self._view(session, context=context, snapshot=snapshot)["requests"]
                    if session["status"] == "consulting" and session["consult"]
                    else {}
                )
                return summary
            return self._view(session, context=context)

    @staticmethod
    def _session_summary(session, *, checked=False, snapshot_id=None):
        """Report saved coordination state without rebuilding scientific evidence."""
        action = (session.get("draft") or {}).get("action") or {}
        binding = session.get("binding") if isinstance(session.get("binding"), Mapping) else {}
        return_id = (
            session.get("return_identity")
            if isinstance(session.get("return_identity"), Mapping)
            else {}
        )
        consulted = [opinion["role"] for opinion in session["advice"]]
        clarifications = set(session.get("pending_clarifications") or {})
        return {
            "team_id": session["team_id"],
            "revision": session["revision"],
            "status": session["status"],
            "view": "saved_session_summary",
            "snapshot_id": snapshot_id or session.get("snapshot_id") or binding.get("snapshot_id"),
            "state_version": session.get("state_version")
            or binding.get("state_version")
            or return_id.get("state_version"),
            "config_version": session.get("config_version")
            or binding.get("config_version")
            or return_id.get("config_version"),
            "model_version": session.get("model_version")
            or binding.get("model_version")
            or return_id.get("model_version"),
            "action_hash": _json_digest(action) if action else None,
            "plan_id": None,
            "proposal_hash": None,
            "stale": False if checked else None,
            "execution_approved": False if checked else None,
            "registered_roles": consulted,
            "remaining_roles": [
                role
                for role in session["consult"]
                if role not in consulted or role in clarifications
            ]
            if session["status"] == "consulting"
            else [],
            "draft": {
                "operation": action.get("operation") or action.get("tool"),
                "target_count": len(action.get("target_ids") or []),
                "simulation_registered": bool(action.get("_simulation_plan")),
                "simulation_program_status": (action.get("_simulation_program") or {}).get(
                    "status"
                ),
                "selection_plot": session["draft"].get("selection_plot"),
            }
            if action
            else None,
        }

    def read_summary(self, team_id=None):
        # Sessions and the current pointer are each published atomically. An ID
        # in the pointer always resolves to an existing, complete saved session.
        # This view is informational and does not certify action freshness.
        return self._session_summary(self._load(team_id))

    def read(self, team_id=None, *, request_role=None, include_history=False):
        with self.control.chat_handler.lock, team_lock(self.control.state_path):
            return self._view(
                self._load(team_id), request_role=request_role, include_history=include_history
            )

    def evidence(
        self, team_id, revision, snapshot_id, source_ref, *, offset=0, limit=25, task_ids=None
    ):
        """Read exact deferred evidence under the original consultation binding."""
        from phase_agent.evidence.advisory_projection import project_advisory_payload
        from phase_agent.persistence.state.read_versions import read_identity

        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("team_evidence_invalid_page")
        if task_ids is not None and (
            not isinstance(task_ids, list)
            or any(not isinstance(t, str) or not t for t in task_ids)
            or len(task_ids) != len(set(task_ids))
        ):
            raise ValueError("team_evidence_invalid_task_ids")
        with self.control.chat_handler.lock, team_lock(self.control.state_path):
            session = self._active(team_id, revision)
            context = self._fresh(session)
            snapshot = self._snapshot(context, session)
            if snapshot["snapshot_id"] != snapshot_id:
                raise ValueError("team_evidence_snapshot_changed")
            raw = self._snapshot_payload(context, session)
            projected = project_advisory_payload(
                raw, source_identity=context.get("team_source_identity")
            )
            permitted = (projected.get("decision_context", {}).get("deferred_evidence") or {}).get(
                "fields"
            ) or []
            if source_ref not in permitted:
                raise ValueError("team_evidence_source_not_deferred")
            value = raw
            for part in source_ref.split("."):
                value = value[part]
            if isinstance(value, dict):
                if task_ids is not None:
                    raise ValueError("team_evidence_task_filter_invalid")
                items = sorted(value.items())
                page = dict(items[offset : offset + limit])
                total = len(items)
            elif isinstance(value, list):
                items = value
                if task_ids is not None:
                    if not source_ref.endswith(".tasks"):
                        raise ValueError("team_evidence_task_filter_invalid")
                    wanted = set(task_ids)
                    items = [
                        row
                        for row in value
                        if isinstance(row, dict) and row.get("task_id") in wanted
                    ]
                    if len(items) != len(wanted) or len({row["task_id"] for row in items}) != len(
                        wanted
                    ):
                        raise ValueError("team_evidence_task_ids_missing_or_duplicated")
                total = len(items)
                page = deepcopy(items[offset : offset + limit])
            else:
                raise ValueError("team_evidence_source_not_pageable")
            if read_identity(self.control.state_path, strong=True) != context.get(
                "team_source_identity"
            ):
                raise ValueError("team_evidence_sources_changed_during_read")
            return {
                "snapshot_id": snapshot_id,
                "source_ref": source_ref,
                "total": total,
                "offset": offset,
                "next_offset": offset + len(page) if offset + len(page) < total else None,
                "items": page,
            }

    def advise(self, team_id, revision, value, *, compact=False):
        with self.control.chat_handler.lock, team_lock(self.control.state_path):
            session = self._active(team_id, revision)
            if session["status"] != "consulting":
                raise ValueError("codex_team_not_consulting")
            role = value.get("role") if isinstance(value, dict) else None
            if role not in session["consult"]:
                raise ValueError("expert_was_not_requested")
            context = self._fresh(session)
            snapshot = self._snapshot(context, session)
            additional_identifiers = set()
            deferred = (snapshot.get("facts", {}).get("decision_context") or {}).get(
                "qbc_candidates"
            )
            if (
                isinstance(deferred, dict)
                and deferred.get("source_ref") == "decision_context.qbc_candidates"
            ):
                from phase_agent.persistence.state.read_versions import read_identity

                identity = context.get("team_source_identity")
                raw = self._snapshot_payload(context, session)["decision_context"]["qbc_candidates"]
                for candidate in raw:
                    if isinstance(candidate, dict):
                        additional_identifiers.update(
                            candidate[key]
                            for key in ("candidate_id", "structure_id", "branch_id")
                            if isinstance(candidate.get(key), str) and candidate[key]
                        )
                if read_identity(self.control.state_path, strong=True) != identity:
                    raise ValueError("codex_team_sources_changed_during_read; reload")
            opinion = validate_advice(
                value, role=role, snapshot=snapshot, additional_identifiers=additional_identifiers
            )
            previous = next((a for a in session["advice"] if a["role"] == role), None)
            pending = session.get("pending_clarifications", {}).get(role)
            if previous and previous != opinion and not pending:
                raise ValueError("expert_already_answered; human revision required")
            if not previous or pending:
                session["advice"] = [a for a in session["advice"] if a["role"] != role]
                session["advice"].append(opinion)
                session.get("pending_clarifications", {}).pop(role, None)
                session["events"].append(
                    {"revision": revision, "kind": "expert_advice", "advice": deepcopy(opinion)}
                )
                self._save(session)
            if compact:
                return {
                    "team_id": team_id,
                    "revision": revision,
                    "snapshot_id": snapshot["snapshot_id"],
                    "status": session["status"],
                    "registered_role": role,
                    "remaining_roles": [
                        name
                        for name in session["consult"]
                        if name not in {a["role"] for a in session["advice"]}
                        or name in session.get("pending_clarifications", {})
                    ],
                    "execution_approved": False,
                    "context_access": "Reuse the tasks from start/read; phase_team_read can retrieve one pending role or full strategic facts.",
                }
            return self._view(session, context=context, snapshot=snapshot)

    def clarify(self, team_id, revision, role, gap_index):
        """One targeted read-only follow-up; new evidence still requires revision."""
        with self.control.chat_handler.lock, team_lock(self.control.state_path):
            session = self._active(team_id, revision)
            if session["status"] != "consulting" or role not in session["consult"]:
                raise ValueError("clarification_requires_active_consultation")
            context = self._fresh(session)
            previous = next((a for a in session["advice"] if a["role"] == role), {})
            gaps = previous.get("evidence_gaps", [])
            if (
                previous.get("recommendation") != "insufficient_evidence"
                or type(gap_index) is not int
                or not 0 <= gap_index < len(gaps)
            ):
                raise ValueError("clarification_requires_declared_evidence_gap")
            counts = session.setdefault("clarification_counts", {})
            if counts.get(role, 0) >= 1:
                raise ValueError(
                    "clarification_limit_reached; retain unknown or revise with new facts"
                )
            snapshot = self._snapshot(context, session)
            counts[role] = 1
            gap = {
                "gap_index": gap_index,
                "gap": gaps[gap_index],
                "snapshot_id": snapshot["snapshot_id"],
                "round": 1,
            }
            session.setdefault("pending_clarifications", {})[role] = gap
            session["events"].append(
                {"revision": revision, "kind": "clarification_requested", "role": role, **gap}
            )
            self._save(session)
            return self._view(session, context=context, snapshot=snapshot, request_role=role)

    def draft(
        self,
        team_id,
        revision,
        action,
        *,
        rationale,
        memory_ids=None,
        simulation_plan=None,
        replace_draft=False,
        compact=False,
    ):
        from phase_agent.agents.common.action_contracts import action_contract_errors

        rationale = _text(rationale, "rationale")
        with self.control.chat_handler.lock, team_lock(self.control.state_path):
            session = self._active(team_id, revision)
            current_draft_action = (session.get("draft") or {}).get("action") or {}
            deterministic_prep = self._is_deterministic_local_prep(
                action
            ) and self._is_deterministic_local_prep(current_draft_action)
            allowed_replace_statuses = {"planning_simulation", "awaiting_human_input"}
            if deterministic_prep:
                allowed_replace_statuses = allowed_replace_statuses | {
                    "ready_to_propose",
                    "validation_rejected",
                }
            replacing = replace_draft and session["status"] in allowed_replace_statuses
            if session["status"] != "consulting" and not replacing:
                raise ValueError(
                    "codex_team_review_required; revise before replacing an accepted draft"
                )
            context = self._fresh(session)
            if session.get("pending_clarifications") or set(session["consult"]) != {
                a["role"] for a in session["advice"]
            }:
                raise ValueError("codex_team_missing_expert_opinion")
            if not isinstance(action, dict) or action.get("_decision_team") is not None:
                raise ValueError("coordinator_action_must_not_supply_team_metadata")
            if any(key.startswith("_simulation") for key in action):
                raise ValueError("draft_requires_new_simulation_plan_via_simulation_argument")
            errors = action_contract_errors(action)
            if errors:
                raise ValueError("invalid_coordinator_action: " + "; ".join(errors))
            if (action.get("tool") or action.get("action_type")) not in context["allowed_tools"]:
                raise ValueError("coordinator_tool_outside_scope")
            if (context.get("required_contracts") or {}).get("scientific_decision"):
                from phase_agent.agents.strategic.scientific_decisions import (
                    scientific_decision_errors,
                )

                decision_context = {
                    **(context.get("decision_context") or {}),
                    **(context.get("team_decision_context") or {}),
                }
                errors = scientific_decision_errors(
                    action,
                    {
                        "config_version": context.get("config_version"),
                        "model_version": context.get("model_version")
                        or (context.get("project") or {}).get("initial_model_version"),
                    },
                    decision_context,
                )
                if errors:
                    raise ValueError(
                        "coordinator_scientific_decision_invalid: " + "; ".join(errors)
                    )
            from phase_agent.runtime.codex.context import dft_audit_preflight_errors

            errors = dft_audit_preflight_errors(action, context.get("decision_context") or {})
            if errors:
                raise ValueError("coordinator_preflight_failed: " + "; ".join(errors))
            from phase_agent.agents.common.proposal_validation import prerequisite_errors

            errors = prerequisite_errors(
                action,
                {
                    **(context.get("decision_context") or {}),
                    **(context.get("team_decision_context") or {}),
                },
            )
            if errors:
                raise ValueError("coordinator_preflight_failed: " + "; ".join(errors))
            from phase_agent.agents.permissions import missing_role_reasons
            from phase_agent.agents.simulation.program_checks import (
                dft_input_record,
                local_batch_input_record,
            )

            checked_state = context.get("team_state") or {}
            if (action.get("parameters") or {}).get("repair_profile") == "dft_recalculation":
                from phase_agent.agents.simulation.dft_recalculation import validate_action

                validate_action(action, checked_state)
            program = dft_input_record(action, checked_state)
            if program is None and self._is_deterministic_local_prep(action):
                from phase_agent.configuration.runtime.build_effective_run_config import (
                    build_effective_run_config,
                )
                from phase_agent.runtime.project_context import current_config_session

                dependencies = getattr(self.control.chat_handler, "workflow_kwargs", None) or {}
                eff_config = (
                    build_effective_run_config(
                        current_config_session(self.control.chat_handler),
                        dependencies.get("run_config"),
                    )
                    if hasattr(self.control.chat_handler, "config_session")
                    or dependencies.get("run_config")
                    else {}
                )
                program = local_batch_input_record(
                    action,
                    checked_state,
                    config=eff_config,
                    manager=dependencies.get("manager"),
                )
            if program is not None and program["status"] != "passed":
                raise ValueError(
                    "simulation_program_blocked: " + "; ".join(program["unmet_conditions"])
                )
            snapshot = self._snapshot(context, session)
            advice_snapshot_id = (
                session.get("reused_advice_snapshot_id")
                if (
                    session.get("human_directed")
                    and self._is_deterministic_local_prep(action)
                    and session.get("reused_advice_snapshot_id")
                )
                else snapshot["snapshot_id"]
            )
            if (
                session.get("human_directed")
                and not session["consult"]
                and self._is_deterministic_local_prep(action)
                and program is not None
                and program["status"] == "passed"
            ):
                missing = []
            else:
                missing = missing_role_reasons(
                    action,
                    session["advice"],
                    snapshot_id=advice_snapshot_id,
                    state=checked_state,
                    context={
                        **(context.get("decision_context") or {}),
                        **(context.get("team_decision_context") or {}),
                    },
                )
            if missing and any(a["role"] in missing for a in session["advice"]):
                raise ValueError("required_expert_evidence_insufficient; clarify once or revise")
            if missing:
                session["consult"] = list(dict.fromkeys([*session["consult"], *missing]))
                session["missing_roles"] = missing
                session["status"] = "consulting"
                session.pop("draft", None)
                session.pop("direction_review", None)
                session["events"].append(
                    {
                        "revision": revision,
                        "kind": "required_consultation",
                        "missing_roles": missing,
                    }
                )
                self._save(session)
                return self._view(session, context=context, snapshot=snapshot)
            session.pop("missing_roles", None)
            memory_ids = list(memory_ids or [])
            available = {
                row["knowledge_id"]
                for row in (context.get("decision_context", {}).get("memory_context") or {}).get(
                    "approved_knowledge", []
                )
            }
            if set(memory_ids) - available:
                raise ValueError("unavailable_memory_reference")
            if (action.get("tool") or action.get("action_type")) == "generate_branches":
                from phase_agent.agents.exploration.branch_pool_review import expand_review_indices

                # Freeze the resolved pool decisions before Simulation binds its hash.
                action = expand_review_indices(action, context.get("decision_context") or {})
            if (action.get("parameters") or {}).get("mode") == "relax_mc_inputs":
                from phase_agent.configuration.runtime.build_effective_run_config import (
                    build_effective_run_config,
                )
                from phase_agent.runtime.project_context import current_config_session
                from phase_agent.tools.relax_mc_batch.plan import attach_preview

                dependencies = self.control.chat_handler.workflow_kwargs
                config = build_effective_run_config(
                    current_config_session(self.control.chat_handler),
                    dependencies.get("run_config"),
                )
                state = read_json(self.control.state_path, {}) or {}
                if state.get("active_model"):
                    config["mlip"] = {
                        **(config.get("mlip") or {}),
                        **deepcopy(state["active_model"]),
                    }
                # Derived budget and purpose must be planned before they are frozen.
                action = attach_preview(
                    action,
                    state,
                    {**dependencies, "config_version": context.get("config_version")},
                    config,
                )
            if (action.get("parameters") or {}).get("mode") == "mc_energy_repair_inputs":
                from phase_agent.configuration.runtime.build_effective_run_config import (
                    build_effective_run_config,
                )
                from phase_agent.runtime.project_context import current_config_session
                from phase_agent.tools.local.prepare_mc_energy_repair import attach_preview

                dependencies = self.control.chat_handler.workflow_kwargs
                config = build_effective_run_config(
                    current_config_session(self.control.chat_handler),
                    dependencies.get("run_config"),
                )
                state = read_json(self.control.state_path, {}) or {}
                if state.get("active_model"):
                    config["mlip"] = {
                        **(config.get("mlip") or {}),
                        **deepcopy(state["active_model"]),
                    }
                action = attach_preview(
                    action,
                    state,
                    {**dependencies, "config_version": context.get("config_version")},
                    config,
                )
            if action.get("tool") == "select_dft_candidates":
                from phase_agent.configuration.runtime.build_effective_run_config import (
                    build_effective_run_config,
                )
                from phase_agent.runtime.project_context import current_config_session
                from phase_agent.tools.workflows.prepare_dft_proposal import prepare_dft_proposal

                dependencies = self.control.chat_handler.workflow_kwargs
                config = build_effective_run_config(
                    current_config_session(self.control.chat_handler),
                    dependencies.get("run_config"),
                )
                state = read_json(self.control.state_path, {}) or {}
                if state.get("active_model"):
                    config["mlip"] = {
                        **(config.get("mlip") or {}),
                        **deepcopy(state["active_model"]),
                    }
                prepared = prepare_dft_proposal(action, state, {}, config, revise=None)
                if prepared.error:
                    raise ValueError("dft_preview_before_simulation: " + prepared.error)
                action = prepared.action
            session.pop("direction_review", None)
            session["draft"] = {
                "action": deepcopy(action),
                "rationale": rationale,
                "memory_ids": memory_ids,
            }
            if program is not None and simulation_plan is None:
                session["draft"]["action"]["_simulation_program"] = program
                session["draft"]["action"]["_simulation_intent"] = deepcopy(action)
            if simulation_plan is not None:
                from phase_agent.agents.simulation.agent import validate_plan

                session["draft"]["action"]["_simulation_plan"] = validate_plan(
                    simulation_plan,
                    action,
                    expected_binding={
                        "team_id": team_id,
                        "revision": revision,
                        "snapshot_id": snapshot["snapshot_id"],
                    },
                )
                session["draft"]["action"]["_simulation_intent"] = deepcopy(action)
            session["status"] = (
                "planning_simulation"
                if action.get("tool", action.get("action_type"))
                not in {"pause_search", "check_convergence"}
                and simulation_plan is None
                and program is None
                else "awaiting_human_input"
            )
            if session["status"] == "awaiting_human_input":
                self._apply_confirmed_policy(session)
            from phase_agent.models.prepare_llm_request import compact_current_proposal

            session["events"].append(
                {
                    "revision": revision,
                    "kind": "draft",
                    "action": compact_current_proposal(session["draft"]["action"]),
                    "rationale": rationale,
                    "memory_ids": memory_ids,
                }
            )
            self._save(session)
            if compact:
                summary = self._session_summary(
                    session, checked=True, snapshot_id=snapshot["snapshot_id"]
                )
                if session["status"] == "planning_simulation":
                    reqs = self._view(
                        session, context=context, snapshot=snapshot, request_role="simulation"
                    )["requests"]
                    if "simulation" in reqs and isinstance(reqs["simulation"], dict):
                        reqs["simulation"].pop("operations", None)
                    summary["requests"] = reqs
                return summary
            return self._view(session, context=context, snapshot=snapshot)

    def _plot_dft_draft(self, session):
        from phase_agent.evidence.model_epoch import model_epoch
        from phase_agent.tools.workflows.dft_selection_plot import plot_dft_selection

        state = read_json(self.control.state_path, {}) or {}
        action = session["draft"]["action"]
        root = state.get("analysis_output_directory")
        epoch = model_epoch(state, state.get("active_model_version"))
        from phase_agent.evidence.model_diagram_scope import selection_diagram

        snapshot = selection_diagram(state)
        if not root or not epoch:
            raise ValueError("dft_selection_analysis_directory_or_epoch_missing")
        folder = Path(root) / "history" / epoch / "draft_selections"
        saved = (session.get("draft") or {}).get("selection_plot") or {}
        target = Path(saved["figure"]) if saved.get("figure") else None
        if (
            target is None
            or target.parent != folder
            or not target.name.startswith("DFT_Selection_")
        ):
            number = 1
            while (folder / f"DFT_Selection_{number:03d}.png").exists():
                number += 1
            target = folder / f"DFT_Selection_{number:03d}.png"
        record = {
            "model_version": state.get("active_model_version"),
            "hull_version": snapshot.get("version"),
            "decisions": (action.get("parameters") or {}).get("decisions") or [],
        }
        result = plot_dft_selection(state, record, target)
        if result["unplotted_count"]:
            raise ValueError("dft_selection_plot_unplotted:" + str(result["unplotted_count"]))
        return result

    def plot_dft_draft(self, team_id, revision):
        with self.control.chat_handler.lock, team_lock(self.control.state_path):
            session = self._active(team_id, revision)
            self._fresh(session)
            if (
                not session.get("draft")
                or session["draft"]["action"].get("tool") != "select_dft_candidates"
            ):
                raise ValueError("dft_selection_draft_required")
            result = self._plot_dft_draft(session)
            session["draft"]["selection_plot"] = result
            self._save(session)
            return result

    def simulate(self, team_id, revision, plan, *, compact=False):
        from phase_agent.agents.simulation.agent import validate_plan

        with self.control.chat_handler.lock, team_lock(self.control.state_path):
            session = self._active(team_id, revision)
            if session["status"] != "planning_simulation":
                raise ValueError("simulation_requires_current_strategic_draft")
            context = self._fresh(session)
            snapshot = self._snapshot(context, session)
            validated = validate_plan(
                plan,
                session["draft"]["action"],
                expected_binding={
                    "team_id": team_id,
                    "revision": revision,
                    "snapshot_id": snapshot["snapshot_id"],
                },
            )
            if session["draft"]["action"].get("tool") == "select_dft_candidates":
                session["draft"]["selection_plot"] = self._plot_dft_draft(session)
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
            if compact:
                return self._session_summary(session, checked=True)
            return self._view(session, context=context, snapshot=snapshot)

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
            if session["status"] not in {"submitting", "submission_uncertain"}:
                raise ValueError("reconcile_requires_uncertain_submission")
            if self._recorded_submission(session) is not None:
                raise ValueError("submission_has_durable_scientific_record")
            stale = (
                session["status"] == "submitting"
                or (session.get("submission") or {}).get("status")
                == "execution_reconciliation_required"
                or (
                    session["status"] == "submission_uncertain"
                    and not session.get("submission")
                    and ((session.get("draft") or {}).get("action") or {}).get("tool")
                    == "generate_branches"
                )
            )
            energy_block = (
                stale
                and ((session.get("draft") or {}).get("action") or {})
                .get("parameters", {})
                .get("mode")
                == "mc_energy_repair_inputs"
            )
            manual_wait = (session.get("submission") or {}).get(
                "status"
            ) == "awaiting_manual_submission"
            if manual_wait:
                from phase_agent.agents.strategic.codex_wait_submission import waiting_checkpoint

                proof = waiting_checkpoint(
                    self.control.state_path,
                    session,
                    checkpoint_id=checkpoint_id,
                    request_id=request_id,
                    checkpoint_sha256=checkpoint_sha256,
                )
            elif stale:
                from phase_agent.agents.strategic.codex_stale_submission import (
                    blocked_energy_submission_checkpoint,
                    stale_submission_checkpoint,
                )

                checker = (
                    blocked_energy_submission_checkpoint
                    if energy_block
                    else stale_submission_checkpoint
                )
                proof = checker(
                    self.control.state_path,
                    session,
                    checkpoint_id=checkpoint_id,
                    request_id=request_id,
                    checkpoint_sha256=checkpoint_sha256,
                )
            else:
                proof = rejection_checkpoint(
                    self.control.state_path,
                    session,
                    checkpoint_id=checkpoint_id,
                    request_id=request_id,
                    checkpoint_sha256=checkpoint_sha256,
                )
            session["status"] = (
                "ready_to_propose"
                if manual_wait
                or (stale and not energy_block)
                or proof.get("classification")
                in {"dft_legacy_stage_limit_rejected", "dft_input_recovery_gate_rejected"}
                else "validation_rejected"
            )
            session["events"].append(
                {
                    "revision": revision,
                    "kind": "verified_manual_wait_before_dispatch"
                    if manual_wait
                    else "verified_blocked_energy_before_dispatch"
                    if energy_block
                    else "verified_stale_lifecycle_block"
                    if stale
                    else "verified_validation_rejection",
                    "proof": proof,
                    "evidence_note": evidence_note,
                }
            )
            self._save(session)
            return self._view(session)

    def restore_dft_audit(self, team_id, revision, *, apply=False):
        """Restore the verified pre-rejection draw under the existing locks."""
        from phase_agent.agents.strategic.codex_dft_audit_recovery import recover

        with (
            self.control.chat_handler.lock,
            team_lock(self.control.state_path),
            FileLock(str(self.root / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
        ):
            session = self._active(team_id, revision)
            result = recover(self.control.state_path, session, apply=apply)
            if result["status"] == "restored":
                self.control._codex_context_cache = None
            return result

    def feedback(
        self,
        team_id,
        revision,
        decision,
        feedback,
        *,
        supersedes=None,
        role_tasks=None,
        human_directed=None,
        compact=False,
    ):
        feedback = _text(feedback, "feedback")
        if decision not in {"accept", "revise", "pause"}:
            raise ValueError("human_decision_must_be_accept_revise_or_pause")
        if (
            supersedes or role_tasks is not None or human_directed is not None
        ) and decision != "revise":
            raise ValueError("supersedes_requires_revision")
        with self.control.chat_handler.lock, team_lock(self.control.state_path):
            session = self._active(team_id, revision)
            if session["status"] in {"submitting", "submission_uncertain", "submitted"}:
                if decision != "pause" or session["status"] == "submitted":
                    raise ValueError(
                        "submission_already_started; inspect existing pending plan and receipts"
                    )
                from phase_agent.tools.state.execution_receipts import recovery_report

                with FileLock(str(self.root / "langgraph_lifecycle.sqlite") + ".lock", timeout=10):
                    state = read_json(self.control.state_path, {}) or {}
                    if (
                        self._recorded_submission(session) is not None
                        or state.get("pending_execution_policies")
                        or recovery_report(self.control.state_path, state)["unsettled"]
                    ):
                        raise ValueError(
                            "interrupted_submission_has_records_or_unsettled_execution"
                        )
                    session["status"] = "paused"
                    session["events"].append(
                        {
                            "revision": revision,
                            "kind": "human_pause_interrupted",
                            "feedback": feedback,
                        }
                    )
                    self._save(session)
                return self._session_summary(session) if compact else self._view(session)
            context = None
            if decision == "accept":
                if session["status"] != "awaiting_human_input" or not session["draft"]:
                    raise ValueError("human_review_requires_coordinator_draft")
                context = self._fresh(session)
                session["status"] = "ready_to_propose"
                from phase_agent.agents.strategic.human_review import direction_review

                session["direction_review"] = direction_review(
                    session["draft"]["action"], feedback={"decision": decision, "comment": feedback}
                )
            elif decision == "revise":
                records = _guidance_records(session)
                supersedes = list(supersedes or [])
                if len(supersedes) != len(set(supersedes)) or set(supersedes) - {
                    row["guidance_id"] for row in _effective_guidance(session)
                }:
                    raise ValueError("supersedes_requires_current_guidance_ids")
                directed = (
                    session.get("human_directed", False)
                    if human_directed is None
                    else human_directed
                )
                consult = list((role_tasks or {}).keys()) if directed else session["consult"]
                tasks = ConsultationPlan(
                    snapshot_id="pending",
                    question=session["question"],
                    consult=consult,
                    role_tasks=role_tasks or {},
                ).role_tasks
                session["human_directed"] = directed
                session["consult"] = consult
                session.pop("missing_roles", None)
                session["events"].append(
                    {
                        "revision": revision,
                        "kind": "role_tasks_superseded",
                        "role_tasks": deepcopy(session.get("role_tasks", {})),
                    }
                )
                session["role_tasks"] = {role: task.model_dump() for role, task in tasks.items()}
                context = self._context(
                    question=session["question"], role_tasks=session["role_tasks"]
                )
                session["revision"] += 1
                records.append(
                    {
                        "guidance_id": "g" + str(len(records) + 1),
                        "text": feedback,
                        "source": "human_revise",
                        "revision": session["revision"],
                        "scope": session["team_id"],
                        "supersedes": supersedes,
                    }
                )
                session["decision_contract_version"] = 2
                session["guidance_records"] = records
                session["guidance"].append(feedback)
                session.pop("direction_review", None)
                session.pop("pending_clarifications", None)
                session.pop("clarification_counts", None)
                prev_action = (session.get("draft") or {}).get("action") or {}
                keep_advice = (
                    directed
                    and not role_tasks
                    and bool(session.get("advice"))
                    and self._is_deterministic_local_prep(prev_action)
                )
                if keep_advice:
                    session["consult"] = [opinion["role"] for opinion in session["advice"]]
                    session["reused_advice_snapshot_id"] = session["advice"][0].get("snapshot_id")
                else:
                    session.pop("reused_advice_snapshot_id", None)
                session["binding"] = self._binding(context)
                session["return_identity"] = context["team_return_identity"]
                session["context_version"] = context["context_version"]
                session["context_digest"] = context["context_digest"]
                if not keep_advice:
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
            if compact and decision in {"accept", "pause", "revise"}:
                summary = self._session_summary(session, checked=decision == "accept")
                if decision == "revise" and session.get("consult") and summary["remaining_roles"]:
                    summary["requests"] = self._view(session, context=context)["requests"]
                return summary
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
            "effective_guidance": _effective_guidance(session),
            "human_review": session["events"],
            "rationale": draft["rationale"],
            "memory_ids": draft["memory_ids"],
            "binding": session["binding"],
        }
        return action

    def propose(self, team_id, revision, *, compact=False):
        # The normal control entry takes the file lock and atomically checks the
        # frozen draft, records submission intent, and enters existing approval.
        session = self._active(team_id, revision)
        if (
            session["status"] == "submission_uncertain"
            and (session.get("submission") or {}).get("status")
            == "execution_reconciliation_required"
        ):
            from phase_agent.tools.state.execution_receipts import recovery_report

            state = read_json(self.control.state_path, {}) or {}
            if not recovery_report(self.control.state_path, state).get("unsettled"):
                with self.control.chat_handler.lock, team_lock(self.control.state_path):
                    session = self._active(team_id, revision)
                    session["status"] = "ready_to_propose"
                    if isinstance(session.get("submission"), dict):
                        session["submission"]["status"] = "reconciled_no_effect"
                    session["events"].append(
                        {
                            "revision": revision,
                            "kind": "reconciled_no_effect",
                        }
                    )
                    self._save(session)
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
        if not compact:
            return result
        from phase_agent.runtime.codex.mcp import compact_pending_response

        reply = result.get("reply") or {}
        pending = result.get("pending")
        compact_pending = (
            compact_pending_response(pending) if isinstance(pending, dict) else pending
        )
        binding = session.get("binding") if isinstance(session.get("binding"), Mapping) else {}
        action = (session.get("draft") or {}).get("action") or {}
        plan_id = (
            (compact_pending or {}).get("plan_id") if isinstance(compact_pending, dict) else None
        ) or reply.get("plan_id")
        proposal_hash = (
            (compact_pending or {}).get("proposal_hash")
            if isinstance(compact_pending, dict)
            else None
        ) or reply.get("proposal_hash")
        state_version = (
            (compact_pending or {}).get("state_version")
            if isinstance(compact_pending, dict)
            else None
        ) or binding.get("state_version")
        return {
            "team_id": session["team_id"],
            "revision": session["revision"],
            "status": reply.get("status") or session["status"],
            "snapshot_id": session.get("snapshot_id") or binding.get("snapshot_id"),
            "state_version": state_version,
            "config_version": binding.get("config_version"),
            "model_version": binding.get("model_version"),
            "action_hash": _json_digest(action) if action else None,
            "plan_id": plan_id,
            "proposal_hash": proposal_hash,
            "reply": {
                k: v
                for k, v in reply.items()
                if k
                not in {
                    "effective_config",
                    "scientific_operations",
                    "calculation_cost_reference",
                    "state",
                }
            },
            **({"pending": compact_pending} if compact_pending is not None else {}),
            **({"pending_error": result["pending_error"]} if "pending_error" in result else {}),
        }


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


def _guidance_records(session):
    return deepcopy(
        session.get(
            "guidance_records",
            [
                {
                    "guidance_id": "g" + str(index + 1),
                    "text": text,
                    "source": "legacy_guidance",
                    "revision": None,
                    "scope": session["team_id"],
                    "supersedes": [],
                }
                for index, text in enumerate(session.get("guidance", []))
            ],
        )
    )


def _effective_guidance(session):
    records = _guidance_records(session)
    replaced = {identifier for row in records for identifier in row["supersedes"]}
    return [row for row in records if row["guidance_id"] not in replaced]


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise ValueError(label + "_must_be_nonempty_text_up_to_4000_characters")
    return value.strip()


def _approved_mc_continuation(state, session, action):
    """Recognize only an already-approved Relax–MC wave awaiting input export."""
    if (
        session.get("status") != "submitted"
        or action.get("tool") != "prepare_local_batch_files"
        or action.get("parameters") != {"mode": "mc_inputs"}
        or action.get("_decision_team") is not None
    ):
        return False
    targets = action.get("target_ids") or []
    if not targets or len(targets) != len(set(targets)):
        return False
    pending = [
        t
        for t in state.get("tasks") or []
        if t.get("stage") == "deep_search"
        and t.get("status") == "pending"
        and not t.get("slurm_batch_id")
    ]
    if {t.get("task_id") for t in pending} != set(targets):
        return False
    bundle_ids = {t.get("relax_mc_bundle_id") for t in pending}
    if len(bundle_ids) != 1 or None in bundle_ids:
        return False
    bundle_id = next(iter(bundle_ids))
    batch = (state.get("relax_mc_batches") or {}).get(bundle_id) or {}
    wave = (batch.get("threshold_transition") or {}).get("mc_first_wave") or {}
    if (
        wave.get("status") != "allocated_not_submitted"
        or set(wave.get("task_ids") or []) != set(targets)
        or wave.get("hull_version")
        != ((state.get("phase_diagrams") or {}).get("mlip") or {}).get("version")
        or batch.get("model_version") != state.get("active_model_version")
        or batch.get("config_version") != state.get("confirmed_config_version")
    ):
        return False
    records = [
        r
        for r in state.get("action_records") or []
        if r.get("record_id") == wave.get("original_action_record_id")
    ]
    if len(records) != 1:
        return False
    record = records[0]
    original = record.get("final_action") or {}
    frozen = (original.get("parameters") or {}).get("relax_mc_preview") or {}
    return (
        record.get("human_feedback", {}).get("decision") == "approve"
        and original.get("tool") == "prepare_local_batch_files"
        and (original.get("parameters") or {}).get("mode") == "relax_mc_inputs"
        and frozen.get("bundle_id") == bundle_id
        and frozen.get("checksum") == batch.get("plan_checksum")
        and (original.get("_decision_team") or {}).get("team_id") == session.get("team_id")
    )


def guarded_submission(control, action, submit):
    """A discussion cannot be bypassed, changed after review, or blindly replayed."""
    team = CodexTeam(control)
    with control.chat_handler.lock, team_lock(control.state_path):
        if not team.current_path.exists():
            if action.get("_decision_team") is not None:
                raise ValueError("codex_team_metadata_without_discussion")
            return submit(None)
        session = team._load()
        if session.get("status") == "submitted" and _approved_mc_continuation(
            read_json(control.state_path, {}) or {}, session, action
        ):
            return submit(None)
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

        events = session.get("events") or []
        prior = next(
            (
                event
                for event in reversed(events)
                if event.get("kind") == "verified_stale_lifecycle_block"
            ),
            None,
        )
        if prior:
            proof = prior.get("proof") or {}
            old_draft = next(
                (
                    event.get("action")
                    for event in reversed(events)
                    if event.get("kind") == "draft"
                    and event.get("revision") == prior.get("revision")
                ),
                None,
            )
            old_intent = (old_draft or {}).get("_simulation_intent") or old_draft
            current_intent = action.get("_simulation_intent") or {
                k: v for k, v in action.items() if not k.startswith("_")
            }
            if (
                (
                    proof.get("resume_from_assessed") is True
                    or proof.get("resume_from_wait_checked") is True
                )
                and proof.get("action_task_key") == action.get("task_key")
                and old_intent == current_intent
            ):
                before_dispatch.resume_checkpoint = {
                    key: proof[key] for key in ("checkpoint_id", "checkpoint_sha256", "request_id")
                }
                if proof.get("resume_next"):
                    before_dispatch.resume_checkpoint["resume_next"] = proof["resume_next"]

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
        report = reply.get("recovery_report") or {}
        unsettled = report.get("unsettled") or []
        stale_lifecycle_block = (
            reply.get("status") == "execution_reconciliation_required"
            and reply.get("submitted") is False
            and reply.get("steps_executed") == 0
            and recorded is None
            and not reply.get("record_id")
            and not reply.get("execution")
            and not reply.get("execution_result")
            and report.get("checks") == []
            and len(unsettled) == 1
            and unsettled[0].get("reason") == "unsafe_inflight_checkpoint"
            and (unsettled[0].get("identity") or {}).get("tool") == "scientific_lifecycle"
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
        ) or stale_lifecycle_block
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
