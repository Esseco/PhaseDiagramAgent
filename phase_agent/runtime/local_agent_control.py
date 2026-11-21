"""Small authenticated control surface; no shell, Python, or arbitrary file tools."""

from copy import deepcopy
import os
from pathlib import Path
from phase_agent.persistence.memory.transaction import memory_transaction


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
from phase_agent.tools.step_runner.build_status_summary import build_status_summary
from phase_agent.tools.step_runner.file_protocol import read_json, write_json_if_version
from phase_agent.persistence.state.read_versions import read_identity
from phase_agent.runtime.control_ui_contracts import (
    StatusResponse,
    PendingResponse,
    TasksResponse,
    ConfigResponse,
    MemoryResponse,
    validate_control_response,
)
from phase_agent.runtime.control_request_contracts import validate_control_request


from phase_agent.tools.workflows.readonly_views import read_only_view


def _scientific_validation_context(action, decision_context, state_path):
    """Use the same current DFT-stage evidence as the frozen native team."""
    if not action.get("_decision_team"):
        return decision_context
    from phase_agent.evidence.post_dft_assessment import post_dft_assessment

    state = read_json(state_path, {}) or {}
    return {
        **decision_context,
        "post_dft_assessment": post_dft_assessment(state, state.get("confirmed_config") or {}),
    }


def _scientific_pending_view(state):
    view = dict(state)
    view["pending_execution_policies"] = dict(state.get("pending_execution_policies") or {})
    view["pending_execution_policies"].pop("__search_mode__", None)
    return view


class LocalAgentControl:
    def __init__(self, chat_handler):
        self.chat_handler = chat_handler
        self.state_path = chat_handler.state_path
        from phase_agent.runtime.ui_operations import UIOperations

        import threading

        self.operations = UIOperations(
            self.state_path, getattr(chat_handler, "lock", threading.RLock())
        )
        self._codex_context_cache = None

    def cancel_unsent_mc_screening(self, *, decision_id, allocation_checksum):
        """Undo a proven local screening mistake through the project Agent."""
        from filelock import FileLock
        from phase_agent.persistence.state.read_versions import read_identity
        from phase_agent.tools.state.cancel_unsent_mc_screening import cancel_unsent_mc_screening
        from phase_agent.tools.state.execution_receipts import execution_events
        from phase_agent.tools.state.state_manager import update_state_snapshot
        from phase_agent.tools.workflows.lifecycle_support import _save_runner_state

        with (
            self.chat_handler.lock,
            FileLock(
                str(self.state_path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10
            ),
        ):
            identity = read_identity(self.state_path, strong=True)
            state = read_json(self.state_path, {}) or {}
            events = execution_events(self.state_path, decision_id)
            state, count = cancel_unsent_mc_screening(
                state,
                events,
                decision_id=decision_id,
                allocation_checksum=allocation_checksum,
                _owned=True,
            )
            state = update_state_snapshot(
                state, config_version=state.get("confirmed_config_version"), _owned=True
            )
            _save_runner_state(state, self.state_path, expected_identity=identity)
            self._codex_context_cache = None
            return {
                "status": "cancelled",
                "task_count": count,
                "decision_id": decision_id,
                "remote_cancel_sent": False,
            }

    def discard_unsent_mc_energy_repair(self, *, decision_id, repair_id, user_message):
        """Preserve an exact prepared handoff while cancelling only its unsent tasks."""
        from filelock import FileLock
        from phase_agent.tools.state.discard_unsent_mc_energy_repair import (
            discard_unsent_mc_energy_repair,
        )
        from phase_agent.tools.state.execution_receipts import execution_events
        from phase_agent.tools.state.workflow_control import commit_control

        with (
            self.chat_handler.lock,
            FileLock(
                str(self.state_path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10
            ),
        ):
            identity = read_identity(self.state_path, strong=True)
            state = read_json(self.state_path, {}) or {}
            events = execution_events(self.state_path, decision_id)
            state, count = discard_unsent_mc_energy_repair(
                state,
                events,
                decision_id=decision_id,
                repair_id=repair_id,
                user_message=user_message,
                workspace_root=self.state_path.parent.parent,
                _owned=True,
            )
            commit_control(
                self.state_path,
                state,
                {},
                evidence={
                    "operation": "discard_unsent_mc_energy_repair",
                    "decision_id": decision_id,
                    "repair_id": repair_id,
                    "user_message": user_message,
                    "remote_cancel_sent": False,
                },
                expected_sha256=identity[2][0],
                desired_state=state,
                restore_allowed=False,
            )
            self._codex_context_cache = None
            return {
                "status": "cancelled",
                "task_count": count,
                "decision_id": decision_id,
                "remote_cancel_sent": False,
            }

    def rebuild_prepared_mc_handoff(self, *, decision_id, allocation_checksum, package_fingerprint):
        """Repack a receipt-bound unsubmitted MC batch via the existing exporter."""
        from filelock import FileLock
        from phase_agent.persistence.state.read_versions import read_identity
        from phase_agent.tools.state.execution_receipts import execution_events
        from phase_agent.tools.remote.repack_prepared_mc import rebuild_prepared_mc_handoff
        from phase_agent.tools.workflows.lifecycle_support import _save_runner_state

        with (
            self.chat_handler.lock,
            FileLock(
                str(self.state_path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10
            ),
        ):
            identity = read_identity(self.state_path, strong=True)
            state = read_json(self.state_path, {}) or {}
            events = execution_events(self.state_path, decision_id)
            state, handoff = rebuild_prepared_mc_handoff(
                state,
                events,
                decision_id=decision_id,
                allocation_checksum=allocation_checksum,
                package_fingerprint=package_fingerprint,
            )
            _save_runner_state(state, self.state_path, expected_identity=identity)
            self._codex_context_cache = None
            from phase_agent.tools.remote.submission_layout import write_submission_index

            base = next(
                (p for p in Path(handoff["import_directory"]).parents if p.name == "submissions"),
                None,
            )
            index_warning = None
            if base:
                try:
                    write_submission_index(base, state)
                except Exception as exc:
                    index_warning = str(exc)
            return {
                "status": "prepared",
                "submitted": False,
                "handoff": handoff,
                "index_warning": index_warning,
            }

    def context_overview(self, *, publish=True):
        """Read or rebuild the small, source-bound project directory page."""
        from phase_agent.runtime.context_overview import project_overview

        handler = self.chat_handler
        kwargs = getattr(handler, "workflow_kwargs", {}) or {}
        run_config = kwargs.get("run_config") or {}
        config_path = getattr(handler, "config_session_path", None) or kwargs.get(
            "config_session_path"
        )
        ledger_path = run_config.get("ledger_path") or kwargs.get("ledger_path")
        with handler.lock:
            return project_overview(self.state_path, config_path, ledger_path, publish=publish)

    def resolve_operation(self, body):
        validate_control_request("/phase/operations/resolve", body)
        from phase_agent.runtime.operation_resolution import resolve_operation

        return resolve_operation(self, body)

    def name_direct_dft_handoff(self, record_id, source_sha256, remote_root):
        from pathlib import Path
        from filelock import FileLock
        from phase_agent.tools.step_runner.file_protocol import read_json, write_json_if_version
        from phase_agent.persistence.state.read_versions import read_identity
        from phase_agent.tools.state.execution_receipts import recovery_report
        from phase_agent.tools.remote.direct_dft_handoff_naming import name_direct_dft_handoff

        path = Path(self.state_path).resolve()
        with (
            self.chat_handler.lock,
            FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
        ):
            version = read_identity(path, strong=True)
            state = read_json(path, {}) or {}
            if state.get("pending_execution_policies") or recovery_report(path, state)["unsettled"]:
                raise ValueError("direct_dft_pending_or_unsettled_action")
            result = name_direct_dft_handoff(
                state, path.parent.parent, record_id, source_sha256, remote_root
            )
            write_json_if_version(path, state, version)
            self._codex_context_cache = None
            return result

    def repair_refresh_handoff(self, source_sha256, task_count, *, runtime_config_path=None):
        from phase_agent.runtime.refresh_handoff_repair_control import repair

        return repair(self, source_sha256, task_count, runtime_config_path=runtime_config_path)

    def prepare_phase_cpu_recovery(self, source_sha256, spec_sha256, task_count, remote_root):
        from phase_agent.runtime.phase_cpu_recovery_control import prepare

        return prepare(self, source_sha256, spec_sha256, task_count, remote_root)

    def repair_dft_handoff(self, record_id, source_sha256, task_count, partition_counts=None):
        from phase_agent.runtime.dft_handoff_repair_control import repair

        return repair(self, record_id, source_sha256, task_count, partition_counts=partition_counts)

    def register_diagnostic_retry(self, input_archive, result_archive):
        from phase_agent.runtime.diagnostic_retry_control import register

        return register(self, input_archive, result_archive)

    def reconcile_recovery_refresh_epoch(self, **args):
        from phase_agent.runtime.epoch_registration import reconcile

        return reconcile(self, **args)

    def adopt_epoch_recovery_policy(self, expected_round_digest):
        from phase_agent.runtime.epoch_recovery_adoption import adopt

        return adopt(self, expected_round_digest)

    def register_same_frame_comparisons(self, input_archive, result_archive):
        from phase_agent.runtime.same_frame_comparison_control import register

        return register(self, input_archive, result_archive)

    def dft_recalculation_review(self, target_ids):
        """Read exact failed tasks and return a self-contained Simulation request."""
        from phase_agent.agents.simulation.dft_recalculation import review
        from phase_agent.persistence.state.read_versions import read_identity

        with self.chat_handler.lock:
            before = read_identity(self.state_path, strong=True)
            state = read_json(self.state_path, {}) or {}
            result = review(state, target_ids)
            if read_identity(self.state_path, strong=True) != before:
                raise ValueError("dft_recalculation_sources_changed; read again")
            return result

    def regenerate_training_preflight(self, **arguments):
        from phase_agent.runtime.training_preflight_control import regenerate_preflight

        return regenerate_preflight(self, **arguments)

    def register_dft_failure_analysis(self, failure_evidence_digest, diagnosis):
        """Record a native Agent diagnosis of the exact current failed DFT batch."""
        from filelock import FileLock
        from phase_agent.persistence.state.read_versions import read_identity
        from phase_agent.agents.simulation.contracts import SimulationDiagnosis
        from phase_agent.evidence.dft_failure_policy import completed_failure_batches
        from phase_agent.tools.workflows.lifecycle_support import _save_runner_state

        validated = SimulationDiagnosis.model_validate(diagnosis).model_dump()
        if "execution.operation_outcome" not in validated["evidence"]:
            raise ValueError("dft_failure_analysis_missing_batch_evidence")
        with self.chat_handler.lock:
            with FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ):
                version = read_identity(self.state_path, strong=True)
                state = read_json(self.state_path, {}) or {}
                batches = completed_failure_batches(state, [])
                batch = next(
                    (
                        row
                        for row in batches
                        if row["failure_evidence_digest"] == failure_evidence_digest
                    ),
                    None,
                )
                if batch is None:
                    raise ValueError("dft_failure_analysis_stale_or_unknown")
                reports = state.setdefault("dft_failure_reviews", [])
                existing = next(
                    (
                        row
                        for row in reports
                        if row.get("batch", {}).get("failure_evidence_digest")
                        == failure_evidence_digest
                        and row.get("diagnosis") == validated
                    ),
                    None,
                )
                created = existing is None
                if created:
                    existing = {
                        "batch": batch,
                        "diagnosis": validated,
                        "analysis_source": "native_agent",
                    }
                    reports.append(existing)
                from phase_agent.evidence.dft_failure_policy import valid_failure_diagnosis

                latest = {
                    row.get("batch", {}).get("failure_evidence_digest"): row
                    for row in reports
                    if valid_failure_diagnosis(row)
                }
                matched = [latest.get(row["failure_evidence_digest"]) for row in batches]
                guard = {
                    "status": "analyzed" if all(matched) else "awaiting_analysis",
                    "threshold": 0.2,
                    "batches": deepcopy(batches),
                    "reports": [deepcopy(row) for row in matched if row],
                    "instruction": "失败已交由 Agent 分析；后续方案保留成功结果并使用既有审批。",
                }
                state["dft_failure_guard"] = guard
                _save_runner_state(state, self.state_path, expected_identity=version)
                return {
                    "status": "recorded" if created else "reused",
                    "failure_evidence_digest": failure_evidence_digest,
                    "guard_status": guard["status"],
                    "submitted": False,
                }

    def settle_cancelled_dft_reservations(self, task_ids):
        """Settle verified cancelled DFT jobs using their existing reservations."""
        from filelock import FileLock
        from phase_agent.persistence.state.read_versions import read_identity
        from phase_agent.tools.state.cancelled_dft_settlement import settle_cancelled
        from phase_agent.tools.workflows.lifecycle_support import _save_runner_state

        with self.chat_handler.lock:
            with FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ):
                identity = read_identity(self.state_path, strong=True)
                state = read_json(self.state_path, {}) or {}
                updated, report = settle_cancelled(state, task_ids)
                if report["settled_count"]:
                    _save_runner_state(updated, self.state_path, expected_identity=identity)
                elif read_identity(self.state_path, strong=True) != identity:
                    raise ValueError("stale_state_file_version")
                return report

    def register_dft_wait_waiver(self, task_ids, user_message):
        """Record the user's exact non-wait decision using the existing recovery transition."""
        from filelock import FileLock
        from phase_agent.persistence.state.read_versions import read_identity
        from phase_agent.tools.state.dft_recovery_decision import update_dft_recovery_question
        from phase_agent.tools.workflows.lifecycle_support import _save_runner_state

        if not isinstance(task_ids, list) or not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("dft_wait_scope_invalid")
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("explicit_dft_wait_decision_required")
        with self.chat_handler.lock:
            with FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ):
                version = read_identity(self.state_path, strong=True)
                state = read_json(self.state_path, {}) or {}
                prepared = update_dft_recovery_question(state)
                question = prepared.get("pending_dft_recovery_question") or {}
                if set(question.get("pending_task_ids") or []) != set(task_ids):
                    raise ValueError("dft_wait_scope_changed_or_unavailable")
                updated = update_dft_recovery_question(
                    prepared,
                    {
                        "decision": "close",
                        "question_id": question["question_id"],
                        "user_message": user_message,
                    },
                )
                _save_runner_state(updated, self.state_path, expected_identity=version)
                return {
                    "status": "recorded",
                    "task_ids": sorted(task_ids),
                    "question_id": question["question_id"],
                    "remote_cancel_sent": False,
                    "submitted": False,
                }

    def register_scheduler_terminal_evidence(self, observations):
        """Record externally observed terminal states; do not settle costs or run work."""
        from filelock import FileLock
        from phase_agent.persistence.state.read_versions import read_identity
        from phase_agent.tools.state.scheduler_terminal_evidence import bind_terminal_observations
        from phase_agent.tools.step_runner.file_protocol import write_json_if_version

        with self.chat_handler.lock:
            with FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ):
                version = read_identity(self.state_path, strong=True)
                state = read_json(self.state_path, {}) or {}
                updated = bind_terminal_observations(state, observations)
                if updated != state:
                    write_json_if_version(self.state_path, updated, version)
                elif read_identity(self.state_path, strong=True) != version:
                    raise ValueError("stale_state_file_version; reload state before reporting")
                return {
                    "status": "recorded" if updated != state else "reused",
                    "task_ids": sorted(row["task_id"] for row in observations),
                    "costs_settled": False,
                    "submitted": False,
                }

    def register_prepared_unsubmitted_evidence(self, observation):
        """Record scoped SSH observations without modifying existing task records."""
        import hashlib
        from filelock import FileLock
        from phase_agent.tools.state.prepared_unsubmitted import FIELD, create_record
        from phase_agent.tools.state.workflow_control import commit_control, read_control

        with self.chat_handler.lock:
            with FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ):
                path = Path(self.state_path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                state = read_json(path, {}) or {}
                record = create_record(state, observation)
                if Path(record["project_root"]) != path.resolve().parent.parent:
                    raise ValueError("unsubmitted_evidence_project_root_mismatch")
                previous = read_control(path)
                updated = {**state, FIELD: [*(state.get(FIELD) or []), record]}
                commit_control(
                    path,
                    state,
                    {"append": {FIELD: [record]}},
                    evidence={"kind": "human_verified_unsubmitted", "record": record},
                    expected_sha256=digest,
                    desired_state=updated,
                    restore_allowed=previous.get("restore_allowed", True) if previous else True,
                )
                return {
                    "status": "recorded",
                    "task_ids": sorted(observation["task_ids"]),
                    "submitted": False,
                    "tasks_modified": False,
                }

    def hpc_startup_recovery(self):
        from phase_agent.tools.relax_mc_batch.hpc_failure import startup_recovery_options

        return {"recovery_options": startup_recovery_options(read_json(self.state_path, {}) or {})}

    def allocate_approved_relax_mc_continuation(self, **arguments):
        from phase_agent.runtime.relax_mc_continuation_allocation import allocate

        return allocate(self, **arguments)

    def preview_imported_relax_mc(self, bundle_id):
        from filelock import FileLock
        from phase_agent.tools.remote.transfer_bundle import checksum
        from phase_agent.tools.relax_mc_batch.transition_imported_preview import preview

        path = Path(self.state_path).resolve()
        kwargs = self.chat_handler.workflow_kwargs
        ledger = Path(
            (kwargs.get("run_config") or {}).get("ledger_path") or kwargs["ledger_path"]
        ).resolve()
        with (
            self.chat_handler.lock,
            FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
        ):
            before = (checksum(path), checksum(ledger))
            state = read_json(path, {}) or {}
            data = read_json(ledger, {}) or {}
        result = preview(state, data, bundle_id)
        with (
            self.chat_handler.lock,
            FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
        ):
            if (checksum(path), checksum(ledger)) != before:
                raise ValueError("transition_preview_project_changed")
        return result

    def import_relax_mc_phase(self, bundle_id):
        from phase_agent.runtime.relax_mc_phase_import import import_relax_phase

        return import_relax_phase(self, bundle_id)

    def register_relax_mc_input_attestation(self, **arguments):
        from phase_agent.runtime.relax_mc_input_attestation_control import register

        return register(self, **arguments)

    def preview_relax_mc_threshold_continuation(self, bundle_id):
        from filelock import FileLock
        from phase_agent.tools.relax_mc_batch.transition_preview import preview

        from phase_agent.tools.remote.transfer_bundle import checksum

        path = Path(self.state_path)
        with (
            self.chat_handler.lock,
            FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
        ):
            state = read_json(path, {}) or {}
            before = checksum(path)
        result = preview(state, bundle_id)
        with (
            self.chat_handler.lock,
            FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10),
        ):
            if checksum(path) != before:
                raise ValueError("transition_preview_project_changed")
        return result

    def register_relax_mc_threshold_transition(self, **arguments):
        from phase_agent.runtime.relax_mc_transition_control import register

        return register(self, **arguments)

    def review_relax_mc_return(self, bundle_id, *, quarantine=False):
        from phase_agent.runtime.relax_mc_return_review import review_return

        return review_return(self, bundle_id, quarantine=quarantine)

    def diagnostics(self):
        from phase_agent.runtime.project_diagnostics import inspect_project

        return inspect_project(self.state_path)

    def recover_phase_evidence(self, **arguments):
        from phase_agent.runtime.phase_evidence_recovery_control import recover

        return recover(self, **arguments)

    def recovery_transaction(self):
        from phase_agent.persistence.result_recovery_commit import recovery_transaction_status

        return recovery_transaction_status(self.state_path)

    def resume_recovery_transaction(self, revision):
        from phase_agent.persistence.result_recovery_commit import resume_result_recovery

        return resume_result_recovery(self.state_path, revision=revision)

    @read_only_view
    def status(self):
        state = read_json(self.state_path, {}) or {}
        from phase_agent.policies.workflow_policy import normalize_workflow_state

        state = normalize_workflow_state(state)
        from phase_agent.runtime.scientific_progress import project_progress

        summary = build_status_summary(state, config_version=state.get("confirmed_config_version"))
        from phase_agent.runtime.project_context import (
            current_config_session,
            handler_project_context,
        )

        session = current_config_session(self.chat_handler)
        summary["current_project"] = handler_project_context(self.chat_handler, state)
        progress = project_progress(state, self.state_path, session=session)
        summary["pending_approvals"] = progress["pending_review_count"]
        summary["waiting_state"] = progress["waiting_state"]
        summary["refresh_result_facts"] = progress.get("refresh_result_facts", {})
        summary["recovery_report"] = progress["recovery_report"]
        summary["ui_operations"] = self.operations.recent()
        summary["result_recoveries"] = progress.get("result_recoveries", [])
        summary["dft_failure_reviews"] = (state.get("dft_failure_reviews") or [])[-5:]
        from phase_agent.evidence.dft_failure_policy import failure_guard_facts

        summary["dft_failure_guard"] = failure_guard_facts(state)
        return validate_control_response(StatusResponse, summary)

    @read_only_view
    def pending(self):
        from phase_agent.persistence.state.field_index import IndexUnavailable, open_indexed_state
        from phase_agent.runtime.review_requests import remember_approval_state_version
        from phase_agent.runtime.turn_process import has_reusable_initial_state

        state = (
            None
            if has_reusable_initial_state(self.state_path)
            else open_indexed_state(self.state_path)
        )
        if state is not None:
            try:
                result = self._pending_with_state(state)
                state.verify(remember=True)
                remember_approval_state_version(state, result["state_version"])
                return result
            except (IndexUnavailable, TypeError, AttributeError):
                pass
        return self._pending_with_state(read_json(self.state_path, {}) or {})

    def _pending_with_state(self, state):
        from phase_agent.runtime.review_requests import approval_state_version, pending_reviews

        state_version = approval_state_version(state)

        from phase_agent.runtime.project_context import current_config_session

        session = current_config_session(self.chat_handler)
        from phase_agent.persistence.state.field_index import IndexedState, PendingState

        science_state = (
            PendingState(state) if type(state) is IndexedState else _scientific_pending_view(state)
        )
        # The internal search-mode proposal is hidden from row-level versions.
        row_version = (
            None
            if "__search_mode__" in (state.get("pending_execution_policies") or {})
            else state_version
        )
        rows = pending_reviews(
            science_state, session, state_path=self.state_path, state_version=row_version
        )
        from phase_agent.tools.state.execution_receipts import recovery_report

        if rows and recovery_report(self.state_path, state)["unsettled"]:
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

    @read_only_view
    def tasks(self):
        from phase_agent.persistence.state.state_view import read_state_view

        view = read_state_view(self.state_path)
        return validate_control_response(
            TasksResponse,
            {"tasks": view.get("task_rows") or []},
        )

    @read_only_view
    def charts(self):
        state = read_json(self.state_path, {}) or {}
        return build_webui_charts(state)

    @read_only_view
    def config(self):
        handler = getattr(self.chat_handler, "config_delegate", None) or self.chat_handler
        from phase_agent.runtime.project_context import current_config_session

        session = current_config_session(handler)
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

    @read_only_view
    def memory(self):
        from phase_agent.persistence.memory.decision_context import decision_memory_context

        state = read_json(self.state_path, {}) or {}
        return validate_control_response(
            MemoryResponse,
            {
                "decision_context": decision_memory_context(state),
                "active": deepcopy((state.get("decision_memory") or {}).get("long_term") or {}),
                "records": deepcopy((state.get("decision_memory") or {}).get("records") or []),
                "candidate_count": len(state.get("memory_candidates") or []),
                "review_queue": deepcopy(state.get("memory_review_queue") or []),
            },
        )

    @memory_transaction
    def propose_memory(self, record):
        validate_control_request("/phase/memory/propose", {"record": record})
        version = read_identity(self.state_path, strong=True)
        state = read_json(self.state_path, {}) or {}
        result = propose_knowledge_record(state, record, source="agent_proposal", _owned=True)
        write_json_if_version(self.state_path, result["state"], version)
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

    @read_only_view
    def domain_skill_matches(self):
        config = (self.config().get("confirmed_snapshot") or {}).get("config") or {}
        signature = build_system_signature(config.get("system") or config.get("system_config"))
        return {"matches": load_matching_domain_skills(self._knowledge_root(), signature)}

    @memory_transaction
    def propose_skill_import(self):
        config = (self.config().get("confirmed_snapshot") or {}).get("config") or {}
        signature = build_system_signature(config.get("system") or config.get("system_config"))
        version = read_identity(self.state_path, strong=True)
        state = read_json(self.state_path, {}) or {}
        result = propose_domain_skill_import(state, self._knowledge_root(), signature)
        if result["proposals"]:
            write_json_if_version(self.state_path, result["state"], version)
            from phase_agent.persistence.memory.publish_memory_views import publish_memory_views

            publish_memory_views(result["state"], self.state_path)
        elif read_identity(self.state_path, strong=True) != version:
            raise ValueError("stale_state_file_version; reload state before reporting")
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

    def codex_context(self):
        from filelock import FileLock
        from phase_agent.runtime.codex.context import build_codex_context, _json_digest
        from phase_agent.persistence.ledger.phase_data_manager import (
            PhaseDataManager,
            _data_digest,
            refresh_if_changed,
        )

        handler = self.chat_handler
        kwargs = getattr(handler, "workflow_kwargs", {}) or {}
        config_path = getattr(handler, "config_session_path", None) or kwargs.get(
            "config_session_path"
        )
        run_config = kwargs.get("run_config") or {}
        ledger_path = run_config.get("ledger_path") or kwargs.get("ledger_path")
        editable = getattr(handler, "editable_config_path", None)
        if not editable:
            editable = (
                Path(self.state_path).resolve().parent.parent
                / "parameters"
                / "search_config.project.json"
            )
        manager = kwargs.get("manager")

        def file_signature(path):
            if not path:
                return None
            try:
                stat = Path(path).stat()
                # Configuration is small; bind contents as well as metadata. Ledger bytes
                # are separately verified by refresh_if_changed before cache lookup.
                content = None
                if path in (config_path, editable):
                    from hashlib import sha256

                    content = sha256(Path(path).read_bytes()).hexdigest()
                return (
                    str(Path(path).resolve()),
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ino,
                    stat.st_ctime_ns,
                    content,
                )
            except FileNotFoundError:
                return (str(Path(path).resolve()), None)

        with handler.lock:
            if ledger_path and isinstance(manager, PhaseDataManager):
                with FileLock(str(Path(ledger_path).resolve()) + ".lock", timeout=30):
                    refresh_if_changed(manager, ledger_path)

            def signature():
                from phase_agent.runtime.codex.context import shared_codex_identity
                from phase_agent.persistence.state.read_versions import read_identity

                return (
                    shared_codex_identity() or read_identity(self.state_path, strong=True),
                    file_signature(config_path),
                    file_signature(editable),
                    file_signature(ledger_path),
                    _data_digest(manager.data) if isinstance(manager, PhaseDataManager) else None,
                    _json_digest(
                        {
                            "session": kwargs.get("config_session"),
                            "phase_references": kwargs.get("phase_references"),
                        }
                    ),
                )

            before = signature()
            cached = self._codex_context_cache
            from phase_agent.runtime.turn_process import record_read_metric

            if cached and cached[0] == before:
                record_read_metric("codex_context_hit")
                result = deepcopy(cached[1])
                if signature() != before:
                    raise ValueError("codex_context_sources_changed_during_build")
                return result
            names = (
                "state_control",
                "session_file",
                "editable_config",
                "ledger_file",
                "ledger_memory",
                "runtime_session",
            )
            changed = (
                [name for name, old, new in zip(names, cached[0], before) if old != new]
                if cached
                else ["empty"]
            )
            record_read_metric("codex_context_miss", changed_sources=changed)
            context = build_codex_context(handler)
            if signature() != before:
                raise ValueError("codex_context_sources_changed_during_build")
            self._codex_context_cache = (before, deepcopy(context))
            return context

    def authorize_unknown_gs_dft(self, **arguments):
        from phase_agent.runtime.unknown_gs_dft_authorization import authorize

        return authorize(self, **arguments)

    def authorize_exact_candidate_relax(self, **arguments):
        from phase_agent.runtime.exact_candidate_relax_authorization import authorize

        return authorize(self, **arguments)

    def register_endpoint_supplement(self, csv_path, archive_path, csv_sha256, archive_sha256):
        """Register verified historical evidence under the normal lifecycle lock."""
        from filelock import FileLock
        from phase_agent.persistence.state.read_versions import read_identity
        from phase_agent.evidence.endpoint_supplement import register_endpoint_supplement
        from phase_agent.tools.workflows.lifecycle_support import _save_runner_state

        with self.chat_handler.lock:
            with FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ):
                version = read_identity(self.state_path, strong=True)
                state = read_json(self.state_path, {}) or {}
                state, receipt = register_endpoint_supplement(
                    state,
                    csv_path,
                    archive_path,
                    csv_sha256=csv_sha256,
                    archive_sha256=archive_sha256,
                )
                if receipt["status"] in {"registered", "relocated"}:
                    _save_runner_state(state, self.state_path, expected_identity=version)
                elif read_identity(self.state_path, strong=True) != version:
                    raise ValueError("stale_state_file_version; reload state before reporting")
                self._codex_context_cache = None
                return receipt

    def remove_phase_quality_records(self, *, expected_hull, targets, user_message):
        """Apply an explicit review to exact saved MLIP records through the project transaction."""
        from filelock import FileLock
        from phase_agent.analysis.phase.phase_quality_persistence import commit_human_quality_review

        if not expected_hull or not isinstance(targets, dict) or not targets:
            raise ValueError("requires_bound_hull_and_exact_review_targets")
        handler = self.chat_handler
        with (
            handler.lock,
            FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ),
        ):
            state = read_json(self.state_path, {}) or {}
            if state.get("pending_execution_policies"):
                raise ValueError("pending_execution_policy_blocks_phase_quality_removal")
            kwargs = handler.workflow_kwargs
            manager = kwargs.get("manager")
            ledger = kwargs.get("ledger_path") or getattr(manager, "_ledger_path", None)
            if manager is None or not ledger:
                raise ValueError("phase_quality_ledger_unavailable")
            result = commit_human_quality_review(
                self.state_path,
                state,
                manager,
                review_ids=list(targets),
                decision="remove",
                user_message=user_message,
                ledger_path=ledger,
                directory=kwargs.get("phase_diagram_directory"),
                expected_hull=expected_hull,
                expected_targets=targets,
            )
            self._codex_context_cache = None
            diagram = result["state"]["phase_diagrams"]["mlip"]
            return {
                "status": result["status"],
                "excluded_record_ids": result["excluded_record_ids"],
                "hull_version": diagram["version"],
                "entry_count": len(diagram["entries"]),
                "submitted": False,
            }

    def refresh_dft_candidates(self):
        """Refresh factual MC candidate bindings without selecting or creating tasks."""
        from filelock import FileLock
        from phase_agent.persistence.state.read_versions import read_identity
        from phase_agent.science.qbc.post_mc_candidates import post_mc_candidates
        from phase_agent.tools.workflows.lifecycle_support import _save_runner_state

        with self.chat_handler.lock:
            with FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ):
                version = read_identity(self.state_path, strong=True)
                state = read_json(self.state_path, {}) or {}
                if state.get("pending_execution_policies"):
                    raise ValueError("pending_execution_policy_blocks_candidate_refresh")
                from phase_agent.evidence.model_diagram_scope import selection_diagram

                diagram = selection_diagram(state)
                model = state.get("active_model_version")
                if (
                    not model
                    or diagram.get("status") != "completed"
                    or diagram.get("model_version") != model
                ):
                    raise ValueError("current_model_phase_diagram_required")
                candidates = post_mc_candidates(state, model)
                if not candidates:
                    raise ValueError("candidate_refresh_empty")
                state["qbc_candidates"] = candidates
                _save_runner_state(state, self.state_path, expected_identity=version)
                self._codex_context_cache = None
                return {
                    "status": "refreshed",
                    "candidate_count": len(candidates),
                    "model_version": model,
                    "hull_version": diagram.get("version"),
                    "tasks_created": 0,
                }

    def perturbation_recovery_candidates(self, max_det_H):
        """Read verified current-hull parents; never select or create a task."""
        from phase_agent.evidence.perturbation_recovery import build_perturbation_recovery_pool

        with self.chat_handler.lock:
            state = read_json(self.state_path, {}) or {}
            kwargs = self.chat_handler.workflow_kwargs or {}
            manager = kwargs.get("manager")
            path = kwargs.get("ledger_path") or (kwargs.get("run_config") or {}).get("ledger_path")
            ledger = read_json(path, {}) if path else getattr(manager, "data", None)
            if not isinstance(ledger, dict) or not ledger:
                raise ValueError("perturbation_recovery_ledger_unavailable")
            return build_perturbation_recovery_pool(state, ledger, max_det_H=max_det_H)

    def direct_review_report(self, scope_id):
        """Publish a derived review of an already registered DIRECT selection."""
        from filelock import FileLock
        from phase_agent.tools.workflows.direct_review_report import publish_direct_review

        with (
            self.chat_handler.lock,
            FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ),
        ):
            state = read_json(self.state_path, {}) or {}
            manager = self.chat_handler.workflow_kwargs.get("manager")
            if manager is None or not isinstance(getattr(manager, "data", None), dict):
                raise ValueError("direct_review_ledger_unavailable")
            return publish_direct_review(state, manager.data, scope_id)

    def bind_direct_candidates(self, scope_id):
        """Register parent metadata for existing selected frames, without DFT tasks."""
        from filelock import FileLock
        from phase_agent.persistence.state.read_versions import read_identity
        from phase_agent.tools.workflows.direct_review_report import bind_direct_candidates
        from phase_agent.tools.workflows.lifecycle_support import _save_runner_state

        with (
            self.chat_handler.lock,
            FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ),
        ):
            version = read_identity(self.state_path, strong=True)
            state = read_json(self.state_path, {}) or {}
            if state.get("pending_execution_policies"):
                raise ValueError("direct_candidate_pending_approval")
            manager = self.chat_handler.workflow_kwargs.get("manager")
            if manager is None or not isinstance(getattr(manager, "data", None), dict):
                raise ValueError("direct_review_ledger_unavailable")
            updated, receipt = bind_direct_candidates(state, manager.data, scope_id)
            _save_runner_state(updated, self.state_path, expected_identity=version)
            self._codex_context_cache = None
            return receipt

    def quarantine_extrapolation_refresh(self, scope_id):
        """Apply the explicitly requested refresh quarantine under the project lock."""
        from filelock import FileLock
        from phase_agent.tools.workflows.quarantine_refresh_publication import (
            quarantine_refresh_publication,
        )

        with (
            self.chat_handler.lock,
            FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ),
        ):
            state = read_json(self.state_path, {}) or {}
            if state.get("pending_execution_policies"):
                raise ValueError("recovery_pending_approval_blocks_quarantine")
            return quarantine_refresh_publication(state, self.state_path, scope_id)

    def archive_failed_epoch(self):
        """Archive the invalid epoch without changing its referenced source paths."""
        from filelock import FileLock
        from phase_agent.analysis.feedback.failed_epoch_archive import archive_failed_epoch
        from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory

        with (
            self.chat_handler.lock,
            FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ),
        ):
            state = read_json(self.state_path, {}) or {}
            root = Path(state["analysis_output_directory"]).resolve()
            lock_root = analysis_metadata_directory(root)
            lock_root.mkdir(parents=True, exist_ok=True)
            with FileLock(str(lock_root / "publication.lock"), timeout=30):
                return archive_failed_epoch(state, self.state_path)

    def codex_plot(
        self,
        *,
        refresh_analysis=False,
        organize_analysis=False,
        training_parity=False,
        round_dft_parity=False,
        round_error_trend=False,
        organize_dft=False,
    ):
        """Render the currently recorded MLIP diagram through Agent's plotter."""
        from phase_agent.runtime.phase_diagram_chat import phase_diagram_reply
        from filelock import FileLock

        with (
            self.chat_handler.lock,
            FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ),
        ):
            if not isinstance(refresh_analysis, bool) or not isinstance(organize_analysis, bool):
                raise ValueError("analysis options must be boolean")
            if (
                organize_dft
                or refresh_analysis
                or organize_analysis
                or training_parity
                or round_dft_parity
                or round_error_trend
            ):
                state = read_json(self.state_path, {}) or {}
                from phase_agent.analysis.feedback.export_dft_products import publish_output_catalog

                directory = state.get("analysis_output_directory") or getattr(
                    self.chat_handler, "analysis_output_directory", None
                )
                if not directory:
                    raise ValueError(
                        "No registered analysis output directory; recover saved results first"
                    )
                if organize_dft:
                    from phase_agent.analysis.feedback.folder_navigation import (
                        publish_folder_navigation,
                    )
                    from phase_agent.analysis.feedback.output_paths import (
                        analysis_metadata_directory,
                    )
                    from phase_agent.analysis.feedback.physical_layout import (
                        organize_physical_layout,
                    )

                    lock_root = analysis_metadata_directory(directory)
                    lock_root.mkdir(parents=True, exist_ok=True)
                    with FileLock(str(lock_root / "publication.lock"), timeout=30):
                        layout = organize_physical_layout(directory, dft_only=True)
                        publish_folder_navigation(directory)
                    return {"dft_layout": layout}
                if refresh_analysis:
                    publish_output_catalog(state, directory)
                trend_report = None
                if round_error_trend:
                    from phase_agent.analysis.feedback.round_error_trend import (
                        publish_round_error_trend,
                    )

                    trend_report = publish_round_error_trend(state, directory)
                    from phase_agent.analysis.feedback.analysis_output_organization import (
                        write_latest_guide,
                    )

                    write_latest_guide(directory, state, {})
                round_report = None
                if round_dft_parity:
                    from phase_agent.analysis.feedback.export_dft_products import (
                        export_dft_products,
                    )

                    export_dft_products(state, directory)
                    round_report = state.get("dft_result_exports") or {}
                    from phase_agent.analysis.feedback.analysis_output_organization import (
                        write_latest_guide,
                    )

                    write_latest_guide(directory, state, {})
                training_report = None
                if training_parity or round_error_trend:
                    from phase_agent.analysis.feedback.training_parity import (
                        publish_training_parity,
                    )

                    training_report = publish_training_parity(state, directory)
                    from phase_agent.analysis.feedback.analysis_output_organization import (
                        write_latest_guide,
                    )

                    write_latest_guide(directory, state, {})
                organization = None
                if organize_analysis:
                    from phase_agent.analysis.feedback.analysis_output_organization import (
                        organize_analysis_outputs,
                    )

                    organization = organize_analysis_outputs(state, directory)
                    # Relocate only navigation metadata, under the same state-write guard
                    # used by the Agent. Reload inside the lock to preserve newer facts.
                    from phase_agent.tools.state.workflow_control import state_write_guard
                    from phase_agent.tools.step_runner.file_protocol import _write_json

                    with state_write_guard(self.state_path, state):
                        fresh = read_json(self.state_path, {}) or {}
                        changed = False
                        for key, name in (
                            ("output_index_path", "output_index.md"),
                            ("round_summary_csv_path", "round_summary.csv"),
                        ):
                            old = fresh.get(key)
                            from phase_agent.analysis.feedback.output_paths import (
                                analysis_metadata_directory,
                            )

                            target = analysis_metadata_directory(directory) / name
                            old_locations = {
                                Path(directory) / name,
                                Path(directory) / "latest" / name,
                            }
                            if old and Path(old) in old_locations and target.is_file():
                                fresh[key] = str(target)
                                changed = True
                        if changed:
                            _write_json(self.state_path, fresh)
                return {
                    "reply": (
                        "已根据已登记的 DFT 与同帧模型预测生成逐轮对比；科学状态未修改。"
                        if round_dft_parity and not refresh_analysis and not organize_analysis
                        else "已更新逐轮 DFT 误差和 K-fold 折外误差；可用性详见报告。"
                        if round_error_trend and not refresh_analysis and not organize_analysis
                        else "已生成已登记训练数据的折外对比图；科学状态未修改。"
                        if training_parity and not refresh_analysis and not organize_analysis
                        else "已刷新已登记数据的分析归档和 latest；科学状态未修改。"
                        if refresh_analysis
                        else "已根据已登记数据整理分析目录；未重复计算相图。"
                    ),
                    "organization": organization,
                    "training_parity": training_report,
                    "round_dft_parity": round_report,
                    "round_error_trend": trend_report,
                    "analysis_publications": state.get("analysis_publications") or {},
                    "phase_epoch_publications": state.get("phase_epoch_publications") or {},
                    "dft_epoch_publications": state.get("dft_epoch_publications") or {},
                    "output_index_path": state.get("output_index_path"),
                }
            from phase_agent.runtime.plot_state_projection import read_plot_state

            state = read_plot_state(self.state_path)
            directory = state.get("analysis_output_directory") or getattr(
                self.chat_handler, "analysis_output_directory", None
            )
            return {
                "reply": phase_diagram_reply(
                    state, operation="plot", method="mlip", directory=directory
                )
            }

    def codex_publish_phase(self):
        """Publish only the current hull from the indexed plotting projection."""
        from filelock import FileLock
        from phase_agent.analysis.feedback.analysis_publication import _phase
        from phase_agent.runtime.plot_state_projection import read_plot_state

        with (
            self.chat_handler.lock,
            FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ),
        ):
            state = read_plot_state(self.state_path)
            directory = state.get("analysis_output_directory") or getattr(
                self.chat_handler, "analysis_output_directory", None
            )
            if not directory:
                raise ValueError("phase_publication_directory_missing")
            return _phase(state, Path(directory).resolve())

    def reopen_completed_refresh(
        self, *, expected_model, expected_plan, expected_hull, expected_preview
    ):
        """Record an explicitly requested restart, without creating or submitting tasks."""
        from filelock import FileLock
        from phase_agent.agents.strategic.codex_team import CodexTeam, team_lock
        from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory
        from phase_agent.evidence.model_epoch import model_epoch
        from phase_agent.persistence.ledger.phase_data_manager import (
            refresh_if_changed,
            _ledger_identity,
        )
        from phase_agent.persistence.state.read_versions import read_identity
        from phase_agent.runtime.project_context import current_config_session
        from phase_agent.tools.state.execution_receipts import recovery_report
        from phase_agent.tools.workflows.preview_completed_refresh import preview_completed_refresh
        from phase_agent.tools.workflows.reopen_completed_refresh import reopen_state
        from phase_agent.tools.workflows.reopen_refresh_publication import commit_reopen

        with (
            self.chat_handler.lock,
            team_lock(Path(self.state_path)),
            FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ),
        ):
            identity = read_identity(self.state_path, strong=True)
            state = read_json(self.state_path, {}) or {}
            team = CodexTeam(self)
            if team.current_path.exists() and team._load().get("status") not in {
                "paused",
                "submitted",
            }:
                raise ValueError("reopen_active_consultation")
            if recovery_report(self.state_path, state)["unsettled"]:
                raise ValueError("reopen_unsettled_execution")
            session = current_config_session(self.chat_handler)
            snapshot = session.get("confirmed_snapshot") or {}
            if session.get("status") != "confirmed" or not snapshot.get("config"):
                raise ValueError("completed_refresh_confirmed_config_required")
            manager = self.chat_handler.workflow_kwargs.get("manager")
            ledger_path = getattr(manager, "_ledger_path", None)
            if manager is None or not ledger_path:
                raise ValueError("completed_refresh_ledger_unavailable")
            refresh_if_changed(manager, ledger_path)
            ledger_version = _ledger_identity(ledger_path)
            preview = preview_completed_refresh(
                state,
                manager,
                snapshot["config"],
                expected_model=expected_model,
                expected_plan=expected_plan,
                expected_hull=expected_hull,
            )
            updated = reopen_state(
                state,
                preview,
                expected_model=expected_model,
                expected_plan=expected_plan,
                expected_hull=expected_hull,
                expected_preview=expected_preview,
            )
            if (
                read_identity(self.state_path, strong=True) != identity
                or _ledger_identity(ledger_path) != ledger_version
            ):
                raise ValueError("completed_refresh_sources_changed")
            epoch = model_epoch(state, expected_model)
            root = Path(state["analysis_output_directory"]).resolve()
            lock_root = analysis_metadata_directory(root)
            lock_root.mkdir(parents=True, exist_ok=True)
            with FileLock(str(lock_root / "publication.lock"), timeout=30):
                commit_reopen(updated, self.state_path, identity, epoch=epoch)
            self._codex_context_cache = None
            return {
                "status": "reopened",
                "preview_checksum": expected_preview,
                "candidate_count": len(preview["plan"]["candidates"]),
                "old_hull_version": expected_hull,
                "submitted": False,
                "next": "review_new_refresh_plan",
            }

    def propose_reopened_refresh(self, *, expected_preview):
        """Send the reopened refresh through the original project approval gate."""
        from phase_agent.runtime.codex.no_project_model import NoProjectModel

        handler = self.chat_handler
        with handler.lock:
            state = read_json(self.state_path, {}) or {}
            refresh = state.get("model_refresh") or {}
            if (
                not expected_preview
                or refresh.get("status") != "approval_required"
                or refresh.get("reopened_preview_checksum") != expected_preview
                or refresh.get("input_authorization")
            ):
                raise ValueError("reopened_refresh_proposal_binding_changed")
            invocation = f"reopened-refresh-{expected_preview}"
            pending = state.get("pending_execution_policies") or {}
            if set(pending) - {invocation}:
                raise ValueError("other_pending_plan_blocks_reopened_refresh")
            del state, refresh, pending
            dependencies = handler.workflow_kwargs
            absent = object()
            previous = dependencies.get("agent_client", absent)
            dependencies["agent_client"] = NoProjectModel()
            try:
                result = handler._run(invocation, None, "按已冻结范围提出本轮刷新审批")
            finally:
                if previous is absent:
                    dependencies.pop("agent_client", None)
                else:
                    dependencies["agent_client"] = previous
            return {
                "status": result.get("status"),
                "reason": result.get("reason"),
                "plan_id": invocation,
                "submitted": False,
            }

    def preview_completed_refresh(self, *, expected_model, expected_plan, expected_hull):
        """Read-only, version-bound preview of an explicitly selected completed refresh."""
        from filelock import FileLock
        from phase_agent.persistence.state.read_versions import read_identity
        from phase_agent.persistence.ledger.phase_data_manager import (
            refresh_if_changed,
            _ledger_identity,
        )
        from phase_agent.runtime.project_context import current_config_session
        from phase_agent.tools.workflows.preview_completed_refresh import preview_completed_refresh

        with (
            self.chat_handler.lock,
            FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ),
        ):
            version = read_identity(self.state_path, strong=True)
            state = read_json(self.state_path, {}) or {}
            session = current_config_session(self.chat_handler)
            snapshot = session.get("confirmed_snapshot") or {}
            if session.get("status") != "confirmed" or not snapshot.get("config"):
                raise ValueError("completed_refresh_confirmed_config_required")
            manager = self.chat_handler.workflow_kwargs.get("manager")
            ledger_path = getattr(manager, "_ledger_path", None)
            if manager is None or not ledger_path:
                raise ValueError("completed_refresh_ledger_unavailable")
            refresh_if_changed(manager, ledger_path)
            ledger_version = _ledger_identity(ledger_path)
            result = preview_completed_refresh(
                state,
                manager,
                snapshot["config"],
                expected_model=expected_model,
                expected_plan=expected_plan,
                expected_hull=expected_hull,
            )
            if (
                read_identity(self.state_path, strong=True) != version
                or _ledger_identity(ledger_path) != ledger_version
            ):
                raise ValueError("completed_refresh_sources_changed")
            result["config_version"] = snapshot.get("config_version")
            return result

    def rebuild_mlip_lineage(
        self,
        *,
        expected_model,
        expected_previous_version,
        required_record_ids,
        reason="correct historical relaxation ordering",
    ):
        """Rebuild a registered hull from saved records without running science."""
        from filelock import FileLock
        from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram
        from phase_agent.persistence.state.read_versions import read_identity
        from phase_agent.tools.step_runner.file_protocol import write_json_if_version

        if reason not in {
            "correct historical relaxation ordering",
            "exclude unidentified phase frames",
            "exclude unidentified refresh inputs",
            "remove superseded outside-boundary frames",
        }:
            raise ValueError("lineage_rebuild_reason_invalid")
        if (
            not isinstance(required_record_ids, list)
            or not required_record_ids
            or len(set(required_record_ids)) != len(required_record_ids)
        ):
            raise ValueError("lineage_required_records_invalid")
        with (
            self.chat_handler.lock,
            FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ),
        ):
            version = read_identity(self.state_path, strong=True)
            state = read_json(self.state_path, {}) or {}
            old = (state.get("phase_diagrams") or {}).get("mlip") or {}
            refresh = state.get("model_refresh") or {}
            if (
                state.get("active_model_version") != expected_model
                or refresh.get("new_model_version") != expected_model
                or old.get("version") != expected_previous_version
                or refresh.get("status")
                != (
                    "completed"
                    if reason
                    in {
                        "remove superseded outside-boundary frames",
                        "exclude unidentified phase frames",
                        "exclude unidentified refresh inputs",
                    }
                    else "waiting_results"
                )
            ):
                raise ValueError("lineage_rebuild_context_changed")
            result = update_phase_diagram(
                state.get("phase_records") or [],
                active_model_version=expected_model,
                output_state=state,
                latest_per_structure=True,
                latest_only=True,
            )
            diagram = (result.get("diagrams") or {}).get("mlip") or {}
            entries = diagram.get("entries") or []
            ids = {entry.get("record_id") for entry in entries}
            if (
                diagram.get("status") != "completed"
                or diagram.get("model_version") != expected_model
                or len(ids) != len(entries)
                or (
                    reason
                    not in {
                        "remove superseded outside-boundary frames",
                        "exclude unidentified phase frames",
                        "exclude unidentified refresh inputs",
                    }
                    and not set(required_record_ids).issubset(ids)
                )
            ):
                raise ValueError("lineage_rebuild_missing_required_evidence")
            if reason == "remove superseded outside-boundary frames":
                previous_ids = {entry.get("record_id") for entry in old.get("entries") or []}
                removed = previous_ids - ids
                if removed != set(required_record_ids) or ids - previous_ids:
                    raise ValueError("lineage_rebuild_outside_scope_changed")
                from phase_agent.analysis.phase.outside_refresh_hull import outside_refresh_inputs

                rounds = (state.get("upload_layout") or {}).get("model_rounds") or {}
                proven = outside_refresh_inputs(
                    state.get("phase_records") or [], state.get("tasks") or [], rounds
                )
                for entry in old.get("entries") or []:
                    if entry.get("record_id") not in removed:
                        continue
                    model = entry.get("model_version") or entry.get("source_version")
                    epoch = rounds.get(model)
                    boundary_epoch = proven.get(
                        (entry.get("structure_id"), entry.get("structure_sha256"))
                    )
                    if type(epoch) is not int or not boundary_epoch or epoch >= boundary_epoch:
                        raise ValueError("lineage_rebuild_outside_evidence_missing")
            if reason == "exclude unidentified refresh inputs":
                previous = {entry.get("record_id"): entry for entry in old.get("entries") or []}
                removed = set(previous) - ids
                if removed != set(required_record_ids) or ids - set(previous):
                    raise ValueError("lineage_rebuild_unidentified_input_scope_changed")
                from phase_agent.analysis.phase.outside_refresh_hull import (
                    unidentified_refresh_inputs,
                )

                rounds = (state.get("upload_layout") or {}).get("model_rounds") or {}
                proven = unidentified_refresh_inputs(
                    state.get("phase_records") or [], state.get("tasks") or [], rounds
                )
                for rid in removed:
                    entry = previous[rid]
                    epoch = rounds.get(entry.get("model_version") or entry.get("source_version"))
                    newer = proven.get((entry.get("structure_id"), entry.get("structure_sha256")))
                    if type(epoch) is not int or not newer or epoch >= newer:
                        raise ValueError("lineage_rebuild_unidentified_input_evidence_missing")
            independent = None
            if reason == "exclude unidentified phase frames":
                previous = {entry.get("record_id"): entry for entry in old.get("entries") or []}
                requested = set(required_record_ids)
                unknown = {
                    rid
                    for rid, entry in previous.items()
                    if entry.get("phase_identification_status") not in (None, "identified")
                    or str(entry.get("phase") or "").strip().lower()
                    in {"", "x", "unknown", "unidentified"}
                }
                if requested != unknown or requested & ids:
                    raise ValueError("lineage_rebuild_unidentified_scope_changed")
                removed = set(previous) - ids
                if removed != requested:
                    raise ValueError("lineage_rebuild_unexpected_removed_records")
                allowed_structures = {previous[rid].get("structure_id") for rid in requested}
                if any(
                    entry.get("record_id") not in previous
                    and entry.get("structure_id") not in allowed_structures
                    for entry in entries
                ):
                    raise ValueError("lineage_rebuild_unexpected_added_records")
                if any(
                    entry.get("phase_identification_status") not in (None, "identified")
                    or str(entry.get("phase") or "").strip().lower()
                    in {"", "x", "unknown", "unidentified"}
                    for entry in entries
                ):
                    raise ValueError("lineage_rebuild_unidentified_remains")
                model_result = update_phase_diagram(
                    state.get("phase_records") or [],
                    active_model_version=expected_model,
                    output_state=state,
                    model_versions_only=[expected_model],
                )
                independent = (model_result.get("mlip_by_version") or {}).get(expected_model) or {}
                if (
                    independent.get("status") != "completed"
                    or independent.get("model_version") != expected_model
                ):
                    raise ValueError("lineage_rebuild_independent_invalid")
                if any(
                    entry.get("phase_identification_status") not in (None, "identified")
                    or str(entry.get("phase") or "").strip().lower()
                    in {"", "x", "unknown", "unidentified"}
                    for entry in independent.get("entries") or []
                ):
                    raise ValueError("lineage_rebuild_independent_unidentified_remains")
            by_hash = {entry.get("structure_sha256"): entry for entry in entries}
            old_by_hash = {
                entry.get("structure_sha256"): entry for entry in old.get("entries") or []
            }
            tasks = {task.get("task_id"): task for task in state.get("tasks") or []}
            for task_id in refresh.get("waived_task_ids") or []:
                task = tasks.get(task_id) or {}
                sha = task.get("input_structure_sha256")
                prior, current = old_by_hash.get(sha), by_hash.get(sha)
                if (
                    not sha
                    or not prior
                    or not current
                    or prior.get("record_id") != current.get("record_id")
                ):
                    raise ValueError("lineage_rebuild_waived_energy_changed")
            state.setdefault("lineage_rebuild_history", []).append(
                {
                    "previous_diagram": old,
                    "new_version": diagram.get("version"),
                    "model_version": expected_model,
                    "required_record_ids": sorted(required_record_ids),
                    "reason": reason,
                    "scientific_calculation": False,
                }
            )
            if independent is not None:
                state["lineage_rebuild_history"][-1]["previous_model_diagram"] = (
                    state.get("phase_diagrams_by_model") or {}
                ).get(expected_model)
                state.setdefault("phase_diagrams_by_model", {})[expected_model] = independent
            state["phase_diagrams"]["mlip"] = diagram
            write_json_if_version(self.state_path, state, version)
            self._codex_context_cache = None
            return {
                "status": "rebuilt",
                "old_version": expected_previous_version,
                "new_version": diagram.get("version"),
                "entry_count": len(entries),
                "required_record_ids": sorted(required_record_ids),
                "submitted": False,
            }

    def codex_report(self, plan_id):
        """Render an existing pending plan; never approve or replay it."""
        from phase_agent.runtime.codex.reports import save_codex_report

        with self.chat_handler.lock:
            state = read_json(self.state_path, {}) or {}
            record = (state.get("pending_execution_policies") or {}).get(plan_id)
            if not isinstance(record, dict) or not record.get("agent_proposal"):
                raise ValueError("pending_plan_not_found")
            result = {
                "status": "awaiting_approval",
                "agent_proposal": record["agent_proposal"],
                "state": state,
            }
            return {"plan_id": plan_id, "report_path": save_codex_report(self.state_path, result)}

    def propose_codex(
        self, action, *, context_version, context_digest=None, conversation_id="codex-control"
    ):
        from phase_agent.agents.strategic.codex_team import guarded_submission

        if not isinstance(action, dict) or not action:
            raise ValueError("Codex structured proposal action must be a nonempty object")
        return guarded_submission(
            self,
            action,
            lambda on_dispatch: self._propose_codex(
                action,
                context_version=context_version,
                context_digest=context_digest,
                conversation_id=conversation_id,
                _on_dispatch=on_dispatch,
            ),
        )

    def _propose_codex(
        self,
        action,
        *,
        context_version,
        context_digest=None,
        conversation_id="codex-control",
        _on_dispatch=None,
    ):
        """Submit a Codex-built action to the existing proposal workflow.

        The adapter supplies only the candidate action.  The normal workflow
        still performs proposal normalization, validation, budget checks,
        project binding, human approval and execution.
        """
        validate_control_request(
            "/phase/codex/propose",
            {
                "action": action,
                "context_version": context_version,
                "context_digest": context_digest,
                "conversation_id": conversation_id,
            },
        )
        if not isinstance(action, dict) or not action:
            raise ValueError("Codex structured proposal action must be a nonempty object")
        from copy import deepcopy
        from phase_agent.graphs.invocation_context import new_invocation
        from phase_agent.runtime.codex.context import (
            build_codex_rule_audit,
            ensure_codex_post_dft_review,
            ensure_codex_round_budget_review,
        )

        snapshot = self.codex_context()
        if context_version != snapshot["context_version"]:
            raise ValueError("codex_context_stale; refresh /phase/codex/context")
        if context_digest is not None and context_digest != snapshot["context_digest"]:
            raise ValueError("codex_context_digest_mismatch; refresh /phase/codex/context")
        candidate = deepcopy(action)
        candidate["decision_source"] = "codex"
        tool = candidate.get("tool") or candidate.get("action_type")
        if tool not in snapshot["allowed_tools"]:
            raise ValueError("tool_not_allowed_in_current_codex_context")
        required_tool = snapshot.get("required_next_tool")
        if required_tool and tool != required_tool:
            raise ValueError(
                f"codex_stage_requires_{required_tool}; {snapshot.get('required_next_reason', '')}"
            )
        candidate = ensure_codex_round_budget_review(candidate, snapshot["decision_context"])
        candidate = ensure_codex_post_dft_review(candidate, snapshot["decision_context"])
        audit = build_codex_rule_audit(
            candidate, snapshot.get("state") or {}, snapshot["decision_context"]
        )
        candidate["codex_rule_audit"] = audit
        preflight_errors = []
        if audit.get("status") == "invalid":
            preflight_errors.extend(
                "codex_rule_audit:" + str(error) for error in audit.get("errors", [])
            )
        from phase_agent.agents.common.action_contracts import action_contract_errors

        preflight_errors.extend(action_contract_errors(candidate))
        from phase_agent.agents.strategic.round_budget_review import round_budget_review_errors

        preflight_errors.extend(round_budget_review_errors(candidate, snapshot["decision_context"]))
        if tool == "generate_branches":
            from phase_agent.agents.exploration.branch_pool_review import review_errors

            preflight_errors.extend(review_errors(candidate, snapshot["decision_context"]))
        if (snapshot.get("required_contracts") or {}).get("scientific_decision"):
            from phase_agent.agents.strategic.scientific_decisions import scientific_decision_errors

            preflight_errors.extend(
                scientific_decision_errors(
                    candidate,
                    {
                        "config_version": snapshot.get("config_version"),
                        "model_version": snapshot.get("model_version")
                        or (snapshot.get("project") or {}).get("initial_model_version"),
                    },
                    _scientific_validation_context(
                        candidate, snapshot["decision_context"], self.state_path
                    ),
                )
            )
        if preflight_errors:
            raise ValueError("codex_preflight_invalid: " + "; ".join(preflight_errors))
        params = candidate.get("parameters") or {}
        if not (
            candidate.get("_simulation_plan")
            or candidate.get("_simulation_program")
            or tool in {"pause_search", "check_convergence"}
            or candidate.get("operation") == "prepare_dft_inputs"
            or params.get("mode") == "dft_inputs"
        ):
            # Same outcome as the Simulation stage (Codex has no project planner),
            # reported before the full lifecycle observation runs.
            raise ValueError(
                "codex_simulation_plan_required; compute actions need a Simulation plan: "
                "use phase_team_start -> phase_team_draft -> phase_team_simulate -> "
                "phase_team_propose. No lifecycle step or scientific action was run."
            )
        candidate["decision_context"] = deepcopy(snapshot["decision_context"])

        class CodexAction:
            _validates_original_facts = True
            _decision_source = "codex"
            _codex_structured = True
            _explicit_direct_scope = (
                (candidate.get("parameters") or {}).get("scope_id")
                if candidate.get("operation")
                in {
                    "sample_direct_extrapolation",
                    "prepare_perturbation_recovery",
                    "prepare_parent_first_recovery",
                }
                else None
            )
            # Set only after the normal project identity, context and rule preflight.
            _frozen_generation_draft = tool == "generate_branches"
            _frozen_mc_action = deepcopy(candidate) if tool == "allocate_mc_bohb" else None
            _frozen_dft_action = (
                deepcopy(candidate)
                if tool == "select_dft_candidates" and candidate.get("_decision_team")
                else None
            )
            _frozen_training_action = (
                deepcopy(candidate)
                if tool == "update_mlip"
                and (candidate.get("parameters") or {}).get("prepare_inputs_only") is True
                and candidate.get("_decision_team")
                else None
            )
            _frozen_exact_relax_action = (
                deepcopy(candidate)
                if tool == "prepare_local_batch_files"
                and candidate.get("operation") == "prepare_mlip_relax"
                and (candidate.get("parameters") or {}).get("mode") == "relax_inputs"
                and (candidate.get("parameters") or {}).get("selection_scope") == "candidate_ids"
                else None
            )
            _frozen_refresh_draft = (
                tool == "prepare_local_batch_files"
                and (candidate.get("parameters") or {}).get("mode") == "model_refresh_inputs"
            )
            _failed_retry_target = (
                (candidate.get("target_ids") or [None])[0]
                if tool == "restart_failed_task"
                and (
                    (candidate.get("parameters") or {}).get("prepare_inputs_after_approval")
                    or (candidate.get("parameters") or {}).get("repair_profile")
                    == "dft_recalculation"
                )
                else None
            )

            def __call__(self, _payload):
                return deepcopy(candidate)

        sentinel = object()
        previous = self.chat_handler.workflow_kwargs.get("agent_client", sentinel)
        previous_resume = self.chat_handler.workflow_kwargs.get(
            "resume_verified_checkpoint", sentinel
        )
        with self.chat_handler.lock:
            self.chat_handler.workflow_kwargs["agent_client"] = CodexAction()
            self.chat_handler.conversation_id = conversation_id
            try:
                if _on_dispatch is not None:
                    _on_dispatch()
                resume = getattr(_on_dispatch, "resume_checkpoint", None)
                if resume:
                    self.chat_handler.workflow_kwargs["resume_verified_checkpoint"] = resume
                result = self.chat_handler._run(
                    resume["request_id"] if resume else new_invocation(),
                    None,
                    "Codex 结构化提案：提交到原有验证与审批流程。",
                )
            finally:
                if previous is sentinel:
                    self.chat_handler.workflow_kwargs.pop("agent_client", None)
                else:
                    self.chat_handler.workflow_kwargs["agent_client"] = previous
                if previous_resume is sentinel:
                    self.chat_handler.workflow_kwargs.pop("resume_verified_checkpoint", None)
                else:
                    self.chat_handler.workflow_kwargs["resume_verified_checkpoint"] = (
                        previous_resume
                    )
        # Do not send the full durable state through the control response.
        # Callers can fetch it through the existing status/pending endpoints.
        response = {
            "reply": {
                key: deepcopy(result.get(key))
                for key in (
                    "status",
                    "reason",
                    "record_id",
                    "agent_proposal",
                    "human_feedback",
                    "validation",
                    "execution",
                    "execution_result",
                    "submitted",
                    "steps_executed",
                    "recovery_report",
                    "execution_authorization",
                    "simulation_report",
                )
                if key in result
            }
        }
        for event in reversed(result.get("events") or []):
            for key in ("execution_authorization", "simulation_report"):
                if key in event:
                    response["reply"].setdefault(key, deepcopy(event[key]))
        if result.get("status") == "awaiting_approval":
            from phase_agent.runtime.codex.reports import save_codex_report

            try:
                response["report_path"] = save_codex_report(self.state_path, result)
            except Exception as error:
                response["report_error"] = (
                    f"记录保存失败：{type(error).__name__}；方案仍保留，勿重复提交。"
                )
        return response

    def decide(
        self,
        decision,
        *,
        plan_id,
        expected_state_version,
        expected_proposal_hash,
        comment="",
        conversation_id="local-control",
        operator="agent",
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
                "operator": operator,
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
            **({"operator": "codex"} if operator == "codex" else {}),
        )

    def propose_recovery(self, body):
        validate_control_request("/phase/recovery/propose", body)
        from filelock import FileLock
        from phase_agent.tools.state.execution_reconciliation import propose_reconciliation
        from phase_agent.persistence.state.read_versions import read_identity
        from phase_agent.tools.step_runner.file_protocol import write_json_if_version

        with self.chat_handler.lock:
            with FileLock(
                str(Path(self.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
                timeout=10,
            ):
                version = read_identity(self.state_path, strong=True)
                state = read_json(self.state_path, {}) or {}
                attestation = {
                    key: body.get(key)
                    for key in (
                        "evidence_note",
                        "external_jobs",
                        "output_inventory",
                        "external_cost",
                        "staging_directory",
                    )
                }
                result = propose_reconciliation(
                    self.state_path, state, body["invocation_id"], body["resolution"], attestation
                )
                write_json_if_version(self.state_path, result["state"], version)
                return {"status": result["status"], "plan_id": result["plan_id"]}

    def migrate_project_dft_nelm(self, expected_old, expected_new, value):
        from phase_agent.runtime.nelm_config_migration import migrate

        return migrate(self, expected_old, expected_new, value)

    def set_project_dft_nelm(self, value):
        """Record a user-requested project-only NELM in a versioned config draft."""
        if type(value) is not int or not 1 <= value <= 10000:
            raise ValueError("invalid_dft_nelm")
        with self.chat_handler.lock:
            session = deepcopy(self.chat_handler.workflow_kwargs.get("config_session") or {})
            if session.get("status") == "confirmed":
                session["status"] = "draft"
                session["setup_stage"] = "json_ready"
                session["previous_confirmed_snapshot"] = deepcopy(session.get("confirmed_snapshot"))
                for key in (
                    "agent_reviewed_revision",
                    "agent_reviewed_config_hash",
                    "agent_reviewed_config_digest",
                    "agent_reviewed_mother_digest",
                ):
                    session.pop(key, None)
                session.setdefault("dialogue", []).append(
                    {
                        "type": "configuration_reopened",
                        "reason": "user requested DFT NELM",
                        "previous_config_version": (session.get("confirmed_snapshot") or {}).get(
                            "config_version"
                        ),
                    }
                )
            if session.get("status") != "draft":
                raise ValueError("configuration_draft_unavailable")
            path = "dft.parameters.incar_settings.static.NELM"
            updated = apply_config_revision(
                session,
                {path: value},
                reasons={path: "user requested electronic iteration limit"},
                author="user_via_codex",
            )
            self._save_session(updated)
            self._codex_context_cache = None
            return {
                "status": "draft_updated",
                "path": path,
                "value": value,
                "previous_config_version": (session.get("confirmed_snapshot") or {}).get(
                    "config_version"
                ),
                "confirmed": False,
            }

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
        if explicit and session.get("status") == "draft":
            editable_path = getattr(self.chat_handler, "editable_config_path", None)
            import_fn = getattr(self.chat_handler, "import_editable_config_json", None)
            hash_fn = getattr(self.chat_handler, "_editable_config_hash", None)
            if (
                editable_path is not None
                and Path(editable_path).is_file()
                and callable(import_fn)
                and callable(hash_fn)
                and (
                    hash_fn() != session.get("last_imported_config_hash")
                    or str(editable_path) != session.get("last_imported_config_path")
                )
            ):
                try:
                    session = import_fn(session)["session"]
                except (OSError, TypeError, ValueError, OverflowError, KeyError) as error:
                    return {
                        "status": "draft",
                        "confirmation_error": "configuration_not_ready",
                        "missing": [f"配置 JSON 导入失败：{error}"],
                        "conflicts": [],
                        "ambiguities": [],
                    }
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

    @memory_transaction
    def review_memory(self, proposal_id, *, approved):
        validate_control_request(
            "/phase/memory/review", {"proposal_id": proposal_id, "approved": approved}
        )
        version = read_identity(self.state_path, strong=True)
        state = read_json(self.state_path, {}) or {}
        result = review_memory_update(
            state, proposal_id, approved=approved, reviewer="local_approval_page", _owned=True
        )
        write_json_if_version(self.state_path, result["state"], version)
        from phase_agent.persistence.memory.publish_memory_views import publish_memory_views

        publish_memory_views(result["state"], self.state_path)
        return {
            "proposal_id": proposal_id,
            **{key: value for key, value in result.items() if key != "state"},
        }

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
