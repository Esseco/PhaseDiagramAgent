"""In-process Codex access to the existing project workflow, without an HTTP service."""

from phase_agent.runtime.agent_api import create_agent_runtime
from phase_agent.runtime.local_agent_control import LocalAgentControl


class CodexOffline:
    """Use the same proposal and review gates without calling the project model."""

    def __init__(self, runtime_config, *, runtime_factory=None):
        self.runtime_config = runtime_config
        self.runtime_factory = runtime_factory or create_agent_runtime
        self._load()

    def _load(self):
        self.handler = (
            self.runtime_factory(self.runtime_config, operator="codex")
            if self.runtime_factory is create_agent_runtime
            else self.runtime_factory(self.runtime_config)
        )
        from phase_agent.runtime.codex.no_project_model import NoProjectModel

        dependencies = getattr(self.handler, "workflow_kwargs", None)
        if isinstance(dependencies, dict):
            dependencies["agent_client"] = NoProjectModel()
        if "agent_client" in getattr(self.handler, "__dict__", {}):
            self.handler.agent_client = NoProjectModel()
        self.control = LocalAgentControl(self.handler)

    def authorize_unknown_gs_dft(self, **arguments):
        return self.control.authorize_unknown_gs_dft(**arguments)

    def cancel_unsent_mc_screening(self, **arguments):
        return self.control.cancel_unsent_mc_screening(**arguments)

    def discard_unsent_mc_energy_repair(self, **arguments):
        return self.control.discard_unsent_mc_energy_repair(**arguments)

    def rebuild_prepared_mc_handoff(self, **arguments):
        return self.control.rebuild_prepared_mc_handoff(**arguments)

    def revise_unsent_relax_mc_delivery(self, *, bundle_id, source_sha256, confirmed_unsubmitted):
        from phase_agent.tools.relax_mc_batch.revise_delivery import revise_registered_delivery

        return revise_registered_delivery(
            self.control,
            bundle_id=bundle_id,
            source_sha256=source_sha256,
            confirmed_unsubmitted=confirmed_unsubmitted,
        )

    def minimize_unsent_relax_mc_delivery(self, *, bundle_id, source_sha256, confirmed_unsubmitted):
        from phase_agent.tools.relax_mc_batch.revise_delivery import minimize_registered_delivery

        return minimize_registered_delivery(
            self.control,
            bundle_id=bundle_id,
            source_sha256=source_sha256,
            confirmed_unsubmitted=confirmed_unsubmitted,
        )

    def config(self):
        return self.control.config()

    def confirm_config(self, *, explicit=False):
        result = self.control.confirm_config(explicit=explicit)
        if result.get("status") == "confirmed":
            self._load()
        return result

    def set_remote_workspace(self, remote_root):
        from phase_agent.runtime.project_remote_workspace import set_remote_workspace

        return set_remote_workspace(self.runtime_config, remote_root)

    def allocate_approved_relax_mc_continuation(self, **arguments):
        return self.control.allocate_approved_relax_mc_continuation(**arguments)

    def preview_imported_relax_mc(self, bundle_id):
        return self.control.preview_imported_relax_mc(bundle_id)

    def import_relax_mc_phase(self, bundle_id):
        return self.control.import_relax_mc_phase(bundle_id)

    def register_relax_mc_input_attestation(self, **arguments):
        return self.control.register_relax_mc_input_attestation(**arguments)

    def preview_relax_mc_threshold_continuation(self, bundle_id):
        return self.control.preview_relax_mc_threshold_continuation(bundle_id)

    def register_relax_mc_threshold_transition(self, **arguments):
        return self.control.register_relax_mc_threshold_transition(**arguments)

    def register_scheduler_terminal_evidence(self, observations):
        return self.control.register_scheduler_terminal_evidence(observations)

    def reopen_completed_refresh(self, **arguments):
        return self.control.reopen_completed_refresh(**arguments)

    def preview_completed_refresh(self, **arguments):
        return self.control.preview_completed_refresh(**arguments)

    def rebuild_mlip_lineage(self, **arguments):
        return self.control.rebuild_mlip_lineage(**arguments)

    def context(self):
        from phase_agent.models.transport_tables import compact_evidence_tables
        from phase_agent.runtime.project_remote_workspace import remote_workspace

        result = self.control.codex_context()
        result["remote_workspace"] = remote_workspace(self.runtime_config)
        result["remote_workspace_required"] = not bool(result["remote_workspace"])
        result = compact_evidence_tables(result)
        result["evidence_encoding"] = (
            "Inline lossless tables: common values apply to every row; columns match row "
            "cells; dictionaries contain indexed cell values; prefixes prepend path suffixes; "
            "schema groups restore original indices. count is the original row count."
        )
        return result

    def propose(self, action, *, conversation_id="codex-offline"):
        context = self.control.codex_context()
        result = self.control.propose_codex(
            action,
            context_version=context["context_version"],
            context_digest=context["context_digest"],
            conversation_id=conversation_id,
        )
        if (result.get("reply") or {}).get("status") == "awaiting_approval":
            try:
                result["pending"] = self.pending()
            except Exception as error:
                # The proposal may already be durable; never invite an automatic retry.
                result["pending_error"] = type(error).__name__
        return result

    def recover_phase_evidence(self, **arguments):
        return self.control.recover_phase_evidence(**arguments)

    def publish_phase(self):
        return self.control.codex_publish_phase()

    def pending(self):
        return self.control.pending()

    def recover(
        self, *, force_metadata_rescan=False, refresh_wait_decision=None, return_delivery=None
    ):
        """Observe returns; optionally admit a verified Relax-MC return before recovery."""
        if return_delivery is not None:
            delivery = dict(return_delivery)
            if delivery.pop("kind", None) == "manual_handoff":
                from phase_agent.runtime.manual_return_delivery import deliver
            else:
                from phase_agent.runtime.relax_mc_delivery import deliver

            deliver(self.control, **delivery)
        if refresh_wait_decision not in {None, "不再等待刷新"}:
            raise ValueError("unsupported_refresh_wait_decision")
        from phase_agent.tools.step_runner.file_protocol import read_json
        from phase_agent.tools.workflows.recover_saved_results import recover_saved_results
        from phase_agent.tools.local.recover_remote_training import recover_remote_training
        from phase_agent.tools.workflows.lifecycle_support import _save_runner_state

        from pathlib import Path
        from filelock import FileLock

        lifecycle_lock = (
            str(Path(self.handler.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock"
        )
        with self.handler.lock, FileLock(lifecycle_lock, timeout=0):
            dependencies = dict(self.handler.workflow_kwargs)
            dependencies["state_path"] = self.handler.state_path
            dependencies["force_metadata_rescan"] = force_metadata_rescan
            state = read_json(self.handler.state_path, {}) or {}
            if state.get("relax_mc_batches"):
                from phase_agent.configuration.runtime.build_effective_run_config import (
                    build_effective_run_config,
                )
                from phase_agent.runtime.project_context import current_config_session
                from phase_agent.tools.relax_mc_batch.results import recover_available

                config = build_effective_run_config(
                    current_config_session(self.handler), dependencies.get("run_config")
                )
                if state.get("active_model"):
                    config["mlip"] = {**(config.get("mlip") or {}), **state["active_model"]}
                recovered = recover_available(
                    state,
                    {
                        **dependencies,
                        "effective_config": config,
                        "config_version": state.get("confirmed_config_version"),
                    },
                )
                _save_runner_state(recovered, self.handler.state_path)
                state = recovered
            prior_magnetism = {
                key: row.get("magnetic_plots")
                for key, row in (state.get("dft_result_exports") or {}).items()
            }
            updated = recover_saved_results(state, **dependencies)
            training_state, training_returns = recover_remote_training(updated)
            training_handoffs = []
            if training_returns and training_state.get("confirmed_config"):
                from phase_agent.configuration.runtime.build_effective_run_config import (
                    build_effective_run_config,
                )
                from phase_agent.tools.local.training_handoff import advance_training_handoffs

                config = build_effective_run_config(
                    {
                        "status": "confirmed",
                        "confirmed_snapshot": {
                            "config_version": training_state.get("confirmed_config_version"),
                            "config": training_state["confirmed_config"],
                        },
                    },
                    dependencies.get("run_config"),
                )
                training_state, training_handoffs = advance_training_handoffs(
                    training_state, config, state_path=self.handler.state_path
                )
                from phase_agent.tools.local.training_agent_review import review_training_choice

                training_state, training_handoffs = review_training_choice(
                    training_state, training_handoffs, None, state_path=self.handler.state_path
                )
            refresh_waiting = None
            refresh = training_state.get("model_refresh") or {}
            if refresh.get("status") == "waiting_results" and refresh.get(
                "new_model_version"
            ) == training_state.get("active_model_version"):
                from phase_agent.tools.workflows.model_refresh_gate import advance_refresh

                training_state, refresh_waiting = advance_refresh(
                    training_state, refresh_wait_decision or ""
                )
            if training_state != updated:
                _save_runner_state(training_state, self.handler.state_path)
            updated = training_state
            self.control._codex_context_cache = None
            return {
                "status": "observed",
                "submitted": False,
                "dataset_count": len(updated.get("dft_dataset_records") or []),
                "comparison_count": len(updated.get("dft_mlip_comparisons") or []),
                "dft_magnetic_analysis": [
                    {"round_scope": row["round_scope"], **row["magnetic_plots"]}
                    for key, row in (updated.get("dft_result_exports") or {}).items()
                    if row.get("magnetic_plots")
                    and row["magnetic_plots"] != prior_magnetism.get(key)
                ],
                "result_collection": updated.get("result_collection"),
                "training_returns": training_returns,
                "training_handoffs": training_handoffs,
                "model_refresh_status": (updated.get("model_refresh") or {}).get("status"),
                "model_refresh_waiting_reason": refresh_waiting,
            }

    def decide(self, decision, *, plan_id, state_version, proposal_hash, comment=""):
        return self.control.decide(
            decision,
            plan_id=plan_id,
            expected_state_version=state_version,
            expected_proposal_hash=proposal_hash,
            comment=comment,
            conversation_id="codex-offline",
            operator="codex",
        )
