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
        self.handler = self.runtime_factory(self.runtime_config)
        self.control = LocalAgentControl(self.handler)

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

    def context(self):
        from phase_agent.runtime.project_remote_workspace import remote_workspace

        result = self.control.codex_context()
        result["remote_workspace"] = remote_workspace(self.runtime_config)
        result["remote_workspace_required"] = not bool(result["remote_workspace"])
        return result

    def propose(self, action, *, conversation_id="codex-offline"):
        context = self.context()
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

    def pending(self):
        return self.control.pending()

    def recover(self):
        """Run the existing observation/feedback stage without a decision or dispatch."""
        from phase_agent.tools.step_runner.file_protocol import read_json
        from phase_agent.tools.workflows.recover_saved_results import recover_saved_results
        from phase_agent.tools.local.recover_remote_training import recover_remote_training
        from phase_agent.tools.workflows.lifecycle_support import _save_runner_state

        with self.handler.lock:
            dependencies = dict(self.handler.workflow_kwargs)
            dependencies["state_path"] = self.handler.state_path
            state = read_json(self.handler.state_path, {}) or {}
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
            refresh_waiting = None
            refresh = training_state.get("model_refresh") or {}
            if refresh.get("status") == "waiting_results" and refresh.get(
                "new_model_version"
            ) == training_state.get("active_model_version"):
                from phase_agent.tools.workflows.model_refresh_gate import advance_refresh

                training_state, refresh_waiting = advance_refresh(training_state)
            if training_state != updated:
                _save_runner_state(training_state, self.handler.state_path)
            updated = training_state
            self.control._codex_context_cache = None
            return {
                "status": "observed",
                "submitted": False,
                "dataset_count": len(updated.get("dft_dataset_records") or []),
                "comparison_count": len(updated.get("dft_mlip_comparisons") or []),
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
        )
