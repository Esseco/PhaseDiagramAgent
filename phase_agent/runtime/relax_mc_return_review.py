"""Public Agent review of an incomplete return prematurely imported by older code."""

from copy import deepcopy
import json
import zipfile

from phase_agent.tools.remote.transfer_bundle import checksum, validate_bundle
from phase_agent.tools.relax_mc_batch.handoff import result_archive
from phase_agent.tools.relax_mc_batch.completion_gate import completion_error
from phase_agent.tools.step_runner.file_protocol import read_json
from phase_agent.persistence.result_recovery_commit import commit_result_recovery


def review_return(control, bundle_id, *, quarantine=False):
    from filelock import FileLock

    with (
        control.chat_handler.lock,
        FileLock(
            str(control.state_path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10
        ),
    ):
        state = read_json(control.state_path, {})
        record = state.get("relax_mc_batches", {}).get(bundle_id)
        if not record or not record.get("import_checksum"):
            raise ValueError("registered_import_required")
        archive = result_archive(record)
        digest = checksum(archive)
        if digest != record["import_checksum"]:
            raise ValueError("review_archive_changed")
        validate_bundle(archive, extra_result_suffixes=(".csv", ".bin"))
        with zipfile.ZipFile(archive) as zipped:
            summary = json.loads(zipped.read("result_summary.json"))
            returned = json.loads(zipped.read("workflow_state.json"))
        for key in ("bundle_id", "plan_checksum", "model_version", "config_version"):
            expected = bundle_id if key == "bundle_id" else record.get(key)
            if summary.get(key) != expected:
                raise ValueError("review_archive_binding_mismatch:" + key)
        error = completion_error(summary, returned.get("tasks") or [])
        response = {
            "bundle_id": bundle_id,
            "archive_sha256": digest,
            "formal_eligible": error is None,
            "reason": error,
            "submitted": False,
            "recomputed": False,
        }
        if not quarantine or error is None:
            return {"status": "reviewed", **response}
        if record.get("formal_recovery_status") != "quarantined_incomplete":
            next_state = deepcopy(state)
            next_state["relax_mc_batches"][bundle_id].update(
                formal_recovery_status="quarantined_incomplete",
                return_error="relax_mc_formal_recovery_blocked:" + error,
            )
            for snapshot in [
                next_state.get("phase_diagrams", {}).get("mlip"),
                next_state.get("phase_diagrams_by_model", {}).get(record.get("model_version")),
            ]:
                if snapshot:
                    snapshot.update(
                        validity="quarantined_incomplete_batch", quarantine_bundle_id=bundle_id
                    )
            commit_result_recovery(
                control.state_path,
                next_state,
                {},
                expected_hashes={str(control.state_path.resolve()): checksum(control.state_path)},
                evidence={"operation": "quarantine_incomplete_relax_mc_import", **response},
            )
            state = read_json(control.state_path, {})
        directory = state.get("analysis_output_directory")
        if directory:
            from phase_agent.analysis.feedback.versioned_products import publication_status
            from phase_agent.evidence.model_epoch import model_epoch

            publication_status(
                directory,
                "phase_diagram",
                state.get("active_model_version"),
                model_epoch(state, state.get("active_model_version")),
                status="not_ready",
                reason="relax_mc_formal_recovery_blocked:" + error,
            )
        control._codex_context_cache = None
        return {"status": "quarantined_incomplete", **response, "raw_results_preserved": True}
