"""Record explicit approval to prepare a frozen set of candidate relax inputs."""

from hashlib import sha256
from pathlib import Path

from filelock import FileLock

from phase_agent.persistence.state.read_versions import read_identity
from phase_agent.tools.step_runner.file_protocol import read_json
from phase_agent.tools.workflows.lifecycle_support import _save_runner_state


def authorize(control, *, candidate_frame_sha256, expected_model, expected_config,
              expected_hull, user_instruction):
    if (not isinstance(user_instruction, str) or not user_instruction.strip()
            or not isinstance(candidate_frame_sha256, dict)
            or not 1 <= len(candidate_frame_sha256) <= 1000):
        raise ValueError("explicit_exact_frame_authorization_required")
    with control.chat_handler.lock, FileLock(
        str(Path(control.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10
    ):
        version = read_identity(control.state_path, strong=True)
        state = read_json(control.state_path, {}) or {}
        if (state.get("active_model_version") != expected_model
                or state.get("confirmed_config_version") != expected_config
                or ((state.get("phase_diagrams") or {}).get("mlip") or {}).get("version") != expected_hull):
            raise ValueError("exact_candidate_authorization_context_changed")
        if state.get("pending_execution_policies"):
            raise ValueError("exact_candidate_authorization_pending_execution")
        candidates = {}
        for candidate in state.get("qbc_candidates") or []:
            cid = candidate.get("candidate_id")
            if cid in candidate_frame_sha256:
                if cid in candidates:
                    raise ValueError("candidate_frame_id_ambiguous")
                candidates[cid] = candidate
        for cid, digest in candidate_frame_sha256.items():
            row = candidates.get(cid) or {}
            path = Path(row.get("structure_path") or "")
            if (not isinstance(cid, str) or not isinstance(digest, str) or len(digest) != 64
                    or row.get("structure_sha256") != digest or not path.is_file()
                    or sha256(path.read_bytes()).hexdigest() != digest):
                raise ValueError("candidate_frame_checksum_mismatch:" + str(cid))
        record = {"status": "approved_for_preparation", "candidate_frame_sha256": dict(candidate_frame_sha256),
                  "model_version": expected_model, "config_version": expected_config,
                  "hull_version": expected_hull, "user_instruction": user_instruction.strip(),
                  "submit_authorized": False}
        old = state.get("exact_candidate_relax_authorization")
        if old and old != record:
            raise ValueError("different_exact_candidate_authorization_exists")
        if old is None:
            state["exact_candidate_relax_authorization"] = record
            _save_runner_state(state, control.state_path, expected_identity=version)
        elif read_identity(control.state_path, strong=True) != version:
            raise ValueError("stale_state_file_version")
        control._codex_context_cache = None
        return {"status": "authorized" if old is None else "already_authorized",
                "candidate_count": len(candidate_frame_sha256), "submit_authorized": False}
