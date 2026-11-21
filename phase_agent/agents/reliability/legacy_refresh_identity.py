"""Reconstruct verified identities for legacy refresh relaxations."""

from __future__ import annotations

import hashlib
import io
import json
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from threading import Lock

_cache_lock = Lock()
_observations: OrderedDict = OrderedDict()
_MAX_OBSERVATIONS = 8192


def _observation_key(task):
    """Task facts plus metadata identities of every artifact the verdict reads."""
    from phase_agent.persistence.artifact_cache import stat_key

    try:
        input_path = Path(task["input_path"])
        result_dir = Path(task["result_path"]).parent
    except (KeyError, TypeError):
        return None
    files = (
        input_path,
        input_path.parent / "initial.vasp",
        result_dir / "result.json",
        result_dir / "task.finished.json",
    )
    stats = tuple(stat_key(path) for path in files)
    if any(item is None for item in stats):
        return None
    facts = json.dumps(
        {
            key: task.get(key)
            for key in (
                "task_id",
                "task_key",
                "task_checksum",
                "model_version",
                "config_version",
                "status",
                "stage",
                "input_structure_sha256",
            )
        }
        | {
            "operation": (task.get("parameters") or {}).get("model_refresh_operation"),
            "monitor": (task.get("outputs") or {}).get("extrapolation_monitor"),
        },
        sort_keys=True,
        default=str,
    )
    return facts, stats


def verified_legacy_refresh_observation(task):
    """Cached wrapper: identical task facts and unchanged files give the same verdict."""
    if (
        task.get("status") != "completed"
        or task.get("stage") != "relax_and_feature"
        or (task.get("parameters") or {}).get("model_refresh_operation") != "relax"
    ):
        return None
    key = _observation_key(task)
    if key is not None:
        with _cache_lock:
            if key in _observations:
                _observations.move_to_end(key)
                return deepcopy(_observations[key])
    result = _verified_legacy_refresh_observation(task)
    if key is not None and _observation_key(task) == key:
        with _cache_lock:
            _observations[key] = deepcopy(result)
            while len(_observations) > _MAX_OBSERVATIONS:
                _observations.popitem(last=False)
    return result


def _verified_legacy_refresh_observation(task):
    """Return an observation only when input, result and completion marker agree.

    This reads existing artifacts without changing project state or rerunning a
    calculation. Missing or inconsistent evidence leaves the round unknown.
    """
    if (
        task.get("status") != "completed"
        or task.get("stage") != "relax_and_feature"
        or (task.get("parameters") or {}).get("model_refresh_operation") != "relax"
    ):
        return None
    try:
        input_path = Path(task["input_path"])
        result_path = Path(task["result_path"]).parent / "result.json"
        input_record = json.loads(input_path.read_text(encoding="utf-8-sig"))
        initial = (input_path.parent / "initial.vasp").read_bytes()
        result_bytes = result_path.read_bytes()
        result = json.loads(result_bytes.decode("utf-8-sig"))
        marker = json.loads(
            (result_path.parent / "task.finished.json").read_text(encoding="utf-8-sig")
        )
        identity_keys = ("task_id", "task_key", "task_checksum", "model_version", "config_version")
        if any(
            not task.get(k)
            or input_record.get(k) != task.get(k)
            or result.get(k) != task.get(k)
            or marker.get(k) != task.get(k)
            for k in identity_keys
        ):
            return None
        expected_input = task.get("input_structure_sha256")
        if (
            not expected_input
            or hashlib.sha256(initial).hexdigest() != expected_input
            or input_record.get("input_structure_sha256") != expected_input
            or marker.get("status") != "completed"
            or result.get("status") != "completed"
            or marker.get("result_checksum") != hashlib.sha256(result_bytes).hexdigest()
        ):
            return None
        monitor = (task.get("outputs") or {}).get("extrapolation_monitor")
        if (
            not isinstance(monitor, dict)
            or (result.get("outputs") or {}).get("extrapolation_monitor") != monitor
        ):
            return None
        from ase.io import read
        from pymatgen.io.ase import AseAtomsAdaptor
        from Process_AL_MC.MC_sample import _relaxation_input_identity

        atoms = read(io.StringIO(initial.decode("utf-8")), format="vasp")
        identity = _relaxation_input_identity(AseAtomsAdaptor.get_structure(atoms))
        if not isinstance(identity, str) or not identity:
            return None
        return {
            "structure_id": identity,
            "identity_scheme": "periodic_input_v1",
            "monitor_complete": True,
            "extrapolation_failure": monitor.get("extrapolation_failure"),
        }
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, ImportError):
        return None
