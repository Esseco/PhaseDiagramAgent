"""Admit verified manual-return archives to an existing registered handoff."""

import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from filelock import FileLock
from phase_agent.tools.remote.transfer_bundle import checksum, validate_bundle, registered_return_locations
from phase_agent.tools.step_runner.file_protocol import read_json


def deliver(control, *, return_directory, archives):
    if not isinstance(archives, list) or not archives:
        raise ValueError("manual_return_empty")
    sources = []
    for item in archives:
        source = Path(item["archive_path"]).resolve(strict=True)
        expected = item["archive_sha256"]
        if not source.name.endswith(".results.zip") or checksum(source) != expected:
            raise ValueError("manual_return_source_checksum_mismatch")
        validate_bundle(source)
        with zipfile.ZipFile(source) as z:
            manifest = json.loads(z.read("transfer_manifest.json"))
        sources.append((source, expected, manifest))
    if len({source.name for source, _, _ in sources}) != len(sources):
        raise ValueError("manual_return_duplicate_names")
    state_path = Path(control.state_path).resolve()
    with control.chat_handler.lock, FileLock(str(state_path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=10):
        from phase_agent.persistence.state.field_index import open_indexed_state
        state = open_indexed_state(state_path) or read_json(state_path, {}) or {}
        location = registered_return_locations(state).get(str(Path(return_directory)))
        if not isinstance(location, dict):
            raise ValueError("manual_return_directory_not_registered")
        directory = Path(return_directory).resolve()
        if not directory.is_relative_to(state_path.parent.parent) or not directory.is_dir():
            raise ValueError("manual_return_directory_invalid")
        binding = {"batch": location["batch"], "input_sha256": checksum(location["input_archive"])}
        for _, _, manifest in sources:
            if manifest.get("kind") != "results" or manifest.get("batch_binding") != binding:
                raise ValueError("manual_return_binding_mismatch")
        if hasattr(state, "verify"):
            state.verify(strong=True)
        # Preflight all targets before publication, then use no-overwrite atomic links/rename.
        for source, expected, _ in sources:
            target = directory / source.name
            if target.exists() and checksum(target) != expected:
                raise ValueError("manual_return_target_conflict")
        staged = []
        try:
            for source, expected, _ in sources:
                target = directory / source.name
                if target.exists():
                    continue
                fd, name = tempfile.mkstemp(prefix=".manual-return-", suffix=".part", dir=directory)
                temp = Path(name)
                staged.append((temp, target))
                with os.fdopen(fd, "wb") as output, source.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output, 1024 * 1024)
                    output.flush(); os.fsync(output.fileno())
                if checksum(temp) != expected:
                    raise ValueError("manual_return_copy_checksum_mismatch")
            for temp, target in staged:
                if os.name == "nt": os.rename(temp, target)
                else: os.link(temp, target)
        finally:
            for temp, _ in staged: temp.unlink(missing_ok=True)
    return {"status": "delivered", "count": len(sources), "return_directory": str(directory)}
