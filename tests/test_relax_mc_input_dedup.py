from hashlib import sha256
from pathlib import Path

import pytest

from phase_agent.tools.relax_mc_batch.prepare import _copy_frozen_inputs


def test_shared_bytes_reuse_one_file_without_changing_source_bindings(tmp_path):
    sources = [tmp_path / name for name in ("a.vasp", "b.vasp", "c.vasp")]
    for path, content in zip(sources, (b"same", b"same", b"different"), strict=True):
        path.write_bytes(content)
    entries = [{"path": str(p), "sha256": sha256(p.read_bytes()).hexdigest()} for p in sources]
    mapping = _copy_frozen_inputs(entries, tmp_path / "out")
    assert mapping[str(sources[0])] == mapping[str(sources[1])]
    assert mapping[str(sources[0])] != mapping[str(sources[2])]
    assert len(list((tmp_path / "out" / "inputs").iterdir())) == 2
    assert all(Path(mapping[str(p)]).read_bytes() == p.read_bytes() for p in sources)


def test_changed_source_is_rejected(tmp_path):
    path = tmp_path / "source.vasp"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="input_file_changed"):
        _copy_frozen_inputs([{"path": str(path), "sha256": "0" * 64}], tmp_path / "out")


def test_existing_payload_rebinds_only_identical_input_paths():
    from phase_agent.tools.relax_mc_batch.minimal_payload import reduced_payload

    entries = [
        {"path": "inputs/a.vasp", "sha256": "a", "size": 4},
        {"path": "inputs/b.vasp", "sha256": "a", "size": 4},
        {"path": "inputs/c.vasp", "sha256": "a", "size": 5},
        {"path": "runtime/phase_agent/tools/README.md", "sha256": "d", "size": 1},
        {
            "path": "runtime/phase_agent/analysis/feedback/templates/workspace.md",
            "sha256": "e",
            "size": 1,
        },
    ]
    batch = {
        "tasks": [{"id": "t1", "structure_path": "inputs/b.vasp"}],
        "phase_records": [{"structure_path": "inputs/b.vasp", "energy": -1.0}],
        "budget_reservations": {"old": {"status": "running"}},
    }
    kept, result, aliases = reduced_payload(entries, batch)
    assert aliases == {"inputs/b.vasp": "inputs/a.vasp"}
    assert result["tasks"][0] == {"id": "t1", "structure_path": "inputs/a.vasp"}
    assert result["phase_records"][0] == {"structure_path": "inputs/a.vasp", "energy": -1.0}
    assert result["budget_reservations"] == batch["budget_reservations"]
    assert "inputs/c.vasp" in {row["path"] for row in kept}
    assert "runtime/phase_agent/analysis/feedback/templates/workspace.md" in {
        row["path"] for row in kept
    }
    assert batch["tasks"][0]["structure_path"] == "inputs/b.vasp"


def test_runtime_inventory_resolves_dynamic_worker_and_rejects_missing(tmp_path):
    from phase_agent.tools.relax_mc_batch.runtime_inventory import (
        ROOTS,
        configured_worker_roots,
        runtime_python_inventory,
    )

    def path(module):
        return module.replace(".", "/") + ".py"

    sources = {path(root): b"" for root in ROOTS}
    sources["phase_agent/custom.py"] = b"from phase_agent.custom_dependency import value"
    sources["phase_agent/custom_dependency.py"] = b"value = 1"
    sources["phase_agent/unrelated.py"] = b"value = 2"
    worker = {"command": ["python3", "--executor", "phase_agent.custom:run"]}
    roots = configured_worker_roots(worker)
    selected = runtime_python_inventory(sources, extra_roots=roots)
    assert "phase_agent/custom.py" in selected
    assert "phase_agent/custom_dependency.py" in selected
    assert "phase_agent/unrelated.py" not in selected
    with pytest.raises(ValueError, match="required_runtime_module_missing"):
        runtime_python_inventory(sources, extra_roots=["phase_agent.missing"])
