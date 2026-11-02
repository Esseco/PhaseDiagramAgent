"""Codex has one implementation package and stable command entry adapters."""

import importlib
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["agent", "control", "mcp"])
def test_legacy_cli_is_only_a_main_entry_to_new_implementation(name):
    legacy = importlib.import_module("phase_agent.runtime.codex_" + name)
    current = importlib.import_module("phase_agent.runtime.codex." + name)
    assert legacy.main is current.main


@pytest.mark.parametrize("old,new", [
    ("codex_agent", "agent"), ("codex_control", "control"), ("codex_mcp", "mcp"),
    ("codex_context", "context"), ("codex_offline", "offline"),
    ("codex_reports", "reports"), ("codex_team_mcp", "team"),
])
def test_historical_descriptors_resolve_to_new_codex_package(old, new):
    from phase_agent.compatibility.module_references import import_saved_module

    assert import_saved_module("phase_agent.runtime." + old) is importlib.import_module(
        "phase_agent.runtime.codex." + new
    )


def test_lifecycle_uses_repo_root_after_package_move(tmp_path):
    from unittest.mock import Mock
    from phase_agent.runtime.codex.agent import CodexAgent

    process = Mock()
    reader = Mock(side_effect=[None, {"status": "ready"}])
    runtime = tmp_path / "agent_runtime.json"
    runtime.write_text("{}", encoding="utf-8")
    agent = CodexAgent(
        runtime, process_factory=process,
        service_reader=reader, readiness_wait=0,
    )
    agent.start()
    assert Path(process.call_args.kwargs["cwd"]) == ROOT
