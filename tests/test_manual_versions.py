from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from threading import Event
from unittest.mock import patch

import pytest

from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)
from phase_agent.configuration.session.create_config_draft import create_config_draft
from phase_agent.configuration.session.confirm_config_snapshot import confirm_config_snapshot
from phase_agent.configuration.session.begin_config_revision import begin_config_revision
from phase_agent.configuration.session.apply_config_revision import apply_config_revision
from phase_agent.runtime.local_project_launcher import start_agent, _check_start_cancelled
from phase_agent.runtime.project_context import project_context_line


def test_open_and_noop_edit_do_not_create_versions():
    confirmed = confirm_config_snapshot(
        create_config_draft(default_layered_search_config()), user_confirmed=True
    )
    opened = begin_config_revision(confirmed, reason="查看设置")
    assert opened["draft_revision"] == confirmed["draft_revision"]
    same = apply_config_revision(
        opened, {"calculation.mlip_version": opened["config"]["calculation"]["mlip_version"]}
    )
    assert same == opened
    repeated = confirm_config_snapshot(same, user_confirmed=True)
    assert repeated["confirmed_snapshot"] == confirmed["confirmed_snapshot"]
    changed = apply_config_revision(
        opened,
        {
            "round_strategy.maximum_mc_budget": opened["config"]["round_strategy"][
                "maximum_mc_budget"
            ]
            + 1
        },
    )
    assert changed["draft_revision"] == opened["draft_revision"] + 1
    assert (
        confirm_config_snapshot(changed, user_confirmed=True)["confirmed_snapshot"][
            "config_version"
        ]
        != confirmed["confirmed_snapshot"]["config_version"]
    )


def test_cancelled_start_does_not_spawn(tmp_path):
    event = Event()
    event.set()
    with patch("phase_agent.runtime.local_project_launcher.subprocess.Popen") as popen:
        with pytest.raises(RuntimeError, match="启动已取消"):
            start_agent(tmp_path / "agent_runtime.json", cancel_event=event)
    popen.assert_not_called()


def test_stop_while_starting_terminates_new_process():
    event = Event()
    event.set()
    process = object()
    with patch(
        "phase_agent.runtime.local_project_launcher._stop_owned_process", return_value=True
    ) as stop:
        with pytest.raises(RuntimeError, match="启动已取消"):
            _check_start_cancelled(event, process)
    stop.assert_called_once_with(process)


def test_current_header_hides_internal_versions():
    card = {
        "project": "FeMn",
        "epoch": "epoch1",
        "active_model": {"version": "remote-long-internal-id"},
        "configuration": {"version": "config-long-hash"},
        "draft": {"status": "pending_confirmation", "revision": 35},
    }
    line = project_context_line(card)
    assert line == "FeMn · epoch1 · 当前模型 微调模型 · 设置修改待确认"
    assert "config-" not in line and "remote-" not in line and "35" not in line


def test_unchanged_file_read_keeps_confirmation(tmp_path):
    from phase_agent.runtime.configuration_read_policy import confirmed_file_unchanged
    from phase_agent.configuration.session.replace_config_from_json import mother_structure_digest

    config = {"system": {"phase_references": {}}}
    source = tmp_path / "settings.json"
    handler = SimpleNamespace(editable_config_path=source, base_directory=tmp_path)
    session = {
        "confirmed_snapshot": {"config": config},
        "last_imported_mother_digest": mother_structure_digest(config),
    }
    before = deepcopy(session)
    with patch(
        "phase_agent.runtime.configuration_read_policy.load_editable_config_json",
        return_value=deepcopy(config),
    ):
        assert confirmed_file_unchanged(handler, session)
    assert session == before
    with patch(
        "phase_agent.runtime.configuration_read_policy.load_editable_config_json",
        return_value={"system": {"phase_references": {}, "boundary": {"P": ["P2"]}}},
    ):
        assert not confirmed_file_unchanged(handler, session)
