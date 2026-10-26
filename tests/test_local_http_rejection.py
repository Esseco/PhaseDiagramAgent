"""Rejected POSTs drain bounded data without invoking application actions."""

from io import BytesIO
from unittest.mock import Mock

import pytest

from phase_agent.runtime.agent_api import create_server


@pytest.fixture
def rejected_handler():
    server = create_server(
        lambda *a, **k: "unused",
        api_key="fixture-tool-key",
        control_api_key="fixture-control-key",
        port=0,
    )
    try:
        handler = object.__new__(server.RequestHandlerClass)
        handler.command = "POST"
        handler.headers = {"Content-Length": "7"}
        handler.rfile = BytesIO(b"payload")
        handler.connection = Mock()
        handler.connection.gettimeout.return_value = None
        handler.wfile = BytesIO()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        yield handler
    finally:
        server.server_close()


def test_rejected_post_drains_body_before_response(rejected_handler):
    handler = rejected_handler
    handler.send_response.side_effect = lambda status: (
        pytest.fail("POST body was left unread") if handler.rfile.read() else None
    )
    handler._json(401, {"error": "unauthorized"})
    assert handler.close_connection
    handler.send_response.assert_called_once_with(401)
    handler.send_header.assert_any_call("Connection", "close")
    assert handler.connection.settimeout.call_args.args == (None,)


@pytest.mark.parametrize("length", ["invalid", "-1", "1000001"])
def test_rejected_post_does_not_read_unbounded_body(rejected_handler, length):
    handler = rejected_handler
    handler.headers["Content-Length"] = length
    handler._json(401, {"error": "unauthorized"})
    assert handler.rfile.tell() == 0
    handler.connection.settimeout.assert_not_called()
    assert handler.close_connection


def test_parsed_post_body_is_not_read_twice(rejected_handler):
    handler = rejected_handler
    assert handler._read_post_body(7) == b"payload"
    handler.rfile = Mock()
    handler._json(400, {"error": "invalid"})
    handler.rfile.read1.assert_not_called()
    handler.connection.settimeout.assert_not_called()


def test_rejected_post_restores_socket_timeout_on_incomplete_body(rejected_handler):
    handler = rejected_handler
    handler.rfile = Mock()
    handler.rfile.read1.side_effect = TimeoutError
    handler._json(401, {"error": "unauthorized"})
    assert handler.connection.settimeout.call_args.args == (None,)
    handler.send_response.assert_called_once_with(401)


def test_rejected_post_has_total_drain_deadline(rejected_handler, monkeypatch):
    handler = rejected_handler
    handler.rfile = Mock()
    handler.rfile.read1.return_value = b"p"
    times = iter([10.0, 10.1, 10.3])
    monkeypatch.setattr("phase_agent.runtime.local_http_server.monotonic", lambda: next(times))
    handler._json(401, {"error": "unauthorized"})
    handler.rfile.read1.assert_called_once()
    assert handler.connection.settimeout.call_args.args == (None,)


def test_each_post_resets_body_read_tracking(rejected_handler):
    handler = rejected_handler
    handler._post_body_read = True
    handler._post = Mock()
    handler.do_POST()
    assert not handler._post_body_read
    handler._post.assert_called_once()
