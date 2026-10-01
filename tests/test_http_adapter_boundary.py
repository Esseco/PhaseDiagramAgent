import ast
from http.client import HTTPConnection
from pathlib import Path
import threading

from run import local_http_server
from run.open_webui_api import create_server


def test_http_adapter_does_not_import_runtime_entry():
    tree = ast.parse(Path(local_http_server.__file__).read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "run.open_webui_api" not in imports


def test_empty_chat_body_is_rejected_without_calling_workflow():
    calls = []
    server = create_server(lambda *args, **kwargs: calls.append(args),
                           api_key="a" * 24, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection(*server.server_address, timeout=5)
    try:
        connection.request("POST", "/v1/chat/completions", body=b"",
                           headers={"Authorization": "Bearer " + "a" * 24})
        response = connection.getresponse()
        assert response.status == 400
        assert b"invalid_request" in response.read()
        assert calls == []
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
