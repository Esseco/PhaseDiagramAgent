"""HTTP routing and authentication; chat and pages are explicit dependencies."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import secrets


def create_local_http_server(
    chat_handler,
    *,
    api_key: str,
    host="127.0.0.1",
    port=8765,
    model_id,
    local_control=None,
    control_api_key=None,
    deepseek_key_setup=None,
    deepseek_model="deepseek-flash",
    request_handler,
    request_error,
    approval_page,
    setup_page,
    error_formatter,
    max_request_bytes,
):
    """Create a dependency-free OpenAI-compatible server, bound locally by default."""
    if not callable(chat_handler):
        raise TypeError("chat_handler must be callable")
    if not isinstance(api_key, str) or len(api_key) < 16:
        raise ValueError("local API bearer key must contain at least 16 characters")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "ok", "config_protocol": 2})
            elif self.path == "/phase/setup":
                if not self._is_loopback_client():
                    self._json(403, error_formatter("forbidden", "key setup is local-only"))
                    return
                from phase_agent.runtime.deepseek_credentials import load_deepseek_api_key

                self._html(
                    200,
                    setup_page(
                        deepseek_model,
                        configured=bool(load_deepseek_api_key()),
                    ),
                )
            elif self.path == "/phase/chat":
                if not self._is_loopback_client():
                    self._json(403, error_formatter("forbidden", "local chat is local-only"))
                    return
                self._json(
                    410,
                    error_formatter(
                        "chat_moved", "旧聊天页已移除，请从项目启动器打开 LangGraph 聊天页面"
                    ),
                )
            elif self.path == "/v1/models" and self._authorized():
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": model_id,
                                "object": "model",
                                "created": 0,
                                "owned_by": "local-project",
                            }
                        ],
                    },
                )
            elif self.path == "/v1/models":
                self._json(401, error_formatter("unauthorized", "invalid local API key"))
            elif self.path == "/phase/approval":
                self._html(200, approval_page())
            elif (
                self.path
                in {
                    "/phase/status",
                    "/phase/pending",
                    "/phase/tasks",
                    "/phase/charts",
                    "/phase/config",
                    "/phase/memory",
                    "/phase/memory/skills",
                }
                and self._control_authorized()
                and local_control is not None
            ):
                name = (
                    "domain_skill_matches"
                    if self.path == "/phase/memory/skills"
                    else self.path.rsplit("/", 1)[-1]
                )
                try:
                    self._json(200, getattr(local_control, name)())
                except (ValueError, KeyError) as error:
                    self._json(400, error_formatter("invalid_request", str(error)))
            elif self.path.startswith("/phase/"):
                self._json(
                    401 if not self._control_authorized() else 404,
                    error_formatter(
                        "unauthorized" if not self._control_authorized() else "not_found",
                        "invalid control API key"
                        if not self._control_authorized()
                        else "endpoint not found",
                    ),
                )
            else:
                self._json(404, error_formatter("not_found", "endpoint not found"))

        def do_POST(self):
            if self.path == "/phase/setup/key":
                if not self._is_loopback_client():
                    self._json(403, error_formatter("forbidden", "key setup is local-only"))
                    return
                if not callable(deepseek_key_setup):
                    self._json(
                        503, error_formatter("not_configured", "本地 DeepSeek 设置入口未配置")
                    )
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 8192:
                        raise request_error("请求内容无效")
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    key = body.get("api_key") if isinstance(body, dict) else None
                    if not isinstance(key, str) or len(key.strip()) < 16:
                        raise request_error("请粘贴完整的 DeepSeek API Key")
                    response = deepseek_key_setup(key.strip())
                    self._json(200, response)
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    request_error,
                    ValueError,
                ) as error:
                    self._json(400, error_formatter("invalid_request", str(error)))
                except Exception as error:
                    safe_message = getattr(error, "safe_message", None)
                    message = safe_message or "连接或本地安全保存失败；密钥未在网页中回显。"
                    self._json(502, error_formatter("setup_failed", message))
                return
            if self.path in {
                "/phase/propose",
                "/phase/decision",
                "/phase/pause",
                "/phase/config/patch",
                "/phase/config/confirm",
                "/phase/memory/review",
                "/phase/memory/propose",
                "/phase/memory/skills/import",
                "/phase/memory/skills/publish",
            }:
                if not self._control_authorized():
                    self._json(401, error_formatter("unauthorized", "invalid control API key"))
                    return
                if local_control is None:
                    self._json(404, error_formatter("not_found", "local control is disabled"))
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > max_request_bytes:
                        raise request_error("invalid body size")
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    from phase_agent.runtime.control_request_contracts import (
                        validate_control_request,
                    )

                    validate_control_request(self.path, body)
                    conversation = str(body.get("conversation_id") or "local-control")[:128]
                    if self.path == "/phase/propose":
                        response = local_control.propose(
                            body.get("instruction", ""), conversation_id=conversation
                        )
                    elif self.path == "/phase/decision":
                        response = local_control.decide(
                            body.get("decision"),
                            plan_id=body.get("plan_id"),
                            expected_state_version=body.get("state_version"),
                            expected_proposal_hash=body.get("proposal_hash"),
                            comment=body.get("comment", ""),
                            conversation_id=conversation,
                        )
                    elif self.path == "/phase/config/patch":
                        response = local_control.patch_config(
                            body.get("patch"),
                            reasons=body.get("reasons"),
                            impacts=body.get("impacts"),
                        )
                    elif self.path == "/phase/config/confirm":
                        response = local_control.confirm_config(
                            explicit=body.get("explicit") is True
                        )
                    elif self.path == "/phase/memory/review":
                        response = local_control.review_memory(
                            body.get("proposal_id"), approved=body.get("approved") is True
                        )
                    elif self.path == "/phase/memory/propose":
                        response = local_control.propose_memory(body.get("record"))
                    elif self.path == "/phase/memory/skills/import":
                        response = local_control.propose_skill_import()
                    elif self.path == "/phase/memory/skills/publish":
                        response = local_control.publish_skill(
                            body.get("draft_directory"),
                            approved=body.get("approved") is True,
                            version=body.get("version", "1.0.0"),
                        )
                    else:
                        response = local_control.pause(
                            body.get("reason", ""), conversation_id=conversation
                        )
                    self._json(200, response)
                except (UnicodeDecodeError, ValueError, KeyError, request_error) as error:
                    self._json(400, error_formatter("invalid_request", str(error)))
                return
            if self.path != "/v1/chat/completions":
                self._json(404, error_formatter("not_found", "endpoint not found"))
                return
            if not self._authorized():
                self._json(401, error_formatter("unauthorized", "invalid local API key"))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > max_request_bytes:
                    raise request_error("request body is empty or too large")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                response = request_handler(payload, chat_handler, model_id=model_id)
            except (UnicodeDecodeError, json.JSONDecodeError, request_error) as error:
                self._json(400, error_formatter("invalid_request", str(error)))
                return
            except Exception as error:
                import traceback

                traceback.print_exc()
                self.log_error("local workflow failed (%s)", type(error).__name__)
                self._json(
                    500,
                    error_formatter(
                        "workflow_error",
                        f"local Agent workflow failed ({type(error).__name__}); inspect server stderr",
                    ),
                )
                return
            if payload.get("stream"):
                self._stream(response)
            else:
                self._json(200, response)

        def _authorized(self):
            value = self.headers.get("Authorization", "")
            return value.startswith("Bearer ") and secrets.compare_digest(value[7:], api_key)

        def _control_authorized(self):
            value = self.headers.get("Authorization", "")
            return (
                isinstance(control_api_key, str)
                and len(control_api_key) >= 16
                and value.startswith("Bearer ")
                and secrets.compare_digest(value[7:], control_api_key)
            )

        def _is_loopback_client(self):
            try:
                return ipaddress.ip_address(self.client_address[0]).is_loopback
            except (ValueError, IndexError):
                return False

        def _json(self, status, value):
            body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _html(self, status, value):
            body = value.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'",
            )
            self.end_headers()
            self.wfile.write(body)

        def _stream(self, response):
            base = {key: response[key] for key in ("id", "created", "model")}
            content = response["choices"][0]["message"]["content"]
            chunks = [
                {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                },
                {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                },
                {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
            body = (
                "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks)
                + "data: [DONE]\n\n"
            )
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(encoded)

    return ThreadingHTTPServer((host, int(port)), Handler)
