"""Durable facts for asynchronous control requests; scientific execution stays in existing handlers."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading

from filelock import FileLock
from phase_agent.tools.step_runner.file_protocol import read_json, write_json


_workers = set()


def _checksum(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _modified_time(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0


class UIOperations:
    def __init__(self, state_path, handler_lock):
        self.directory = Path(state_path).parent / "ui_operations"
        self.handler_lock = handler_lock

    def progress_url(self):
        service = self.directory.parent / "studio/service.json"
        if service.is_file():
            try:
                port = json.loads(service.read_text(encoding="utf-8")).get("studio_port")
                if type(port) is int and 1 <= port <= 65535:
                    return f"http://127.0.0.1:{port}/phase/process"
            except (OSError, ValueError):
                pass
        return "/phase/process"

    def _path(self, identity):
        if (
            not isinstance(identity, str)
            or len(identity) != 32
            or any(c not in "0123456789abcdef" for c in identity)
        ):
            raise ValueError("invalid_operation_id")
        return self.directory / (identity + ".json")

    def get(self, identity):
        path = self._path(identity)
        resolution = self.directory / "resolutions" / (identity + ".json")
        try:
            if resolution.is_file():
                resolved = read_json(resolution)
                if (
                    not isinstance(resolved, dict)
                    or resolved.get("status") != "resolved"
                    or resolved.get("operation_id") != identity
                ):
                    raise ValueError("invalid_operation_resolution")
                return resolved
            record = read_json(path, None)
            if record is not None and (
                not isinstance(record, dict)
                or record.get("status")
                not in {"pending", "running", "completed", "blocked", "interrupted"}
            ):
                raise ValueError("invalid_operation_record")
        except (OSError, ValueError, TypeError) as error:
            from phase_agent.runtime.boundary_diagnostics import describe_failure

            return {
                "operation_id": identity,
                "record_sha256": _checksum(path),
                "status": "unavailable",
                "message": "此请求记录不可读取；已隔离，不能确认执行效果，不会重放",
                "diagnostic": describe_failure(
                    error, boundary="operation_record", object_id=identity
                ),
            }
        if record is None:
            raise ValueError("unknown_operation")
        if record["status"] in {"running", "pending"}:
            if not isinstance(record.get("owner_pid"), int) or not isinstance(
                record.get("owner_started"), (int, float)
            ):
                return {
                    **record,
                    "status": "unavailable",
                    "message": "请求进程身份缺失，需核对原执行收据；未自动重放",
                }
            import psutil

            try:
                alive = (
                    abs(psutil.Process(record["owner_pid"]).create_time() - record["owner_started"])
                    < 0.01
                )
            except psutil.NoSuchProcess:
                alive = False
            except psutil.AccessDenied:
                alive = True
            if (
                alive
                and record["owner_pid"] == os.getpid()
                and (str(self.directory.resolve()), identity) not in _workers
            ):
                alive = False
            if not alive:
                return {
                    **record,
                    "status": "interrupted",
                    "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "message": "请求进程或处理线程已结束；先核对执行收据和已有结果，未自动重放",
                    "next_action": "/phase/pending",
                }
        return {**record, "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def recent(self, *, limit=20):
        if not self.directory.is_dir():
            return []
        paths = sorted(self.directory.glob("*.json"), key=_modified_time, reverse=True)[:limit]
        return [self.get(p.stem) for p in paths]

    def submit(self, callback, *, request_id, request):
        identity = hashlib.sha256(str(request_id).encode()).hexdigest()[:32]
        fingerprint = hashlib.sha256(
            json.dumps(request, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.directory / ".requests.lock"), timeout=2):
            path = self._path(identity)
            if path.is_file():
                record = self.get(identity)
                if record.get("request_fingerprint") != fingerprint:
                    raise ValueError("idempotency_key_reused_for_different_request")
                return {**record, "idempotent_replay": True}
            recent = self.recent(limit=None)
            if any(
                r["status"] in {"unavailable", "interrupted"}
                or (
                    r["status"] == "blocked"
                    and (r.get("diagnostic") or {}).get("execution_effect") == "unknown"
                )
                for r in recent
            ):
                raise ValueError("unreadable_operation_requires_receipt_review")
            active = next((r for r in recent if r["status"] in {"running", "pending"}), None)
            if active:
                return {
                    **active,
                    "busy": True,
                    "message": "项目已有后台请求，未重复执行；可继续查看进度",
                }
            import psutil

            record = {
                "operation_id": identity,
                "status": "pending",
                "request_fingerprint": fingerprint,
                "route": request["route"],
                "owner_pid": os.getpid(),
                "owner_started": psutil.Process().create_time(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status_url": "/phase/operations/" + identity,
                "progress_url": self.progress_url(),
                "message": "后台处理已登记，尚不表示科研任务完成",
            }
            write_json(path, record)
            _workers.add((str(self.directory.resolve()), identity))
            try:
                self._start_worker(identity, callback)
            except Exception:
                _workers.discard((str(self.directory.resolve()), identity))
                raise
            return record

    def _start_worker(self, identity, callback):
        threading.Thread(
            target=self._execute,
            args=(identity, callback),
            daemon=True,
            name="phase-ui-" + identity[:8],
        ).start()

    def _execute(self, identity, callback):
        try:
            self._run(identity, callback)
        except Exception as error:
            from phase_agent.runtime.boundary_diagnostics import record_failure

            record_failure(
                self.directory.parent / "state.json",
                error,
                boundary="control_operation",
                object_id=identity,
            )
        finally:
            _workers.discard((str(self.directory.resolve()), identity))

    def _run(self, identity, callback):
        path = self._path(identity)
        record = read_json(path)
        if not self.handler_lock.acquire(blocking=False):
            write_json(
                path,
                {
                    **record,
                    "status": "blocked",
                    "message": "项目正在处理另一条Studio请求，本次未执行；查看进度后再操作",
                },
            )
            return
        try:
            write_json(path, {**record, "status": "running"})
            result = callback()
            # Control callbacks return bounded presentations, never a whole scientific state.
            if isinstance(result, dict):
                result = {k: v for k, v in result.items() if k not in {"state", "manager", "tasks"}}
            json.dumps(result, ensure_ascii=False, allow_nan=False)
            write_json(
                path,
                {
                    **record,
                    "status": "completed",
                    "result": result,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "message": "交互请求已返回；科研结果以任务与回收记录为准",
                },
            )
        except Exception as error:
            from phase_agent.runtime.boundary_diagnostics import record_failure

            diagnostic = record_failure(
                self.directory.parent / "state.json",
                error,
                boundary="control_operation",
                object_id=identity,
            )
            write_json(
                path,
                {
                    **record,
                    "status": "blocked",
                    "error_type": type(error).__name__,
                    "message": diagnostic["message"],
                    "diagnostic": diagnostic,
                    "next_action": "核对现有收据与结果；不要重复提交计算",
                },
            )
        finally:
            self.handler_lock.release()
