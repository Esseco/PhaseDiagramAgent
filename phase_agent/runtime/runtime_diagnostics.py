"""Observational diagnostics only: no workflow calls, restarts or file removal."""

from datetime import datetime, timezone
from html import escape
from itertools import islice
import json
import logging
from pathlib import Path
import shutil
import sys
import threading
from time import monotonic, time

MIB = 1024**2
GIB = 1024**3
_monitors = {}
_lock = threading.RLock()
logger = logging.getLogger(__name__)


def snapshot_path(config):
    root = Path(config).resolve().parent
    path = root / "workflow_state/diagnostics/runtime_diagnostics.json"
    if not path.resolve().is_relative_to(root):
        raise ValueError("Diagnostics path is outside project")
    return path


def capacity_snapshot(root, previous=None, *, now=None):
    """Stat known runtime files only; never open databases or scan structures/backups."""
    root = Path(root).resolve()
    now = time() if now is None else now
    workflow = root / "workflow_state"
    paths = [workflow / "state.json"]
    errors, warnings, files = [], [], []
    directories = (workflow, workflow / "studio/.langgraph_api")
    for directory in directories:
        try:
            if not directory.resolve().is_relative_to(root):
                errors.append("诊断目录超出项目范围，未扫描")
                continue
            entries = list(islice(directory.iterdir(), 257))
            if len(entries) > 256:
                errors.append("目录条目过多，容量统计可能不完整")
            paths.extend(
                p
                for p in entries[:256]
                if p.name.endswith(
                    (
                        ".pckl",
                        ".pckl.tmp",
                        ".sqlite",
                        ".sqlite-wal",
                        ".sqlite-shm",
                        ".sqlite3",
                        ".db",
                        ".db-wal",
                        ".db-shm",
                    )
                )
            )
        except FileNotFoundError:
            continue
        except OSError:
            errors.append("部分运行文件目录无法读取")
    if len(paths) > 256:
        errors.append("运行文件超过 256 个，本次仅统计前 256 个，容量不是完整总量")
    for path in paths[:256]:
        try:
            if not path.resolve().is_relative_to(root):
                errors.append("跳过项目外的文件链接")
                continue
            info = path.stat()
            if not path.is_file():
                continue
            files.append({"file": str(path.relative_to(root)), "bytes": info.st_size})
            threshold = 100 * MIB if path.name == "state.json" else GIB
            if info.st_size >= threshold:
                warnings.append(f"{path.name} 为 {info.st_size / MIB:.0f} MiB，达到容量提醒阈值")
        except FileNotFoundError:
            continue  # Atomic replacement may briefly change a directory entry.
        except OSError:
            errors.append("部分运行文件大小无法读取")
    total = sum(row["bytes"] for row in files)
    pickle_bytes = sum(row["bytes"] for row in files if row["file"].endswith(".pckl"))
    temporary_bytes = sum(row["bytes"] for row in files if row["file"].endswith(".tmp"))
    if total >= 2 * GIB:
        warnings.append(
            f"运行文件磁盘占用 {total / GIB:.2f} GiB（含保留历史及临时文件）；"
            "不代表内存占用或每轮读写量，分类明细见过程页"
        )
    free = None
    try:
        free = shutil.disk_usage(root).free
        if free < 10 * GIB:
            warnings.append(f"项目磁盘剩余 {free / GIB:.1f} GiB，低于 10 GiB")
    except (OSError, ValueError, SystemError):
        errors.append("磁盘剩余空间无法获取")
    delta = None
    if previous and previous.get("complete") and not errors:
        elapsed = now - previous.get("sampled_at_unix", now)
        if 10 <= elapsed <= 900:
            delta = total - previous.get("total_bytes", total)
            if delta >= 256 * MIB:
                warnings.append(
                    f"最近 {elapsed:.0f} 秒运行文件增加 {delta / MIB:.0f} MiB，增长偏快；尚不能据此断定异常"
                )
    return {
        "sampled_at_unix": now,
        "total_bytes": total,
        "pickle_history_bytes": pickle_bytes,
        "temporary_bytes": temporary_bytes,
        "free_bytes": free,
        "growth_bytes": delta,
        "complete": not errors,
        "files": files,
        "warnings": warnings,
        "unavailable": sorted(set(errors)),
    }


def sampled_location(thread_id):
    """Expose function names only, never source text, locals, arguments or secrets."""
    frame = sys._current_frames().get(thread_id)
    locations = []
    while frame is not None and len(locations) < 8:
        locations.append(
            {
                "file": Path(frame.f_code.co_filename).name,
                "function": frame.f_code.co_name,
                "line": frame.f_lineno,
            }
        )
        frame = frame.f_back
    return locations


def describe_wait(item):
    category = item.get("category", "node")
    seconds = item.get("step_seconds", 0)
    threshold = {"model": 30, "io": 10, "checkpoint": 15, "copy": 10}.get(category, 30)
    cause = {
        "model": "该模型请求尚未返回；服务响应慢、网络等待或内部重试尚不能区分",
        "io": "项目文件读取或解析步骤尚未完成；是否受文件大小或磁盘影响仍需核对",
        "checkpoint": "检查点读写步骤尚未完成；是否为压缩、磁盘或锁等待仍需核对",
        "copy": "请求数据复制或序列化步骤尚未完成",
    }.get(category, "该业务步骤尚未返回；仅凭耗时不能确认卡死或具体原因")
    locations = sampled_location(item.get("thread_id")) if seconds >= threshold else []
    name = item.get("name", "")
    for prefix, explanation in (
        ("json_encode:", "JSON 序列化步骤尚未完成"),
        ("json_decode:", "JSON 解析步骤尚未完成"),
        ("json_write:", "文件保存步骤尚未完成"),
    ):
        if name.startswith(prefix):
            cause = explanation + "；是否受数据量或磁盘影响仍需核对"
    if locations:
        top = locations[0]
        if top["file"] == "threading.py" and top["function"] in {"wait", "join"}:
            cause += "；采样时线程正在等待同步信号"
        elif any(row["function"] in {"acquire", "_acquire"} for row in locations):
            cause += "；采样位置包含锁获取调用"
        elif any(
            row["file"] in {"compact_checkpoints.py", "checkpoint_blobs.py", "studio_storage.py"}
            for row in locations
        ):
            cause += "；采样位置位于检查点存储代码"
    return {
        **{k: v for k, v in item.items() if k != "thread_id"},
        "level": "warning" if seconds >= threshold else "normal",
        "cause": cause,
        "locations": locations,
        "warning_after_seconds": threshold,
    }


class RuntimeMonitor:
    def __init__(self, config):
        self.config = str(config)
        self.root = Path(config).resolve().parent
        self.started = monotonic()
        self.stage_started = self.started
        self.stage_name = "初始化服务"
        self.stage_thread = None
        self.ready = False
        self.status = "starting"
        self.last_turn = None
        self.failure = None
        self.stop = threading.Event()
        self.data_lock = threading.RLock()
        self.latest = {}
        self.capacity = None
        self.capacity_time = -60.0
        self.thread = threading.Thread(
            target=self.run, name="phase-runtime-diagnostics", daemon=True
        )

    def stage(self, label, *, ready=False):
        with self.data_lock:
            self.stage_name, self.stage_started = label, monotonic()
            self.stage_thread = threading.get_ident()
            self.ready = ready
            if ready:
                self.status = "ready"

    def tick(self):
        from phase_agent.runtime.turn_progress import active_turns

        now = monotonic()
        if self.capacity is None or now - self.capacity_time >= 60:
            self.capacity = capacity_snapshot(self.root, self.capacity)
            self.capacity_time = now
        with self.data_lock:
            startup = (
                None
                if self.ready or self.status in {"stopped", "failed"}
                else describe_wait(
                    {
                        "label": self.stage_name,
                        "category": "node",
                        "thread_id": self.stage_thread,
                        "step_seconds": round(now - self.stage_started, 1),
                        "total_seconds": round(now - self.started, 1),
                    }
                )
            )
        turns = [describe_wait(row) for row in active_turns(self.config)[:10]]
        data = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "service_status": self.status,
            "service_failure": self.failure,
            "last_turn": self.last_turn,
            "startup": startup,
            "turns": turns,
            "capacity": self.capacity,
            "details_file": str(snapshot_path(self.config)),
            "automatic_actions": "仅观察；不重启、不删除文件、不执行计算",
        }
        with self.data_lock:
            self.latest = data
        path = snapshot_path(self.config)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        except OSError as error:
            with self.data_lock:
                self.latest = {
                    **data,
                    "diagnostic_error": "诊断快照保存失败：" + type(error).__name__,
                }

    def run(self):
        while not self.stop.is_set():
            try:
                self.tick()
            except Exception as error:
                # Optional observation must never fail scientific execution.
                with self.data_lock:
                    self.latest = {"diagnostic_error": "诊断采样失败：" + type(error).__name__}
                logger.warning("Runtime diagnostics unavailable: %s", type(error).__name__)
            self.stop.wait(15)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        if not self.thread.is_alive():
            if self.status != "failed":
                self.status = "stopped"
            try:
                self.tick()
            except Exception:
                pass


def start_monitor(config):
    monitor = RuntimeMonitor(config)
    key = str(Path(config).resolve())
    with _lock:
        if key in _monitors:
            raise RuntimeError("Runtime diagnostics already owns this project")
        _monitors[key] = monitor
    try:
        monitor.thread.start()
    except Exception:
        with _lock:
            _monitors.pop(key, None)
        raise
    return monitor


def stop_monitor(config, monitor):
    monitor.close()
    with _lock:
        if _monitors.get(str(Path(config).resolve())) is monitor:
            _monitors.pop(str(Path(config).resolve()), None)


def service_stage(config, label, *, ready=False):
    with _lock:
        monitor = _monitors.get(str(Path(config).resolve()))
    if monitor:
        monitor.stage(label, ready=ready)


def observe_completed(config, record):
    with _lock:
        monitor = _monitors.get(str(Path(config).resolve()))
    if monitor is None:
        return
    timings = record.get("timings") or []
    slowest = max(timings, key=lambda row: row.get("seconds", 0), default={})
    with monitor.data_lock:
        monitor.last_turn = {
            "time": record.get("time"),
            "status": record.get("status"),
            "duration_seconds": record.get("duration_seconds"),
            "slowest_step": {k: slowest.get(k) for k in ("name", "category", "seconds")},
            "failure": record.get("failure_summary"),
        }


def get_snapshot(config):
    with _lock:
        monitor = _monitors.get(str(Path(config).resolve()))
    if monitor:
        with monitor.data_lock:
            data = dict(monitor.latest)
        try:
            age = max(0, time() - datetime.fromisoformat(data["updated_at"]).timestamp())
            data.update(snapshot_age_seconds=round(age, 1), stale=age > 45)
        except (KeyError, ValueError, TypeError):
            pass
        return data
    try:
        path = snapshot_path(config)
        if path.stat().st_size <= 256 * 1024:
            return {**json.loads(path.read_text(encoding="utf-8")), "saved_snapshot": True}
    except (OSError, ValueError, TypeError):
        pass
    return {}


def monitoring_active(config):
    with _lock:
        monitor = _monitors.get(str(Path(config).resolve()))
    return monitor is not None and monitor.thread.is_alive()


def diagnostic_notice(config):
    data = get_snapshot(config)
    warnings = (data.get("capacity") or {}).get("warnings") or []
    return "\n运行提醒：" + warnings[0] + "。详情见本轮过程页 /phase/process。" if warnings else ""


def render_diagnostics(config):
    data = get_snapshot(config)
    if not data:
        return (
            "<section><h2>运行诊断</h2><p>尚无诊断快照；下次手动启动服务后自动记录。</p></section>"
        )
    lines = ["<section><h2>运行诊断与容量提醒</h2>"]
    if data.get("saved_snapshot"):
        lines.append("<p>以下是已保存快照，不能据此判断当前服务仍在运行。</p>")
    if data.get("stale"):
        lines.append(
            "<p>诊断已 "
            + str(data["snapshot_age_seconds"])
            + " 秒未更新，无法确认当前状态；采样线程或进程是否仍响应尚需核对，不会自动重启。</p>"
        )
    lines.append("<p>记录时间：" + escape(str(data.get("updated_at", "未知"))) + "</p>")
    for item in ([data["startup"]] if data.get("startup") else []) + data.get("turns", []):
        lines.append(
            "<p>当前阶段："
            + escape(str(item["label"]))
            + "；已等待 "
            + str(item["step_seconds"])
            + " 秒。"
            + ("耗时偏长。" if item["level"] == "warning" else "")
            + escape(item["cause"])
            + "。</p>"
        )
        if item.get("locations"):
            top = item["locations"][0]
            lines.append(
                "<p>采样位置："
                + escape(f"{top['file']} · {top['function']} · 行 {top['line']}")
                + "</p>"
            )
    capacity = data.get("capacity") or {}
    if capacity:
        lines.append(
            f"<p>磁盘文件：pickle 历史源文件 {capacity.get('pickle_history_bytes', 0) / MIB:.0f} MiB；"
            f"临时文件 {capacity.get('temporary_bytes', 0) / MIB:.0f} MiB；"
            "以上按文件统计，不等于正在加载或写入的大小。</p>"
        )
    for warning in capacity.get("warnings", []) + capacity.get("unavailable", []):
        lines.append("<p>" + escape(warning) + "</p>")
    last = data.get("last_turn") or {}
    failure = last.get("failure") or {}
    if failure:
        reason = {
            "TimeoutError": "已记录请求超时",
            "PermissionError": "已记录文件权限错误",
            "FileNotFoundError": "已记录缺失文件",
            "ConnectionError": "已记录连接错误",
        }.get(failure.get("type"), "已记录异常，具体原因需核对本轮过程记录")
        if failure.get("http_status") == 503:
            reason = "已记录服务返回 HTTP 503（暂时不可用）"
        lines.append(
            "<p>最近异常阶段："
            + escape(str(failure.get("stage", "未知")))
            + "；"
            + escape(reason)
            + "；异常类型："
            + escape(str(failure.get("type", "未知")))
            + "。</p>"
        )
    if capacity and not capacity.get("warnings") and capacity.get("complete"):
        lines.append(
            f"<p>运行文件 {capacity['total_bytes'] / MIB:.0f} MiB，未达到容量提醒阈值。</p>"
        )
    if data.get("diagnostic_error"):
        lines.append("<p>" + escape(data["diagnostic_error"]) + "</p>")
    if data.get("service_failure"):
        failure = data["service_failure"]
        lines.append(
            "<p>服务异常阶段："
            + escape(failure["stage"])
            + "；已记录 "
            + escape(failure["type"])
            + "。具体异常请核对项目 logs 文件夹内本次启动日志。</p>"
        )
    lines.append(
        "<p>诊断文件："
        + escape(str(data.get("details_file", snapshot_path(config))))
        + "</p><p>仅观察；不会自动重启、删除文件或执行计算。</p></section>"
    )
    return "".join(lines)
