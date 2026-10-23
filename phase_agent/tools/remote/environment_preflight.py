"""Portable stdlib-only compute-entry checks; never import scientific libraries."""

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import os
import hashlib


class EnvironmentPreflightError(RuntimeError):
    def __init__(self, code, *, missing=(), file=None):
        self.report = {
            "status": "failed",
            "code": code,
            "missing": list(missing),
            "file": file,
            "submitted": False,
            "model_calls": 0,
            "next_step": "修复当前计算环境或所列脚本后，在原目录运行同一入口；保留回执，已提交作业不会重新提交。",
        }
        super().__init__(
            "environment_preflight_failed:"
            + code
            + (":" + ",".join(missing) if missing else "")
            + (":" + str(file) if file else "")
        )


def preflight_environment(root, scripts, *, required_modules=()):
    """Module discovery + bash syntax, per invocation; no stale environment cache."""
    if sys.version_info < (3, 10):
        raise EnvironmentPreflightError("python_version", missing=["Python>=3.10"])
    missing = []
    for name in sorted(set(required_modules)):
        try:
            found = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            found = None
        if found is None:
            missing.append(name)
    if missing:
        raise EnvironmentPreflightError("missing_modules", missing=missing)
    bash = shutil.which("bash")
    if not bash:
        raise EnvironmentPreflightError("missing_command", missing=["bash"])
    root = Path(root).resolve()
    checked, syntax_checked = set(), set()
    environment = {k: v for k, v in os.environ.items() if k not in {"BASH_ENV", "ENV"}}
    for script in scripts:
        target = (root / script).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise EnvironmentPreflightError("missing_script", file=str(script))
        if target in checked:
            continue
        if target.stat().st_size > 2 * 1024**2:
            raise EnvironmentPreflightError("shell_size_limit", file=str(script))
        # Slurm header labels differ per job but do not affect Bash syntax.
        lines = target.read_bytes().splitlines(keepends=True)
        start = next(
            (
                i
                for i, line in enumerate(lines)
                if line.strip() and not line.lstrip().startswith(b"#")
            ),
            len(lines),
        )
        data = b"".join(lines[start:])
        signature = hashlib.sha256(data).digest()
        if signature in syntax_checked:
            checked.add(target)
            continue
        try:
            result = subprocess.run(
                [bash, "-n", str(target)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=environment,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise EnvironmentPreflightError("shell_check_unavailable", file=str(script)) from None
        if result.returncode:
            raise EnvironmentPreflightError("shell_syntax", file=str(script))
        checked.add(target)
        if len(syntax_checked) < 2048:
            syntax_checked.add(signature)
    return {
        "status": "passed",
        "modules_checked": len(set(required_modules)),
        "scripts_checked": len(checked),
        "submitted": False,
        "model_calls": 0,
        "scope": "module_presence_and_shell_syntax",
        "gpu_runtime": "not_checked",
    }
