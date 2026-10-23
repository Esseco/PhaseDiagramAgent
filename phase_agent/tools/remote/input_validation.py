"""Bounded admission of final upload bytes; no execution, extraction or model calls."""

import argparse
import ast
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import posixpath
import shlex
from threading import RLock
import time
import zipfile

from phase_agent.tools.remote.transfer_bundle import MANIFEST, _safe_name, checksum
from phase_agent.tools.remote.transfer_limits import MAX_MANIFEST_BYTES, validate_capacity

VALIDATOR_VERSION = 1
PACKAGE_CACHE_SIZE = 16
TEXT_CACHE_SIZE = 2048
MAX_TEXT_BYTES = 16 * 1024**2
MAX_CODE_BYTES = 2 * 1024**2
_packages, _texts = OrderedDict(), OrderedDict()
_lock = RLock()


class InputValidationError(ValueError):
    def __init__(self, report):
        self.report = report
        issue = report["issues"][0]
        super().__init__(f"input_admission_failed:{issue['code']}:{issue['file']}")


def clear_validation_cache():
    with _lock:
        _packages.clear()
        _texts.clear()


def validation_cache_info():
    with _lock:
        return {"packages": len(_packages), "texts": len(_texts)}


def verification_summary(report):
    """Keep successful state/model context small; details stay in explicit checks."""
    if report["status"] != "passed":
        return report
    return {key: report[key] for key in ("version", "status", "fingerprint", "remote_environment")}


def _remember(cache, key, value, limit):
    with _lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)


def _cached(cache, key):
    with _lock:
        if key in cache:
            cache.move_to_end(key)
            return deepcopy(cache[key])
    return None


def _structure(text):
    # VASP layout only: physical validity still belongs to scientific adapters.
    lines = [line.strip() for line in text.splitlines()]
    if len(lines) < 8:
        raise ValueError("incomplete POSCAR")
    for line, count in [(lines[1], None), *[(line, 3) for line in lines[2:5]]]:
        values = [float(part) for part in line.split()]
        if not values or (count is not None and len(values) != count):
            raise ValueError("invalid lattice")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite lattice")
    index = 5 if all(word.isdigit() for word in lines[5].split()) else 6
    counts = [int(word) for word in lines[index].split()]
    if not counts or min(counts) < 0 or sum(counts) < 1:
        raise ValueError("invalid atom counts")
    index += 1
    if lines[index].lower().startswith("s"):
        index += 1
    if not lines[index].lower().startswith(("d", "c", "k")):
        raise ValueError("coordinate mode missing")
    coordinates = lines[index + 1 : index + 1 + sum(counts)]
    if len(coordinates) != sum(counts):
        raise ValueError("atom coordinates missing")
    for line in coordinates:
        values = [float(value) for value in line.split()[:3]]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError("invalid coordinates")


def _text_kind(name):
    path = PurePosixPath(name)
    if path.suffix in {".py", ".json", ".sh", ".sbatch", ".vasp", ".poscar"}:
        return path.suffix
    if path.name in {"POSCAR", "CONTCAR"}:
        return ".vasp"
    return None


def _check_text(name, data, digest):
    kind = _text_kind(name)
    key = (VALIDATOR_VERSION, kind, digest)
    cached = _cached(_texts, key)
    if cached is not None:
        return cached, True
    problem = {}
    try:
        text = data.decode("utf-8-sig")
        if kind == ".py":
            ast.parse(text, filename=name, feature_version=(3, 10))
            # compile also detects scope errors that ast.parse alone accepts.
            compile(text, name, "exec", dont_inherit=True)
        elif kind == ".json":
            json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        elif kind in {".sh", ".sbatch"}:
            if not text.startswith("#!") or "\r" in text or "\x00" in text:
                raise ValueError("shell requires shebang and LF; bash -n runs on compute node")
        elif kind in {".vasp", ".poscar"}:
            _structure(text)
    except (ValueError, SyntaxError, IndexError, OverflowError, RecursionError) as error:
        problem = {
            "code": {
                ".py": "python_syntax",
                ".json": "invalid_json",
                ".sh": "shell_format",
                ".sbatch": "shell_format",
            }.get(kind, "structure_format"),
            "line": getattr(error, "lineno", None),
            "column": getattr(error, "offset", None) or getattr(error, "colno", None),
        }
    _remember(_texts, key, deepcopy(problem), TEXT_CACHE_SIZE)
    return problem, False


def _inventory(archive):
    infos = archive.infolist()
    manifest_info = archive.getinfo(MANIFEST)
    validate_capacity(len(infos), manifest_info.file_size, sum(i.file_size for i in infos))
    if manifest_info.file_size > MAX_MANIFEST_BYTES:
        raise ValueError("manifest exceeds limit")
    manifest = json.loads(archive.read(MANIFEST))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != 1
        or manifest.get("kind") != "inputs"
    ):
        raise ValueError("not input manifest")
    rows = manifest["files"]
    if not isinstance(rows, list) or any(
        not isinstance(row, dict)
        or not isinstance(row.get("path"), str)
        or type(row.get("size")) is not int
        or row["size"] < 0
        or not isinstance(row.get("sha256"), str)
        or len(row["sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in row["sha256"])
        for row in rows
    ):
        raise ValueError("invalid inventory fields")
    names = [row["path"] for row in rows]
    if (
        len({name.casefold() for name in names}) != len(names)
        or len(infos) != len(rows) + 1
        or set(names) | {MANIFEST} != {i.filename for i in infos}
    ):
        raise ValueError("inventory mismatch")
    for info in infos:
        _safe_name(info.filename)
        if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("archive directories/symlinks unsupported")
    return rows


def _missing_script_reference(name, data, names):
    """Resolve literal Python entry files used by fixed templates, without running Bash."""
    for line in data.decode("utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped.startswith(("python ", "python3 ")):
            continue
        try:
            words = shlex.split(stripped, comments=True)
        except ValueError:
            continue  # Full Bash syntax is checked by bash -n on the target host.
        if len(words) < 2 or words[1] in {"-m", "-c", "-"}:
            continue
        target = next((word for word in words[1:] if not word.startswith("-")), "")
        if not target.endswith(".py") or "$" in target or target.startswith("/"):
            continue
        relative = posixpath.normpath(posixpath.join(str(PurePosixPath(name).parent), target))
        if relative not in names:
            return relative
    return None


def _validate_handoff(folder, *, kind="dispatch"):
    """Validate immutable package + final entry files. Cache facts, never authority."""
    started = time.perf_counter()
    folder = Path(folder)
    entry = "dispatch_tasks.py" if kind == "dispatch" else "run_upload.py"
    if kind not in {"dispatch", "relax_mc"}:
        raise ValueError("unknown input package kind")
    report = {
        "version": VALIDATOR_VERSION,
        "status": "failed",
        "model_calls": 0,
        "submitted": False,
        "remote_environment": "not_checked",
        "cache_hit": False,
        "files_checked": 0,
        "text_cache_hits": 0,
        "max_text_bytes": MAX_TEXT_BYTES,
        "issues": [],
        "next_step": "修复所列文件或依赖后重新检查原交付目录；不要重投已登记的作业。",
    }
    current = "inputs.zip"

    def fail(code, name, **detail):
        report["issues"] = [{"code": code, "file": name, **detail}]
        report["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        raise InputValidationError(report)

    try:
        archive_path = folder / current
        signature = [VALIDATOR_VERSION, kind, checksum(archive_path)]
        for name in ("submit_all.sh", entry):
            current = name
            if (folder / name).stat().st_size > MAX_CODE_BYTES:
                fail("text_size_limit", name)
            signature.append(checksum(folder / name))
        fingerprint = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
        cached = _cached(_packages, fingerprint)
        if cached is not None:
            return {
                **cached,
                "cache_hit": True,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        report["fingerprint"] = fingerprint
        for name in ("submit_all.sh", entry):
            current = name
            issue, hit = _check_text(name, (folder / name).read_bytes(), checksum(folder / name))
            report["text_cache_hits"] += hit
            if issue:
                fail(issue.pop("code"), name, **issue)
        current = "inputs.zip"
        with zipfile.ZipFile(archive_path) as archive:
            rows = _inventory(archive)
            names = {row["path"] for row in rows}
            required = (
                {"submission_plan.json"}
                if kind == "dispatch"
                else {"batch.json", "run_batch.py", "submit_all.sh"}
            )
            for name in sorted(required - names):
                fail("missing_entry", name)
            active_scripts = {"submit_all.sh"}
            if kind == "dispatch":
                current = "submission_plan.json"
                if archive.getinfo(current).file_size > MAX_TEXT_BYTES:
                    fail("text_size_limit", current)
                plan = json.loads(archive.read(current))
                scripts = plan.get("scripts") if isinstance(plan, dict) else None
                if (
                    not isinstance(scripts, list)
                    or not scripts
                    or any(
                        not isinstance(name, str)
                        or name not in names
                        or PurePosixPath(name).suffix not in {".sh", ".sbatch"}
                        for name in scripts
                    )
                    or len(set(scripts)) != len(scripts)
                ):
                    fail("invalid_job_references", current)
                active_scripts = set(scripts)
            for row in rows:
                current = row["path"]
                text_kind = _text_kind(current)
                text_limit = (
                    MAX_CODE_BYTES if text_kind in {".py", ".sh", ".sbatch"} else MAX_TEXT_BYTES
                )
                if text_kind and row["size"] > text_limit:
                    fail("text_size_limit", current)
                digest, size, chunks = hashlib.sha256(), 0, []
                with archive.open(current) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        size += len(chunk)
                        digest.update(chunk)
                        if text_kind:
                            if size > text_limit:
                                fail("text_size_limit", current)
                            chunks.append(chunk)
                if size != row["size"] or digest.hexdigest() != row["sha256"]:
                    fail("checksum_mismatch", current)
                if text_kind:
                    data = b"".join(chunks)
                    issue, hit = _check_text(current, data, digest.hexdigest())
                    report["text_cache_hits"] += hit
                    if issue:
                        fail(issue.pop("code"), current, **issue)
                    if text_kind in {".sh", ".sbatch"} and current in active_scripts:
                        missing = _missing_script_reference(current, data, names)
                        if missing:
                            fail("missing_script_reference", current, reference=missing)
                report["files_checked"] += 1
        # Refuse a result assembled while any of the final input bytes changed.
        if signature[2:] != [
            checksum(folder / name) for name in ("inputs.zip", "submit_all.sh", entry)
        ]:
            fail("input_changed_during_check", current)
    except InputValidationError:
        raise
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError) as error:
        fail(
            "input_read_error" if isinstance(error, OSError) else "invalid_input_inventory",
            current,
            retryable_read=getattr(error, "winerror", None) in {32, 33},
        )
    report.update(status="passed", elapsed_ms=round((time.perf_counter() - started) * 1000, 2))
    _remember(_packages, fingerprint, deepcopy(report), PACKAGE_CACHE_SIZE)
    return report


def validate_handoff(folder, *, kind="dispatch"):
    """Retry one transient Windows sharing violation; never rerun preparation."""
    for attempt in range(2):
        try:
            return _validate_handoff(folder, kind=kind)
        except InputValidationError as error:
            if attempt or not error.report["issues"][0].get("retryable_read"):
                raise
            time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--kind", choices=("dispatch", "relax_mc"), default="dispatch")
    args = parser.parse_args()
    try:
        report = validate_handoff(args.folder, kind=args.kind)
    except InputValidationError as error:
        report = error.report
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
