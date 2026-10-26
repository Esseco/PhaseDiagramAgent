"""Explicit file-only maintenance; move data directly without filesystem aliases."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_files(root):
    for directory, children, files in os.walk(root, followlinks=False):
        base = Path(directory)
        children[:] = [
            name
            for name in children
            if not (base / name).is_symlink() and not (base / name).is_junction()
        ]
        for name in files:
            path = base / name
            if path.is_symlink():
                raise ValueError(f"Unexpected file link: {path}")
            yield path


def _epoch_from_results(directory, rounds):
    versions = set()
    for path in directory.rglob("result.json"):
        version = json.loads(path.read_text(encoding="utf-8")).get("model_version")
        if version:
            versions.add(version)
    if len(versions) == 1 and next(iter(versions)) in rounds:
        return f"epoch{int(rounds[next(iter(versions))]) - 1}"
    return "unassigned"


def _english_interaction_stem(stem):
    mapping = (
        ("交互异常", "interaction_error"),
        ("已回收285个刷新结果", "model_refresh_results_collected"),
        ("无运行中任务", "workflow_check"),
        ("本轮新branch已登记", "branch_registered"),
        ("输入准备方案待确认", "input_preparation_review"),
        ("DFT恢复", "dft_recovery"),
        ("DFT回收", "dft_results_collected"),
        ("DFT选点", "dft_selection"),
        ("结构生成", "structure_generation"),
        ("MC分配", "mc_allocation"),
        ("MC取消", "mc_cancelled"),
        ("模型刷新", "model_refresh"),
        ("模型训练", "model_training"),
        ("MC回收", "mc_recovery"),
        ("Relax回收", "relax_recovery"),
        ("流程检查", "workflow_check"),
    )
    for old, new in mapping:
        if stem == old or stem.startswith(old + "_"):
            return new + stem[len(old) :]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_").lower() or "interaction"


def folder_plan(root, state):
    root = Path(root).resolve()
    rounds = (state.get("upload_layout") or {}).get("model_rounds") or {}
    moves = []

    def add(old, new):
        old, new = root / old, root / new
        if old.exists() and new.exists() and old.samefile(new):
            return
        if old.exists() and not old.is_junction() and old != new:
            moves.append((old, new))

    for version, number in rounds.items():
        epoch = f"epoch{int(number) - 1}"
        for category in ("analysis_outputs", "submissions"):
            add(f"{category}/{epoch}_{version}", f"{category}/{epoch}")
        for recovery in sorted((root / "analysis_outputs/recovery").glob("*")):
            add(
                recovery / f"{epoch}_{version}",
                root / "analysis_outputs" / epoch / "recovery" / recovery.name,
            )
    submission = root / "submissions"
    for category, target in (("relax_mc_batches", "relax_mc"), ("recoveries", "recovery")):
        for path in sorted((submission / category).glob("*")):
            match = re.match(r"(epoch\d+)_(.+)", path.name)
            if match:
                lane = "relax_mc" if target == "relax_mc" else "recovery"
                bucket = "delivery" if lane == "relax_mc" else "evidence"
                add(path, submission / match[1] / bucket / lane / match[2])
    for path in sorted((submission / "verified").glob("*")):
        summary = json.loads((path / "result_summary.json").read_text(encoding="utf-8"))
        number = rounds.get(summary.get("model_version"))
        epoch = f"epoch{int(number) - 1}" if number else "unassigned"
        add(path, submission / epoch / "evidence" / "verified_results" / path.name)
    for path in sorted((submission / "_recovery").glob("*")):
        add(
            path,
            submission / _epoch_from_results(path, rounds) / "evidence" / "recovery" / path.name,
        )
    for kind in ("01-upload", "02-results"):
        for path in sorted((submission / "archives" / kind).glob("*")):
            match = re.match(r"(epoch\d+)_(.+)", path.name)
            epoch, name = (match[1], match[2].lstrip("_")) if match else ("unassigned", path.name)
            if not match and path.is_dir() and (path / "inputs.zip").is_file():
                with zipfile.ZipFile(path / "inputs.zip") as zipped:
                    versions = {
                        json.loads(zipped.read(member)).get("model_version")
                        for member in zipped.namelist()
                        if member.endswith("/task.json")
                    }
                if len(versions) == 1 and next(iter(versions)) in rounds:
                    epoch = f"epoch{int(rounds[next(iter(versions))]) - 1}"
                    name = path.name.split("__", 1)[0] + "_legacy_package"
            if path.is_file():
                target = submission / epoch / "evidence" / "storage_archives" / name
            else:
                target = submission / epoch / "delivery" / "transfers" / name
            add(path, target)
    legacy = root / "交互记录"
    sequences = {}
    for path in sorted(legacy.glob("*.md")):
        if path.name == "README.md":
            continue
        match = re.match(r"(epoch\d+)_(.+)", path.stem)
        if match:
            epoch, name = match[1], match[2]
        elif re.match(r"\d{8}T", path.name):
            epochs = set(re.findall(r"\bepoch\d+\b", path.read_text(encoding="utf-8")[:1600]))
            epoch = next(iter(epochs)) if len(epochs) == 1 else "unassigned"
            date = path.stem[:8]
            key = (epoch, date)
            sequences[key] = sequences.get(key, 0) + 1
            name = f"conversation_{date}_{sequences[key]:02d}"
        else:
            epoch, name = "project", path.stem
        add(path, root / "interaction_records" / epoch / (name + ".md"))
    targets = [b for _, b in moves]
    if len(targets) != len(set(targets)):
        raise ValueError("Duplicate migration destination")
    for old, new in moves:
        if not old.resolve().is_relative_to(root) or not new.resolve().is_relative_to(root):
            raise ValueError(f"Migration escapes workspace: {old}")
        if new.exists():
            raise FileExistsError(new)
        if old.is_dir():
            for directory, children, files in os.walk(old):
                if any(
                    (Path(directory) / n).is_junction() or (Path(directory) / n).is_symlink()
                    for n in children + files
                ):
                    raise ValueError(f"Nested link requires explicit handling: {old}")
    return moves


def _attributes(path, hidden):
    if os.name == "nt":
        value = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if value == 0xFFFFFFFF:
            raise ctypes.WinError()
        value = value | 2 if hidden else value & ~2
        if not ctypes.windll.kernel32.SetFileAttributesW(str(path), value):
            raise ctypes.WinError()


def _write_index_bytes(index, data):
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

    hidden = os.name == "nt" and bool(index.stat().st_file_attributes & 2)
    _publish_bytes(index, data)
    _attributes(index, hidden)


def _submission_root(root):
    root = Path(root).resolve()
    submission_root = root / "submissions"
    if not submission_root.is_dir() and root.name == "submissions":
        submission_root = root
    return submission_root


def _archive_move_pairs(moves, submission_root):
    """Normalize tuple and dict move records to submission-relative pairs."""
    pairs = []
    for move in moves or ():
        if isinstance(move, dict):
            old, new = move.get("from"), move.get("to")
        else:
            try:
                old, new = move
            except (TypeError, ValueError):
                continue
        if not old or not new:
            continue
        old, new = Path(old).resolve(), Path(new).resolve()
        if old.is_relative_to(submission_root) and new.is_relative_to(submission_root):
            pairs.append(
                (
                    old.relative_to(submission_root).as_posix(),
                    new.relative_to(submission_root).as_posix(),
                )
            )
    return sorted(pairs, key=lambda pair: -len(pair[0]))


def _normalize_archive_mapping(submission_root, mappings, moves=()):
    """Return safe canonical archive keys and targets without guessing payloads."""
    pairs = _archive_move_pairs(moves, submission_root)

    def relocated(value):
        for old, new in pairs:
            if value == old or value.startswith(old + "/"):
                return new + value[len(old) :]
        return value

    def canonical_key(value):
        path = Path(value)
        if len(path.parts) >= 3 and re.fullmatch(r"epoch\d+", path.parts[0]):
            if path.parts[1] not in {"calculations", "delivery", "evidence", "previously_used"}:
                candidate = "/".join((path.parts[0], "calculations", *path.parts[1:]))
                if (submission_root / Path(candidate)).exists():
                    return candidate
        return value

    def canonical_target(value):
        target = submission_root / Path(value)
        if target.is_file():
            return value
        path = Path(value)
        if (
            len(path.parts) >= 3
            and re.fullmatch(r"epoch\d+", path.parts[0])
            and path.parts[1] == "storage_archives"
        ):
            candidate = "/".join((path.parts[0], "evidence", "storage_archives", *path.parts[2:]))
            if (submission_root / Path(candidate)).is_file():
                return candidate
        return value

    normalized = {}
    for raw_key, raw_value in mappings.items():
        key = canonical_key(relocated(str(raw_key)))
        value = canonical_target(relocated(str(raw_value)))
        prior = normalized.get(key)
        if prior is not None and prior != value:
            raise ValueError(f"Archive index collision after migration: {key}")
        normalized[key] = value
    return normalized


def relocate_archive_index(root, moves):
    submission_root = _submission_root(root)
    index = submission_root / ".archive_index.json"
    if not index.is_file():
        return
    original = json.loads(index.read_bytes())
    mappings = _normalize_archive_mapping(submission_root, original, moves)
    _write_index_bytes(index, (json.dumps(mappings, indent=2) + "\n").encode("utf-8"))


def repair_archive_index(root):
    """Repair stale pre-bucket archive keys/targets when their files still exist."""
    relocate_archive_index(root, ())


def migrate_folders(root, state, *, apply=False):
    root = Path(root).resolve()
    moves = folder_plan(root, state)
    report = {"moves": [{"from": str(a), "to": str(b)} for a, b in moves]}
    if not apply:
        return report
    completed = []
    index = root / "submissions/.archive_index.json"
    original_index = index.read_bytes() if index.is_file() else None
    try:
        for old, new in moves:
            before = (
                {str(f.relative_to(old)): file_hash(f) for f in regular_files(old)}
                if old.is_dir()
                else {"": file_hash(old)}
            )
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)  # An occupied directory fails visibly; never create a link.
            completed.append((old, new))
            _attributes(new, False)
            after = (
                {str(f.relative_to(new)): file_hash(f) for f in regular_files(new)}
                if new.is_dir()
                else {"": file_hash(new)}
            )
            if before != after:
                raise ValueError(f"Migration changed file bytes: {new}")
        relocate_archive_index(root, moves)
    except BaseException:
        try:
            if original_index is not None and index.read_bytes() != original_index:
                _write_index_bytes(index, original_index)
        finally:
            for old, new in reversed(completed):
                new.rename(old)
                _attributes(old, False)
        raise
    report["verified_files"] = len(
        {
            path.absolute()
            for _, new in moves
            for path in (regular_files(new) if new.is_dir() else [new])
        }
    )
    return report


def prune_archived_runtime(root, *, apply=False):
    """Remove only runtime source copies proven identical to their retained input ZIP."""
    root = Path(root).resolve()
    report = []
    candidates = {
        *root.glob("submissions/epoch*/delivery/relax_mc/*/inputs.zip"),
        *root.glob("submissions/epoch*/relax_mc/*/inputs.zip"),
    }
    for archive in sorted(candidates):
        if not any(re.fullmatch(r"epoch\d+", parent.name) for parent in archive.parents):
            continue
        directory = archive.parent
        if directory.name.endswith(("_input", "_output")):
            continue
        rows = []
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                relative = Path(member.filename)
                if member.is_dir() or not relative.parts or relative.parts[0] != "runtime":
                    continue
                path = directory / relative
                if not path.resolve().is_relative_to(directory.resolve()):
                    raise ValueError("Archive member escapes batch directory")
                if path.is_file():
                    expected = hashlib.sha256(zipped.read(member)).hexdigest()
                    if file_hash(path) == expected:
                        rows.append(
                            {
                                "member": member.filename,
                                "sha256": expected,
                                "bytes": path.stat().st_size,
                            }
                        )
        if not rows:
            continue
        entry = {"archive": str(archive), "sha256": file_hash(archive), "files": rows}
        report.append(entry)
        if apply:
            for row in rows:
                path = directory / row["member"]
                if file_hash(path) != row["sha256"]:
                    raise ValueError(f"Runtime source changed during cleanup: {path}")
            if file_hash(archive) != entry["sha256"]:
                raise ValueError(f"Input ZIP changed during cleanup: {archive}")
            for row in rows:
                (directory / row["member"]).unlink()
            for path in sorted((directory / "runtime").rglob("*"), key=lambda p: -len(p.parts)):
                if path.is_dir() and not path.is_junction() and not any(path.iterdir()):
                    path.rmdir()
            if not any((directory / "runtime").iterdir()):
                (directory / "runtime").rmdir()
    return report
