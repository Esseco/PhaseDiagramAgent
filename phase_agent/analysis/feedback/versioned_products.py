"""Immutable complete analysis bundles, with checked per-epoch and project latest copies."""

from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from filelock import FileLock

SCHEMA_VERSION = 1


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def digest(value):
    return sha256(json_bytes(value)).hexdigest()


def write_json(path, value):
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

    _publish_bytes(path, json_bytes(value))


def validate_bundle(directory):
    """Never silently repair or overwrite an immutable archive."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not set(manifest["required_files"]).issubset(manifest["files"]):
        raise ValueError("Incomplete analysis manifest")
    for name, checksum in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"Missing analysis artifact: {name}")
        if sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Analysis checksum mismatch: {name}")
        if path.suffix == ".png":
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
    return manifest


def _replace_latest(source, target):
    """Stage a full copy before swapping; restore the previous copy on failure."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".latest-", dir=target.parent) as temporary:
        work = Path(temporary).resolve()
        # Both moved directories are explicitly contained in this publication root.
        if target.resolve().parent != work.parent or not source.resolve().is_relative_to(
            work.parent.parent
        ):
            raise ValueError("Analysis latest target outside publication root")
        staged, previous = work / "new", work / "previous"
        shutil.copytree(source, staged)
        validate_bundle(staged)
        existed = target.exists()
        if existed:
            target.rename(previous)
        try:
            staged.rename(target)
        except BaseException:
            if existed:
                previous.rename(target)
            raise


def publication_status(root, kind, model, epoch, *, status, reason=None, **details):
    result = {"status": status, "model_version": model, "epoch": epoch, **details}
    if reason:
        result["reason"] = reason
    previous = Path(root) / "latest" / kind / "manifest.json"
    if previous.is_file():
        try:
            saved = json.loads(previous.read_text(encoding="utf-8"))
            key = (
                "quarantined_latest" if "relax_mc_formal" in str(reason or "") else "last_complete"
            )
            result[key] = {
                "version": saved["version"],
                **{k: saved["identity"][k] for k in ("model_version", "epoch")},
            }
        except (OSError, ValueError, KeyError):
            pass
    write_json(Path(root) / "latest" / f"{kind}_status.json", result)
    if epoch:
        write_json(Path(root) / epoch / kind / "latest_status.json", result)
    return result


def publish_bundle(root, *, kind, epoch, model, identity, required, build, current=True):
    try:
        return _publish_bundle(
            root,
            kind=kind,
            epoch=epoch,
            model=model,
            identity=identity,
            required=required,
            build=build,
            current=current,
        )
    except Exception as error:
        if (
            kind in {"phase_diagram", "dft_error"}
            and isinstance(epoch, str)
            and epoch.startswith("epoch")
            and epoch[5:].isdigit()
        ):
            result = {
                "status": "not_ready",
                "model_version": model,
                "epoch": epoch,
                "reason": f"{type(error).__name__}: {error}",
            }
            try:
                if current:
                    publication_status(
                        root, kind, model, epoch, status="not_ready", reason=result["reason"]
                    )
                else:
                    write_json(Path(root) / epoch / kind / "latest_status.json", result)
            except OSError:
                pass  # Caller retains the failure in workflow state if disk publication also fails.
        raise


def _publish_bundle(root, *, kind, epoch, model, identity, required, build, current=True):
    """Only complete bundles advance latest. Caller records failed attempts explicitly."""
    root = Path(root).resolve()
    if (
        kind not in {"phase_diagram", "dft_error"}
        or not epoch
        or not epoch.startswith("epoch")
        or not epoch[5:].isdigit()
    ):
        raise ValueError("Explicit epoch and supported product kind required")
    identity = {"schema": SCHEMA_VERSION, "epoch": epoch, "model_version": model, **identity}
    version = digest(identity)[:24]
    product = root / epoch / kind
    versions = product / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".analysis-publication.lock"), timeout=30):
        if current:
            publication_status(
                root, kind, model, epoch, status="publishing", requested_version=version
            )
        archive = versions / version
        if archive.exists():
            manifest = validate_bundle(archive)
            if manifest["identity"] != identity:
                raise ValueError("Conflicting immutable analysis identity")
        else:
            with TemporaryDirectory(prefix=".build-", dir=versions) as temporary:
                staged = Path(temporary) / "bundle"
                staged.mkdir()
                build(staged)
                missing = [
                    name
                    for name in required
                    if not (staged / name).is_file() or not (staged / name).stat().st_size
                ]
                if missing:
                    raise ValueError(f"Incomplete {kind} bundle: {missing}")
                files = {
                    p.relative_to(staged).as_posix(): sha256(p.read_bytes()).hexdigest()
                    for p in sorted(staged.rglob("*"))
                    if p.is_file()
                }
                previous = product / "latest" / "manifest.json"
                parent = (
                    json.loads(previous.read_text(encoding="utf-8")).get("version")
                    if previous.is_file()
                    else None
                )
                manifest = {
                    "version": version,
                    "identity": identity,
                    "parent_version": parent,
                    "files": files,
                    "required_files": list(required),
                }
                write_json(staged / "manifest.json", manifest)
                validate_bundle(staged)
                staged.rename(archive)
        for target in [product / "latest", *([root / "latest" / kind] if current else [])]:
            if target.is_dir():
                try:
                    if validate_bundle(target) == manifest:
                        continue
                except (OSError, ValueError, KeyError):
                    pass
            _replace_latest(archive, target)
        result = {
            "status": "completed",
            "model_version": model,
            "epoch": epoch,
            "version": version,
            "directory": str(archive),
            "latest_directory": str(product / "latest"),
            "manifest_path": str(archive / "manifest.json"),
        }
        write_json(product / "latest_status.json", result)
        if current:
            write_json(root / "latest" / f"{kind}_status.json", result)
        return result
