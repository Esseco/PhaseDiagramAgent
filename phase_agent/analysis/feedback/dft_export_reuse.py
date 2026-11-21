"""Validate reusable per-round DFT report products against their source evidence."""

import hashlib
import json
from pathlib import Path

from phase_agent.analysis.feedback.dft_parity_plots import PLOT_VERSION


REPORT_VERSION = 1


def report_version(*inputs):
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    for value in (REPORT_VERSION, PLOT_VERSION, *inputs):
        # Bound temporary serialization to one evidence record at a time.
        if isinstance(value, list):
            digest.update(b"[")
            for row in value:
                digest.update(encoder.encode(row).encode("utf-8"))
                digest.update(b",")
            digest.update(b"]")
        else:
            digest.update(encoder.encode(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def product_hashes(root, paths, entry, *, capture=None):
    resolved_root = root.resolve()
    for key, subdir, manifest, empty_reasons in (
        (
            "parity_plots",
            "comparisons/plots",
            "parity_metrics.json",
            {"无完整同帧能量/受力配对，未绘图"},
        ),
        (
            "magnetic_plots",
            "diagnostics/magnetic_plots",
            "magnetic_distribution.json",
            {"no_finite_transition_metal_moments"},
        ),
    ):
        plot = entry[key]
        if plot["status"] != "completed":
            if plot.get("reason") not in empty_reasons:
                return None
            continue
        files = [Path(path).resolve() for path in plot["files"].values()]
        if not files or any(
            not path.is_relative_to(resolved_root) or not path.is_file() for path in files
        ):
            return None
        # The plot publishers validate figure presence using their own manifest.
        paths = [*paths, root / subdir / manifest]
    hashes = {}
    for path in paths:
        relative = path.resolve().relative_to(resolved_root).as_posix()
        if capture is not None and path in capture:
            data = path.read_bytes()
            capture[path] = data
            hashes[relative] = hashlib.sha256(data).hexdigest()
        else:
            with path.open("rb") as stream:
                hashes[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes


def reusable_metrics(root, paths, entry, version, manifest, metrics_path):
    if not entry or version is None:
        return None
    try:
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["version"] == version and saved["entry"] == entry and saved["files"]:
            # Decode the same bytes whose digest matched the saved manifest.
            captured = {metrics_path: None}
            if saved["files"] == product_hashes(root, paths, entry, capture=captured):
                data = captured[metrics_path]
                if data is not None:
                    return json.loads(data)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    return None


def report_manifest(root, paths, entry, version):
    if version is None:
        return None
    try:
        files = product_hashes(root, paths, entry)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    if files:
        return json.dumps(
            {"version": version, "entry": entry, "files": files},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
