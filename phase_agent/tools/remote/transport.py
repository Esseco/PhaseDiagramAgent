"""Explicit file-transfer boundary; no scheduler or scientific behavior."""

from __future__ import annotations

import shutil
from pathlib import Path


class LocalMirrorTransport:
    """Dry-phase_agent/runtime/mock transport backed by a second local directory."""

    def upload_tree(self, local_path, remote_path):
        source, target = Path(local_path), Path(remote_path)
        if target.exists():
            return {"status": "already_synced", "path": str(target)}
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        return {"status": "synced", "path": str(target)}

    def download_tree(self, remote_path, local_path):
        source, target = Path(remote_path), Path(local_path)
        if not source.exists():
            return {"status": "not_found", "path": str(source)}
        target.mkdir(parents=True, exist_ok=True)
        for path in source.rglob("*"):
            if path.is_file():
                destination = target / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        return {"status": "synced", "path": str(target)}


class CommandTransferAdapter:
    """Adapter for a caller-supplied SSH/SFTP/rsync/scp implementation."""

    def __init__(self, *, upload, download):
        if not callable(upload) or not callable(download):
            raise TypeError("upload and download must be callables")
        self._upload, self._download = upload, download

    def upload_tree(self, local_path, remote_path):
        return self._upload(str(local_path), str(remote_path))

    def download_tree(self, remote_path, local_path):
        return self._download(str(remote_path), str(local_path))
