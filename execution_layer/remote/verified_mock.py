"""Deterministic dry-run adapters that enforce the production file policy."""
import json
import shutil
from pathlib import Path
from execution_layer.remote.scheduler import MockRemoteScheduler

DEFAULT_RESULT_NAMES = {"result.json", "task.finished.json", "CONTCAR", "final.vasp",
                        "checkpoint.json", "checkpoint.json.gz", "log_index.json",
                        "status_summary.json"}


class VerifiedLocalMirrorTransport:
    def __init__(self, *, result_names=None):
        self.result_names = set(result_names or DEFAULT_RESULT_NAMES)

    def upload_tree(self, local_path, remote_path):
        source, target = Path(local_path), Path(remote_path)
        local_snapshot = _snapshot(source)
        if target.exists():
            if _snapshot(target) != local_snapshot: raise ValueError("immutable remote batch checksum mismatch")
            return {"status": "already_synced", "path": str(target)}
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copytree(source, target)
        return {"status": "synced", "path": str(target)}

    def download_tree(self, remote_path, local_path):
        source, target = Path(remote_path), Path(local_path)
        if not source.exists(): return {"status": "not_found", "path": str(source)}
        copied = []
        for path in source.rglob("*"):
            if not path.is_file() or path.name not in self.result_names: continue
            destination = target / path.relative_to(source); destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination); copied.append(str(path.relative_to(source)))
        return {"status": "synced", "path": str(target), "files": copied}


class FileBackedMockRemoteScheduler(MockRemoteScheduler):
    def __init__(self, status_directory):
        super().__init__(); self.status_directory = Path(status_directory); self.status_directory.mkdir(parents=True, exist_ok=True)

    def submit(self, **kwargs):
        result = super().submit(**kwargs); self._save(result); return result

    def query(self, **kwargs):
        result = super().query(**kwargs); self._save(result); return result

    def cancel(self, **kwargs):
        result = super().cancel(**kwargs); self._save(result); return result

    def _save(self, result):
        job_id = result.get("job_id")
        if not job_id: return
        path = self.status_directory / f"{job_id}.json"; temporary = Path(f"{path}.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)


def _snapshot(root):
    path = Path(root) / "batch.snapshot.json"
    if not path.is_file(): raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
