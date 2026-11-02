"""Memory mutations share the scientific lifecycle lock for the entire transaction."""

from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from filelock import FileLock


def memory_transaction(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        path = Path(self.state_path).resolve().parent / "langgraph_lifecycle.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        with getattr(self.chat_handler, "lock", nullcontext()):
            with FileLock(str(path) + ".lock", timeout=10):
                return method(self, *args, **kwargs)

    return locked
