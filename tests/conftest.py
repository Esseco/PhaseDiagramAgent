"""Keep default pytest artifacts away from workspace and drive roots."""
from pathlib import Path
import tempfile
import uuid


def pytest_configure(config):
    # Respect a deliberately supplied path; use an isolated run directory by
    # default so concurrent test sessions cannot erase each other's artifacts.
    if config.option.basetemp is None:
        directory = Path(tempfile.gettempdir()) / "pdt"
        directory.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(directory / uuid.uuid4().hex[:8])
