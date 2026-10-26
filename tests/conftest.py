"""Own and remove test artifacts; real scientific fixtures are opt-in."""

import os
from pathlib import Path
import tempfile
import warnings

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-real-project-tests",
        action="store_true",
        default=False,
        help="Run real_project tests that read/copy large local scientific projects",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    previous_tempdir = tempfile.tempdir
    config.addinivalue_line("markers", "real_project: reads/copies a real local project (opt-in)")
    # Treat an explicit basetemp as a parent, never as a directory pytest may erase.
    parent = (
        Path(config.option.basetemp).expanduser().resolve()
        if config.option.basetemp
        else Path(tempfile.gettempdir()) / "pdt"
    )
    parent.mkdir(parents=True, exist_ok=True)
    spelling = str(parent)
    if os.name == "nt" and not spelling.startswith("\\\\?\\"):
        parent = Path(
            "\\\\?\\UNC\\" + spelling[2:] if spelling.startswith("\\\\") else "\\\\?\\" + spelling
        )
    run = tempfile.TemporaryDirectory(prefix="", dir=parent)
    directory = Path(run.name)
    # Keep the extended spelling for cleanup, but subprocesses (notably Node)
    # need a conventional Windows path.
    spelling = str(directory)
    if spelling.startswith("\\\\?\\UNC\\"):
        directory = Path("\\\\" + spelling[8:])
    elif spelling.startswith("\\\\?\\"):
        directory = Path(spelling[4:])
    config.option.basetemp = str(directory / "t")
    previous_environment = {name: os.environ.get(name) for name in ("TEMP", "TMP")}
    for name in previous_environment:
        os.environ[name] = str(directory)
    tempfile.tempdir = str(directory)

    def cleanup():
        tempfile.tempdir = previous_tempdir
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        try:
            run.cleanup()
        except OSError as error:
            warnings.warn(
                f"Test artifacts could not be removed from {directory}: {error}",
                pytest.PytestWarning,
                stacklevel=2,
            )

    config.add_cleanup(cleanup)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-real-project-tests"):
        return
    skip = pytest.mark.skip(reason="large real-project fixture; use --run-real-project-tests")
    for item in items:
        if item.get_closest_marker("real_project") is not None:
            item.add_marker(skip)
