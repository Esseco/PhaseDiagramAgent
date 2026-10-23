"""Exercise pytest lifecycle in tiny subprocesses, without scientific backends."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["pass", "fail", "collection_error", "interrupt"])
@pytest.mark.parametrize("explicit", [False, True])
def test_run_artifacts_are_removed_and_parent_is_preserved(tmp_path, mode, explicit):
    case = tmp_path / "case"
    case.mkdir()
    shutil.copy2(Path(__file__).with_name("conftest.py"), case / "conftest.py")
    parent = tmp_path / "scratch"
    parent.mkdir()
    sentinel = parent / "keep.txt"
    sentinel.write_text("existing user file", encoding="utf-8")
    observed = tmp_path / "observed.txt"
    setup = (
        "import tempfile\nfrom pathlib import Path\n"
        f"Path({str(observed)!r}).write_text(tempfile.gettempdir(), encoding='utf-8')\n"
        "Path(tempfile.gettempdir(), 'child-output.bin').write_bytes(b'fixture')\n"
    )
    if mode == "collection_error":
        body = setup + "raise RuntimeError('intentional collection failure')\n"
    else:
        ending = {
            "pass": "assert True",
            "fail": "assert False, 'intentional failure'",
            "interrupt": "raise KeyboardInterrupt()",
        }[mode]
        body = (
            setup
            + "def test_case(tmp_path):\n    (tmp_path / 'state.json').write_text('{}')\n    "
            + ending
            + "\n"
        )
    (case / "test_case.py").write_text(body, encoding="utf-8")
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if explicit:
        command.append("--basetemp=" + str(parent))
    result = subprocess.run(
        command,
        cwd=case,
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "TEMP": str(parent),
            "TMP": str(parent),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_ADDOPTS": "",
        },
    )
    assert result.returncode == (0 if mode == "pass" else 1 if mode == "fail" else 2), (
        result.stdout + result.stderr
    )
    run = Path(observed.read_text(encoding="utf-8"))
    assert run != parent
    assert not run.exists(), result.stdout + result.stderr
    assert sentinel.read_text(encoding="utf-8") == "existing user file"
    assert not list(parent.rglob("state.json"))
    assert not list(parent.rglob("child-output.bin"))


def test_real_project_tests_require_explicit_option(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    shutil.copy2(Path(__file__).with_name("conftest.py"), case / "conftest.py")
    (case / "test_case.py").write_text(
        "import pytest\n@pytest.mark.real_project\ndef test_real(): assert False\n"
        "def test_small(): assert True\n",
        encoding="utf-8",
    )
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    env = {**os.environ, "PYTEST_ADDOPTS": "", "PYTHONDONTWRITEBYTECODE": "1"}
    default = subprocess.run(command, cwd=case, env=env, capture_output=True, text=True, timeout=30)
    assert default.returncode == 0, default.stdout + default.stderr
    assert "1 passed, 1 skipped" in default.stdout
    enabled = subprocess.run(
        command + ["--run-real-project-tests"],
        cwd=case,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert enabled.returncode == 1, enabled.stdout + enabled.stderr
    assert "1 failed, 1 passed" in enabled.stdout


def test_cleanup_restores_environment_for_in_process_pytest(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    shutil.copy2(Path(__file__).with_name("conftest.py"), case / "conftest.py")
    (case / "test_case.py").write_text("def test_small(): assert True\n", encoding="utf-8")
    code = (
        "import os, tempfile, pytest\n"
        "before = ({k: os.environ.get(k) for k in ('TEMP', 'TMP')}, tempfile.gettempdir())\n"
        "assert pytest.main(['-q', '-p', 'no:cacheprovider']) == 0\n"
        "after = ({k: os.environ.get(k) for k in ('TEMP', 'TMP')}, tempfile.gettempdir())\n"
        "assert before == after, (before, after)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=case,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTEST_ADDOPTS": "", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_concurrent_runs_do_not_erase_each_others_artifacts(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "keep.txt").write_text("keep", encoding="utf-8")
    processes = []
    try:
        for index in range(2):
            case = tmp_path / f"case{index}"
            case.mkdir()
            shutil.copy2(Path(__file__).with_name("conftest.py"), case / "conftest.py")
            observed = tmp_path / f"observed{index}.txt"
            other = tmp_path / f"observed{1 - index}.txt"
            body = (
                "import tempfile, time\nfrom pathlib import Path\n"
                "def test_small(tmp_path):\n"
                f"    Path({str(observed)!r}).write_text(tempfile.gettempdir(), encoding='utf-8')\n"
                "    deadline = time.monotonic() + 10\n"
                f"    while not Path({str(other)!r}).exists() and time.monotonic() < deadline: time.sleep(0.05)\n"
                f"    assert Path({str(other)!r}).exists()\n"
                "    (tmp_path / 'state.json').write_text('{}')\n"
                "    time.sleep(0.2)\n"
                "    assert (tmp_path / 'state.json').is_file()\n"
            )
            (case / "test_case.py").write_text(body, encoding="utf-8")
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "-p",
                        "no:cacheprovider",
                        "--basetemp=" + str(shared),
                    ],
                    cwd=case,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env={**os.environ, "PYTEST_ADDOPTS": "", "PYTHONDONTWRITEBYTECODE": "1"},
                )
            )
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
    runs = [Path((tmp_path / f"observed{i}.txt").read_text(encoding="utf-8")) for i in range(2)]
    assert runs[0] != runs[1]
    assert all(not run.exists() for run in runs)
    assert [p.name for p in shared.iterdir()] == ["keep.txt"]
