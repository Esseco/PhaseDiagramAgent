"""The optional regression guard propagates to child Python processes."""

import os
from pathlib import Path
import subprocess
import sys


def test_guard_blocks_real_model_loader_but_allows_explicit_fake(tmp_path):
    guard = Path(__file__).parent / "fake_only_guard"
    env = {**os.environ, "PYTHONPATH": str(guard), "TEMP": str(tmp_path), "TMP": str(tmp_path)}
    blocked = subprocess.run(
        [sys.executable, "-c", "import chgnet.model"], env=env, capture_output=True, text=True
    )
    assert blocked.returncode != 0 and "real_model_loader_blocked:chgnet.model" in blocked.stderr
    fake = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, types; sys.modules['maml.describers'] = types.ModuleType('maml.describers'); "
            "from maml.describers import __name__; assert __name__ == 'maml.describers'",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert fake.returncode == 0, fake.stderr
