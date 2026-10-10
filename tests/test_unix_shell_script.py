from phase_agent.tools.remote.write_unix_shell_script import write_unix_shell_script


def test_slurm_script_normalizes_windows_and_old_mac_endings(tmp_path):
    path = tmp_path / "GPU.sh"
    write_unix_shell_script(path, b"#!/bin/sh\r\n#SBATCH -N 1\r\necho ok\r")
    assert path.read_bytes() == b"#!/bin/sh\n#SBATCH -N 1\necho ok\n"
