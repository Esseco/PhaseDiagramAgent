from phase_agent.tools.remote.write_unix_shell_script import write_unix_shell_script


def test_slurm_script_normalizes_windows_and_old_mac_endings(tmp_path):
    path = tmp_path / "GPU.sh"
    write_unix_shell_script(path, b"#!/bin/sh\r\n#SBATCH -N 1\r\necho ok\r")
    data = path.read_bytes()
    assert b"\r" not in data
    assert data.startswith(b"#!/bin/sh\n#SBATCH -N 1\n")
    assert data.endswith(b"echo ok\n")
    assert data.count(b"# GPU_USAGE_CLOCK") == 1
    assert (tmp_path / "_gpu_accounting.py").is_file()
    assert (tmp_path / "_gpu_transfer_bundle.py").is_file()
