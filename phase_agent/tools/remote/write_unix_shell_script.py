"""Write Slurm shell scripts with LF endings on every host OS."""

from pathlib import Path


def write_unix_shell_script(path, content):
    data = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    Path(path).write_bytes(data)
