"""SQLite URI spelling for normal and Windows extended filesystem paths."""

from pathlib import Path
from urllib.parse import quote


def readonly_uri(path):
    value = str(Path(path).resolve())
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    if value.startswith("\\\\"):
        # SQLite treats a nonempty URI authority as a host, not a Windows UNC path.
        uri = "file://" + quote(value.replace("\\", "/"), safe="/:")
    else:
        uri = Path(value).as_uri()
    return uri + "?mode=ro"
