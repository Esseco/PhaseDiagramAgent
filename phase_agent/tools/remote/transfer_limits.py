"""Versioned transfer capacity shared by producers and both receivers."""
MAX_ARCHIVE_FILES = 100000  # Includes the transfer manifest itself.
MAX_MANIFEST_BYTES = 64 * 1024**2
MAX_ARCHIVE_BYTES = 8 * 1024**3


def validate_capacity(file_count, manifest_bytes, total_bytes, *, max_bytes=MAX_ARCHIVE_BYTES):
    if file_count > MAX_ARCHIVE_FILES:
        raise ValueError(f"Transfer file count exceeds limit ({file_count}/{MAX_ARCHIVE_FILES}); split the package before transfer")
    if manifest_bytes > MAX_MANIFEST_BYTES:
        raise ValueError(f"Transfer manifest exceeds size limit ({manifest_bytes}/{MAX_MANIFEST_BYTES}); split the package before transfer")
    if total_bytes > max_bytes:
        raise ValueError(f"Result archive exceeds size limit ({total_bytes}/{max_bytes}); split the package before transfer")


def bootstrap_limits():
    # The compute node must verify inputs before it can import the bundled package.
    return (
        f"MAX_ARCHIVE_FILES = {MAX_ARCHIVE_FILES}\n"
        f"MAX_MANIFEST_BYTES = {MAX_MANIFEST_BYTES}\n"
        f"MAX_ARCHIVE_BYTES = {MAX_ARCHIVE_BYTES}\n"
    )
