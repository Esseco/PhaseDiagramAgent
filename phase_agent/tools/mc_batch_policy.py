"""Validate configured MC tasks per job; no hidden capacity override."""


def mc_batch_limit(configured_size):
    size = int(configured_size)
    if size <= 0:
        raise ValueError("MC batch size must be positive")
    return size
