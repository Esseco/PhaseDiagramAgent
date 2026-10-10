"""Shared capacity rule for packaging independent MC simulations into one job."""

MAX_MC_TASKS_PER_JOB = 10


def mc_batch_limit(configured_size):
    size = int(configured_size)
    if size <= 0:
        raise ValueError("MC batch size must be positive")
    return min(size, MAX_MC_TASKS_PER_JOB)
