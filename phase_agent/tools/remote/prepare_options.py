"""Negotiate optional preparation optimizations without retrying side effects."""

from phase_agent.interface_compatibility import supported_optimizations


def supported_prepare_options(prepare, **options):
    return supported_optimizations(prepare, options)
