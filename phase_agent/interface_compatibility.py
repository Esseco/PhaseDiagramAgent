"""Negotiate performance-only keywords without retrying side effects."""

from inspect import Parameter, signature

_PERFORMANCE_OPTIONS = {"_owned", "_approval_only", "_defer_index", "_defer_archive"}


def supported_optimizations(function, options):
    if options.keys() - _PERFORMANCE_OPTIONS:
        raise ValueError("Only performance options may be omitted for interface compatibility")
    parameters = signature(function).parameters
    if any(item.kind == Parameter.VAR_KEYWORD for item in parameters.values()):
        return options
    return {
        key: value
        for key, value in options.items()
        if key in parameters
        and parameters[key].kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
    }


def call_with_optimizations(function, /, *args, _optimizations, **kwargs):
    return function(*args, **kwargs, **supported_optimizations(function, _optimizations))
