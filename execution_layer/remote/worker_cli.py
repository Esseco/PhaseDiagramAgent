"""Offline compute-node CLI for one portable remote manifest task."""
import argparse
import importlib
from execution_layer.remote.worker import run_remote_task


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", required=True)
    parser.add_argument("--array-index", required=True, type=int); parser.add_argument("--executor", required=True)
    args = parser.parse_args(argv); module, separator, name = args.executor.partition(":")
    if not separator: raise ValueError("executor must use package.module:function")
    executor = getattr(importlib.import_module(module), name)
    result = run_remote_task(args.manifest, args.array_index, executor=executor)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__": raise SystemExit(main())
