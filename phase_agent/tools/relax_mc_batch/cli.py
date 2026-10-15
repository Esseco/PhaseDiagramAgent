"""Remote run/dry-run and local read-only result inspection."""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--root", required=True)
    run.add_argument("--submit", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--phase", type=int)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--archive", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        from phase_agent.tools.relax_mc_batch.run import run_batch

        result = run_batch(args.root, submit=args.submit, dry_run=args.dry_run, phase=args.phase)
    else:
        from phase_agent.tools.relax_mc_batch.results import inspect_archive

        result = inspect_archive(args.archive)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
