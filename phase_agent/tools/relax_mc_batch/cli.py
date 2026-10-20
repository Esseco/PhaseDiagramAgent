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
    recover = sub.add_parser("recover", help="collect existing verified results only; never submit")
    recover.add_argument("--original-archive", required=True)
    recover.add_argument("--evidence-directory", required=True)
    recover.add_argument("--destination", required=True)
    recover.add_argument("--job-evidence", required=True)
    recover.add_argument("--remote-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        from phase_agent.tools.relax_mc_batch.run import run_batch

        result = run_batch(args.root, submit=args.submit, dry_run=args.dry_run, phase=args.phase)
    elif args.command == "recover":
        from pathlib import Path
        from phase_agent.tools.relax_mc_batch.recovery import build_recovery_return

        result = build_recovery_return(
            args.original_archive,
            args.evidence_directory,
            args.destination,
            job_evidence=json.loads(Path(args.job_evidence).read_text(encoding="utf-8")),
            remote_root=args.remote_root,
        )
    else:
        from phase_agent.tools.relax_mc_batch.results import inspect_archive

        result = inspect_archive(args.archive)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
