"""Codex offline entry for an explicitly authorized Relax–MC policy review."""

import argparse
import json
from phase_agent.runtime.codex.offline import CodexOffline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "runtime-config",
        "bundle-id",
        "raw-path",
        "raw-sha256",
        "receipts-path",
        "receipts-sha256",
        "expected-review-digest",
        "user-instruction",
    ):
        parser.add_argument("--" + key, required=True)
    args = vars(parser.parse_args(argv))
    runtime_config = args.pop("runtime_config")
    result = CodexOffline(runtime_config).register_relax_mc_threshold_transition(**args)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
