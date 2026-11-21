"""Allocate an already-approved Relax–MC continuation through Codex Agent."""

import argparse, json
from phase_agent.runtime.codex.offline import CodexOffline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("runtime-config", "bundle-id", "preview-path", "preview-sha256"):
        parser.add_argument("--" + key, required=True)
    args = vars(parser.parse_args())
    runtime = args.pop("runtime_config")
    print(
        json.dumps(
            CodexOffline(runtime).allocate_approved_relax_mc_continuation(**args),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
