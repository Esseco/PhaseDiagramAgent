"""Single command-line entry for split supercomputer orchestration."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from config_layer.defaults.default_slurm_cluster_config import default_slurm_cluster_config
from decision_layer.agent.create_deepseek_client import create_deepseek_client
from execution_layer.slurm.slurm_batch_runner import SlurmBatchRunner
from execution_layer.step_runner import (
    advise_next_actions, confirm_action_plan, prepare_confirmed_plan,
    read_runner_status, recover_results, submit_prepared_jobs,
)
from execution_layer.step_runner.file_protocol import read_json
from execution_layer.step_runner.scheduler_adapter import (
    CommandSchedulerAdapter, UnconfiguredSchedulerAdapter,
)


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    config = _runtime_config(args.config)
    paths = config["supercomputer"]["paths"]
    state_path = args.state or paths["state"]
    if args.command == "recover":
        runner = _batch_runner(config)
        result = recover_results(
            state_path, args.summary or paths["summary"], result_collector=runner,
            config_version=args.config_version, node_role=args.node_role,
        )
    elif args.command == "advise":
        deepseek = config.get("deepseek") or {}
        client = create_deepseek_client(
            api_key=deepseek.get("api_key"), model=deepseek.get("model", "deepseek-v4-pro"),
            base_url=deepseek.get("base_url", "https://api.deepseek.com"),
            max_tokens=int(deepseek.get("max_tokens", 800)),
            timeout=int(deepseek.get("timeout", 120)),
        )
        result = advise_next_actions(
            args.summary or paths["summary"], args.plans or paths["plans"],
            agent_client=client, config_version=args.config_version,
            node_role=args.node_role,
        )
    elif args.command == "confirm":
        result = confirm_action_plan(
            args.plan, user_confirmed=args.approve, comment=args.comment,
            node_role=args.node_role,
        )
    elif args.command == "prepare":
        result = prepare_confirmed_plan(
            state_path, args.plan, batch_runner=_batch_runner(config),
            action_executor=_load_optional(config["supercomputer"]["worker"].get("action_executor")),
            expected_config_version=args.config_version, node_role=args.node_role,
        )
    elif args.command == "submit":
        result = submit_prepared_jobs(
            state_path, scheduler=_scheduler(config), node_role=args.node_role, limit=args.limit,
        )
    else:
        result = read_runner_status(state_path, node_role=args.node_role)
    print(json.dumps(_compact(result), ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") != "not_configured" else 2


def _parser():
    parser = argparse.ArgumentParser(description="Split login/compute-node phase-diagram runner")
    parser.add_argument("--config", help="confirmed runtime JSON or confirmed config session")
    parser.add_argument("--state")
    parser.add_argument("--node-role", choices=["login", "compute"])
    sub = parser.add_subparsers(dest="command", required=True)
    recover = sub.add_parser("recover"); recover.add_argument("--summary"); recover.add_argument("--config-version")
    advise = sub.add_parser("advise"); advise.add_argument("--summary"); advise.add_argument("--plans"); advise.add_argument("--config-version", required=True)
    confirm = sub.add_parser("confirm"); confirm.add_argument("--plan", required=True); confirm.add_argument("--approve", action="store_true"); confirm.add_argument("--comment")
    prepare = sub.add_parser("prepare"); prepare.add_argument("--plan", required=True); prepare.add_argument("--config-version", required=True)
    submit = sub.add_parser("submit"); submit.add_argument("--limit", type=int)
    sub.add_parser("status")
    return parser


def _runtime_config(path):
    from run.default_run_config import default_run_config
    config = default_run_config()
    if not path:
        return config
    payload = read_json(path)
    snapshot = (payload or {}).get("confirmed_snapshot") or {}
    if (payload or {}).get("status") != "confirmed" or not snapshot.get("config") or not snapshot.get("config_version"):
        raise ValueError("step_runner requires a confirmed config session snapshot")
    supplied = snapshot["config"]
    return _deep_merge(config, supplied)


def _batch_runner(config):
    cluster = config["supercomputer"]
    command = cluster.get("worker", {}).get("command")
    if not command:
        command = ["python3", "-m", "execution_layer.slurm.run_slurm_array_task"]
    return SlurmBatchRunner(
        cluster["paths"]["batches"], worker_command=command,
        stage_batch_sizes=cluster.get("batch_sizes"),
        stage_profiles=default_slurm_cluster_config(), submit=False,
    )


def _scheduler(config):
    settings = config["supercomputer"].get("scheduler") or {}
    if not settings.get("submit_command"):
        return UnconfiguredSchedulerAdapter()
    return CommandSchedulerAdapter(
        submit_command=settings["submit_command"],
        query_command=settings.get("query_command"), cancel_command=settings.get("cancel_command"),
    )


def _load_optional(reference):
    if not reference:
        return None
    module, separator, name = reference.partition(":")
    if not separator:
        raise ValueError("action_executor must use package.module:function")
    return getattr(importlib.import_module(module), name)


def _deep_merge(base, update):
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _compact(result):
    return {key: value for key, value in result.items() if key != "state"}


if __name__ == "__main__":
    raise SystemExit(main())
