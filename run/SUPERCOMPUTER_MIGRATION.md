# Supercomputer staging migration map

The local workspace root is the only maintained source tree and the only owner
of confirmed configuration, task identity, budget transactions, search state,
and the writable scientific ledger. `_supercomputer_stage` was not present in
the inspected workspace, so no unseen staging code was copied or assumed.

| Staging responsibility | Canonical root implementation | Status |
|---|---|---|
| Confirmed configuration gate | `config_layer/session`, `config_layer/runtime`, `run.main` | canonical |
| Open WebUI proposal/approval | `run/open_webui_api.py`, `run/local_agent_control.py` | canonical |
| Offline plan confirmation | `execution_layer/step_runner/confirm_action_plan.py` | canonical |
| Split recover/advise/confirm/prepare/submit/status | `run/step_runner.py` | canonical |
| Portable immutable batch | `execution_layer/remote/batch_runner.py` | canonical |
| Manual upload bundle | `execution_layer/remote/manual_upload_runner.py` | canonical |
| SSH/OpenSSH adapters | `execution_layer/remote/ssh_adapter.py`, `openssh_runtime.py` | adapter boundary |
| Submit/query/cancel and uncertain submission | `execution_layer/remote/workflow.py`, `step_runner/submit_prepared_jobs.py` | canonical |
| Atomic result then completion marker | `execution_layer/remote/worker.py` | canonical |
| Integrity/version validation | `execution_layer/remote/integrity.py` | canonical |
| Idempotent reconciliation and settlement | `execution_layer/state/reconcile_task_results.py`, `execution_layer/budget` | canonical |
| Restart guidance | `execution_layer/remote/workflow.resume`, `run/resume_pipeline.py` | canonical |
| Offline legality/dedup gate | `execution_layer/workflows/prepare_dedup_batch.py`, `accept_dedup_results.py` | canonical |

Remote copies are deployment artifacts only. They may contain immutable task
snapshots and job-status/result files, but they must never become a second
writable master state or accept an unconfirmed configuration as effective.
