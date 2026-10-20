# Result recovery and asynchronous control

The scientific graph remains the only workflow owner. Recovery and UI transports reuse its result verification, task reconciliation, scientific quality gates and approval bindings. They never authorize another scientific action.

## Collect an existing interrupted batch

Use `python -m phase_agent.tools.relax_mc_batch.cli recover --original-archive OLD_RESULTS.zip --evidence-directory DOWNLOADED_WORK --destination NEW_RECOVERY_DIRECTORY --job-evidence JOB.json --remote-root ORIGINAL_REMOTE_WORK`.

Evidence must include the original remote workflow_state.json submission-to-job mapping and the task result/input/pool directories. JOB.json contains job_id, task_ids, terminal state, source, elapsed_seconds and allocated_gpus. For automatic GPU clock accounting source must be exactly sacct. Missing or invalid evidence cannot imply completed calculation. The builder preserves signed result bytes, validates existing task/model/config/checksum bindings and local final structures, and never submits or runs a model. Earlier phases do not imply later MC waves completed. An interrupted build is retained under interrupted-* and a fresh isolated attempt is used. Repeat requests for identical evidence return the existing checksum-verified archive.

Import the generated return through the existing Relax/MC importer. It records result_recoveries, retains unknown per-task costs, and deduplicates verified whole-job clocks through the normal GPU ledger. A controller failure remains historical evidence, not a new failed calculation.

## Administrative result admission

`commit_result_recovery` accepts only canonical state and project ledger destinations with compare-and-swap SHA256 checks. It stages desired data, preserves originals, publishes a workflow-control revision barrier, writes ledgers/state, and retains a durable transaction journal. Python exceptions restore originals; a process crash can finish only the staged facts using `resume_result_recovery`. Any changed source/staged checksum blocks recovery. No scientific tool is replayed.

Long-lived managers refresh their ledger once per changed administrative revision. Existing durable lifecycle threads already include that revision and cannot restore an earlier business state over the committed results. This is not a transaction over every external service or LangGraph database.

Authenticated GET `/phase/recovery/transaction` inspects admission without reading/re-writing interrupted state. POST `/phase/recovery/transaction/resume` requires the displayed revision and only finishes that existing admission. The approval page offers this explicit recovery before attempting to load interrupted scientific state.

## UI requests and factual states

Existing Studio runs remain asynchronous. Approval/control POSTs may opt in with `Prefer: respond-async` and `Idempotency-Key`; the built-in approval page does so. Requests return HTTP 202 and a durable operation/status URL, and completion is polled separately. The same key cannot change request content. Concurrent requests return the active operation or are blocked by the handler lock; they do not queue duplicate scientific effects. A dead owner is shown as interrupted and is never automatically replayed. Legacy synchronous API clients remain supported.

Authenticated GET `/phase/operations` and `/phase/operations/<id>` are read-only. Interaction completed means its callback returned, not that a scientific search converged. Project recovery facts distinguish awaiting verification, local recovery blocked, completed admission, and an early phase requiring next-step review. Current errors and historical controller diagnostics are separate fields.

All status/plan/flow GET presentations use request-local read-only views. Reading a branch or DFT proposal never creates, rewrites or migrates JSON/TXT review artifacts. Proposal creation and approved execution retain their existing artifact writers.

## Release verification

Run `python tests/recovery_release_gate.py --basetemp SHORT_WRITABLE_PATH --junitxml REPORT.xml` in py1. The gate uses fake models/Slurm and an isolated random-port HTTP handler, covering receipts, interrupted/duplicate collection, scheduler history, bad evidence, long paths, old checkpoint restoration, approvals, manager refresh, GPU deduplication, async controls and read-only views.

Additionally replay actual project-state copies (Co/Cr/FeMn for this release) through durable lifecycle serialization/resume and UI projections, compare scientific fields and approval counts, and confirm original data/service metadata hashes. Do not invoke live inference/HPC execution in regression tests. Deploy only when baseline source hashes match; preserve a source backup. Deployment does not hot reload running services. Service lifecycle remains exclusively user-managed.
