# Project-scoped shared MCP service

Status: design and recovery-lock change in progress; no shared service has been deployed.

## Problem and constraints

The existing MCP adapter uses stdio and each connection starts its own Python runtime. Multiple clients can therefore load the same scientific project and compete over memory and project state. Existing conversations can select different projects, and both the original Agent and Codex must remain functional. Existing approved work and returned results must not be replayed.

## Intended architecture

Use one project-owned long-lived service per runtime configuration. A shared connection layer must retain a separate selected project for each client session and route requests to the matching project service; it must not replace the selected project in another conversation. Existing stdio clients require an explicit compatibility route; simply rejecting the second stdio process breaks active clients. Reuse the existing project service identity and process-lifetime lock, and existing Agent/Codex control APIs rather than creating a second scientific workflow.

Any HTTP transport must follow the MCP Streamable HTTP specification, including per-client session handling, explicit initialization, localhost binding, Origin validation and authentication. Handle client disconnects separately from cancellation: a timed-out client must not start an additional recovery while the original operation is still running. Publish bounded operation status and make retries return or inspect the original operation rather than replay it.

Use the same project lifecycle lock for all state-mutating entry points, including recovery through both Agent and Codex. Re-read and verify the project version after acquiring the lock; a stale state must not overwrite a newer state. Preserve existing ledger and result-commit guards. Locks must be scoped to the resolved project identity, released on crash, and must not block unrelated projects.

## Implemented so far

CodexOffline.recover uses the existing project lifecycle file lock before reading state. A multi-process synthetic test verifies that a competing same-project lock blocks recovery while an unrelated project remains usable. Existing running processes do not load this change until restarted. This does not implement shared transport or protect all Agent recovery paths.

## Required validation before migration

- Two simultaneous clients for one project reuse one service; separate projects do not share selected state.
- Reconnect, timeout, client disconnection, process crash, stale metadata and concurrent startup do not duplicate or silently replay a scientific operation.
- Agent and Codex recoveries and decisions contend on the same project lock; old configuration, approvals and completed results remain valid.
- Check protocol compatibility, authorization, Origin handling, project isolation, bounded memory and repeated status-read performance with synthetic projects.
- Preserve active operations during migration. Coordinate explicit service lifecycle and configuration changes before activating the shared endpoint.
