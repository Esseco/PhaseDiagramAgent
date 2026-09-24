# Open WebUI integration

Open WebUI is the browser chat client; this project remains the Agent and the
only authority that can approve, validate, budget, and dispatch actions. The
local endpoint implements the OpenAI-compatible `/v1/models` and
`/v1/chat/completions` routes, so no Open WebUI tools with model-generated
approval arguments are exposed. The server reads the latest `user` message
from the request and sends it to the existing interactive workflow.

## Runtime composition

Copy `run/open_webui_runtime.example.json` to the git-ignored/local file
`run/open_webui_runtime.json`, then replace its paths with the site's real local
paths. Relative paths are resolved from the JSON file. The built-in factory
loads the confirmed config session, state, `PhaseDataManager` ledger, phase
references and DeepSeek client. It never scans scientific data directories.

The JSON may contain these non-secret settings:

- required `state_path` and `ledger_path`;
- optional `phase_references_path` and `config_session_path`;
- optional `new_runs_directory`, runtime path settings and non-secret DeepSeek settings;
- optional `local_action_directory`; approved file actions may write only below this
  directory and stop on non-identical existing content;
- optional `dispatcher_factory`, `task_runner_factory`, and `runtime_adapters`, whose
  values use `package.module:function` and are called without arguments.
- optional DeepSeek `configuration_thinking` (`disabled` by default for compact
  first-run JSON replies) and `thinking` (API default when omitted, normally
  enabled) settings. Configuration dialogue and search decisions can therefore
  use different reasoning modes.
- optional `manual_upload`. When enabled, it uses the built-in portable batch
  exporter, writes `manifest.json`, per-task JSON, `submit.sbatch`, immutable
  snapshots, `SHA256SUMS`, and `UPLOAD_AND_SUBMIT.md`, but never calls `sbatch`.
  Its `worker_command`, stage profiles and any task preparer must be supplied by
  the site. `manual_upload.submit=true` is rejected.

Do not put API keys, passwords, SSH secrets, structures, or guessed cluster
paths in this file. Secret-like JSON fields are rejected. `DEEPSEEK_API_KEY`
and the local Open WebUI bearer token are read only from environment variables.
If a configured config-session file is missing, recovery is allowed only when
the configured state explicitly contains both `confirmed_config` and a config
version. A missing ledger can be created only when the confirmed config has a
real `system.boundary`; the runtime will report the exact missing setting rather
than inventing a system, structure, cluster, or compute adapter.

Start the endpoint under the project's `py1` environment:

```powershell
$env:OPENWEBUI_TOOL_TOKEN = "<a random local token of at least 16 characters>"
$env:OPENWEBUI_CONTROL_TOKEN = "<a different random local control token>"
$env:DEEPSEEK_API_KEY = "<your local API key>"
python -m run.open_webui_api
```

Use `--runtime-config PATH` for another local JSON. The older
`--handler-factory package.module:function` remains an explicit override.

The factory can return a `RunWorkflowChatHandler(workflow_kwargs)` or a custom
callable accepting `(messages, conversation_id=...)`. It must return only the
runtime objects needed by the existing workflow; keys and SSH credentials stay
in local environment/configuration and are never included in chat state.

In Open WebUI, add an OpenAI-compatible connection with base URL
`http://127.0.0.1:8765/v1`, the same local token as API key, and select model
`phase-search-agent`. The endpoint binds to loopback by default. If Open WebUI
runs in Docker, use `http://host.docker.internal:8765/v1` where supported (or the
explicit host-gateway address), bind the service only to the required host
interface, and add firewall/token protection. Do not expose it to the public
internet.

The separate `OPENWEBUI_CONTROL_TOKEN` protects the deliberately small control API:
`GET /phase/status`, `/phase/pending`, `/phase/tasks`, `/phase/charts`,
`/phase/config`, `/phase/memory`; and authenticated `POST /phase/propose`,
`/phase/decision`, `/phase/pause`, `/phase/config/patch`,
`/phase/config/confirm`, `/phase/memory/review`. It exposes no shell,
Python evaluation, arbitrary path, upload, or file-reading endpoint. Decisions
still enter the existing interactive workflow and policy gate. Sensitive DFT
Relax/model update actions require `confirm_sensitive` on the local approval
page instead of ordinary batch approval.

Open `http://127.0.0.1:8765/phase/approval` in the host browser. The page itself
contains no project data until the distinct control token is entered. It shows
the plan ID/hash, target IDs, parameters, reason, missing evidence, cost and
config/model/state versions, plus project-generated SVG charts. Approval posts
the displayed proposal hash and state version; stale pages are rejected and
must be refreshed. Repeated approval of an already processed invocation is
idempotent.

## Human approval behavior

Each chat turn advances at most one action. While a proposal is pending, the
actual latest user message is forwarded as revision feedback. Chat text,
including `同意`, `approve`, `可以`, `继续`, `拒绝` or `确认敏感操作`, never
authorizes or rejects execution. Decisions are accepted only by the authenticated
local approval page. The workflow reruns its existing Execution Policy,
config/frozen-parameter checks, permissions, budget and duplicate validation at
approval time. An assistant message or generated tool call cannot approve an
action. To change a proposal, send the requested changes, inspect the revised
proposal, then approve that exact revision on the page.

The registered `prepare_local_batch_files` action is the minimal safe file
generation action. After approval it creates an immutable dry-run manifest,
`DRY_RUN_ONLY.txt` and `SHA256SUMS` below `local_action_directory`; it submits
nothing. A matching task ID and content checksum is reused, while any other
existing target is reported as a conflict. Windows-local/Linux-remote path
mappings belong in the versioned configuration as explicit
`supercomputer.path_mappings` rows with `windows_local` and `linux_remote`.
Remote-facing payloads reject Windows absolute paths.

When the configured state or ledger contains branch, structure, action, or task
history, the first turn shows a short summary and accepts only `继续`/`continue`
or `新建`/`new`. Continue reloads the existing state, ledger, recent actions and
pending proposal. It never counts as proposal approval. New creates unique
state and ledger paths below `new_runs_directory`, keeps the same confirmed
configuration, and leaves all old files untouched. Confirm a separate config
session first if the system or boundary must change. With no history, the first
user instruction enters the normal confirmed-config workflow directly.

The handler serializes calls and binds its in-memory approval interaction to one
Open WebUI conversation. This is a single-user debugging service, not a
multi-user approval server. Start separate processes with separate state paths
for independent users/runs.

## Local secrets and Open WebUI data

Set `DEEPSEEK_API_KEY`, `OPENWEBUI_TOOL_TOKEN`, and the distinct
`OPENWEBUI_CONTROL_TOKEN` in the local process
environment, never in runtime JSON, state, task payloads, or remote manifests.
Disable Open WebUI configuration/chat persistence when your deployment supports
it, and protect its data directory with OS permissions and disk encryption.
Open WebUI versions and deployment settings differ, so the project does **not**
claim that a key can never reach process logs, browser storage, container
metadata, backups, or a misconfigured WebUI database; verify your deployment.
Remote logs and recovered text are treated only as untrusted data fields, not as
Agent instructions.

Open WebUI does not reduce DeepSeek's API unit price. Cost control comes from the
bounded state snapshot, five-item recent history, one-step interaction, call and
token budgets, and reuse of persisted analyses instead of replaying full logs or
chat history.

## First-run configuration dialogue

On a clean first start, the service creates only a small resumable dialogue session at `config_session_path` and asks for the local workspace root. It does not create the workspace, editable settings JSON, scientific backend, ledger, or jobs yet. The user enters a path; the service previews the settings file and all derived output destinations. Only after the user replies “确认存储路径” does it create the workspace and annotated settings JSON. The user can then edit/import the JSON and have the Agent review missing or conflicting fields. Only a later, separate “确认配置” creates a versioned snapshot; neither confirmation starts a calculation.

After confirming the workspace root, the service creates `<workspace_root>/search_config.draft.json` (or the filename configured by `editable_config_draft_path`; it is always placed under the selected root). Open that file in an editor and change values directly instead of typing every parameter into chat. Keep the outer `_format`, `_instructions`, `_section_help`, and `config` keys; edit values inside `config`. The file accepts JSONC `//` and `/* ... */` comments, but not trailing commas. `system.boundary.P` and `TM_ratio` are prefilled from the current layered-oxide defaults for review; fill the per-phase `H` matrix lists. The mother-structure directory is a single `system.phase_reference_directory`; on import, the project resolves each phase to `<directory>/<phase>.vasp` (for example, `O3.vasp`). You can still use `system.phase_references` to override individual files. Use JSON `null` for unknown values; do not delete fields or put API keys and passwords in the file. The import checks syntax, field shape, and secret-like keys.

The same JSON includes `config.storage`: the selected `workspace_root` and relative destinations for config snapshots, state, the main ledger, structures, phase diagrams, QBC results, approvals, task work and upload batches. New workspaces use the standard relative layout shown in the file. Relative output paths are constrained to remain under the workspace root. If you select a new root while existing state or ledger files remain elsewhere, startup stops rather than silently moving or ignoring them; manually migrate and verify those files, or point the entries back to their original locations.

The dialogue session remains at the bootstrap path selected by `config_session_path`, so the local service can resume setup before the chosen workspace is loaded. The editable JSON and immutable confirmed config copy are stored under the selected workspace root; snapshots go to `workspace_root/config_snapshots/` by default. Runtime setting `editable_config_draft_path` controls only the JSON filename, not its directory.

After saving the file, send the exact chat message `读取配置 JSON`. The service imports it into the unconfirmed session, records a revision/change audit, and asks the Agent to review missing items and conflicts. Agent suggestions are not applied to the JSON or draft automatically; edit the file and import it again. Importing does not confirm the snapshot or launch work. Only after the Agent reports the draft ready, send `确认配置`. This writes an immutable `config_snapshots/<config_version>.json` under the selected workspace root and prints every resolved output path; it does not submit or start calculations. Runtime startup creates the JSON template only when absent, so it will never overwrite your edits.

The first-run draft includes candidate setup hints from the project owner: local mother structures under E:/0-FM-PhaseDiagram/InitFile/Struct and the mh-1 model at /data/home/lichaoyue/Py-lzy/MLIP_Model/mace-mh-1.model on the supercomputer. These are checkable suggestions, not silently confirmed values. The local dialogue must keep the cluster model path as remote metadata; it must never try to load that file on the local computer. Boundary, phase-to-file mapping, budget, convergence criteria, and stage settings still require explicit review. Empty DFT parameters are accepted only when parameter_source is explicitly atomate_defaults.

If a configured session is missing but the state already has run data and no recoverable confirmed configuration, startup fails closed rather than replacing the state. A missing ledger is created only after a confirmed configuration provides system.boundary. Mother-structure paths can be kept in system.phase_references; a separate phase-reference JSON file is optional.

## Manual upload output

Set `manual_upload.enabled` only after replacing `YOUR_PACKAGE:YOUR_TASK_EXECUTOR`
and providing the actual stage profiles required by the destination cluster.
For DFT stages, `dispatcher_factory` must produce the existing atomate-based
dispatcher; the exporter refuses to invent VASP inputs. Approved tasks with a
valid budget reservation are materialized below `batches_directory`. Copy the
whole generated batch directory to the cluster, verify `SHA256SUMS`, inspect the
script and inputs, and submit manually. Result recovery remains a separate,
explicit local operation.
