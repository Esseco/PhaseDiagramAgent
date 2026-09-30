# Open WebUI integration

## Debug-mode Relax handoff

When the confirmed ledger already has deduplicated electrostatic initial
structures, the next proposal prepares `relax_and_feature` inputs for every
registered structure without a completed or prepared Relax task. Existing tasks
keep their IDs and budget reservations. New tasks from the same branch share
one upload batch directory and have separate `GPU.sh` scripts. The generated
`RELAX_UPLOAD_PLAN.json` groups all pending tasks by branch, including older
single-task batches. It does not regenerate branches, run MACE locally, or
submit a cluster job. After approval, inspect the generated directory under
the configured `storage.paths.upload_batches`: each bundle contains copied
`initial.vasp` files, per-task JSON, a manifest, checksums, and a Slurm script
using the confirmed remote worker command and the MACE environment profile.
Only upload and submit after checking the remote model path, project import
path, resource directives, and cluster dependencies. The stored budget covers
the individual Relax tasks; repeating preparation does not reserve them twice.
An older pending `run_calculation_stage` proposal is revised to this safe
preparation step before it can be approved in debug mode.

## One-click local project launcher (Windows)

Double-click `start_phase_agent.cmd` in the repository root. It opens a small
project picker under `py1`. Choose **继续所选项目**, **新建独立项目**, or
**添加已有项目** (select its `open_webui_runtime.json`). A new project creates
only its own runtime JSON, resumable config session, and annotated science
draft in the chosen workspace; no scientific task starts. Edit
`search_config.project.json`, then tell the Agent `读取配置 JSON 并继续`. The first
GUI workspace selection replaces the duplicate path-confirmation chat step;
scientific settings still need Agent/program review and user consent.

The launcher displays the chosen workspace, config status, and whether the
project state contains approved long-term or recent short-term memory. It
rejects a second project pointing at the same state, and refuses to attach to
an already occupied Agent port because it cannot prove which project owns it.
It starts the local Agent endpoint, or safely reconnects when the same project's
Agent already owns port 8765 and accepts the stored connection token. A different
or unidentifiable process is never reused. The launcher prefers Open WebUI at
`open_webui_url` (default `http://127.0.0.1:3000`). If it is not running and no
startup command is configured, the launcher opens its own lightweight chat at
`http://127.0.0.1:8765/phase/chat`; the Agent, configuration conversation and
approval rules are the same. Click **从剪贴板连接** once on that page. New projects
therefore remain usable without installing or restarting Open WebUI.

If you use Open WebUI, set `open_webui_start_command` as an argument list and
`open_webui_workdir` when required by its actual installation. Shared desktop
startup settings can be kept in
`%LOCALAPPDATA%/PhaseSearchAgent/open_webui_startup.json`; explicit project
settings take precedence, and newly created projects inherit the shared
settings. A successful one-shot startup command may exit before the web page is
ready; the launcher waits for the page. It does not invent Docker or installer
commands. If Open WebUI startup fails, the launcher opens the local chat and
shows the startup error in its window. The DeepSeek key stays in Windows Credential
Manager. Separate local connection and control tokens are created there once;
the connection token is copied to the clipboard on launch for Open WebUI's
one-time OpenAI-compatible connection setup. Do not paste the control token
into Open WebUI. The endpoint log is `agent_server.log` in the chosen workspace.

An Open WebUI **new chat is not a new project**. Its account Memory may also
be injected across chats. For this Agent, turn off the model's Memory injection
in Open WebUI and use the project's reviewed long-term memory and state-derived
short-term memory instead. Change projects with the launcher after stopping the
old local Agent service; never point two projects at one state/ledger. Existing
runtime JSON and scientific result files are not overwritten by the launcher.

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

On first setup, the assistant asks for the local workspace root and Agent model
version. The user can provide both in one message or separately. It previews
only those choices and the settings JSON path, then waits for “确认”. That
confirmation creates the annotated JSON and lists the fields to complete. The
user edits it and sends “读取配置 JSON”; the Agent reviews it and the program
validates its fields. If both pass, the user replies “同意” to save the config
snapshot and enter the search workflow. No scientific calculation is dispatched
by setup or config approval. The LLM model is local runtime configuration, not a
material/search parameter.

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
- `deepseek.model` selects the local Agent's DeepSeek model. Supported IDs are
  `deepseek-flash` (V4.1 Flash) and `deepseek-v4-pro`. During first setup, select
  the version alongside the workspace path (or omit it to keep the current
  version); both are committed only after “确认”. This updates the
  local runtime JSON, not the scientific search config or cluster files. API
  keys remain environment-only.
- optional `manual_upload`. When enabled, compatible Relax tasks are grouped up
  to 100 per job and MC tasks up to 20 per job. Submit the batch root's `GPU.sh`
  once; each MLIP task has its own directory, `run_mlip_task.py`, result and log.
  DFT remains one task per job, with atomate inputs and `GPU.sh` in that task directory.
  The parent directory keeps `manifest.json`, an immutable snapshot, `SHA256SUMS`,
  and `UPLOAD_AND_SUBMIT.md`. No array submit script or `sbatch` invocation is made.
  Its `worker_command` must specify an `--executor` reference for MLIP tasks.
`manual_upload.submit=true` is rejected.

After a Relax input batch is prepared in debug mode, the Agent pauses and lists
the batch directories. Upload each complete batch directory, submit its root
`GPU.sh` once, then copy `result.json` and `task.finished.json` back into the
same local task directory. On the next “继续”, the local runner verifies the
markers, records each result and settles its budget once. Until results return,
it reports the remaining tasks instead of proposing the same preparation again.

Do not put API keys, passwords, SSH secrets, structures, or guessed cluster
paths in this file. Secret-like JSON fields are rejected. The DeepSeek key is
read from the local Windows Credential Manager or, if explicitly set, the
`DEEPSEEK_API_KEY` environment variable. Open WebUI bearer tokens remain local
environment-only.
If a configured config-session file is missing, recovery is allowed only when
the configured state explicitly contains both `confirmed_config` and a config
version. A missing ledger can be created only when the confirmed config has a
real `system.boundary`; the runtime will report the exact missing setting rather
than inventing a system, structure, cluster, or compute adapter.

Start the endpoint under the project's `py1` environment:

```powershell
$env:OPENWEBUI_TOOL_TOKEN = "<a random local token of at least 16 characters>"
$env:OPENWEBUI_CONTROL_TOKEN = "<a different random local control token>"
python -m run.open_webui_api
```

The endpoint can start without a DeepSeek key. Open the printed local setup URL
(`http://127.0.0.1:8765/phase/setup`) and enter the key there instead.

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

`OPENWEBUI_TOOL_TOKEN` and the distinct `OPENWEBUI_CONTROL_TOKEN` remain local
endpoint access tokens. The DeepSeek key no longer needs to be entered in a
terminal: start the local service, open `http://127.0.0.1:8765/phase/setup`,
paste the key, and choose “测试并启用”. The page makes a small JSON-mode test
request first, then saves the key in the current Windows user's Credential
Manager and activates it without restarting the service. If `DEEPSEEK_API_KEY`
is already set, that environment value takes precedence. The DeepSeek key is
never written to runtime JSON, state, task payloads, or remote manifests.
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

On a clean first start, the service creates only a small resumable dialogue session at `config_session_path`. It asks for the local workspace root and Agent model version (V4.1 Flash or V4 Pro), together or separately. It previews those two choices and the JSON path. Reply “确认” to create the annotated JSON and receive separate required-field and recommended-review lists. After editing the file, send “读取配置 JSON” for review only; after both Agent and program checks pass, reply “同意” to save the immutable config snapshot and enter the search Agent. Alternatively, “读取配置 JSON 并继续” is conditional consent to do that immediately if both checks pass. Either route enters search advice only; it does not submit scientific tasks.

If no DeepSeek key is available, the Agent replies with the local setup-page link. Open it in the same computer's browser, paste the key once, and test it. The setup endpoint is restricted to loopback clients; the page does not echo the submitted value or write it to web storage. Browser password-manager behavior depends on browser settings. A successful test stores the key in Windows Credential Manager and makes the existing Agent usable immediately. Replacing the key follows the same test-before-save flow.

After confirming the workspace root, the service creates `<workspace_root>/search_config.project.json` (or the filename configured by `editable_config_draft_path`; it is always placed under the selected root). This short file inherits the version-pinned `layered-oxide-v1` defaults. Edit `config` for the main project settings and `overrides` only for advanced fields; do not copy the full default configuration into the file. The file accepts JSONC comments. Set the mother-structure directory once; phase files are resolved as `<directory>/<phase>.vasp`. For layered supercells, choose indices in `selected_recommendation_indices` or provide actual integer matrices under `additional_containment_matrices`; the two recommendation matrices are examples, not automatically selected. The import checks schema, matrix shapes, template version and secret-like fields. Legacy `search_config.draft.json` remains readable and is never overwritten by the new format.

The same JSON includes `config.storage`: the selected `workspace_root` and relative destinations for config snapshots, state, the main ledger, structures, phase diagrams, QBC results, approvals, task work and upload batches. New workspaces use the standard relative layout shown in the file. Relative output paths are constrained to remain under the workspace root. If you select a new root while existing state or ledger files remain elsewhere, startup stops rather than silently moving or ignoring them; manually migrate and verify those files, or point the entries back to their original locations.

The dialogue session remains at the bootstrap path selected by `config_session_path`, so the local service can resume setup before the chosen workspace is loaded. The editable JSON and immutable confirmed config copy are stored under the selected workspace root; snapshots go to `workspace_root/config_snapshots/` by default. Runtime setting `editable_config_draft_path` controls only the JSON filename, not its directory. A confirmed snapshot stores the fully expanded settings and the default-profile identifier and digest, so later template edits cannot silently change an existing run.

After saving the file, send `读取配置 JSON`. The local service reads the configured file, resolves `<phase>.vasp` mother structures, generates per-phase H matrices when `H_generation` is enabled, and validates the resulting full configuration before asking the Agent to review it. The Agent receives the verified source path, file digest, allowed phase union and H counts; it never needs a hand-copied H list. Import replaces the entire unconfirmed in-memory draft so old conversation fields cannot survive. Ordinary configuration chat can inspect the current short file without importing it, and Agent suggestions do not silently edit that file. When both checks pass, reply `同意`; `读取配置 JSON 并继续` gives conditional consent. Restart the local Agent service after updating its code or runtime path; an already-running process keeps its previous imports and settings. Legacy long drafts remain untouched. Either confirmation route saves `config_snapshots/<config_version>.json` and enters search advice without submitting calculations.

If a short config's `profile_digest` is older than the installed code, the service uses the saved full config session as the migration baseline, preserving inherited values while reading. An explicitly requested config write rebases the short file to the current profile and keeps the original beside it as `*.pre-profile-migration.bak`; without a saved full baseline, it stops with a recovery message instead of guessing defaults.

The first-run draft includes candidate setup hints from the project owner: local mother structures under E:/0-FM-PhaseDiagram/InitFile/Struct and the mh-1 model at /data/home/lichaoyue/Py-lzy/MLIP_Model/mace-mh-1.model on the supercomputer. These are checkable suggestions, not silently confirmed values. The local dialogue must keep the cluster model path as remote metadata; it must never try to load that file on the local computer. Boundary, phase-to-file mapping, budget, convergence criteria, and stage settings still require explicit review. Empty DFT parameters are accepted only when parameter_source is explicitly atomate_defaults.

If a configured session is missing but the state already has run data and no recoverable confirmed configuration, startup fails closed rather than replacing the state. A missing ledger is created only after a confirmed configuration provides system.boundary. Mother-structure paths can be kept in system.phase_references; a separate phase-reference JSON file is optional.

## Existing run configuration migration

Restarting with an unchanged confirmed configuration reuses its existing snapshot; a repeated confirmation of identical effective parameters does not create another version. A plain “继续” resumes the saved run and is not classified as a configuration edit. If a newly confirmed revision changes only allowed generation/MC scheduling limits, an idle run can rebind to it on startup; completed tasks keep their original version labels. Active tasks, decreases in resource caps, and scientific-setting changes are not automatically rebound.

When a populated run is bound to an older snapshot, a direct “批准迁移” message is handled as a configuration-migration approval, never as approval of a scientific action. It permits non-decreasing resource limits and consistent positive finite values for the three `mc_step_cost_factor` estimates. Legacy snapshots that omit the factor use the historical one-MLIP-relaxation-per-step basis (effective factor `1.0`); for example, changing that legacy estimate to `0.1` is a real cost-policy change and is recorded as such. The confirmed `mc_policy.second_segment_enabled` change from false to true can use the existing idle generation-policy rebind. Any MC cost-estimate migration requires explicit approval and no active task, reservation, or pending execution policy. All other settings must be identical; decreases in resource limits, changes to the MLIP head, relaxation parameters, phase boundaries, or other scientific settings are rejected with changed fields listed. Migrations are saved as audit events and old tasks/results keep their original config-version labels. A migration-only reply does not submit or run tasks.

## Manual upload output

Set `manual_upload.enabled` only after replacing `YOUR_PACKAGE:YOUR_TASK_EXECUTOR`
and providing the actual stage profiles required by the destination cluster.
For DFT stages, `dispatcher_factory` must produce the existing atomate-based
dispatcher; the exporter refuses to invent VASP inputs. Approved tasks with a
valid budget reservation are materialized below `batches_directory`. Copy the
whole generated batch directory to the cluster, verify `SHA256SUMS`, inspect the
script and inputs, and submit manually. Result recovery remains a separate,
explicit local operation.
