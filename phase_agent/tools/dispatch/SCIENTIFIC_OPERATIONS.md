# Scientific operations and node contracts

The Agent chooses one `operation` from `scientific_operations`, keeping its existing canonical `tool`. The operation supplies fixed routing parameters; conflicting tool/mode/action values fail before side effects. Older proposals without `operation` retain their routing and receipt identity.

| Operation | Existing tool | Routing parameter |
|---|---|---|
| generate_branches | generate_branches | confirmed generation plan |
| prepare_mlip_relax | prepare_local_batch_files | mode=relax_inputs |
| allocate_mlip_mc | allocate_mc_bohb | approved MC allocation |
| prepare_mlip_mc | prepare_local_batch_files | mode=mc_inputs, all pending MC tasks |
| select_dft_candidates | select_dft_candidates | recorded QBC/Ehull and categorical decisions |
| prepare_dft_inputs | prepare_local_batch_files | mode=dft_inputs, registered task IDs |
| prepare_mlip_training | update_mlip | action=RETRAIN_MLIP, prepare_inputs_only=true |
| activate_mlip_candidate | update_mlip | action=ACTIVATE_CANDIDATE_MODEL, candidate_model_version |
| prepare_model_refresh | prepare_local_batch_files | mode=model_refresh_inputs, frozen refresh preview |
| check_convergence | check_convergence | current recorded evidence |

Example proposal (still requires normal validation and approval):

```json
{"operation":"prepare_mlip_relax","tool":"prepare_local_batch_files","task_key":"relax-inputs:unique-plan-id","target_ids":["registered-structure-id"],"parameters":{},"budget":0,"reason":"Prepare the reviewed Relax plan","operation_context":{"config_version":"current-confirmed-version","model_version":"current-active-model"}}
```

The example budget must be replaced by the actual existing preview; a zero value is not a budget bypass. IDs and versions are references to existing project facts, never model-generated scientific values.

The action graph exposes `operation__<name>` nodes after the existing refresh barrier, proposal preparation, approval and validation. Grouped production graphs expose these inside the existing tool groups. Cached graphs declare operation nodes independently of runtime handler injection; the executable registry catalog excludes unconfigured handlers. No new executor, service or submission path is introduced.

`operation_context` is optional for old proposals. Supplied config/model versions must match the actual execution context and are checked both in action validation and immediately before dispatch. Candidate activation binds the current model as input; the candidate model version is a separate target parameter.

`operation_outcome` records operation/tool/task key, input config/model versions, resulting model version when available, dispatch status, actual result status, task IDs and submission fact. `dispatch_status=completed` only means the adapter returned; `result_status=prepared` or pending task results do not prove computation completed.

Recovery and hull construction remain the existing observation nodes `collect_and_reconcile` and `scientific_feedback`, with serializable `observation_contracts` on the compiled project graph. They reconcile recorded paths, update ledgers and model-isolated hulls before Agent decisions; they do not launch new calculations. Low-level scientific implementations remain callable Python modules. Expensive jobs keep the prepare/wait/recover stages rather than blocking one node.

Approvals, model-refresh barriers, task reservations and persistent execution receipts continue to govern every scientific operation. Existing projects need no snapshot or task migration. New code loads on the user's next manual service start.
