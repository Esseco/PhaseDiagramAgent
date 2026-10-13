# Shared feedback-tier MC continuation

The unified workflow reuses the existing MC scheduler, State, task/approval/preview and HPC interfaces. Initial selection is unchanged: protected hull ground and near-degenerate competitors, phase/composition/structure coverage, favor low Ehull plus about10%exploration. Initial Ehull<=10,<=20,<=40,>40meV/atom maps to100/10,80/8,60/6,40/4steps/patience.

## Same feedback standard for both additions
After each completed wave update the existing phase diagram and classify each branch using current-model latest Ehull and previous-segment energy improvement per atom. Missing improvement is explicitly unknown, conservatively classified with weak; never fabricated as measured zero.

| Latest Ehull (meV/atom) | Strong >3 | Moderate1-3inclusive | Small0.5<=improvement<1 | Weak<0.5 |
| --- | --- | --- | --- | --- |
| <=10 |high|high|medium|low|
| >10 to20 |high|medium|low|exploration|
| >20 to40 |medium|low|exploration|normally omit|
| >40 |low|exploration|normally omit|normally omit|

| Resource tier | New segment maximum steps | Patience |
| --- | --- | --- |
| high |200|20|
| medium |150|15|
| low |100|10|
| exploration |50|5|

Both additions use this identical table; budget can decrease or increase with feedback. It no longer doubles with segment number. At most3segments per branch/model. Restart from the prior result structure with a recorded seed; no exact Markov-chain checkpoint claim.

## Protected90/10selection
Protection reuses convergence.phase_competition_gap_ev_per_atom(default0.003eV/atom):ground/near-degenerate candidates are retained regardless of improvement and receive at least medium resource. Their actual count can enlarge the batch.

Every valid high-tier candidate is mandatory in BOTH additions, with no candidate-count cap. Take the union of high-tier and protected candidates so protected high candidates are counted once. Half/reference counts,Agent branch_count and90%slot targets cannot drop a high-tier candidate. Enlarge the batch for all mandatory candidates and approximately10%lower-tier exploration where available. Medium candidates fill remaining main slots. Current-model validity,latest-wave membership,max2additions,inflight andduplicate gates still apply.

Ordinary retained-count reference stays max(10,ceil(previous completed wave/2)). Agent branch_count can propose a larger batch with coverage reasons. Roughly90%task slots select protected/high/medium;10%task slots sample valid low/exploration. This is task-count allocation, not step or cost allocation. Exploration count rounds up (10tasks=>1,20=>2). If no exploration candidates, slots go to high/medium. If main candidates are scarce, useful valid low/exploration candidates can supplement above10%; preview records exploration_fill_count and displays the reason. Matrix-ineligible candidates never pad a batch. Without protected/main search or exploration improvement/coverage evidence, stop continuation and consider Branch/Relax or DFT; minimum10is a dispatch condition, not a mandate to search.

Random sampling records a seed and deterministically stratifies resource tier,phase,composition andH/Tstructure family. Identical evidence and seed reproduce the same plan independently of input-list order. Exploration rotates low and exploration tiers when both exist. Main priorities favor high then medium; within the same resource tier balance coverage, then Ehull/improvement. Only latest-wave survivors seed the next addition; eliminated branches are not revived. Fewer than10eligible selected tasks yields zero actions/no partial reservations. Active work, stale source/model/hull or allocation checksums retain existing barriers.

## Library default stopping threshold and deployment
Improvement thresholds above only classify whole-segment feedback. They do not override the backend stop threshold:HBtask parameters omit min_improvement;worker payload and constructor preserve Process_AL_MC defaults. Canonical task patience_steps supersedes stale inherited patience. Budgets remain advisory by default.

New work under old saved configs uses the approved shared effective runtime policy. Historical tasks/config versions/results/models are unchanged. Existing stale unexecuted approvals fail allocation validation and need a new preview. Deployment does not manage services; new source loads at the user's next manual restart. This update changes continuation selection/resources only; remote batch self-execution is a separate future capability.

## Offline verification
Test four exact improvement boundaries and all16matrixcells, same feedback resources on both additions, increasing/decreasing resources,90/10counts,seed reproducibility,stratified exploration,protected weak competitors,main-shortage exploration fills,no-value stop,min-task rollback,current-model/latest-wave scope,preview/dispatch parity,approval duplication andbothmode dry-run. Forbid all tool handlers/subprocesses in dry-run; verify zero newtasks,reservations,Slurm batches. No real service or HPC submission. Offline tests do not validate physical phase-diagram accuracy or the remote library's numeric default.
