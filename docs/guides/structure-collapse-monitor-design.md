# Structure collapse monitoring

Goal: monitor consecutive maximum atomic forces during each relaxation. On an increase greater than 1 eV/Å, compare the initial and current structures with Process_MLIP M3GNet embeddings; a difference-vector L2 norm greater than 1 pauses the workflow.

Reuse: Process_MLIP.direct_sampling.features.M3GNetEmbedding, existing optimizer callbacks and workflow wait/result contracts.

Scope: Process_MLIP relaxation, Process_AL_MC relaxation, phase_agent MLIP workers and result/wait handling, configuration and targeted tests. Inspect exact integration points before changes.

Risks: compatible M3GNet checkpoint/environment must be available on the execution host; remote batches require evidence transfer and propagation into a durable project pause, without claiming that unrelated already-running jobs are stopped. Nonfinite values or missing dependencies must be explicit failures.

Verify using synthetic forces and fake feature extractors, local and remote result propagation, MC and standalone relax, and blocked hull publication. Do not run real projects or jobs.

Implementation status: detector and synthetic unit tests exist in E:/Py-Code/Process_MLIP; optimizer integration, remote result transport, batch stop and project-wide persistent pause remain unfinished. The existing result protocol converts exceptions to ordinary failures and the batch runner continues, so do not present the detector as an active safeguard. Verify availability of the M3GNet checkpoint and compatible TensorFlow environment on each execution host before enabling.

## Follow-up: catastrophic-forgetting integration (unfinished)
# Collapse and catastrophic forgetting integration

Goal: monitor per-step maximum atomic force increases >1 eV/Å, compare initial and trigger structure by M3GNet difference L2, and pause on score >1. At recovery, Reliability assesses the complete set of distinct structures actually relaxed in the current round. If at least 10% meet the collapse indicator, DIRECT sampling is mandatory, with prior DFT structures used only as coverage reference and at most 100 new structures selected for DFT supplementation. Strategic may assess the subsequent training route but cannot veto the mandatory sampling step; existing authorization rules for DFT submission continue to apply. Incomplete evidence cannot clear the round.

Reuse: Process_MLIP collapse_monitor and direct_sampling; ASE callbacks, Process_AL_MC optimizer, cumulative DFT records, phase_agent result collection and strategic action/approval protocol.

Work: instrument relaxers and MC; preserve start, triggering structure, evidence and all required candidates; propagate explicit collapse through remote batch transport and durable wait/hull guard; aggregate distinct structure identities with explicit denominator; register a mandatory reliability-triggered DIRECT action with immutable input provenance and selection cap; add synthetic tests and user documentation. No real projects or jobs.

Risks: M3GNet and checkpoint must be available in each compute environment. Concurrent remote tasks cannot be instantly stopped merely through a local wait state. DIRECT requires sufficient distinct features. Existing result export excludes most structures, so exact user-requested sampling inputs require an explicit transfer contract. Missing evidence/dependencies must block, never be treated as safe.

Current implementation: Process_MLIP collapse monitor, optimizer callbacks for standalone and MACE MC, a distinct-structure 10% assessment, and DIRECT feature-space reference filtering have targeted tests. No worker automatically installs the callback and no agent workflow currently consumes the evidence. The following remain mandatory: durable result identity and archive transfer, fail-closed pause before any hull publication, batch recovery aggregation including incomplete results, reliability evidence binding and mandatory DIRECT execution under existing authorization, complete source structure enumeration and historical DFT extraction, and authenticated DIRECT invocation. The 10% denominator must be the complete identified calculation scope, and incomplete evidence must not be presented as a complete proportion. CHGNet's current backend cannot install a monitor and must not silently proceed as monitored. No real-project execution was performed.


## Current mandatory policy
Reliability computes the fraction from all distinct structures actually relaxed in the current round. At or above 10%, DIRECT sampling and DFT supplementation are mandatory. Strategic receives the observed fraction, QBC evidence, budget and candidate provenance to choose the number of new DFT candidates (at most 100) and to give higher QBC candidates more allocation as the fraction rises. This is a decision guideline, not a fixed mathematical rule; absent or invalid QBC values must remain unknown, not be fabricated. Historical DFT structures are reference-only. The mandatory rule does not bypass existing scientific action and DFT submission authorization. None of this agent integration is active yet.


The mandatory DIRECT and QBC allocation rule applies only when the complete round collapse fraction is at least 10%. Below 10% it does not apply. Incomplete evidence remains unresolved rather than being treated as below threshold.

## Implementation update (2026-11-05)

The earlier implementation-status paragraphs record the original plan, not the current code status. The reliability evidence module now evaluates each recovered epoch-refresh or Relax–MC scope against its active model and complete, distinct relaxation observations. A proven failure fraction of at least 10% requires the existing reference-aware DIRECT workflow before ordinary DFT selection or a model update can proceed. Incomplete monitoring remains unknown; an explicit user instruction may still request DIRECT. The sampling pool consists of all verified final frames in that round, including extrapolation failures and other results; missing or unverified final frames are reported rather than silently omitted. The DIRECT action selects at most 100 representatives, records exact input and feature-model hashes and a reviewable plan, and does not create or submit DFT jobs. Historical validated DFT frames are coverage references. Extrapolation failures remain DFT candidates but are excluded from phase diagrams.

The monitoring changes span Process_PhaseDiagram and Process_AL_MC in E:/Py-Code. They must both be deployed to the execution environment before a remote run can provide the complete new evidence, and a running service loads source changes only when the user next restarts it. Historical endpoint comparisons do not reconstruct missing step observations. If analysis files were relocated without updating registered paths, repair the project through its Agent and revalidate checksums before sampling; do not silently substitute a similarly named file. NiCo's relocated endpoint evidence was repaired through the Agent and revalidated; 100 representatives from its complete 1,293-frame round were generated for review.

## Requested recovery cycle (implementation in progress)

For an epoch refresh or Relax–MC round, once Reliability identifies a large extrapolation failure and Strategic agrees, sample the entire verified round using DIRECT. The affected epoch's diagram must be marked invalid and excluded from latest and subsequent lineage; preserve its files and the previous valid diagram for provenance. Review the DIRECT selections, obtain valid DFT energy and force labels, and fine-tune. DIRECT structures alone are not labelled training examples. After the replacement model is validated and activated through the existing lifecycle, refresh the last valid phase-diagram lineage with it, following the established full-data/latest-valid-per-structure and D/H refresh rules. Incomplete observations must remain unknown; an explicit user direction can initiate the recovery even when the automatic fraction is unknown.

A complete-monitoring high-failure result now blocks the affected current-model diagram from being published by the phase update function. It cannot infer a high failure fraction from incomplete step evidence. An explicit Agent operation can quarantine an epoch refresh, retain its historical files, restore the checked previous publication, and record the selected frames and invalidated model. The recovery gate requires matching, completed, trainable DFT labels before retraining; a replacement model's refresh begins from the prior valid diagram and excludes quarantined model records. These pieces have synthetic regression coverage; NiCo's epoch-refresh quarantine was applied through the Agent. Automatic handoff from DIRECT to this operation, the corresponding Relax–MC recovery and transfer of every relaxed MC frame, and an end-to-end real DFT/training/activation cycle remain unverified or unimplemented. Do not infer those steps from the prompt text. Track remaining work in the repository's temporary plan.


## DIRECT 恢复和 MC 逐帧回收（2026-11-05）

已确认大量外推的刷新或 Relax–MC 轮次，DIRECT 对本批全部可核验弛豫末态采样（含失败帧），目标 100，去重后不足时报告实际数；历史 DFT 仅作参考。该恢复批次不强制追加 GS，普通 DFT 选点仍须全选未验证 GS。DIRECT 完成后，原 Agent 生命周期调用已有隔离入口恢复上一有效相图；隔离失败时保留采样回执并阻止后续选点及模型更新，不重算采样。失效模型不能自动覆盖 latest，历史作废图保留。

新 MC 运行将逐次弛豫的末态存于远端并记录 SHA256；完整轮次确认需要 DIRECT 时，既有 results.zip 只增加这批 DIRECT 所需末态。未达到阈值的正常批次不传这些候选帧。初始 Relax 和各次 MC 弛豫均属于采样范围；帧缺失、哈希不符或监控不完整不会被当作已核验。已有旧任务若没有这些逐帧文件，不能重建为完整证据，需要通过原流程补充可核验数据。

变更涉及本仓库和 E:/Py-Code/Process_AL_MC，必须通过各自版本管理及远端同步后才对新 HPC 批次生效；正在运行或已冻结的旧任务不会加载新代码。合成测试不能证明真实 HPC 全链路或 DFT 与微调完成。Relax–MC 正式回收仍要求整批 completed；单项失败的批次不能假称完成，完整失败证据的单独登记/补算流程仍需核验，自动恢复不能覆盖缺失证据。首次 epoch 若不存在可核验的上一有效相图，自动隔离会明确阻断等待处理。
