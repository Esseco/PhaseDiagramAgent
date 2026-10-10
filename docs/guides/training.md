# 训练、验证与恢复

按下方目录定位需要的说明；源码入口链接保留在对应章节。项目科学参数以确认配置为准。

- [Training result handoff](#doc-training_handoff)
- [远端训练结果回收](#doc-remote_training_recovery)
- [原轮微调输入重生成](#doc-finetune_input_regeneration)
- [跨体系统一训练与评估](#doc-unified_training_evaluation)
- [Portable DFT training results](#doc-training_results)

<a id="doc-training_handoff"></a>

## Training result handoff

The existing scientific graph collects training artifacts, then advances a durable training_handoff in each remote_finetune_job. State is saved before returning a manual wait. Repeated continue reads returned files and preserves identical generated scripts. No scheduler calls or local model inference are performed.

1. Missing paths/hashes: generates inputs/GPU_manifest.sh and refreshes collect_training_results.py. Upload both to the original HPC inputs directory; submit sbatch GPU_manifest.sh. Return results/models.json.
2. Complete manifest: checks independent validation prerequisites. It never uses K-fold results or the full training set as an independent test.
3. Configured independent validation: generates validation.xyz, validation_request.json, validate_remote_training.py and GPU_validation.sh. Upload to original inputs; submit sbatch GPU_validation.sh. Return the request-specific results/validation-<request_id>.json.
4. Recovery binds the report to the request, model manifest and dataset hashes, checks finite metrics, then registers candidate_models using the existing validate_mlip criteria. Failed validation/rejected candidates return control to downstream Agent decision-making.
5. Successful candidates require an explicit chat command: 激活候选 <candidate-version> 原因：<review reason>. 拒绝候选 <candidate-version> 原因：<reason> records rejection. Generic continue/approve never activates. Activation rechecks returned artifact fingerprints, uses existing governance and marks model-dependent results stale. Next continue enters existing model-refresh approval, followed by round-budget comparison.

Job scripts reuse phase_agent/tools/remote/mlip_gpu_template.sh (the user's v100m3 template). Metadata operations also run on compute nodes. An optional confirmed-config remote_training_job_template may point to a compatible template with python3 run_mlip_task.py and conda activate mace placeholders.

Configure remote_training_validation in the confirmed scientific configuration (not only agent_runtime.json, whose unknown runtime keys are ignored):

```json
{
  "remote_training_validation": {
    "data_path": "E:/project/validation/independent.xyz",
    "data_version": "independent-v1",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "ranking_pairs": [[0, 1]],
    "criteria": {
      "max_energy_mae": null,
      "max_force_rmse": null,
      "max_critical_failure_fraction": null,
      "max_near_hull_ranking_reversals": null
    }
  }
}
```

Null thresholds are intentionally unconfigured; obtain scientifically justified user-confirmed limits. Energy MAE uses eV/atom and force RMSE uses eV/Angstrom. Ranking pairs are zero-based indices of independently labeled, same-composition near-hull structures, selected before seeing predictions. Active old-model metadata must include version, remote model_path and sha256 (mace_head if needed). Keep original inputs/_shared_data/train.xyz on HPC. The runner checks file hashes and rejects exact training-geometry overlap; the dataset owner must also exclude shared parent/trajectory provenance, as geometry comparison cannot establish that independence. Evaluation covers old and new main models, not a full committee calibration or dynamics-stability benchmark. Any failed/nonfinite prediction fails the job; no structures are silently dropped and no partial pass is returned.

Each handoff transition enters training_handoff_history and the existing evidence-verified memory-candidate pipeline. These are factual observations, not automatically approved durable advice or estimated scientific gains. Do not overwrite live state externally; restart the service and use continue to recover and persist transitions.

Validation: py1 tests cover scheduled metadata repair, repeated/restarted handoffs, prerequisite waits, request identity, nonfinite reports, candidate registration, explicit activation, changed-artifact rejection, memory extraction, reply rendering and mocked remote evaluation/overlap checks. Actual HPC inference remains to be verified by a submitted job.

### Native graph persistence

Training handoff now runs in the discoverable `scientific_lifecycle/training_lifecycle` subgraph. Collect, manifest checking, metadata job preparation, validation prerequisites, validation job preparation, validation recovery/candidate registration and transition recording are independent graph nodes. Waiting for manifest return, validation return, explicit activation review or missing configuration uses native LangGraph interrupt. The project-specific SQLite checkpoint database is `workflow_state/langgraph_training.sqlite`, with one thread per state-path/training-job identity and a file lock. Graph state contains serializable business/configuration snapshots, never manager/client/runtime objects.

Continue resumes the interrupted node using fresh canonical business state and configuration, then collects and checks current artifacts again. The checkpoint does not authorize activation or override state.json. A crash between nodes restarts reconciliation before preparing idempotent local artifacts. Explicit activation/rejection still uses the existing governance entry; the next continue reconciles that decision and leaves the subgraph for normal model refresh and round-budget review. Terminal rejected/failed validations do not force repeated waiting. The whole scientific graph is not globally checkpointed by this change; the training lifecycle has its own durable checkpoint boundary, while other stages retain their existing persistence.

Tests use actual SQLite saver close/reopen, native interrupt snapshots, fresh-state resume, changed files, validation wait to approval wait to explicit activation, and production Studio child discovery. HPC inference has not been executed locally.

### 实现入口

[phase_agent/graphs/training_handoff_graph.py](../../phase_agent/graphs/training_handoff_graph.py)。

<a id="doc-remote_training_recovery"></a>

## 远端训练结果回收

每轮 `results` 必需包含 training.finished.json、models.json、kfold_metrics.csv、energy_comparison.csv、force_comparison.csv。模型本体保留在超算。

状态查询只读取并校验文件，不写业务状态。发送“继续”由科学生命周期回收节点登记结果。已回传结果优先于旧微调输入方案，不重复生成或提交。回收前后都不会自动激活模型。

校验包括完成标记、原模型版本、训练轮、委员会成员、唯一主模型、逐折成员、逐点数量、有限数值；模型路径必须是超算绝对路径，并记录SHA256。相对路径旧清单允许登记已回传指标，但标为 metadata_required，不能据此激活。可在更新过收集脚本的超算 inputs 目录运行 `python collect_training_results.py --manifest-only` 补清单，无需重训。

### 当前边界

本次接通的是回收、校验与状态展示，不是完整远端模型验证/激活通道。validation_required 表示需要独立验证，而不是验证已通过。现有训练器型验证接口尚未与仅远端路径的回收候选自动连接；生命周期在这里安全停止并说明原因，不重新建议微调输入。后续需要实现远端验证输入生成、同版本验证回收及独立激活审批，然后才恢复新模型结构刷新。

K折汇总来自回传CSV，仅供评估；不得冒充最终全数据主模型的独立测试结果。

### 验证记录

初版针对性测试15项通过。之后增补逐折完整性校验及对应测试数据；完整回归和该增补复测因工具审批额度耗尽未执行，不声称全量验证完成。未处理真实工作区数据。

[phase_agent/tools/local/recover_remote_training.py](../../phase_agent/tools/local/recover_remote_training.py)。

<a id="doc-finetune_input_regeneration"></a>

## 原轮微调输入重生成

训练参数/数据变化且已有未完成输入时，prepare_remote_finetune记录finetune_input_conflict。
“重新生成原轮输入”优先承接该训练冲突，或定位唯一未完成训练输入，不再按branch轮次枚举Relax/MC/DFT action。
明确MC、DFT、Relax、branch重做仍走各自旧流程；多个微调任务且无冲突锚点时不猜测目标。

先展示目录及方案并记录pending_finetune_regeneration，不移动或生成文件。
仅单独“确认重新生成原轮微调输入”执行；“拒绝”取消计划，不改变文件和任务。
执行前复核训练数据、配置、模型、job记录与文件内容指纹；变化要求重新展示方案。
已登记提交/激活，或输入目录含结果、日志、未知文件、链接时不自动替换。
人工仍需确认是否曾在超算提交，项目不能从本地输入记录证明远端作业未运行。

旧输入移入同级.inputs-backup-*目录；原编号目录用当前数据/参数重新生成，未提交、不训练、不激活。
目录不存在则无需移动，仍用原编号。失败恢复旧目录，失败新文件保留在.failed-inputs-*便于排查。
原始DFT、误差产品和其他计算阶段不受影响。替换目录必须是当前epoch下登记且已废弃的微调目录。

验证使用py1与临时fixture；未操作生产工作区文件。

[phase_agent/tools/local/regenerate_finetune_inputs.py](../../phase_agent/tools/local/regenerate_finetune_inputs.py)。

<a id="doc-unified_training_evaluation"></a>

## 跨体系统一训练与评估

不按结构数量自动选择方法。所有体系固定来源分组5折，至少5个独立组，且每折训练元素覆盖该折留出结构元素；不满足时报告原因、等待补充数据，不偷偷改成3折或随机留出。来源按branch/framework/structure ID分组，用户提供的来源标识应保证相关帧不会跨组。

正式committee全部合格数据参与训练；com_1是主模型，不另训main_final。默认4成员+5折评估共9任务。E0s estimated、AMSGrad、rms_forces_scaling、patience20保留已确认设置。

K折留出集参与早停/模型选择，指标名为grouped_cross_validation，并非完全独立测试。正式成员同数据validation仅训练监视，训练指标不能冒充泛化误差。

下一轮DFT回收复用原轮次冻结模型同帧对比链，evaluation_type=original_round_model_same_frame。只在确认样本未参与该冻结模型训练时，才能进一步解释为新数据/独立测试误差；不因回收成功自动宣称独立。旧CSV列和既有结果不改写。

本次没有改变生产state、处理实际数据或生成输入。模型回收/激活仍须独立验证与批准。

[phase_agent/science/training/prepare_mace_finetune.py](../../phase_agent/science/training/prepare_mace_finetune.py)。

<a id="doc-training_results"></a>

## Portable DFT training results

VASP results embed matched ionic-frame labels in `result.json.outputs`.
The existing result checksum and results-directory exporter cover this JSON; no
remote absolute path is needed to recover training structure or labels.

- `structure`: pymatgen Structure JSON (cell, species, coordinates).
- `training_energy`: raw frame `e_0_energy`, eV total, not Eform or Ehull.
- `forces`: N x 3, eV/angstrom.
- `stress`: optional 3 x 3, eV/angstrom^3, ASE tension-positive convention;
  converted from VASP compression-positive kbar.
- `training_frame_index`, `training_schema`, `training_ready`: provenance/status.
- `training_frames`: all valid electronically converged frames, original indices preserved.
- `rejected_training_frames`: explicit reasons for skipped frames.
- `xml_complete`, `ionic_converged`: completion evidence, separate from electronic convergence.
- `final_frame_index`, `final_frame_valid`: actual last ionic step, not the last
  surviving training example. A rejected actual last step cannot be replaced by
  an earlier valid step for phase classification, hull or final-frame metrics.

Each official DFT export also includes a separate `training.json` list, usable
by the existing MACE data preparation interface without unpacking outputs.
`training_file`, `training_frame_count`, and `training_checksum` identify and
verify it. The result envelope still embeds the same frames for local recovery.
The completion marker is published after training/structure/result files.

For single-task DFT submissions, the downloadable directory preserves the
original numbered name, e.g. `results/DFT-single-point-submission-0006_remote-000029/`.
New manifests point there. Local recovery supports these names and legacy
`00000-DFT-...` folders, while checking task identity and training-file checksum.
New DFT exports contain only result.json, training.json and task.finished.json;
CONTCAR/final_structure.vasp are not generated or exported. Phase identification,
Na-layer checks and MLIP evaluation read the embedded final-frame structure.
Classification hashes canonical structure content for JSON input; unchanged
JSON structures reuse the existing persistent scope-aware cache. Legacy files
remain readable with their original file hashes.
The one-off script prints version `json-only-final-frame-v4`, per-task frame
counts, and reports the input-task/submission/output mapping. Three legacy
files alone do not prove failed extraction: training frames were embedded in
result.json before the independent training file was added.

Recovery expands `training_frames` into `new_dft_records`; existing MACE preparation
converts the JSON structure and writes extxyz with REF_energy/REF_forces and,
when requested, REF_stress. Final-frame raw total energy remains in outputs for
phase-diagram use. Duplicate task recovery does not duplicate training records.

Missing/invalid labels mark training unavailable without fabricating values or
preventing valid energy-only phase analysis. Stress-required training rejects
missing stress. Electronically converged frames from unfinished ionic optimization
may be training records, but the unfinished calculation cannot enter final hull
or final-frame MLIP error metrics. Training `converged` refers to electronic
convergence; `ionic_converged` preserves optimization status.

This change does not retroactively modify old remote scripts or result files.
Existing jobs need the updated project parser available on the remote machine
and result re-extraction from their retained vasprun.xml (possibly compressed).
Energy/CONTCAR-only old results cannot reconstruct forces or stress.
Per-frame filtering follows the inspected py1 CHGNet `parse_vasp_dir` logic:
electronic step count must be below NELM. XML-contained structure/e_0_energy/
forces/stress are used directly, without requiring OUTCAR/OSZICAR or CHGNet.
Stress conversion is explicit; raw CHGNet/VASP kbar must not be mistaken for
the ASE eV/angstrom^3 convention used here. Incomplete XML is reparsed with
partial-data support and explicitly marked incomplete, never completed.
Each training frame has a task+frame data_id; frames from one branch remain in
one split group and are not collapsed by structure_id during dataset deduplication.

### Official workflow integration

`run_workflow` now supplies a default same-frame evaluator through the existing
py-mace subprocess. It performs prediction only, never re-relaxes the DFT frame.
It uses the configured main model (committee main index when applicable), and
requires the DFT task model version to match. Missing model/environment or
prediction failures are explicit `not_evaluated` records, not fabricated errors.
An explicit runtime evaluator still overrides the default.

Scientific feedback retains task summaries in `dft_dataset_records`, frame history
in `dft_training_records`, adds
training-ready data to `new_dft_records`, and keeps paired evidence in
`dft_mlip_comparisons`. Existing phase identification and hull refresh remain
authoritative. Phase identification is reused in dataset exports.

Under the configured phase-diagram directory, generated products are:
`<MLIP-version>/dft_rounds/Search-group-0001/DFT-round-0001_<operation>/`.
This folder contains `training.json`, `dft_records.json`, `mlip_dft_metrics.json`,
`energy_comparison.csv`, `force_comparison.csv`, and `metrics.csv`.
The model-version directory also has `round_metrics.csv` for all its DFT rounds,
beside the existing MLIP `phase_diagram.csv` and `history/`.
The cumulative DFT hull remains at `dft/phase_diagram.csv`, with a separate
DFT energy basis; predictions and DFT reference energies are never mixed in a hull.

Scope uses the saved upload operation/search group/parent Relax lineage.
Readable group/round names come from saved task paths or the upload operation
registry, not remote submission numbers. Missing evidence is labelled
`unassigned` with a stable scope hash; conflicting lineage/destinations raise
an explicit error before publication. Legacy `dft_results/<version>/...` files
remain untouched; the next feedback call exports saved products in the new layout.
Unchanged files are not rewritten; duplicate recovery is not re-evaluated.

Energy CSV rows hold raw DFT/MLIP total eV and per-atom eV/atom, signed
`MLIP - DFT` errors, absolute errors, task/structure/branch identifiers,
Na/O2, cached actual phase and the actual final frame index.
Force CSV rows hold one atom/Cartesian component per row, with a zero-based
atom index and original element/order, signed DFT and MLIP forces in eV/angstrom.
Use `dft_energy_eV_per_atom` versus `mlip_energy_eV_per_atom` for energy parity,
and `dft_force_eV_per_A` versus `mlip_force_eV_per_A` for force parity.
Only `comparison_status=completed` rows are paired points. Missing predictions
remain blank with a reason, never zero. Export reads saved predictions only,
does not re-run a model or classify structures.

The metrics CSV and JSON share statistics calculated from exactly these paired
rows. Pending tasks and missing comparisons are explicit coverage counts;
partial recovery does not prevent publishing valid pairs but is labelled `partial`.

Energy MAE/RMSE are reported as total eV and per-atom eV/atom, equally weighted
per structure, without fitted offsets. Force MAE/RMSE average all atomic Cartesian
components. RMSE is sqrt(mean squared error), in the same units as MAE.
MSE is not exported. Incomplete prediction coverage is listed.
These are validation errors of the task's model, not committee disagreement.

Existing finalized result files are not automatically regenerated; old jobs still
need re-extraction or the independent recovery script. If the task model is no
longer configured, matching-version evaluation must be provided explicitly.

### Fe/Mn 磁矩诊断

新 DFT 解析在 `result.json.outputs.magnetic_moments` 保存最后 OUTCAR 局域磁矩：全部元素、逐原子索引、标量/矢量、μB、来源和最后帧索引；远端不做范围验收、不改 checks_passed，不因磁矩异常拒收。文件回传/身份、校验和与预算回收维持原流程。

Agent 先按最终结构识别/复用相，再生成 `magnetic_check` 与 `spin_state_check`；完整原数据和诊断进入 `dft_records.json`，原子清单进入 `magnetic_moments.csv`，对话说明合理范围、实际合理原子统计、异常任务和详细清单。只判定 Fe/Mn，不要求 Na 必须非零；其他元素及非层状体系只保存磁矩而不检查。范围为 Fe |m|=3.5–4.5、Mn |m|=3–4 μB，是经验标准，不是基态证明。旧 magnetic_check 内嵌磁矩继续兼容。

层状范围不再依赖固定 system_id 名称；最终明确相优先于项目提示，配置可显式提供 is_layered_oxide。含 Fe/Mn/O 而最终范围尚无法确定时记 unknown 并暂缓科学使用，不将 unknown 当成非层状豁免。结构内容与分类条件未变时仍使用现有相识别缓存。

局域投影之和不等同总体系磁矩。诊断只对应最终计算帧，不代表较早帧的磁矩标签，不新增逐帧磁矩训练目标；电子收敛帧仍按原流程提取。

#### 读取后的科学使用标准

对明确层状氧化物中的 Fe/Mn，上述规则是本地科学使用门槛，不是下载/回收拒收条件。`outputs.spin_state_check` 保存 applies/status/criterion/原因；通过称为“符合预期自旋态”，`magnetic_ground_state_proven=false` 明确没有比较多个磁性初态/磁序的能量，不能当作严格基态证明。

`passed` 方可按原收敛与其他校验进入有效阶段历史、相图、训练和末帧误差评估。`warning` 对应自旋验收 rejected；缺失/非法、SOC/非共线、末帧或原子顺序不匹配为 unknown，均设 `checks_passed=false` 并暂缓科学使用。其他元素/非层状体系为 not_applicable，不额外要求磁矩。已有其他质量检查失败时，自旋通过不会覆盖为成功。

计算执行的 status、converged、实际成本和原能量不改，仍正常回收/入账；台账 calculation_attempts、原始结果及 dft_records.json 保留异常和标签。不合格帧保留于内部 dft_training_records 作审计，但不进入 new_dft_records、可用 training.json、微调触发或 MAE/RMSE 样本。较早帧并未逐帧证明基态，整项任务按末帧标准决定是否允许训练；要逐帧自旋筛选需另外采集每帧证据。不会自动改 INCAR 或提交修正任务。轮次简表分列 dft_spin_passed/rejected/unknown；energy/force_comparison.csv 增加 spin_check_status，不混淆运行成功与科学验收。

远端生成原始磁矩需同步更新项目代码与 Py-Code/Process_Vasp；旧 JSON 没有磁矩时本地无法恢复，无须回传整个 OUTCAR。部分回收提示按当前轮显示磁矩缺失、未评估、合理、异常、未知与不适用数量，不隐去缺少证据的旧结果。只有结构、磁矩证据和适用条件一致才复用诊断指纹；CSV 内容不变不重写。独立 training.json 中的 task_final_magnetic_moments 是任务最后帧元数据，不误称为每个早期训练帧的磁矩标签。

#### 已回收任务补传磁矩

重新提取同一任务并回传 result.json、匹配其校验和的 task.finished.json，以及引用的 training.json。回收器仍验证原 task 身份、文件校验和、结构/E/F/应力/最后帧一致性；仅补充磁矩，不覆盖计算结果或成本。冲突标签显式拒绝，不靠改 task ID 规避。原始结果和预算结算不重复入账，阶段结果 ID 保持原值。

下一次正常 Agent 回收会刷新缺失/变化的磁矩或相范围诊断，并同步任务、有效相图记录、训练数据和导出；同一证据再次读取不刷新。不合格数据退出后续科学使用，已训练模型不会被悄悄回滚。已完成的同版本同结构 MLIP 预测复用；之前因自旋缺失未评估、现在首次合格的结果才进行正常的首次误差评估。已被消费的合格训练样本不因仅更新元数据再次加入新数据队列。

若更新后全部相图点失去资格，历史快照保留，已有当前 phase_diagram.csv 改为仅表头，状态为无有效相图；不让旧的不合格点继续伪装成当前结果。初始无数据时不创建空相图 CSV，重复读取不改写。

[phase_agent/tools/remote/collect_training_results.py](../../phase_agent/tools/remote/collect_training_results.py)。

