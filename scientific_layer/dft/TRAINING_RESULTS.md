# Portable DFT training results

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

## Official workflow integration

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

## Fe/Mn 磁矩诊断

新 DFT 解析在 `result.json.outputs.magnetic_moments` 保存最后 OUTCAR 局域磁矩：全部元素、逐原子索引、标量/矢量、μB、来源和最后帧索引；远端不做范围验收、不改 checks_passed，不因磁矩异常拒收。文件回传/身份、校验和与预算回收维持原流程。

Agent 先按最终结构识别/复用相，再生成 `magnetic_check` 与 `spin_state_check`；完整原数据和诊断进入 `dft_records.json`，原子清单进入 `magnetic_moments.csv`，对话说明合理范围、实际合理原子统计、异常任务和详细清单。只判定 Fe/Mn，不要求 Na 必须非零；其他元素及非层状体系只保存磁矩而不检查。范围为 Fe |m|=3.5–4.5、Mn |m|=3–4 μB，是经验标准，不是基态证明。旧 magnetic_check 内嵌磁矩继续兼容。

层状范围不再依赖固定 system_id 名称；最终明确相优先于项目提示，配置可显式提供 is_layered_oxide。含 Fe/Mn/O 而最终范围尚无法确定时记 unknown 并暂缓科学使用，不将 unknown 当成非层状豁免。结构内容与分类条件未变时仍使用现有相识别缓存。

局域投影之和不等同总体系磁矩。诊断只对应最终计算帧，不代表较早帧的磁矩标签，不新增逐帧磁矩训练目标；电子收敛帧仍按原流程提取。

### 读取后的科学使用标准

对明确层状氧化物中的 Fe/Mn，上述规则是本地科学使用门槛，不是下载/回收拒收条件。`outputs.spin_state_check` 保存 applies/status/criterion/原因；通过称为“符合预期自旋态”，`magnetic_ground_state_proven=false` 明确没有比较多个磁性初态/磁序的能量，不能当作严格基态证明。

`passed` 方可按原收敛与其他校验进入有效阶段历史、相图、训练和末帧误差评估。`warning` 对应自旋验收 rejected；缺失/非法、SOC/非共线、末帧或原子顺序不匹配为 unknown，均设 `checks_passed=false` 并暂缓科学使用。其他元素/非层状体系为 not_applicable，不额外要求磁矩。已有其他质量检查失败时，自旋通过不会覆盖为成功。

计算执行的 status、converged、实际成本和原能量不改，仍正常回收/入账；台账 calculation_attempts、原始结果及 dft_records.json 保留异常和标签。不合格帧保留于内部 dft_training_records 作审计，但不进入 new_dft_records、可用 training.json、微调触发或 MAE/RMSE 样本。较早帧并未逐帧证明基态，整项任务按末帧标准决定是否允许训练；要逐帧自旋筛选需另外采集每帧证据。不会自动改 INCAR 或提交修正任务。轮次简表分列 dft_spin_passed/rejected/unknown；energy/force_comparison.csv 增加 spin_check_status，不混淆运行成功与科学验收。

远端生成原始磁矩需同步更新项目代码与 Py-Code/Process_Vasp；旧 JSON 没有磁矩时本地无法恢复，无须回传整个 OUTCAR。部分回收提示按当前轮显示磁矩缺失、未评估、合理、异常、未知与不适用数量，不隐去缺少证据的旧结果。只有结构、磁矩证据和适用条件一致才复用诊断指纹；CSV 内容不变不重写。独立 training.json 中的 task_final_magnetic_moments 是任务最后帧元数据，不误称为每个早期训练帧的磁矩标签。

### 已回收任务补传磁矩

重新提取同一任务并回传 result.json、匹配其校验和的 task.finished.json，以及引用的 training.json。回收器仍验证原 task 身份、文件校验和、结构/E/F/应力/最后帧一致性；仅补充磁矩，不覆盖计算结果或成本。冲突标签显式拒绝，不靠改 task ID 规避。原始结果和预算结算不重复入账，阶段结果 ID 保持原值。

下一次正常 Agent 回收会刷新缺失/变化的磁矩或相范围诊断，并同步任务、有效相图记录、训练数据和导出；同一证据再次读取不刷新。不合格数据退出后续科学使用，已训练模型不会被悄悄回滚。已完成的同版本同结构 MLIP 预测复用；之前因自旋缺失未评估、现在首次合格的结果才进行正常的首次误差评估。已被消费的合格训练样本不因仅更新元数据再次加入新数据队列。

若更新后全部相图点失去资格，历史快照保留，已有当前 phase_diagram.csv 改为仅表头，状态为无有效相图；不让旧的不合格点继续伪装成当前结果。初始无数据时不创建空相图 CSV，重复读取不改写。
