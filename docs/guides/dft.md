# DFT 输入、采样与分析

按下方目录定位需要的说明；源码入口链接保留在对应章节。项目科学参数以确认配置为准。

- [Manual DFT preparation](#doc-dft_interface)
- [DFT 后对话顺序](#doc-post_dft_dialogue)
- [四路决策的执行边界](#doc-post_dft_flow_repairs)
- [重新准备输入](#doc-rebuild_inputs)
- [DFT / MLIP 双文件回传](#doc-dft_mlip_results)
- [DFT 回收后的决策接口](#doc-post_dft_flow)

<a id="doc-dft_interface"></a>

## Manual DFT preparation

Approved pending DFT tasks use prepare_local_batch_files mode dft_inputs. The
interface reuses Process_Vasp.generation.generate_atomate_input from Py-Code on
the local Python path. dft_relax uses RelaxMaker; dft_single_point uses one StaticMaker
directly, without preceding relaxation. Static overrides enforce NSW=0, IBRION=-1.

Parameters are stage-keyed incar_settings and kpoints_settings; relax/static keys
are allowed. GGA=None is forced in both input-set overrides (omit GGA tag).
All project-generated INCAR files use ALGO=Normal. This layered-oxide project
also enforces AMIX=0.2, BMIX=0.0001, AMIX_MAG=0.8, BMIX_MAG=0.0001 in both
Py-Code relax input-set overrides and legacy atomate materialized inputs.
The chosen candidate structure_path takes precedence over ledger source_path.

Upload the complete DFT allocation folder. Submit each task's GPU.sh once. The
remote Python environment must provide atomate2, jobflow, pymatgen, VASP command
configuration, and POTCAR data. VASP inputs are generated remotely by atomate2.
The workflow ends after the selected relaxation or single point, without follow-on jobs.
Final energy/structure and integrity marker are exported to the allocation results
folder through the existing finalize_vasp and export_batch_result protocol.

Preparation never submits or runs calculations. MC/QBC candidate construction is
not changed here. Existing confirmed single-point-first policy and relaxation fraction
cap are preserved; calculation types are never silently converted.

After a recorded second MC allocation is fully recovered, continuation no longer
requires another MC allocation. DFT candidate context is rebuilt from the latest
recovered segment per branch and unique identified current-model hull entries.
Candidate structure paths are retained; QBC is copied only from real outputs and
is explicitly not_configured when missing. This does not evaluate a committee or
authorize a DFT calculation. Existing policy/approval validation remains mandatory.

New interactive DFT proposals preview separate single-point/optimization counts, selection
reasons and configured nonzero cost. Approval registers child tasks and immediately
calls the same local upload preparer; no second input-preparation proposal is needed.
Empty, unsupported, partially rejected or uncosted plans cannot be approved as input
generation. Legacy pending DFT proposals without this preview must be rejected and
requested again. Existing sensitive-operation approval safeguards are unchanged.

### 实现入口

[phase_agent/science/dft/parse_vasp_result.py](../../phase_agent/science/dft/parse_vasp_result.py)。

<a id="doc-post_dft_dialogue"></a>

## DFT 后对话顺序

默认聊天将完整分析与备选取舍放入可展开详情，正文保留回收数量、能量/受力误差、
磁矩摘要、受力尺度提醒、综合相图partial/未校正数量、主要判断、收敛与数据局限。
审批条件与回复方式保持可见。verbose=True仍输出完整纯文本。确认、失败、等待回传
和训练等后续状态只报告当前进展及完整状态记录路径，不重复上一轮科学报告。
confirmation_required明确表示流程仍需确认，不能当作已经生成训练输入；未提供
具体原因时显示原因缺失。本次仅修改展示，不改变审批条件或推进生产任务。

回收并校验原模型比较后，post_dft_lines先报告本轮回收/合格配对数量、能量与受力MAE/RMSE、轮次输出位置；之后才展示正常审批方案。

LLM通过post_dft_assessment接收指标、轮次、数据门槛和记忆，比较微调、补DFT、新branch、收敛/停止。确需微调且数据足够时建议update_mlip，parameters.prepare_inputs_only=true；批准后仅生成超算提交文件，不需先启用配置，不调用本机训练适配器或激活模型。非输入准备的训练与激活仍保留敏感审批。

历史恢复必须重新进入工作流核对阶段，不能直接回显缓存建议。旧泛化branch/微调建议缺少_post_dft_review_version或_post_dft_scope_key不匹配当前DFT轮次时重新提议，旧批准不传递。标记代表已走新版决策上下文，不代表科学校验已证明正确。所有原有预算与审批仍有效。

测试：test_post_dft_assessment的回复次序和阶段路由；不操作生产数据。

模型代际显示为epoch0、epoch1……，对应持久化model_rounds编号1、2……；初始模型为epoch0，模型升级后的新代际依次增加。同一模型内追加branch、Relax、MC或DFT不增加epoch。对话保留模型版本名，round_summary.csv增加model_epoch列。既有目录与模型身份不改名，避免破坏回传路径；缺少持久编号时显示“模型轮次未记录”，不猜测。

回复按“结果回收与分析 → 本轮总结 → 下一步建议（待批准）”排序。总结包括round_findings（稳定相发现与搜索收益）及limitations（缺失、自旋排除与证据局限），没有依据时明确未知；不能只列误差就直接给动作。缺少总结字段的旧方案重新提议。分析无需行动审批，生成任务或微调仍需批准。

DFT后生成branch、微调或策略修订方案必须包含post_dft_review：choice、error_assessment、coverage_assessment、finetune_assessment、search_assessment、reference_assessment。choice分别对应search、finetune、revise_strategy。聊天展示误差、覆盖、两种选择及参考意见；缺项则不进入审批。版本2标记不能代替该对比。代码仅校验分析完整性，不声称已证明科学判断正确，也不自动设定误差阈值。

Agent先进行回收校验与相图更新、原轮模型同帧比较，并保存误差/训练/诊断文件，再调用LLM提出下一轮建议。分析报告与下一步建议可在同一回复依次展示，不需要用户先批准一次数据分析；新计算或微调仍须正常审批。开发检查只使用合成测试，不代替Agent分析生产数据。
## 方案格式校验与修正

首次建议和人工修订共用DFT后分析要求；报错明确指出字段、生成分配或微调选择冲突。
已解析方案不合格时，LLM最多修正一次，使用同一证据及具体错误；代码不补写科学结论。
仍失败则不执行、不批准，保留具体原因；两次请求的token消耗合并记账。

[phase_agent/runtime/post_dft_presentation.py](../../phase_agent/runtime/post_dft_presentation.py)。

<a id="doc-post_dft_flow_repairs"></a>

## 四路决策的执行边界

主模型设置更新：正式主模型复用com_1（main_model_index=0），与committee共用一个训练任务；不生成main_final。默认4正式成员+5折共9个训练任务；results/models.json标注is_main_model和is_committee_member，不重复导出。已有10任务目录不会自动改写。

训练设置更新：用户指定所有训练使用patience=20，保留配置值，不再覆盖为max_num_epochs+1；committee/最终模型的同数据validation仅作训练监视，早停不构成独立泛化评估。原先禁用早停的说明不再适用。

LLM基于结果与记忆比较微调、补DFT、新branch、收敛/停止；代码不替代科学判断。
生成策略必须使用现有五种注册名称和正JSON整数配额，校验反馈定位到具体项。
方案参数失败时保留已完整通过分析字段校验的review，不能批准错误方案。
微调输入方案的choice必须finetune。工具名/拼写别名可归一化；旧revise_strategy仅在input-only update_mlip、明确now和continue时视为等价执行标签，保留归一化记录。search、defer等真实科学冲突不能改写，交给LLM修正；错误同时显示期望值和实际值。

新的微调方案使用update_mlip、parameters.prepare_inputs_only=true，展示“建议微调，生成超算训练提交文件”，普通批准只准备文件。正式训练、激活仍需敏感审批。兼容旧的单独mlip_finetune.enabled=true建议，但版本5刷新旧建议，不传递旧批准。其他配置修订仍走草稿。
adjust_strategy可请求配置修订：request_configuration_revision=true、patch为现有点分路径。
审批后在state.requested_config_revision保存草稿，确认配置仍不改变；配置对话中确认patch后，
再批准迁移。仅申请过且明确批准的false→true微调启用变更可迁移，不放宽科学配置迁移规则。

默认微调执行复用prepare_mace_finetune：只生成远端输入、分组train/valid/test数据、run_training.sh以及复用现有MLIP GPU模板的GPU.sh。Agent返回上传目录与提交说明；不提供硬编码真实数据的一次性生成入口。未登记的既有目录不可直接覆盖。
沿用已配置committee；未配置时默认4成员。正式committee全部合格数据参与训练，不做可能遗漏结构的bootstrap抽样，成员保留不同seed和配置。主模型默认5折，按branch/framework/structure来源分组，同来源不跨折泄漏。目录main_cv_1等保存各折输入，main_final全部合格数据参与训练。最终模型和committee的valid_file同train_file仅满足MACE训练监视要求，不是独立验证；patience大于固定训练epoch上限，泛化评估只使用K折留出数据。

旧配置缺少remote_mlip时，训练入口只复用实际远端GPU模板唯一明确的conda activate环境（现有mace），记录remote_environment.source；绝不从本机模型环境推断。明确remote_mlip配置优先，不修改确认配置。

run_training.sh训练后自动运行便携collect_training_results.py。results/kfold_metrics.csv包含各折与合并留出样本能量/atom和逐分量力MAE/RMSE；energy_comparison.csv和force_comparison.csv保存eV单位逐点对比，力每原子3行。最终主模型与committee复制到results/models，models.json记录相对路径；training.finished.json只在完整评估和导出后写入。下载完整results即可取得模型和误差。缺失或歧义模型不伪造结果；本地模型回收/激活接口仍需独立审批。
用户自行将完整目录上传，在目录内sbatch GPU.sh一次；复用现有集群资源模板，不提交作业、不本机训练。
路径使用相对数据路径及已确认远端原模型/环境。模型回传后仍需独立验证适配器和单独激活审批。
已有mlip_trainer/model_update_handler适配器继续沿用训练→验证→候选→审批链。
没有远端模型回收/独立验证适配器时，不声称已完成训练或自动切换模型；这属于明确执行条件。

补DFT只有成功产生新任务才消费来源DFT轮次，参数失败不消费；新结果进入其自己的轮次分析。

[phase_agent/analysis/state/post_dft_assessment.py](../../phase_agent/analysis/state/post_dft_assessment.py)。

<a id="doc-rebuild_inputs"></a>

## 重新准备输入

聊天发送“重新生成 MLIP 输入”，程序绕过等待回传提示，生成清理及重建建议。用户核实旧作业尚未提交，批准后才删除清理计划中列出的旧 Relax 输入批次并重新分组生成。普通“继续”不清理。自动模式也不能跳过重建审批。

清理范围严格限定在当前 upload_batches 下、台账对应的未完成 Relax 批次。只允许生成的输入文件；已有结果、日志、checkpoint、未知文件、运行状态或 job_id 均阻止删除。部分完成的混合批次不整批删除，需单独处理。原母结构、候选结构、主台账、历史成本保留，任务编号和预算预留复用。

新输入格式为每个 Relax 作业最多 100 个结构、一个 run_mlip_batch.py 和 GPU.sh。409 个兼容结构对应五组。脚本生成不代表提交成功。首次重建失败后不会伪造成功，原结构仍可用于恢复。

当前实现的是 Relax/MLIP 输入重建，不是 MC/DFT/训练结果删除或真实计算重跑；这些需要独立的任务取消、预算授权和结果归档策略，不能套用输入清理。

[phase_agent/tools/local/regenerate_dft_files.py](../../phase_agent/tools/local/regenerate_dft_files.py)。

<a id="doc-dft_mlip_results"></a>

## DFT / MLIP 双文件回传

新生成 DFT 输入包含 comparison_model.json，固定任务所属轮次模型版本、超算模型路径、head 和环境。VASP 提取后自动对同一最终帧做一次原模型预测，不额外弛豫。

results 每个任务目录：result.json（DFT 结构、能量、受力、应力、磁矩及副文件校验）、mlip_result.json（同帧 MLIP 能量和受力、任务身份、模型与结构 SHA256）、training.json（训练帧）、task.finished.json（结果完整性标记）。最后两项仍须保留，并非只下载两个文件。

本地默认只读远端预测，不运行 MLIP。校验任务、模型版本/已知模型指纹、最后一帧结构指纹、单位、数组形状和副文件校验和；合格 DFT 才参与 MAE/RMSE。首次未知模型指纹从有效回传记录绑定，后续同版本不同权重被拒绝。严谨核验可预先在模型注册表配置原模型指纹。

已完成批次：在超算项目根目录运行下列命令，指定原轮次模型；只补预测，不重跑 VASP。请先备份 results，因为需要更新 result.json 的副文件引用和完成标记校验和。

```bash
python -m phase_agent.tools.remote.predict_dft_final_frame \
  --results /data/home/lichaoyue/26-10-PhaseDiagramAgent/upload_batches/MLIP-round-0001_mace-mh-1/Search-group-0001/DFT-round-0001_43c800068af6/DFT-single-point/results \
  --model-path /data/home/lichaoyue/Py-lzy/MLIP_Model/mace-mh-1.model \
  --model-version mace-mh-1 --head omat_pbe --environment py-mace
```

使用已更新项目代码，入口环境须能导入 pymatgen/numpy；MACE 在指定 conda 环境内执行。成功预测缓存复用，失败可重试。完整回传该 results 文件夹后说“继续”，已回收任务可补充预测，不增加 DFT 成本或训练标签。

本地参考模型已确认存在：E:\1-guihub库\MLIP_model\mace-mh-1.model。未修改生产配置或运行评估；超算预测使用超算路径，不使用 Windows 路径。显式 dft_comparison_location=local 才允许旧本地预测方式。

[phase_agent/tools/remote/integrity.py](../../phase_agent/tools/remote/integrity.py)。

<a id="doc-post_dft_flow"></a>

## DFT 回收后的决策接口

回收完整，或人工对该轮剩余任务选择不再等待后，按 model_version、search_group_index、parent_relax_round、upload_operation_id 隔离评估。未回传任务保留原状态，不取消作业。

`post_dft_assessment` 复用保存的最终帧预测计算能量和受力 MAE/RMSE。缺失原轮次模型或预测时明确报告未评估；继续时只重试缺失预测，不重新识相、回收、增加训练标签或记账。不同模型不得混合比较。

评估完整后，LLM 根据评估、现有相图、覆盖及记忆提出下一步，仍需正常审批。旧 MC→DFT 建议失效，不能复用其批准。生成新 branch 成功才消费本轮决策；后续新 MC 任务仍按正常阶段推进。

`update_mlip` 正式工具复用 `create_model_update_handler`。运行时可提供 `model_update_handler`，或提供 `mlip_trainer`、`mlip_validation_evaluator`、`mlip_reevaluation_predictor` 及对应数据 providers。未配置训练器时明确报告，不假装完成。

微调须先启用已确认的 `mlip_finetune.enabled` 并达到新增合格数据门槛。数量门槛不是科学误差触发阈值；LLM 决策须说明误差、覆盖与记忆依据。不自动填写误差容忍值；既有独立验证和单独激活审批保持不变。

验证：`tests/test_post_dft_assessment.py` 覆盖轮次/模型隔离、部分回收关闭、缺失误差、重试缓存、训练门禁、正式 LLM 阶段路由。完整回归不包含暂不启用的 BOHB 集成测试。

