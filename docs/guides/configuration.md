# 配置、边界与预算

按下方目录定位需要的说明；源码入口链接保留在对应章节。项目科学参数以确认配置为准。

- [Initial and run configuration](#doc-initial_and_run_configuration)
- [Agent生成策略分配](#doc-generation_allocation)
- [扩胞前规划与审批](#doc-generation_cost_preflight)
- [单任务与批次成本粗估](#doc-calculation_cost)
- [计算成本](#doc-cost_model)
- [两端 Python 环境](#doc-python_environments)

<a id="doc-initial_and_run_configuration"></a>

## Initial and run configuration

`parameters/search_config.project.json` is the editable initial configuration. It contains local/remote Python environments, system boundaries and composition, mother-structure paths, the remote MLIP model, DFT/Slurm settings and storage paths. This existing filename is retained so project launch bindings remain valid.

`parameters/run_config.project.json` contains editable run overrides: budgets, branch selection/sampling, MC, fine-tuning, validation, convergence and later Agent policies. Missing values inherit the pinned profile; startup audit reports later-stage missing thresholds separately rather than demanding them before any search advice. Invalid supplied values and missing initial facts still require correction. Calculation stages retain their own checks; startup readiness is not permission to submit jobs or activate a model.

Both files are read together. Change either manually and say `读取配置 JSON` (review only), or `读取配置 JSON 并继续` (review and proceed if valid). `读取初始配置` and `读取运行配置` also check the combined configuration. During a running search, these commands open a new configuration revision without executing the old proposal. Agent edits route initial fields to the initial file and policy fields to the run file. Run overrides cannot replace initial environment/structure/boundary fields.

Every approved complete effective configuration still has an immutable version snapshot in `parameters/snapshots/`. Historical jobs keep their configuration versions. Editing a file never silently changes an executing job. Review identity includes the bytes of both files, plus the existing checks on imported effective configuration and mother structures, so a manual edit to either file requires a fresh review.

New projects generate both files automatically. The initial draft does not inherit another project's TM composition or HPC model path. Default local ordinary environment remains py1; remote environments must be supplied. Existing single-file project-v2 configurations remain supported. Explicit splitting preserves effective values and keeps `<initial-file>.before-split.bak`; existing live states and confirmed snapshots are not rewritten.

Later-stage independent validation uses `remote_training_validation` in the run file (data_path, data_version, ranking_pairs, criteria and optional label keys), as described in training_handoff.md. Initial local/remote environments and model paths remain in the initial file. DeepSeek credentials remain in the existing credential store/environment, not in editable scientific JSON.

The combined configuration is the effective next-run source; the initial file records current startup facts, while immutable snapshots preserve historical startup and run settings. Run overrides contain persistent user settings, not automatically promoted per-round Agent suggestions or memory. At runtime an Agent may still propose a one-round action without changing either file.

Verification: py1 regression tests cover splitting round trips, legacy compatibility, field routing, manual edit detection across both files, unknown/forbidden fields, deferred startup thresholds, running-chat import, configuration review, launcher, Studio, model handoff and refresh behavior. Actual searches or HPC computations were not launched by this change.

### 实现入口

[phase_agent/configuration/session/split_project_config.py](../../phase_agent/configuration/session/split_project_config.py)。

<a id="doc-generation_allocation"></a>

## Agent生成策略分配

DFT分析后选择搜索时，LLM必须提出generation_plan：每项包含strategy、quota、phase（合法相名或all）、可选na_min/na_max（Na/O2）和简短reason。按策略累计数必须等于quotas，总数等于total_quota。最多12项，保持决策紧凑。支持较小候选批次，不自动抬回300或补齐默认策略。

根据相/Na覆盖、有效相图与QBC、历史收益和人工记忆选择比例；缺少稳定相不等于该相必须补样，不用默认配额冒充智能分配。全域探索可用phase=all，需说明理由。计划字段只验证一致性，不声称科学判断已得到证明。

执行复用原有五种生成策略，在计划限定的合法相、Na范围和超胞边界内产生候选，之后统一历史去重与选择。竞争相保留其他相的合法父branch；固定TM排布禁止tm_ordering。目标无合法区域时报错，不静默换成其他区域。配额是候选目标，不是必然入选数；产量可能受父结构和可行组合限制，生成历史保留计划及实际产量。

对话显示中文策略与目标小清单、候选数、最终入选上限、det(H)与初态上限。旧DFT后通用方案无plan需刷新，批准不转移。首次尚无父branch时仍只支持coverage；现有初态初始化、筛选、计算预算和审批均保留。

[phase_agent/decisions/agent/generation_plan.py](../../phase_agent/decisions/agent/generation_plan.py)。

<a id="doc-generation_cost_preflight"></a>

## 扩胞前规划与审批

生成方案先读取母胞元素数量，以 det(H) 推算原子数上界，不创建超胞。
LLM获得合法框架的原子规模和成本菜单，按相、Na区间、策略、数量与每项
max_det_H 分配任务。每项上限不能突破整批上限。

审批前及实际生成前复核：最多三个Relax初态、一个MC按确认策略最大步数的
成本上界，对照剩余总预算；超出时不生成，要求调整数量/超胞或修订预算。
这不是预算预留，也不是后续计算授权。DFT展示每个入选branch一个单点的
独立情景上界，未包含第二段MC、DFT优化及排队。无实测耗时不推测小时数。

DFT后先独立判断微调科学需要：now/defer/insufficient_evidence。
now且微调未启用时建议修订策略，不能以未启用为理由转向搜索；搜索必须
明确defer并提供科学理由。旧版审批方案重新刷新，不继承批准。

默认对话保留结果、指标、取舍、限制和下一步预算；完整七项评估仍保存在
审批记录，详细模式可展开。此功能不自动处理生产数据。

[phase_agent/analysis/cost/estimate_proposal_cost.py](../../phase_agent/analysis/cost/estimate_proposal_cost.py)。

<a id="doc-calculation_cost"></a>

## 单任务与批次成本粗估

审批方案增加“耗时粗估”：实际任务规模与实测样本完整时显示串行累计小时及观测范围；缺少规模或样本时明确暂无法估算，不以relative_cost换算小时。不包含队列等待，不猜并行完成时间。branch生成尚未展开后续计算任务时，不预报整条科学流程的时间。

### 轻量实测优先

新生成任务在执行前后各取一次时间，不轮询、不调用额外科学计算。result.json 的 runtime_observation 在校验和计算前写入，回收后进入 cost_history。CPU/GPU 数量从已有 Slurm 环境取得；缺失保持空值。partition 只作为近似硬件类别，不声称识别了 GPU 型号。Atomate wrapper 与直接 VASP 脚本均记录任务运行时间，不含队列等待。

预测优先采用同后端、同硬件/资源类别、原子数在目标 0.5–2 倍范围内的最近最多30条完成任务中位数，并作粗略规模折算。报告样本数和观测跨度，不把跨度当统计置信区间。MC 按实测每步耗时估算，只有 patience 和最大步数相同的记录用于典型总步数预测。没有对应样本时回退初始参考值。

运行秒数、GPU小时和CPU核时是主要物理成本；relative_cost仍为独立预算参考，未经指定换算基准不自动将秒数写成预算费用。MLIP批次另有 `.runtime.json` 总耗时，与任务时间片不可重复相加。已有生成的输入不被此次修改重写；缺失历史运行时间不补造。

复用现有原子规模/阶段成本模型，入口为 `phase_agent.analysis.cost.estimate_task_cost` 的 `estimate_task_cost` 和 `estimate_batch_cost`。成本单位为 relative_cost，不是时间或金额。报告包括初始粗估、历史校准比例、有效实测样本数、情景成本和局限。

Relax 支持初态数量；Relax/DFT 可显式提供优化步数、k点数量、电子步数及各自参考值，使用粗略线性倍率。只给工作量不提供参考值会报错，避免假精确。原子数未知时采用配置参考规模。

MC 必须提供最大步数，可提供 patience 和自行假设的典型步数。patience 情景表示从开始连续不改善的成本，不是预期总步数、实际下界或提前停止保证；最大步数情景用于保守比较。报告不修改任务设置。

批次 JSON 是对象列表，例如：

```json
[
  {"stage": "relax_and_feature", "atom_count": 80, "backend": "mace"},
  {"stage": "deep_search", "atom_count": 80, "patience": 20, "max_mc_steps": 100},
  {"stage": "dft_single_point", "atom_count": 80},
  {"stage": "dft_relax", "atom_count": 80, "relax_steps": 40, "reference_relax_steps": 30}
]
```

在 py1 下运行 `python -m phase_agent.runtime.estimate_calculation_cost --tasks 任务规格.json --state 当前state.json`，只读取文件并输出报告，不生成任务、不改数据。可通过 `--config` 显式提供带 budgets 的配置。

Agent 决策上下文自动包含参考规模报告；每个实际任务仍以已有科学成本核验为准。指定 backend 时校准只采用同 backend 实测样本，未知后端样本不混入。无可靠时间标定时核时保持空值，不能把 planned_cost 或 MC 实际步数折算值当作实测成本。

[phase_agent/analysis/cost/estimate_task_cost.py](../../phase_agent/analysis/cost/estimate_task_cost.py)。

<a id="doc-cost_model"></a>

## 计算成本

Proposal 中的下一轮成本以规模模型为初值，并按 stage 使用最近最多 10 个成功任务的
`actual_cost / planned_cost` 校准：30% 初始估计加 70% 历史比例，校准因子限制在
0.25–4.0。审批文件同时展示任务数/MC步数、分阶段成本、已用与预留成本以及执行后
预计剩余预算。

默认值是可校准的相对资源代理，不是实测秒数、核时或费用。预算上限表示用户投入政策，不代表真实后端性能。

基准结构 40 原子：简单检查 0.05，单次 MLIP 弛豫与特征 1，每个“提议并完成 MLIP 弛豫”的 MC 步骤按 MC 成本系数折算，当前默认系数为 0.1，DFT-SP 30，DFT-relax 900。DFT-relax/SP 固定为 30；DFT-SP/单次 MLIP 弛豫在基准规模也是 30。实际比值仍受离子步数、电子收敛、硬件和模型影响。

统一估算入口 `phase_agent/tools/budget/estimate_stage_cost.py`：

```
cost = task_cost × (N / 40)^p × initial_state_count × search_factor × scale
```

简单检查默认 p=1，MLIP 弛豫和 MLIP–MC 默认 p=1.2，DFT-SP 与 DFT-relax 默认 p=3。这里用立方增长表达 DFT 随体系规模快速增长，比数学意义上的指数函数更符合常见平面波 DFT 的经验标度；它仍是可调代理，并非理论复杂度保证。每个 MC step 都包含一次 MLIP 弛豫，因此 `search_factor=计划 MC 步数/参考步数 × MC成本系数`，参考步数为1且系数为0.1时，10/30/90 fidelity 在40原子基准下分别估算为1/3/9，而不是一次弛豫。未提供原子数时使用40原子并标记 `reference_size_assumed`。

BOHB action 同时保存 `incremental_budget`（本段新增的完整 MC 弛豫步数）和 `planned_relative_cost`（结构规模折算后的相对资源成本）。BOHB 用前者比较 fidelity 与晋级，执行层和总预算使用后者；二者不能互换。

全流程默认总上限为20000。阶段上限可重叠但共享总预算：MLIP–MC 4000，DFT-SP 6000/最多100任务，DFT-relax 15000/最多2任务。40原子的一个 Relax 计900；80原子按立方规模因子计7200，因此两个80原子 Relax 已接近阶段上限。任务数量仍受全局成本和其他阶段开销约束。

DFT action_costs 从 default_budget_rules 派生，正式主动学习以 run_config.budgets 为权威。DFT 检查同时计入运行中的预留和同批次新任务，分别检查全局、阶段成本和任务数。

建议在同一后端、模型和DFT参数下记录原子数、MC步数、离子步数、核数/GPU数、耗时及任务状态，用同组任务资源消耗的中位数校准 task_cost 和指数。CPU核时与GPU时需明确换算权重；不能直接相加。后端 actual_cost 必须换算成同一 relative_cost 单位再提交。优先使用可靠调度器记录中的 GPU 数与实际运行秒数计算 `actual_gpu_core_hours`。没有实测值时 `actual_cost` 保持 null；预算台账另存 `estimated_cost`、`accounted_cost` 和估计依据。已知实际步数可用于释放部分预留，但不能改名为实测 GPU 成本；连实际步数也未知时以预留额作为保守估计上界。

MC 任务分别保存 `max_mc_steps/requested_max_mc_steps`、`patience_steps`、
`min_improvement`、`actual_mc_steps`、`stop_reason`、请求预算、预留成本、实测成本
和估计成本。`patience_steps` 是早停参数，不是实际执行步数。失败、超时和取消若有
已消耗成本同样结算；settlement ID 与 task ID 保证重复回收不重复扣费。

旧状态和已确认配置不自动换算或改写。旧 DFT action_costs=1/4、20/200 与新30/900账本不能直接相加；旧 MC“1000步=4”与新“每个含弛豫的MC步=1”也不兼容。建议新运行采用新默认值，旧运行继续使用原配置完成或明确迁移预算。独立直接调用仍可显式提供旧 action_costs。

[phase_agent/tools/budget/estimate_stage_cost.py](../../phase_agent/tools/budget/estimate_stage_cost.py)。

<a id="doc-python_environments"></a>

## 两端 Python 环境

初始化配置 python_environments：local_python（本地管理/提取，默认py1）、local_mlip（本地科学后端，默认py-mace）、remote_python（超算DFT/提取环境）、remote_mlip（超算MLIP环境）。两项远端默认空，必须人工确认真实名称；不能从本地推断。current 表示入口已在正确环境运行。

DFT 输入 comparison_model.json 显式写入 remote_mlip，远端预测不再默认本机py-mace。补充预测命令 --environment 必填。本地显式比较使用 local_mlip。

remote_python 记录并在初始化校验；现有用户提交脚本/集群 shell_preamble 仍须按该环境激活。项目不自动改写用户的集群 module/source 设置。Relax/MC同样须在remote_mlip环境提交。旧的已生成GPU.sh和comparison_model.json不会被配置修改自动覆盖，应按正常重生成确认流程刷新。

旧配置缺字段会提示补齐，不修改历史模型、结果或科学参数。先在超算 conda env list 确认真实环境，不把候选默认当成存在性验证。未连接超算时，初始化只能校验填写格式，不能声称环境已可用。

[phase_agent/configuration/schema/python_environments.py](../../phase_agent/configuration/schema/python_environments.py)。



### 对话写入与审核分开

字段目录声明可编辑源字段和派生字段：相边界写 `system.boundary.P`，元素比例写 `system.boundary.TM_ratio`；constraints 与 species 由程序同步。统一相边界可以用相列表，按组分分相可以用对象。

对话写入只更新文件及会话草稿，列出缺少的母结构；不自动枚举 H，也不视为审核通过。补齐后通过配置审核入口检查母结构和 H，再确认快照。文件保存成功不能显示成草稿未改，审核失败也不能恢复旧会话值。


## MLIP 优先的预算上限

新项目总相对成本上限为 50000。阶段累计上限：初筛 300，MLIP 弛豫 9000（最多1500任务），MLIP/MC 搜索 27000（最多300任务），DFT 单点 7200（最多120任务），DFT 弛豫 2280（最多2任务）。阶段额度共享总预算，不能相加视为已获执行授权。

轮级 MC 上限为3000，DFT上限为6000；DFT单点每轮最多120个、成本6000。科学工具仍核对剩余总预算和阶段额度。已有项目按自身旧上限扩大，总预算2.5倍、MLIP3倍、DFT1.2倍，保留已有项目的差异。

这些数值是上限，不是建议一次使用完的数量。Agent依据阶段结果、有效版本、收益和成本提出小批方案，优先MLIP筛选及结果复用，再使用有代表性的DFT验证；保留独立抽检、DFT单点优先、弛豫比例和原有精度/收敛条件。额度扩大不保证实际成本自动下降，实际消耗由审批后的方案和回收结果决定。

默认模板更新时，现有项目保留非预算参数并迁移模板标识。配置写入草稿不修改旧确认快照、任务或待审批方案；读取并审核新草稿后，后续方案才使用新版本。
