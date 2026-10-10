# 搜索决策与模型刷新

按下方目录定位需要的说明；源码入口链接保留在对应章节。项目科学参数以确认配置为准。

- [DFT sampling and previews](#doc-dft_sampling)
- [轮次路径选择与预算经验](#doc-round_budget_review)
- [新模型结构刷新：实施状态](#doc-model_refresh_implementation)

<a id="doc-dft_sampling"></a>

## DFT sampling and previews

Per search round, single points are capped at 100 structures and 3000 relative cost
unless explicitly configured otherwise. Registered same-model/same-generation
single points count toward both caps. Global/stage budgets remain separate.
Existing confirmed configuration and production task state are not edited by this
change; missing new per-round settings use these approved project defaults.

Recommendations reuse select_dft_candidates: hull score, available QBC, phase
representatives, Na composition spread and independent audit. Missing QBC is omitted,
never represented as zero uncertainty. Sufficient-size proposals missing available
phases require revision. Budget cannot guarantee every phase is affordable; missing
coverage requires an explicit revised plan, not silent selection of cheap candidates.

Previews display selected structure IDs, x_Na_per_O2, identified phase and current
Ehull eV/atom, then one QBC availability count. Candidate costs use actual atom counts
from diagram composition when the diagram lacks an explicit atom_count.

Agent context includes real per-candidate single-point/optimization costs. An
unapproved draft rejected for DFT budget may be replaced by a visibly labeled
budgeted single-point draft using the same phase/Na sampler. It is validated again,
requires fresh human approval, and never alters an approved plan or raises budgets.
Global, stage and same-round prior usage are deducted before draft sampling.

### 实现入口

[phase_agent/tools/workflows/create_dft_selection_handler.py](../../phase_agent/tools/workflows/create_dft_selection_handler.py)。

<a id="doc-round_budget_review"></a>

## 轮次路径选择与预算经验

Agent在已有轮次结果时比较两条完整路径：已有候选补DFT（含必要的再微调），以及新branch/Relax/MC后选DFT（含必要的再微调）。以预期增量收益和后续成本作取舍。覆盖、验证误差、QBC与模型风险提供证据，不是新搜索的硬门槛。缺失成本为null；预期收益标为hypothesis。

round_budget_review记录choice、两方案预期收益和相对成本依据、reason、uncertainty与revisit_when。可以先准备证据、微调或停止，不强制DFT优先。既有post_dft_review保持有效。

每次状态快照读取登记训练目录下轻量results：完成标记、折外误差、模型清单及源模型版本；报告可见不等于模型文件已核验，不自动激活或改写训练作业状态。旧待审批建议缺少比较时重新评估，旧审批不沿用到新选择。

预算经验按明确任务ID或parent_decision_id归因。记录所选路径的预期、实际完成情况、完整实测相对成本及可归因收益；成本缺失或任务未结束保持未知。保留模型与配置版本，不把未执行路径记作实测结果，不跨能量基准直接比较收益。自动生成training_report和round_budget_outcome记忆候选，经现有审阅机制再成为长期建议；后续决策可读取近期预算结果。

验证：py1中新增路径比较、报告一致性、幂等、来源核对、未知成本和任务归因测试；回归DFT后决策、LangGraph与记忆流程。当前项目只读回放识别27结构报告和82个尚无QBC的候选。没有提交计算、激活模型或修改实际工作流状态。运行中的Python服务需重载代码后生效。

[phase_agent/decisions/agent/round_budget_review.py](../../phase_agent/decisions/agent/round_budget_review.py)。

<a id="doc-model_refresh_implementation"></a>

## 新模型结构刷新：实施状态

### 已实现并测试

- 远端微调输入选用累计合格 DFT 记录，保留层状 Fe/Mn 自旋检查。
- 只合并累计队列与新增队列中同一 data_id 的镜像记录；不增加 DFT 科学去重。
- committee 继续全量训练；分组五折继续保留，增加本轮新增子集 OOF 统计。
- XYZ 与逐结构/逐力分量误差输出携带 data_id、task_id、is_current_round。
- 纯规则模块 phase_agent/analysis/feedback/model_refresh_plan.py：旧版本 Ehull <1 meV/atom 弛豫，1–10 单点，>10 按 Na×相分层抽取总量10%（向上取整）。
- 历史 near-hull 标记、未评估结构保留；同结构哈希合并；异版本排序能量拒绝。
- 补充弛豫判据：新 Ehull <1，或1–10且最大原子力范数 >0.05 eV/Å。

### 已接通的工作流

1. 激活后刷新门控、旧相图快照和跨代潜力结构证据提取。
2. 基于实测成本的首批及补充上限预览、审批和预算预留。
3. Model-refresh 目录下远端 Relax/Single-point 输入生成、committee 预测与结果回收。
4. 最多一轮补充弛豫的授权与自动准备，失败/缺结果等待处理。
5. 新版本相图 partial/覆盖标记、旧能量隔离和搜索/收敛决策门控。

上述五项现已接通。激活后先单独审批刷新，不本机自动重算。复用现有预算、ManualUploadBatchRunner、回收校验和相识别缓存；预算不足不静默截断。

目录为 submissions/epochN_模型/Model-refresh-0001/Relax 和 Single-point，各阶段批次共享父目录 results。初始批准同时授权所展示数量和成本上限内最多一轮补充输入生成；缺少批准记录或预算预留不得自动追加，不授权本机计算或自动提交。

成本复用规模估计与历史成本校准，单点暂按弛豫上界；有同类实测时估计串行耗时，否则明确未知。失败/缺结果可以继续等待或明确回复“不再等待刷新”，不把缺失任务标为完成。新相图CSV标注已刷新、暂缓与partial，不能单独据partial宣称全局收敛。

没有处理实际工作区数据、运行真实MACE训练或提交作业。超算仍需更新项目代码并配置正确的模型路径、环境和调度模板。

[phase_agent/analysis/feedback/model_refresh_plan.py](../../phase_agent/analysis/feedback/model_refresh_plan.py)。

