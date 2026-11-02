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

相关回归覆盖路径比较、报告一致性、幂等、来源核对、未知成本和任务归因。模型/API/HPC 的现场有效性需独立验证，源码更新由用户手动重启服务后加载。

[phase_agent/agents/strategic/round_budget_review.py](../../phase_agent/agents/strategic/round_budget_review.py)。

<a id="doc-model_refresh_implementation"></a>

## 新模型结构刷新

候选模型经独立验证和明确激活后，先保留旧相图，再通过 prepare_model_refresh 审阅结构清单、模型/参数、校验和与成本，准备远端 Relax/Single-point 输入。实际范围以冻结方案为准，不因继续消息自动提交或扩大。

选择复用 analysis/feedback/model_refresh_plan.py：按旧模型正式相图 EHull（eV/atom）冻结首批分层；小于 0.001 的结构全部弛豫，0.001–0.010（含两端）全部单点，大于 0.010 的结构按组分与相分层抽取 10%（总数向上取整）做单点。没有可用 EHull 的已登记结构保留单点，不将其假定为远凸包。刷新后只用新模型结果构建其凸包，暂缓和缺失项标记 partial；不能以 partial 相图单独宣称全局收敛。

新方案在同一远端刷新任务中先做单点，再按首批冻结分层检查最大原子力模长：0.001–0.010 eV/atom 严格大于 0.1 eV/Å，>0.010 eV/atom 的抽样点严格大于 0.2 eV/Å 时立即弛豫；无旧 EHull 的结构保守使用 0.1 eV/Å。单点先保存与输入结构和模型身份绑定的受力缓存，弛豫失败后在同一任务目录恢复时复用，不重新计算已完成单点。刷新交付包携带必要的受控执行器；一批审批及一次提交覆盖该条件弛豫范围。回收后校验逐原子受力、阈值、结构身份和弛豫收敛，全部有效后更新本地凸包、版本归档及 latest；缺力、身份不符或失败时保持等待并保留结果，不宣称完成。旧冻结方案按原语义恢复。

刷新完成后，若新模型没有有效新增 Branch，先生成并展示实际数量和分布，再单独审批 Relax/MC；后续比较旧 Branch 在新模型下续搜、新 Branch 和 DFT。详见 [epoch 工作流](../architecture/workflow.md)。

实现：[model_refresh_plan.py](../../phase_agent/analysis/feedback/model_refresh_plan.py)、[model_refresh_gate.py](../../phase_agent/tools/workflows/model_refresh_gate.py)、[training_context.py](../../phase_agent/evidence/training_context.py)。
