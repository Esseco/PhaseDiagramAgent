# Agent 选 Branch、Relax/Hull 预筛选与分档 MC

正式入口默认采用方案二（`selection_policy=relax_hull_uncertainty`）：每个 Branch
从静电能最低的前 10 个合法占位中按固定种子无放回选择至多 3 个（不足时保留实际数量），先预留 Relax 预算并
等待全部结果。取该 Branch 各 Relax 最终能量的总体标准差
`branch_energy_std_per_atom` 作为 branch 不确定性。它不是 QBC；QBC 的三模型力分歧
只供后续 DFT 选点使用。缺少 Relax 能量时不假定为零，也不直接进入 MC。

`branch_hull_batches` 保存按 MLIP 版本隔离的本批次能量池，能量为实际胞总 eV、
组成为实际胞计数。任意覆盖域内的组成通过非负相混合线性优化获得凸包能量/原子；
覆盖域外返回未知。首次筛选以 `E_relax/N - E_hull` 为主，并以较小权重使用 σ，
保留随机探索与 Agent 区域约束。缺失 σ、Ehull 或凸包参考均记录为 unknown，不填成零。
选中 Branch 从最低能 Relax 构型开始 MC。这里不训练 Branch 最低能预测器。

MC fidelity 指实际 MC proposal 步数上限。根据 `Process_AL_MC` 源码，`max_steps`
就是实际步数上限，`patience` 才是可选的无改善提前停止窗口。完成状态按 Relax 正常
返回和有效 pool 判断，不把 MC 搜索误当作几何或能量“收敛”测试。

以下有限池 KDE 接口保留用于历史实验；正式入口由上述预弛豫筛选替代 KDE 初始选择。

正式路径不是 BOHB 或标准 Hyperband：Agent 先从合法台账中给出本轮 Branch 批次；
仅对该批次做 Relax/Hull 预筛选；随后使用 small/medium/large 分档 MC。近 hull 且因
patience 早停时优先换种子开结构级新搜索段；只有持续改善才晋级。每 branch 的段数和
累计成本都有上限。原有限池 KDE/BOHB 接口保持关闭，仅供离线对照。

固定 scope 包含 `mlip_version`、`hull_reference_version` 和 `candidate_set_version`。任一项改变都开启新 scope，旧观测只保留作历史，不能直接混入新密度模型。

BOHB 配置中的预算档位只代表可比较的 MLIP+MC 搜索投入，例如 `[10, 30, 90]` 个约定 MC 单位。它与检查、Relax、MC、DFT 单点、DFT 弛豫五阶段无关。DFT 选点和模型微调仍由 `active_learning` 管理。

损失默认按组分计算：`(最低已发现能量/原子 - 固定组分参考能)/固定尺度`。也可以把 `value_key` 改成相对固定凸包的能量差。缺少参考或尺度时损失为 unknown，不参与晋升。

`run_bohb_iteration` 产生或执行一轮动作；没有 evaluator 时写入 pending，之后由 `collect_bohb_results` 回收。任务以 scope、branch、预算和种子唯一标识，重复回收不重复计费。状态可用 `save_bohb_state` 和 `load_bohb_state` 保存恢复。

每个 MC 结果分别记录请求最大步数、实际步数、patience、停止原因、预留成本和实际 GPU
核时。当前后端只能从上一段结构重新开段，不能恢复完整 Markov 链随机状态，因此明确
标记为结构级新搜索段，不称为严格 checkpoint 续跑。

模拟运行：

```powershell
conda run -n py1 python -m experiments.bohb.example_bohb_simulation
```

先使用 `evaluate_budget_correlation` 检查低/高预算排序相关性。相关性不足时不应采用激进晋升，也不能宣称 BOHB 优于固定预算或随机 Hyperband。

离线回放固定同一候选池、初始种子和 MC 总成本，报告 `random`、
`default_relax_hull_tiered_mc` 与 `bohb_variant`。BOHB 特征只能来自做出 MC 决策时已知的
字段；缺少手工特征或评价指标时返回 `insufficient_data_cannot_determine_benefit`，不补值、
不编造收益。旧的 `fixed_budget/random_hyperband/bohb` 键暂时保留给历史调用方。
