# analysis_layer

全局科学收敛只使用已确认配置中的 MAE、相图变化、稳定模型更新 epoch 数和预算规则。
默认配置建议比较两个连续 MLIP 更新 epoch，但用户可在初始配置确认其他正整数；同一
模型版本的 MC 小轮次不计 epoch。缺失阈值或结果不会按零处理。相 × SOC 覆盖只生成
证据和风险提示，不是自动硬门槛；数值达标后仍为待用户确认。任务待回收、证据不足、
预算耗尽、数值达标待确认和用户最终接受分别使用不同状态。

`state/build_decision_context.py` 从当前相图、最近五条收益及动作记录构建 Agent 上下文，单独展示人工长期建议；保留版本来源，不重新计算或猜测科学结果。

负责对已保存或显式传入的结果进行分析：凸包/Ehull、覆盖率、搜索收益、模型重评估、收敛证据和 Agent 状态摘要。

子目录：`phase/` 凸包和相图，`convergence/` 收敛证据，`feedback/` 收益和重评估，`state/` Agent 状态摘要，`cost/` proposal 成本展示。本层不派发任务、不替 Agent 选择 action。

每次保存相图 JSON 时，同目录生成同版本的 `phase_diagram_<mlip|dft>_<version>.csv`。CSV 逐条列出组分、原始/归一化能量、Ehull、相、结构来源与 Na 层均匀性。Na 判定复用 `Process_Vasp.structure.check_layer_equal`（默认 3 个 Na 层且层间 Na 数相等）；没有最终结构或本地无法导入 Py-Code 时标记 unknown，不影响凸包。纯脱钠结构标记 not_applicable。
# 本地 MLIP Relax 凸包池

每次回收已完成的 Relax 任务后，`update_local_mlip_hull_pool` 将有最终结构、实际组成和 eV 总能量的结果按 MLIP 版本整理为本地能量池，保存到运行配置的 `branch_energy_pool_ledger_path`。池中保存原始能量，不把同组成最低能直接当凸包能；`hull_energy_per_atom` 从所有已覆盖组成的结构构建凸包，覆盖范围外返回未知。MC 结果进入常规相图更新，但不回写已冻结的 Relax 预筛参考池。尚无可回收 Relax 结果时不会创建空池或虚构 Ehull。
