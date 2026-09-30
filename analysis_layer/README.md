# analysis_layer

Relax 与 MC 的最终结构在本地回收时识别相，保存原始相、实际相及未知原因，保留原 branch。
MC 审批前使用当前 MLIP 相图快照中与 CSV `ehull_eV_per_atom` 同源的 Ehull，逐一核对 Relax 最低能结构 ID、能量与路径，按现有分档规则预估完整方案步数及规模相关成本。相图未就绪或结构不能唯一对应时不派发；预览冻结相图版本，批准执行时复核。
建议同时展示完整预算与目标预算；低于目标正常审批运行，超过目标由用户选择完整运行（先核对硬上限）
或缩减 branch。逐项参数保存在审批记录 budget_preview 中；批准后复核分配校验值并准备文件。

全局科学收敛只使用已确认配置中的 MAE、相图变化、稳定模型更新 epoch 数和预算规则。
默认配置建议比较两个连续 MLIP 更新 epoch，但用户可在初始配置确认其他正整数；同一
模型版本的 MC 小轮次不计 epoch。缺失阈值或结果不会按零处理。相 × SOC 覆盖只生成
证据和风险提示，不是自动硬门槛；数值达标后仍为待用户确认。任务待回收、证据不足、
预算耗尽、数值达标待确认和用户最终接受分别使用不同状态。

`state/build_decision_context.py` 从当前相图、最近五条收益及动作记录构建 Agent 上下文，单独展示人工长期建议；保留版本来源，不重新计算或猜测科学结果。

负责对已保存或显式传入的结果进行分析：凸包/Ehull、覆盖率、搜索收益、模型重评估、收敛证据和 Agent 状态摘要。

子目录：`phase/` 凸包和相图，`convergence/` 收敛证据，`feedback/` 收益和重评估，`state/` Agent 状态摘要，`cost/` proposal 成本展示。本层不派发任务、不替 Agent 选择 action。

每次保存相图 JSON 时，同目录生成同版本的 `phase_diagram_<mlip|dft>_<version>.csv`。对同一 TM/O2 组分、只变化 Na 的结构，取已观测最小/最大 Na 含量处最低的 eV/O2 能量作端点；逐结构计算 Eform/O2，按同一 Na 含量最低 Eform 构建下凸包，再算 Ehull/O2 和供 MC 使用的 Ehull/atom。TM/O2 不一致会明确报错；只有一个 Na 含量时相图状态为 unknown，不臆造端点。CSV 含结构 ID/路径、实际相及识别状态、组分、Na 含量、端点、Eform、凸包 Eform、Ehull，并分别标记同组分最低能结构和凸包稳定结构。相图仍按 MLIP 版本与 DFT 分开。Na 层均匀性判定复用 `Process_Vasp.structure.check_layer_equal`；缺最终结构时标记 unknown。聊天中请求“导出当前相图CSV”只导出已保存的当前版本，不推进搜索。
新生成的 MLIP 相图直接按模型版本放在 `phase_diagrams/<模型版本>/`，DFT 放在 `phase_diagrams/dft/`。旧版根目录或 `mlip/<模型版本>/` 快照不移动、同版本不重复导出；CSV 的 `x_Na_per_O2` 写成十位小数。聊天中的“导出当前相图”优先返回已有 CSV；若文件丢失但 state 内当前相图快照完整，则只从快照恢复 CSV 到对应版本目录并更新 `csv_path`，不重算凸包、不重复识别相、不推进任务。

# 本地 MLIP Relax 凸包池

每次回收已完成的 Relax/MC 任务后，先对最终结构识别实际相；相识别按结构文件内容哈希缓存，未变化不重复计算。无 Na 端点用母结构与合法超胞核对；无法识别的结果保留在台账，但暂不进入凸包。已识别结果按 MLIP 版本建立独立能量池，保存到运行配置的 `branch_energy_pool_ledger_path`；新 MC 能量更新当前版本凸包，旧版本和已审批批次的冻结参考仍保留。池中保留原始 eV 总能量，`hull_energy_per_atom` 只在已覆盖组成范围内求凸包，范围外返回未知。Relax 预筛的最低能与三个初态能量 σ 仍只取 Relax 样本，MC 仅影响后续凸包参考。MLIP 和 DFT 相图分开保存，MLIP 相图按模型版本隔离。
