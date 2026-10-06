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

## 每轮 DFT 评估与绘图 CSV

远端 DFT 结果先保存全部元素的逐原子 `magnetic_moments`，不按范围拒收。Agent 本地读取后才生成 `magnetic_check` 和 `spin_state_check`：仅明确层状氧化物中的 Fe/Mn 做合理性判定，通过才用于相图、训练和误差；异常/缺失保留原始数据并暂缓科学使用，其他元素只保存不判断。简述含合理范围/实际合理值及异常任务，完整原子清单为 `magnetic_moments.csv`。复用 Process_Vasp 和证据指纹，不将经验通过当作严格基态证明；不改能量/完成状态，不删原始帧或自动重跑。轮次简表分列自旋通过/拒用/未知数量。规则与远端依赖见 `scientific_layer/dft/TRAINING_RESULTS.md`。

回收反馈在相识别后自动导出已保存的数据和同几何预测；导出本身不调用模型、不重复识别相。布局为：

```text
phase_diagrams/
  round_summary.csv
  <MLIP版本>/
    phase_diagram.csv
    history/
    round_metrics.csv
    dft_rounds/
      Search-group-0001/
        DFT-round-0001_<upload_operation_id>/
          energy_comparison.csv
          force_comparison.csv
          magnetic_moments.csv
          metrics.csv
          mlip_dft_metrics.json
          dft_records.json
          training.json
  dft/
    phase_diagram.csv
    history/
```

- `energy_comparison.csv`：每个已回收任务一行，含任务/结构/branch、Na/O2、实际相、原子数、末帧索引、DFT/MLIP 总能及每原子能、带符号误差和绝对误差。绘图推荐 x=`dft_energy_eV_per_atom`、y=`mlip_energy_eV_per_atom`。
- `force_comparison.csv`：每个原子的 x/y/z 分量各一行，含零起始 `atom_index`、元素、DFT/MLIP 力和误差，单位 eV/angstrom。绘图 x=`dft_force_eV_per_A`、y=`mlip_force_eV_per_A`；保持原结构原子顺序，不按元素重排。
- `magnetic_moments.csv`：每个原子一行，保留模型/任务/结构/最后帧索引、元素与原子索引、原始标量或 xyz 矢量磁矩（μB）、合理性判断和范围。reasonable/out_of_range/unknown/not_checked 分别为合理/异常/未知/未检查；其他元素的原始值仍保留。
- `metrics.csv`：每轮三行（总能、每原子能、力），含 MAE/RMSE、单位、样本数及回收/待完成/已配对数量。能量按结构等权，力按全部原子 xyz 分量等权；不输出 MSE，不拟合能量偏移。
- `round_metrics.csv`：该 MLIP 版本各轮汇总，带 Search-group、DFT-round 和操作标识，可直接比较各轮误差。
- `round_summary.csv`：每个 MLIP 轮次一行，记录 branch 提出/入选/登记数量、Relax/MC 完成数量、DFT 入选结构与任务数、回收率、成功/失败/未回传数量及该轮能量/力 MAE、RMSE。误差另列有效配对结构数和力分量数；无法证明来源的旧 branch 数量留空，不猜测为零。

### 原轮模型配对与部分回收

第 n 轮误差固定为第 n 轮 MLIP 对第 n 轮 DFT 真实末帧的预测误差；模型后续更新不重标历史评估。模型引用与本地文件 SHA256 保存于 `model_registry`，预测和 CSV 同时携带版本与指纹。原轮模型不在本地、版本不匹配或同版本文件被替换时明确标为未评估，不使用当前新模型补位。登记引用不等于备份模型权重，原模型文件仍需保留；可在对应模型配置中提供 `local_model_path`，历史版本可显式登记到 `mlip.comparison_models`。此前已入账但未评估的结果不会因重新导出 CSV 自动调用模型补算。

部分 DFT 回收后，Agent 显示回收数量/总数、比例、成功和失败数量，询问剩余是否继续回收。单说“继续”不代表放弃；明确回复“否”或“不再回收”才解除所询问任务的等待。该决定不改变原始任务状态、不取消远端计算、不释放未知实耗预算；不同轮次及后续新增任务不继承。迟到结果仍正常回收，收敛证据保留缺失风险；下一步仍由 LLM 提案并正常审批。失败结果已回传与结果尚未回传分别统计，不混为一类。

画对角线图时只取 `comparison_status=completed` 的配对行。预测缺失、模型/单位/末帧不匹配时，MLIP 值和误差留空，原因保留；未知不写零、不计入误差。剩余任务未回传或尚有未配对结果时汇总标为 `partial`，已有有效配对仍可导出。

DFT-round 和 Search-group 名称复用已保存的提交路径/上传轮次登记，不以 remote submission 序号猜测轮次；缺证据标为 `unassigned`，关联冲突明确拒绝混写。跨模型/搜索组/轮次不合并。DFT 凸包继续使用独立 DFT 能量口径，不能与 MLIP 能量混用。旧 `dft_results/<版本>/...` 产品不删除、不迁移；下一次正常反馈刷新导出路径。JSON/CSV 内容不变不重写，已有相图路径与历史快照不变。

# 本地 MLIP Relax 凸包池

每次回收已完成的 Relax/MC 任务后，先对最终结构识别实际相；相识别按结构文件内容哈希缓存，未变化不重复计算。无 Na 端点用母结构与合法超胞核对；无法识别的结果保留在台账，但暂不进入凸包。已识别结果按 MLIP 版本建立独立能量池，保存到运行配置的 `branch_energy_pool_ledger_path`；新 MC 能量更新当前版本凸包，旧版本和已审批批次的冻结参考仍保留。池中保留原始 eV 总能量，`hull_energy_per_atom` 只在已覆盖组成范围内求凸包，范围外返回未知。Relax 预筛的最低能与三个初态能量 σ 仍只取 Relax 样本，MC 仅影响后续凸包参考。MLIP 和 DFT 相图分开保存，MLIP 相图按模型版本隔离。
