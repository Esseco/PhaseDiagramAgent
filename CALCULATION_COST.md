# 单任务与批次成本粗估

## 轻量实测优先

新生成任务在执行前后各取一次时间，不轮询、不调用额外科学计算。result.json 的 runtime_observation 在校验和计算前写入，回收后进入 cost_history。CPU/GPU 数量从已有 Slurm 环境取得；缺失保持空值。partition 只作为近似硬件类别，不声称识别了 GPU 型号。Atomate wrapper 与直接 VASP 脚本均记录任务运行时间，不含队列等待。

预测优先采用同后端、同硬件/资源类别、原子数在目标 0.5–2 倍范围内的最近最多30条完成任务中位数，并作粗略规模折算。报告样本数和观测跨度，不把跨度当统计置信区间。MC 按实测每步耗时估算，只有 patience 和最大步数相同的记录用于典型总步数预测。没有对应样本时回退初始参考值。

运行秒数、GPU小时和CPU核时是主要物理成本；relative_cost仍为独立预算参考，未经指定换算基准不自动将秒数写成预算费用。MLIP批次另有 `.runtime.json` 总耗时，与任务时间片不可重复相加。已有生成的输入不被此次修改重写；缺失历史运行时间不补造。

复用现有原子规模/阶段成本模型，入口为 `analysis_layer.cost.estimate_task_cost` 的 `estimate_task_cost` 和 `estimate_batch_cost`。成本单位为 relative_cost，不是时间或金额。报告包括初始粗估、历史校准比例、有效实测样本数、情景成本和局限。

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

在 py1 下运行 `python -m run.estimate_calculation_cost --tasks 任务规格.json --state 当前state.json`，只读取文件并输出报告，不生成任务、不改数据。可通过 `--config` 显式提供带 budgets 的配置。

Agent 决策上下文自动包含参考规模报告；每个实际任务仍以已有科学成本核验为准。指定 backend 时校准只采用同 backend 实测样本，未知后端样本不混入。无可靠时间标定时核时保持空值，不能把 planned_cost 或 MC 实际步数折算值当作实测成本。
