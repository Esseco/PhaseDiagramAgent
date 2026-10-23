# 结果与分析

按 `epoch0/`、`epoch1/` 分轮；`output_index.md` 索引已有产物，尚无结果时不会生成索引。

| 目录或文件 | 用途 |
| --- | --- |
| phase_diagrams/mlip/ | 当前模型的机器学习势相图 |
| phase_diagrams/dft/ | 第一性原理相图 |
| phase_diagrams/combined/ | 合并相图 |
| phase_diagrams/*/history/ | 历史相图快照 |
| Search-group-*/DFT-round-*/comparisons/ | 能量、受力逐点比较与误差指标，plots/ 为图片 |
| Search-group-*/DFT-round-*/training/ | 训练结构和标签 |
| Search-group-*/DFT-round-*/diagnostics/ | 磁矩、验收与检查信息 |
| round_metrics.csv | 当前模型各 DFT 轮误差汇总 |

根部 `round_summary.csv` 和 `gpu_usage.csv` 是跨轮汇总。原始回传和已校验结果在 submissions/epochN/，此处保存分析产物。
