# 结果与分析

按 `epoch0/`、`epoch1/` 分轮；`output_index.md` 索引已有产物，尚无结果时不会生成索引。

| 目录或文件 | 用途 |
| --- | --- |
| phase_diagram/versions/<指纹>/ | MLIP 主图 main.png 与 HD 打叉图 hd_cross.png、数据、报告及校验清单；不可变归档 |
| phase_diagram/latest/ | 本 epoch 最新完整 MLIP 双图与数据副本 |
| dft_error/versions/<指纹>/ | 本模型累计同帧 DFT 能量/受力误差图、MAE/RMSE、比较表与数据清单 |
| dft_error/latest/ | 本 epoch 最新完整 DFT 误差副本 |
| phase_diagrams/mlip/ | MLIP 底层快照与兼容导出 |
| phase_diagrams/dft/ | 旧第一性原理相图历史，普通更新不生成 |
| phase_diagrams/combined/ | 旧合并相图历史，保留可读 |
| phase_diagrams/*/history/ | 历史相图快照 |
| Search-group-*/DFT-round-*/comparisons/ | 能量、受力逐点比较与误差指标，plots/ 为图片 |
| Search-group-*/DFT-round-*/training/ | 训练结构和标签 |
| Search-group-*/DFT-round-*/diagnostics/ | 磁矩、验收与检查信息 |
| round_metrics.csv | 当前模型各 DFT 轮误差汇总 |

根部 `round_summary.csv` 和 `gpu_usage.csv` 是跨轮汇总。原始回传和已校验结果在 submissions/epochN/，此处保存分析产物。

项目结果根部 latest/phase_diagram/ 与 latest/dft_error/ 同步当前模型的最新完整副本。先读 latest/*_status.json（epoch 内为 latest_status.json）：not_ready / publishing 时，保留的旧文件不代表当前输入已发布。任一必需图片失败均不发布半套版本；重复输入复用不可变版本。DFT 只累计本模型已核验的同帧配对，不用旧模型预测替代新模型，不自动视为独立测试集。
