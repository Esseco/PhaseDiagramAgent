# Epoch 与数据类型归档

submissions 与 analysis_outputs 共用 state.upload_layout.model_rounds 的模型映射。epoch 表示模型代际，不是 MC、DFT 或微调尝试次数。模型版本不变时 epoch 不变；不按文件夹个数推测。

新版 Agent 输出目录：

    analysis_outputs/epoch0_mace-mh-1/
      phase_diagrams/mlip/phase_diagram.csv
      phase_diagrams/dft/phase_diagram.csv
      phase_diagrams/combined/phase_diagram.csv
      Search-group-0001/DFT-round-0001_<operation>/
        comparisons/{energy_comparison,force_comparison,metrics}.csv
        comparisons/plots/
        diagnostics/magnetic_moments.csv
        training/training.json
      round_metrics.csv
    analysis_outputs/round_summary.csv
    analysis_outputs/output_index.md

output_index.md 是查找入口：列出已登记的相图、误差、磁矩、绘图、训练输入路径及文件是否存在；不扫描原始计算数据、不计算新结果。round_summary.csv 用于跨 epoch 总览，round_metrics.csv 用于单 epoch 多 DFT 轮误差比较。索引在 Agent 发布 DFT 分析产品时更新，内容相同不重写。

三个相图各自保留 history。DFT 相图是该 epoch 时刻累计的合格 DFT 凸包，不表示只含该 epoch 的 DFT 数据；具体来源仍由记录列追踪。综合相图明确保留能量来源和校正状态，不与纯 DFT/MLIP 相图混同。

CSV 内增加 epoch；误差与磁矩表同时保留模型版本、search group 和 DFT 轮次。超算训练结果 CSV 保留 epoch、原模型版本、training_round 和 fold，不混同本轮原模型同帧误差。

历史 csv_path 仍可读；导出已有 CSV 不重算、不搬迁。科学数据未变时沿用当前文件；后续产生新版数据由 Agent 按新版目录发布。目录内名称简洁，完整归属由路径与 CSV 列共同说明。用户明确要求整理工作区时可运行可恢复迁移器，科学数据不重新计算，旧索引归档。

训练方案状态必须与实际目录一致。旧目录缺失或同模型已有未完成方案而配置/数据变化时，返回 confirmation_required，明确请求核对或重生成，不通过记录数量直接追加训练轮次。不自动删除、恢复或重命名历史任务；提交过的作业也不自动取消。新轮次使用已有合法轮次最大编号加一，不以记录条数代替编号。
