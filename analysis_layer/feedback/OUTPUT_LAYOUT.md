# 输出布局

默认分析根目录为工作区 outputs（内部兼容参数名 phase_diagram_directory 保留）。相图由 phase_snapshot_directory 管理；DFT文件位置统一通过 dft_product_path 解析，导出索引保存实际CSV路径。模型版本与Search-group/DFT-round生成身份仍保留，不重新编号。

新版使用 epochN_模型版本目录，与上传目录共用轮次登记。phase_diagrams 下按 mlip、dft、combined 分别保存当前CSV与history；Search-group/DFT-round下按comparisons、training、diagnostics分文件。根目录round_summary.csv与epoch目录round_metrics.csv保留。内容不变不重写，历史路径仍可读。完整规则见 docs/EPOCH_OUTPUT_LAYOUT.md。

旧输出可通过 execution_layer.local.migrate_analysis_outputs.migrate_analysis_outputs(workspace) 做一次迁移，要求目标outputs尚不存在。先备份分析文件与配置/状态，再移动并重写路径字符串，不改科学值、任务编号或upload_batches。备份保存于output_layout_backups。迁移期间应停止本地Agent，避免并发写入；异常时从备份恢复，不重复运行迁移覆盖。
