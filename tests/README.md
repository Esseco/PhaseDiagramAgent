# tests

单元测试覆盖配置、决策、执行、科学计算、数据与分析功能；`integration/` 覆盖跨层闭环。测试使用 mock、小结构和轻量 evaluator，不启动正式 DFT、训练或集群任务。

默认运行：`C:\ProgramData\anaconda3\envs\py1\python.exe -m pytest -q`。迁移或修改恢复协议后，至少运行导入收集、全量测试和多轮恢复集成测试。
