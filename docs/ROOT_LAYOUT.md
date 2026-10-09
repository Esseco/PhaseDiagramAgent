# 根目录管理

项目根目录保留代码模块、测试、AGENTS.md、README.md 和启动入口。架构与接口说明集中在 docs；开发备份在 backups/development。local 中配置和 outputs 中既有运行文件保留原位，避免改变启动与导入接口。

计算工作目录（例如 E:\0-FM-PhaseDiagram）分工如下：

- InitFile：初始结构。
- upload_batches：提交输入与原始回传 results；不放分析副本。
- current：运行状态、账本与审批。
- config_snapshots：不可变配置快照。
- outputs：相图、轮次统计、误差、训练和诊断数据。
- logs：服务日志。
- backups/configs、dft_inputs、upload_layout、output_layout：对应历史备份，仅用于追溯。

根目录的 config_session.json、search_config.project.json、agent_runtime.json 保持原位，兼容既有启动方式。OUTPUTS.md 是数据导航入口。整理目录不推进搜索、不重算、不提交任务。

2026-10-06 整理记录：旧 r68 与 .pytest_cache 存在访问权限限制，未强制移动、删除或修改权限；可在关闭相关进程并确认权限后另行归档。测试已使用独立临时目录，不再向这些目录写入。

测试临时输出默认位于系统临时目录/pdt/<唯一ID>，由tests/conftest.py设置，短路径用于减少Windows长度限制，避免污染项目或E盘根目录。不要指定E:\为--basetemp；显式覆盖时应使用专用临时子目录。此前E盘phase-*测试残留集中归档在backups/test_artifacts，保留用于恢复，非计算结果。
