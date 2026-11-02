# tests

单元测试覆盖配置、决策、执行、科学计算、数据与分析功能；跨层闭环由 test_complete_training_epoch_flow.py 等集成用例覆盖。测试使用 mock、小结构和轻量 evaluator，不启动正式 DFT、训练或集群任务。

默认运行：`C:\ProgramData\anaconda3\envs\py1\python.exe -m pytest -q`。迁移或修改恢复协议后，至少运行导入收集、全量测试和多轮恢复集成测试。

## 临时文件与大型项目测试

- 普通 pytest 使用系统临时目录下的独立短目录；成功、失败、收集失败或正常中断后均清理本次产物。TEMP/TMP 和 Python tempfile 同时隔离，并在结束后恢复。
- `--basetemp=E:/pdt` 现在指定父目录；每次运行使用独立子目录，只删除本次拥有的子目录，原有文件不会被 pytest 清空。并发测试不能互删目录。
- `real_project` 标记的测试会读取并复制本机真实项目，默认跳过。确需验证时显式使用 `python -m pytest --run-real-project-tests ...`；断言和真实历史覆盖保持不变。
- `python tests/recovery_release_gate.py --basetemp E:/pdt --junitxml E:/reports/recovery.xml` 使用同样的生命周期；JUnit 报告放在临时运行目录之外。默认门禁不包含真实项目测试；真实项目回归用上述 pytest 开关单独运行。
- 不再把唯一命名的 basetemp、项目拷贝、回放或压缩包堆到 build。临时验证工作区使用 TemporaryDirectory；普通用例只生成最小必要数据。默认不保留失败副本，确需排错时单独导出最小失败样本并明确保留范围。
- Windows 文件句柄未释放导致清理失败时明确警告并给出路径。强制杀进程、系统崩溃或断电无法保证清理回调执行。

## 按主题阅读

| 主题 | 主要文件 |
| --- | --- |
| 四角色推理与执行 | test_decision_team.py、test_expert_workflow_transport.py、test_simulation_agent.py、test_four_agent_closed_loop.py |
| Codex 包与协议 | test_codex_package_layout.py、test_codex_mcp.py、test_codex_control.py |
| 原生专家、人的介入与记忆 | test_codex_native_team.py、test_memory_experience.py、test_domain_memory.py |
| 新旧项目格式 | test_codex_offline_cold.py、test_new_project_rounds.py、test_project_resume_compatibility.py |
| 生命周期与跨进程恢复 | test_complete_training_epoch_flow.py、test_lifecycle_checkpoint_resume.py |
| 单动作与审批 | test_single_action_dispatch.py、test_retired_action_loop.py、test_native_approval.py |
| 诊断与状态只读 | test_diagnostics_archive_readonly.py、test_startup_reliability.py、test_status_presentation.py |

保留已有测试文件路径，维持跨测试的最小合成夹具引用；不用复制正式项目或大型检查点来组织普通测试。
