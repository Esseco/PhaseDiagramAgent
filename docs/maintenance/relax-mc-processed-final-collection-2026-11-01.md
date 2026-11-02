# 最终汇总复用已处理结果

NiCo补交114项完成后，原已完成且在processed_task_ids中的20项被collect_phase误报缺失：BatchRunner按设计跳过processed的MLIP任务，但阶段汇总仅统计新增返回。

修复：当前阶段processed且completed的任务必须重新通过原始result.json/完成标记、任务身份、SHA与最终结构验证，才计入覆盖；不重复feedback。损坏结果与科学验证错误仍阻止完成。

旧paused最终阶段可通过原CPU入口恢复：

```bash
python3 run_batch.py run --root WORK --submit --phase 3 --recollect-final
```

限阶段3且全部任务completed；原阶段GPU及CPU须不活跃；未知调度状态、活动任务及其他阶段拒绝。completed幂等返回。复用原collect/finish/publish/publish_output；不提交GPU，不另建打包器。实际代码上传和CPU汇总提交需用户批准，正式回收仍要求完整completed包。

验证：test_relax_mc_processed_collection.py、test_relax_mc_resume_and_gate.py、test_relax_mc_phase_dependencies.py共44项通过（68.06秒）；全应用ruff check及三个修改模块格式检查通过。覆盖已处理结果、损坏JSON/结构、科学验证错误、错误阶段、未完成任务、活动作业、原自动打包与正式回收门槛。

2026-11-01：源码已测试；真实NiCo未部署本次修复、未重提CPU汇总、未正式回收paused包。审批资料：E:/0-electerm_mcp/tasks/20261101/O3-NiCo_RelaxMC_Batch01/recollect_final_v1/APPROVAL.md。
