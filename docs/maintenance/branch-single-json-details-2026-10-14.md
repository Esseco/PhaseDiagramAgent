# Branch 单 JSON 详情

2026-10-14。按用户最终要求，每次 branch 生成仅使用一个 JSON；聊天不展开细节，只给读取链接。审批前记录相/端点规则、过滤后的 H 矩阵及 det(H)、Na 占位规则、候选配额、选取上限、初态数。去重后实际数量未知时不补造。生成后在同一 JSON 的 actual 中填实际 phase、H、det(H)、Na_fraction、branch/structure 数量、ID 和结构路径。计划信息保留。完成记录不因重复显示审批被覆写成待审批。

修正了包含 module_execution_scope 的 branch 卡片边界：其实际范围含输入准备，不包含超算提交。未改变既有执行或批准机制。当前 FeMn 待审批方案只导出文件，未执行生成，原 state.json 字节核对不变。文件：E:/0-PhaseDiagram/O3-2ele-NaFeMn/workflow_state/branch_plans/branch_plan_a30bbc76c7c8681d.json。

生产核验：24 passed / 1 deselected，涵盖单文件生命周期、相/H/Na信息、当前模型 branch 前置条件、scope 及简洁链接；Ruff 检查通过。被排除的 MC 审批预算展示测试在原始基线也复现相同失败，未改与本任务无关的 MC 展示。未调用真实计算、LLM 或 Slurm，未操作真实服务。

部署前哈希检查拦截同期生成处理器更新，已保留 adaptive_branch_count 改动并只合入导出逻辑。原件在 build/branch_review_artifacts/baseline，早期版本在 baseline_initial。用户手动重启服务后加载新提示和生成后回写机制。
