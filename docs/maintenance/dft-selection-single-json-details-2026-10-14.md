# DFT 选点单 JSON 详情

2026-10-14。页面保持简洁，审批前追加一个 JSON 链接；每次选点一个文件，保存于项目 workflow_state/dft_plans。文件含候选/结构/branch ID、路径、相、已知 H 和 det(H)、原子数、Na 含量、组成、选点动作及理由、Ehull/QBC 及版本、预算和输入预览。缺失信息为 null/unknown，训练集独立性不作猜测，未见DFT记录不等于从未计算。

批准执行后，原文件增加实际任务、成本、输入准备结果/目录和验证记录，保存审批前的计划数据。输入准备拒绝/失败不标为 prepared。记录链接进入 qbc_selection_history，后续可读取。未改变科学选点、审批、预算或提交策略；未运行真实 LLM/DFT/HPC，未操作真实服务。

验证：生产细节/branch兼容/DFT预览及协议测试18 passed / 1 deselected；现有预算化DFT子任务集成检查1 passed。共19项通过。排除的旧详细聊天断言在修改前基线也不满足，不恢复与用户简洁偏好相冲突的详情展开。Ruff检查通过。当前Co/Cr/FeMn无待审批DFT选点，未创建空报告。

原件备份位于工作区 build/dft_review_artifacts/baseline；部署哈希核对完成。服务由用户手动重启加载新提示。
