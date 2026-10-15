# Cr DFT 待审详情修复（2026-10-15）

修复 artifact_path 在缺少 workspace_layout 时读取已确认 storage.workspace_root；空响应提案改为展示现有待审方案并复用原有身份/中断校验，不再创建“0目标”的虚假卡片。DFT 卡片和简洁答复明确显示选中结构及单点/优化数量。详情从现有账本读取超胞 H，不创建结构或任务。

补出 `E:\0-PhaseDiagram\O3-1ele-NaCrO2\workflow_state\dft_plans\dft_selection_94fddada09f1ec05.json`，15个原结构、15个单点、0个结构优化；相、H、detH、Na含量、原子数、已有选点理由来自当前保存证据。保留已记录的路线比较，不重新做科学决策。epoch0 完成初次搜索后可直接选点DFT；本次不新增收敛/停止审批，不改变下一轮决策或审批权限。

状态文件与导出前备份逐字节一致，仍为1个待审方案；未批准、准备输入、执行计算、提交或启停服务。源代码下次用户手动启动服务加载，历史消息不会被改写。

验证：生产代码27 passed，6个旧展示断言失败在未改生产基线复现后排除。Cr真实导出15个ID、中文数量、超胞信息、状态、文件路径和简洁回复验证通过。源码备份 `E:\jupyter notebook\1-AL_for_PhaseDiagram\_Process_PhaseDiagram_source_backups\cr_dft_review_20261015_172932`；项目状态备份 `E:\0-PhaseDiagram\O3-1ele-NaCrO2\workflow_state\maintenance_backups\dft_review_20261015_172958\state_before.json`。
