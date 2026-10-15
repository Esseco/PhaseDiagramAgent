# 模型切换后的旧验证指令范围修复（2026-10-14）

问题：training_validation_planning 原样进入决策上下文，且只要非空就触发“微调后策略判断”。FeMn 新模型已激活、285项刷新已回收，旧独立验证指令却仍引导准备20个DFT单点。

改动：模型刷新完成后的无版本旧计划/跨模型计划投影为历史证据，不再发送旧执行指令、followup_action 或激活限制。保留CV指标、独立测试局限和训练排除数据；未知模型归属不猜测。明确绑定当前模型的有效验证请求继续保留。当前模型若又有新回收、未激活候选，仍进入训练审阅，不被旧刷新完成标记覆盖。

新增当前模型branch/MC事实，排除刷新任务与旧模型MC；科学请求和紧凑传输保留这些事实与决策权限说明。仅改变依据范围，不写死下一个动作，不禁用DFT，不绕过审批。缓存快照的请求入口重新提取这些范围事实，历史快照不改写。新训练计划登记模型/配置范围。

FeMn：撤下未审批、未执行的20个DFT单点待确认方案，完整归档；将旧训练验证计划标记为历史。任务、预算、模型、审批、已有结果等其他顶层科学状态逐项相等。状态修改采用备份、前后哈希检查及原子替换。备份：build/training_scope_fix/FeMn.state.before.json；审计：FeMn.maintenance_verified.json。

验证：生产源码60 passed，2 deselected；两条既有展示文案断言已单独复现（test_review_is_presented_not_only_purpose、test_header_is_rebuilt_from_model_activation_result_and_pending_draft），与本次范围修复无关，未改动相应展示模块。8个修改源码lint和format检查通过。真实FeMn只读核对：145个branch候选、新模型MC完成0、待确认方案0、旧计划historical。未调用真实LLM、运行计算、提交HPC或管理服务。

新逻辑需用户手动重启FeMn服务加载，之后“继续”按当前模型证据重新提案。
