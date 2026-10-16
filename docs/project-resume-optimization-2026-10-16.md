# 三个现有项目续跑兼容优化

本次复用已安装的 phase_agent，不操作真实服务，不调用真实 LLM、MLIP、DFT 或 HPC 后端。

## 结果与规则

- Co、Cr、FeMn 的已确认配置均通过检查；按实际运行中的显式目录迁移映射，各相母结构文件均存在。此前只检查原始快照路径导致的 FeMn 路径警告不成立，无需改配置或搬文件。
- 三个项目原待审批方案分别是 Co/Cr 的 DFT 选点和 FeMn 的新 branch，均通过当前原验证器。保留原方案、审批身份和内容。
- 用户本次明确授权取消 FeMn 的 `DFT-6d305f7a57be`、`DFT-bb5ae677f250`、`DFT-0dcd5e7cb3bb`，不再等待、不再回收。仅取消本地跟踪，不发送 scancel 或其他远端终止命令。保存取消前状态和用户原话。
- 以后明确关闭当前 DFT 回收问题，执行同样的本地取消：区分真实回收、取消和未返回；取消解除等待，但不算成功、失败、DFT验证或零成本。保留原成本未知信息。迟到回传不能自动重开或计费。
- 复用 FeMn 已保留训练计划、训练回传指纹、主模型 SHA256 和精确数据 ID，可恢复 27 条训练身份。仅在内存副本中派生，不改原模型注册/审批记录，不把新数据追加到旧训练集。缺失、冲突、模型不匹配或祖先来源不明仍保守判缺证据。
- 独立交付误差排除这些训练 ID 及相同几何；下一次训练继承完整来源。此处恢复来源并不意味着已达到科学收敛。

## 实现复用与变更

`relocate_workspace_paths` 沿用；新增只读 `runtime.project_readiness` 可通过 `python -m phase_agent.runtime.project_readiness --runtime <agent_runtime.json>` 检查。

`training_provenance` 从原计划与回传模型恢复来源，接入既有 `independent_evidence`、远程训练输入和通用训练处理器。

既有 `dft_recovery_decision` 增加精确取消 helper；`dft_round_status` 区分 cancelled/resolved/recovered。既有回收、反馈、收敛循环、摘要和短记忆接入该事实。不新建调度器，不更改封装算法，不追加模型调用。

## 验证

首批 12 项契约测试先红后绿；随后相关回归 125 项通过。最终扩大回归 **199 项通过**；取消说明与短记忆补充 **32 项通过**（部分用例重叠）；封装假后端单条完整链路通过。详细命令与结果以 `build/project_resume_20261016/logs` 与 `verification.json` 为准。另以假后端验证 Relax→三段 MC→安全恢复，不执行真实计算。

扩大检查发现的三个旧测试断言单独排除：旧简短模板期待长说明的两项（其中一项已在前次任务确认），以及独立远端脚本测试未设置 `transfer_bundle` 导入路径的一项。新增两项旧问题已在未修改生产源码复现；不借此任务改科学后端或重设计模板。

## 保护与回退

源码逐文件检查基线、备份与安装后哈希。三个项目共149个控制文件建立本次基线；除用户明确授权的FeMn state.json外，其余148个保持哈希不变。FeMn状态完整备份到工作区，逻辑差异限定三条任务的取消字段、取消审计和相关pending投影；配置、审批、模型、科学数据及其他任务完全保留。

`source-install-receipt.json` 和 `cancellation-receipt.json` 是实际同步/取消凭据；无成功凭据时不能宣称已执行。最终只读诊断见 `project-readiness-final.json`。生产原 TEMP_PLAN 不覆盖。新源码由用户后续手动加载；本次不声称服务已加载更新，也不承诺未运行的真实集群/API行为。
