# LangGraph 单框架迁移进度

目标：LangGraph 统一编排，保留科学模块与 Pydantic 数据校验库，最终移除 Pydantic AI Agent/UI 依赖。

## 已完成：中央动作循环与决策图

- py1 已安装 LangGraph 1.2.14，当前依赖统一写入 requirements.txt。
- run_event_loop 已用 StateGraph 编排 execute_validated_action → persist_action_outcome → 条件继续/结束。
- 原配置、预算、审批和工具验证仍在 run_tool_step 中；审批等待、拒绝、失败、预算耗尽、待运行任务都会结束当前图调用。
- 新增 LangGraph proposal 图：scientific_proposal → validate_proposal；保留原科学修正、用量和 typed contracts，不新增执行权限。
- 正式启动器已切到 run.studio_service，决策后端固定langgraph；不再加载Pydantic AI。
- 针对性测试：模型刷新/事件循环/展示19项通过；决策/事件循环/原聊天49项及5项子测试通过。两组有重叠，不相加。

## 尚待完成

1. 完整科学图持久化尚未实现：审批子图已有SQLite checkpointer和原生interrupt；业务JSON/state/ledger仍负责科学执行恢复。
   不能在具有提交副作用的节点上直接启用自动重放；先明确 action key、版本、结果入账和提交不确定状态的协议。
2. 真实浏览器与实际模型接口联调验收（隔离回归不等于生产验收）。

## Runtime上下文分离

生命周期和工具图使用LangGraph Runtime context保存调用期frame；公开run入口为每次调用创建独立上下文。图状态记录阶段、工具选择和结果。低层build入口要求调用方显式传入新上下文。
验证包含并发调用隔离、运行对象不进入节点更新、缺少上下文显式报错，以及原审批/嵌套图回归：820项测试和5项子测试通过（py1，52.01秒，排除BOHB集成测试）。
后续已接入原生持久审批子图；完整科学流程恢复仍依赖业务快照。见NATIVE_APPROVAL_RECOVERY.md。

## 已完成：整体生命周期与工具编排

- 完整隔离回归：818项测试、5项子测试通过（py1，51.77秒）；按当前范围排除BOHB集成测试。未做实际模型API与生产浏览器验收。
- run.main入口真实拆分为初始化、回收对账、科学反馈、结果等待屏障、评估导出、动作子图、输入准备收尾节点，而非单个节点调用整个旧workflow。
- run_tool_step拆分为动作初始化、新模型刷新屏障、提案、审批、科学/预算校验、具体工具分支和审计入账。
- 所有registry工具均有独立tool__名称分支；不使用允许模型自行循环执行的无审批ToolNode。
- 结果等待会在决策前终止；分析/导出先于下一步建议。拒绝或审批等待不会调用工具；无效动作只入账，不执行。
- 正式handler与无显式后端的决策请求均默认LangGraph；旧legacy决策旁路已移除。
- 原业务invocation_id、pending审批和JSON快照保留，磁盘恢复/重复执行测试继续使用相同协议。
- 嵌套动作图显式设置独立的32步上限，不继承外层很小的单轮步数限制；不存在无限ReAct循环。
- 图frame已移入独立WorkflowRuntime上下文，含调用期manager/client/handler；图状态不再携带frame。上下文不得作为checkpointer状态保存；原生审批恢复独立于科学流程。

## 已完成：默认入口与单框架清理

- 当前页面为 LangSmith Studio，通过本机 Agent Server 连接；不再使用旧自建聊天页的用户名/密码。启动见 STUDIO_LOCAL.md。
- run/agent_web_service共用handler；新项目确认配置后控制服务跟随delegate，不重复加载运行。
- 删除Pydantic AI专用UI、pilot桥接、旧测试和依赖声明。Pydantic只保留为数据校验库。
- 重复依赖入口已删除，安装统一使用 requirements.txt；旧pydantic_web_port仅兼容读取端口值。
- 保留原工作流、审批、台账和兼容API，未重写科学算法。
- 已安装在py1中的旧框架包未卸载（避免影响其他项目）；本项目源码和依赖集不再导入/要求它们。

启动说明见[LangGraph聊天入口](STUDIO_LOCAL.md)。整体运行编排已迁移；原生审批检查点已启用；不能将科学业务快照恢复说成任意LangGraph节点恢复。

没有启动/停止生产服务、迁移业务状态、分析实际数据或生成/提交科学任务。
