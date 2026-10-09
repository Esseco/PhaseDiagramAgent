# 类型化决策接口：渐进迁移

当前Agent框架为LangGraph；本文的Pydantic数据契约继续保留，但旧Pydantic AI桥接已移除。正式进度见LANGGRAPH_MIGRATION.md。

当前迁移DFT后四路分析及生成策略条目契约，未重写执行流程或引入第二套Agent状态机。

## 已实施

- decision_contracts.PostDFTReview统一字段、枚举、严格类型与长度；拒绝额外字段和空白分析。
- post_dft_review_errors复用契约验证；保留动作/choice、微调建议、停止依据以及生成参数的科学一致性校验。
- 请求附带该对象的JSON schema，修正请求沿用同一契约；仍只允许原有一次方案修正。
- 错误不包含原始输入，避免把模型输出或敏感上下文复制进诊断。
- 当前本地依赖统一见 requirements.txt，要求Pydantic 2；不改超算科学环境。

schema作为提示中的格式描述发送，不代表DeepSeek或其他后端支持服务端严格Structured Outputs，也不是动作授权。

## 后续范围

生成策略条目已使用GenerationAllocation，保留quotas/total_quota一致性检查，严格拒绝字符串/布尔/浮点配额。请求携带对应条目schema。原参数字典不被改写。

DFT采点条目已使用DFTDecision，严格验证候选ID、分类动作和reason类型，拒绝额外字段；保留旧方案reason可省略。预览和执行共用的validate_dft_agent_decisions先验证契约，再检查身份、去重和预算，格式错误不会进入成本处理。请求携带条目schema；DFT后补采点纳入原有一次修正，不新增重试或授权。原始参数不被改写。

MCAllocationFields已校验公共字段：mc_budget为非负严格整数，seed为严格整数，DFT预算和探索比例为有限数，区域ID为字符串列表。分配入口在读取科学配置及执行筛选前拒绝无效字段。只投影公共字段校验，不丢弃或重写原参数；内部budget_preview、第二轮来源等继续由现有逻辑校验。请求中的mc_allocation_common_fields schema仅描述公共字段，不是完整参数外壳。没有修改BOHB算法。

ActionEnvelope已接入LLM提议入口，在normalize_action之前验证工具名、参数对象、目标/证据ID列表、有限非负budget和说明类型，并拒绝tool/action_type冲突。保留action_type别名及省略可选字段；仅投影公共字段，不改写元数据。历史状态和规则调用的normalize_action未改变。schema中的action_common_fields仅描述公共字段，不是完整动作元数据白名单。未知工具仍由注册表校验；证据真实性、预算上限与审批仍由原逻辑负责。没有增加修正次数。

后续继续迁移剩余动作专用字段。Pydantic AI可选试点已实现，生产接口尚未切换；没有安装OpenAI SDK或新增自主工具执行。

采用框架前应比较历史失败样例的方案有效率、有限修正成功率、token成本与审批一致性。任何框架仍不得绕过现有预算、配置冻结、结构身份、证据版本、去重与人类批准。

验证：py1全项目回归707项及5项子测试通过（排除BOHB）；最后新增的schema贯穿一次修正测试另行通过，契约模块共5项通过。未调用真实模型。

生成策略迁移后的验证：37项相关测试通过，覆盖配额类型、Na范围、超胞参数、实际生成策略传递和方案修正。该阶段全量回归会话被中断，未取得最终结果，不能沿用前一阶段707项通过结论作为本阶段全量通过证据。

DFT采点迁移验证：py1下55项相关测试通过，覆盖契约、旧reason兼容、身份校验、预算、审批、DFT输入预览和一次修正schema一致性。未调用真实模型或处理生产数据；不是全项目回归结论。

MC公共字段迁移验证：31项契约及决策接口测试、25项现有MC闭环/第二轮配置/预算反馈/主动学习测试通过。未处理生产数据，也未进行全项目回归。

动作公共外壳验证：72项契约/决策测试、32项提议调用方/失败阻拦/token策略测试通过。测试未调用真实模型，未处理生产数据；尚非全项目回归结论。

## Pydantic AI 可选试点

decision_layer/agent/pydantic_ai_pilot.py提供异步propose_with_pydantic_ai(payload, model=...)，必须显式传入模型；默认Agent不会调用它。使用Agent类型化输出和现有动作/DFT后分析/MC/DFT采点/生成契约。模型最多请求一次，框架自动重试为0，无执行工具；输出只是待校验方案，不具备审批权。

历史试点曾固定 pydantic-ai-slim 2.54.0，依赖入口现已删除，不再作为安装指南。以下试点验证记录仅供历史参考；当前正式框架为 LangGraph，依赖统一见 requirements.txt。

试点与公共契约联合验证44项通过，包含无执行工具、单次请求、无效字段和DFT后缺分析拒绝；不是生产API验证或全项目回归。

现有JSON客户端桥接入口request_with_pydantic_ai_client(agent_client, payload)已实现：在工作线程复用request_validated_action及原客户端，因此保留现有token压缩、思考设置、传输重试、一次科学方案修正及真实_llm_usage。最终结果用FunctionModel重放给框架校验；重放不请求后端，其模拟用量不进入成本账本。返回原始dict，不把框架默认值/类型转换写入状态。最终校验失败保留真实调用用量，用户可见异常不包含无效响应正文。客户端异常原样向上传递。

这仍是渐进桥接，不是原生Pydantic AI provider或生产默认入口；框架并未管理真实传输和科学修正。桥接4项及客户端/token/试点相关回归共30项离线通过。下一步需历史失败样例对照及显式启用开关；当前没有切换生产配置或调用真实API。

桥接边界进一步优化：pilot_validation统一请求预检与动作验证，严格验证allowed_tools、instruction、decision_context，坏请求不调用后端。请求深复制后交给工作线程；最终外壳失败保留有效post_dft_review与真实usage，安全错误不链接框架响应异常，避免日志默认打印响应正文。后端传输异常仍沿用原客户端行为。
新增6项预检/分析保留测试后，与试点、旧客户端和token管理共36项离线测试通过。没有新增科学重试或执行工具。异步取消不能强行终止已有同步HTTP线程；若取消，调用方不能假定远端请求未发生，真实接口启用前还需处理这一记账边界。

参考官方FunctionModel文档：https://pydantic.dev/docs/ai/api/models/function/

## 实际服务的显式入口

服务启动增加`--decision-backend legacy|pydantic_ai`，默认legacy。选择pydantic_ai时启动先检查可选依赖；没有依赖明确报错，不静默降级。启动日志显示后端。RunWorkflowChatHandler把选择传入自主动作提议入口；意图识别、配置对话等仍使用原客户端。自定义非RunWorkflowChatHandler不支持此开关时明确拒绝。

决策分派decision_backend.request_decision_action复用原桥接，后端不会形成重复修正循环。同步入口检测已运行事件循环，要求异步调用者使用request_with_pydantic_ai_client，避免嵌套asyncio.run。当前运行配置未改、服务未重启或切换。

相关入口/桥接/聊天/契约60项及5项子测试通过（隔离依赖、离线客户端）。这是实际可选入口的代码验证，不代表真实API质量或全项目回归已验证。
