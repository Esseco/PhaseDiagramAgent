# UI 渐进类型化

Pydantic/Pydantic AI不是页面框架，不直接替代HTML或Open WebUI。当前保留原页面与HTTP认证，从接口边界逐步迁移。

第一步run.ui_contracts定义ChatRequest/ChatResponse：严格验证角色、消息内容、会话ID和stream，拒绝错误类型且不回显输入值。文字分片必须有字符串text；图像等扩展分片保留，不在本接口解释或执行。验证后原消息字典传入旧handler，不因model_dump补默认值而改变用户上下文。普通客户端扩展字段仍允许。

聊天响应仍为OpenAI兼容格式，SSE包装沿用现有实现；响应输出前验证，科学state不直接转为聊天正文。模型schema校验不是用户批准，不变更认证、配置确认或动作审批。

下一步：为状态/待审批方案/控制请求逐个建立契约，再将页面展示与结构化响应分离。禁止把所有控制操作统一改成模型可执行tool。Pydantic AI决策后端通过服务启动--decision-backend显式选择，默认legacy，UI迁移不会自动启用。

第二步已实施：control_ui_contracts.StatusResponse验证公共版本、摘要ID、任务/阶段/作业计数、审批计数及有限非负预留成本；科学summary扩展字段保持原模块负责。PendingResponse验证待审批条目的身份、revision、目标列表、参数对象、proposal_hash以及count/state_version一致性。不改写原响应，不补默认字段，不触发工作流或审批；校验错误仅含字段路径和错误类型。

状态/待审批/旧控制/聊天接口61项及5项子测试离线通过。读取测试确认state字节不变，审批hash稳定。下一步迁移控制请求，而非将控制操作交给模型执行。

第三步已实施：control_request_contracts覆盖/phase/propose、/phase/decision和/phase/pause，严格拒绝错误类型及未知字段。decision必须带plan_id、state_version、proposal_hash；格式正确仍须通过原review_pending的当前版本/hash/敏感操作确认。HTTP认证之后才解析和验证；直接LocalAgentControl调用也验证。pause仍提出方案，不直接停作业。其他配置/记忆路由本阶段仅验证body是JSON对象，专用字段契约尚待迁移。

控制请求/状态/旧控制/聊天接口57项及5项子测试通过。坏请求测试确认不会进入chat/review_pending；未修改生产文件、启用后端或执行审批。

第四步已实施：配置patch/confirm、记忆propose/review、技能import/publish均有请求契约。确认标志严格布尔，不接受字符串或数字；缺省仍不表示批准。补丁必须是对象，路径不能含空段或首尾空白；reasons为字符串映射，impacts为对象。记忆record暂只验证对象外壳，领域结构、证据及收敛要求继续由knowledge_record/review_queue检查，未重复实现科学校验。技能发布仍检查项目目录范围及人工确认收敛。

HTTP和直接控制调用均验证；patch_config深复制补丁后处理路径映射，避免修改调用方对象。此阶段62项及5项子测试通过，坏请求无state写入，配置/记忆未自动确认或批准。后续迁移剩余控制响应和页面展示层，并做全量回归。

测试使用py1离线请求，不启动真实服务或修改生产数据。

更正：Pydantic AI提供官方Agent.to_web聊天UI。此前“不是页面框架”的说明不代表没有内置UI。现已新增独立可选入口run.pydantic_web_chat，详见PYDANTIC_WEB_CHAT.md；类型化API继续服务原审批/配置界面，不是官方UI替代实现。

此前只读响应收尾的全项目回归被用户中断，未取得最终结论；只读响应新增6项测试通过，但不能宣称全量通过。仍需重跑全量（排除BOHB）并核对先前失败记录。
