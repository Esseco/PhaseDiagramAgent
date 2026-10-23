# Cr 上下文分层、操作级取证与截断续写（2026-10-20）

## 审计结论

用户指出单次正式分析84417输入token仍过高。本次沿用原科学状态，只读构造真实请求，在内存审计字段大小，未复制项目、数据库或正式状态到工作区。字符数与供应商token分开记录。

初始约20万字符请求中，branch_pool_facts约30238字符（238条完整branch观测，含源记录ID）、scientific_decision_brief约20073、round_budget_evidence约15581、available_branches约10521、qbc_candidates约9363；输出契约约10888。它们包含不同层面的度量、覆盖、任务状态、成本与格式约束。相同branch标识出现在多个视图不代表数据完全相同，不能仅按ID删除整行。完整候选、浮点精度、来源身份与未知限制保留。

查明的额外开销：

- 通用 prepare_local_batch_files 混合新计算准备和既有结果回收；只按工具名取证会把回收也扩成科学候选/收敛/全部历史。
- history 是未分类字段的兜底主题。前置并集将精确字段选择重新扩成大主题，带入历史解释、封装轨迹及重复经验。
- 已加载字段所属主题在后续无新增同主题字段时可能被误标为 unavailable，触发重复补读或错误停止。
- 现有表格编码只接部分字段且8行起用，遗漏少量但内容重复的大型报告。
- 当前模型已有效登记新branch，原阶段说明仍重复“刷新后先首次生成”，与登记完成事实并存，诱发重新论证。
- 截断重试原先从头发送同一科学请求，没有续接已生成的JSON和推理。

## 最小修正

1. AnalysisPlan 增加 comparison_operations，复用既有操作目录；程序校验操作对应工具仍在 allowed_tools。recover_relax_mc_batch/recover_ordering_search 只预载配置、模型、任务、预算、登记计划与回执，科学选择仍使用原有完整前置条件。实际 ordering 登记字段为 ordering_searches。最终方案若改选科学路线，原完整候选及前置证据先补齐，随后仍做完整原状态校验和审批。没有关键词路由或新增动作选择规则。
2. decision_feedback、recent_experience、recent_generation_allocations、relax_mc_batches 作为支持层完整索引，标注 supporting_memory/on_demand，必要时按精确字段完整读取；回收操作强制读取相应登记回执。当前候选、实测成本/结果摘要、人工约束、原始科学decisions和预算预留、未知新科学字段不默认裁掉。未加载历史保持未知；不能据此宣称不存在结果、限制或因果证据。Agent可以主动补读，也可以升级 high/max。
3. 已加载证据参与主题可用性判断，避免重复补读的假缺失。task_observations/reports/comparable_cohorts/observed_budget_outcomes/decision_feedback/model_round_summary/stages 复用原表格编码，2行起仅在实际更小时启用。布尔、整数、浮点不能因Python相等而被合并；缺失/null、顺序及精确数字均保留。操作目录公开列表结构保持原样。
4. 原模型生成事实与 post_refresh_search 说明明确：当前模型及刷新校验和匹配的有效新增记录已完成首次生成前置；首次搜索未完成不撤销该事实。首次登记确实缺失时，原门槛继续生效；撤回、旧版本记录不满足前置条件。后续科学方向仍由Agent判断。
5. 官方 DeepSeek 的 thinking length 截断已有部分JSON时，第二次请求使用 beta prefix continuation，保留原全部证据、部分正文及原 reasoning_content（仅内存）。最多一次恢复、同一deadline、取消与严格最终JSON解析保持。真实小探针发现 prefix 不兼容 response_format=json_object，现仅续写请求省略该参数，不放松本地解析。失败不再追加第三次兜底。自定义非官方端点不把请求发送到官方域，保留兼容的有限重试；纯思考且无正文仍走原有限扩容/强度恢复。

官方接口依据：[Prefix completion](https://api-docs.deepseek.com/guides/chat_prefix_completion/)、[API schema](https://api-docs.deepseek.com/api/create-chat-completion/)。此机制复用进度，仍需发送原证据，不能宣称第二次请求零输入token。

## 回归与控制审计

- py1 最新相关回归337 passed，1 deselected；覆盖实际登记/旧版本失效、原门槛、操作校验、原样精确读取支持层、完整候选与未知科学字段、后续改选路线、思考升级、审批/预算/取消、严格JSON、原deadline和一次重试上限。
- 排除的已知原有失败仍为 test_transport_optimization.py::test_pending_activation_continue_shows_saved_plan_without_scientific_call，审批链接展示断言，先前已用修改前源码复现。未更改该断言。
- ruff check、目标format检查、git diff --check通过。不格式化其他既有测试或覆盖其他未提交工作。
- 同一科学范围配对：200802 → 171890字符；238个branch及58个DFT候选仍完整。差额包括支持层改为按需，不能把该项声称为所有传输字段都相等。
- 单独可逆编码配对：189077 → 181988字符，解码后的科学内容逐项一致（含JSON类型）；仅transport_metrics测量值不同。
- 同一结果核验范围配对：160452 → 76828字符；72个来源字段均可读取。核验范围不冒充科学选择范围。

## 真实只读测量

| 指标 | 上一轮成功探针 | 本轮最终科学快照探针 |
|---|---:|---:|
| 单次正式输入token | 84417 | 73781 |
| 本轮总输入token | 178389 | 83992 |
| 本轮总输出token | 16851 | 7473 |
| 本轮总token | 195240 | 91465 |
| API次数/重试 | 3/1 | 2/0 |
| 测量耗时秒 | 85.750 | 40.781 |

本轮模型正确识别58个branch已登记、首次搜索未完成，选择 low 科学比较，主动保留生成、MC与收敛等路线，最终提出 select_dft_candidates 并通过原方案验证。没有通过强迫选择核验模式获取低token。单次正式输入仍有约7.38万token，仍是宽范围科学比较，不能称已降到轻量级。

合成小型真实接口探针：首次1024输出token截断后，prefix continuation成功，仅新增421输出token，600个连续整数严格相等；2次API合计3879token，5.781秒。证明续写参数及合并解析可用，不等同于所有科学截断均保证恢复。开发中还执行了参数兼容性失败探针，未混入上述性能对照；本表不是开发阶段全部API累计费用。

科学探针只构造、读取、形成和校验JSON，没有工具dispatch、工作流执行、输入准备或提交计算。state.json前后SHA256均为 9d51316fb2da2a3d3e059783e802ef3f732d4f13c6aee03a8d798b1313b452a6。不是完整Studio端到端对话：未重建全部会话记忆或自动回收链。两次科学探针最终选择不同，数据为单例观测，不是严格统计性能基准。耗时包含网络、排队、思考与输出，不能单独推断纯思考时长。

## 生效与边界

真实服务未启动、停止或重启；须用户下次手动重启加载最终代码。旧代码保护保留。源文件基线及备份为同级 _Process_PhaseDiagram_source_backups/context_layers_20261020/baseline.json。交付前移除本次临时脚本、对应pyc与TEMP_PLAN，保留小型指标报告，不保留项目快照或原始模型思考。

宽范围科学问题仍会加载大量精确记录；模型可因新证据补读历史或升级推理，因此不保证每轮节省相同比例或永不截断。当前未强制压缩科学判断范围，也未硬设候选数量或输入token上限来伪造性能改善。
