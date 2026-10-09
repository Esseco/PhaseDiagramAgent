# Deep Agents 分析适配

py1已安装deepagents 0.7.23、langchain-openai 1.6.7。适配位于decision_layer/agent/deepagents_proposal.py，返回兼容客户端接口，仍经LangGraph提案校验与原审批执行。

搜索客户端参数proposal_harness显式设为deepagents才使用新适配，默认legacy保持现有行为；配置与意图识别接口不切换。不修改现有生产配置、不调用付费API。

安全边界：传输使用摘要白名单，不发送state、原始结构/轨迹或路径字段。Deep Agents只有调用期StateBackend，不接入生产文件系统；已审核参考记忆通过/memory/AGENTS.md装载。task与execute工具明确拒绝，无提交/删除/训练/激活工具。新调用不复用上一次草稿；长期审核记录仍由现有业务记忆管理。

已补齐：程序生成的输出schema、修正请求的校验错误，以及候选ID/Ehull/标量误差与预算字段；离线DFT方案经过真实Deep Agents子图和LangGraph方案校验。工具调用次数计入_llm_usage。

限制：当前是显式试用适配，不是全量生产迁移。白名单仍需与微调、MC和generation_plan的实际摘要逐项对齐；材料不足时不能跳过方案校验或静默回退。真实DeepSeek工具调用与长方案验收尚未运行。不自动修改生产配置。
