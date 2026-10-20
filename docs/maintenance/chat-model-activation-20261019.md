# 已展示模型切换方案的聊天审批修复

2026-10-19。仅修改应用源码，不执行真实审批、模型切换、刷新、计算或服务管理。

已绑定当前会话、唯一且版本/哈希匹配的 model_activation 方案，明确同意后复用原审批所有者 confirm_sensitive；不再跳转令牌页面。初次未展示仅展示，多方案、跨会话或过期仍不执行；拒绝保持拒绝。其他敏感操作和HTTP接口认证不变。训练提示及方案查看文案一致。

源码：phase_agent/graphs/dialogue/support.py、phase_agent/runtime/training_reply.py、phase_agent/runtime/plan_queries.py。新增tests/test_chat_model_activation_approval.py使用假审批处理器，不访问真实项目。

验证：先在原副本复现6失败/4通过；安装后43项通过、3项真实项目副本检查主动排除；ruff check及修改文件格式检查通过。额外发现旧test_presented_binding_persistence的最小summary夹具缺budget_usage；未修改版本同样失败，本次不改该旧夹具，新增测试独立覆盖持久绑定、跨会话、状态与哈希过期校验。

部署前逐文件校验基线并备份，凭据位于工作区build/chat_activation_fix。项目11个受保护文件中Co/Cr与所有服务配置记录保持基线；FeMn state在期间发生外部变化（只观察到哈希变化，不推断来源），本次工具与测试未写入该真实文件，不回写或覆盖。不能声称三项目全部状态哈希不变。

本次修复没有回滚此前误执行的Co模型激活。运行服务未重启；新源码由用户下次手动启动加载。
