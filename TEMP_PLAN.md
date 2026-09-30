# 架构梳理与展示边界拆分

目标：建立层级与功能入口地图，把聊天回复展示从 HTTP/运行装配入口分离。
复用：现有全部回复规则、状态展示和审批判定，保持输出及旧导入兼容。
拟改：run/open_webui_api.py；新增 workflow_reply_presentation.py、chat_approval_rules.py；架构地图与测试。
接口影响：旧 format_workflow_reply 导入继续可用，新模块不依赖 HTTP 入口，避免循环依赖。敏感操作判定单一来源。
风险：原入口被测试 monkeypatch 时需检查调用依赖；不改任务状态、科学算法、目录和 BOHB。
验证：展示独立导入、旧新接口一致、审批/聊天定向测试及全量回归。
