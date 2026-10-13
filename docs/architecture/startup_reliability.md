# 启动与恢复可靠性

项目启动器启动一个属于该项目的 Studio 服务。旧对话恢复只结束残留对话节点，不执行科学动作；科学任务和执行收据保留。

LangGraph dev 的恢复接口包含同步检查点操作和延迟导入。因此恢复在独立工作线程的事件循环中执行。该工作线程内暂时设置已安装 Blockbuster 的上下文开关，结束后恢复；ASGI 主循环和其他调用继续保留阻塞检测，没有全局启用 allow-blocking。初始化导入和路径检查也放入工作线程。

`agent_startup_timeout_seconds` 默认为90秒，可配置5–600秒。启动器将同一 monotonic 截止时间传给 Studio wrapper，避免两层超时不一致。首次启动时等待初始化完成，不重复启动；失败和取消时沿用原进程回收与项目锁机制。

每次启动生成 `logs/agent_start_<UTC时间>.log` 和 `logs/studio_start_<UTC时间>.log`。`workflow_state/studio/service.json` 保存本次日志及异常摘要。失败页面展示当前尝试的关键异常及日志位置，不把历史失败当作本次原因，不删除项目检查点。

验证使用假进程、假控制服务器、假旧会话与定向回归，不自动启动或停止真实服务。Windows测试需要允许asyncio内部loopback套接字；这与运行实际项目服务不同。
