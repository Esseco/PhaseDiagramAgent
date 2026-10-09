# 原生审批恢复

## 范围

有state_path的交互工作流在同级目录自动创建langgraph_approvals.sqlite，使用官方SqliteSaver持久化审批子图。
审批通过interrupt暂停，通过Command(resume=...)恢复。状态只含动作ID、配置版本、方案hash、修订号和决定；不保存模型客户端、结构数组或工具执行上下文。
无state_path的调用不创建数据库，仍可使用原无持久化接口。

## 授权与恢复

原执行策略先判断当前人工回复是否合法，然后才调用原生子图。检查点中的批准不会自行成为本轮授权。
旧pending记录仍由业务层定位和校验；首次遇到时建立原生等待检查点，只有本轮明确审批才恢复。审批页仍校验state版本、proposal hash和敏感操作确认。
不同方案hash、配置版本、revision分别创建审批身份；不同工作区有独立数据库。

审批决定交付工具流程前，先持久标记delivered。此后若业务invocations已有完成记录，原幂等逻辑直接返回；若缺少完成记录，再次交付会返回approval_reconciliation_required，要求检查既有文件、任务和台账，不自动再执行。
此策略宁可把“批准后、执行前崩溃”也视为需人工对账，不推测工具是否已执行。它不是分布式exactly-once，也不是任意节点恢复。
文件锁串行保护同一审批数据库；工具执行仍受原运行服务和state/ledger协议约束，不支持多个服务同时写同一业务工作区。

## 文件管理

数据库属于流程状态，不是相图分析输出。等待审批时不要删除、移动或复制它到另一个运行中。
数据库故障会显式报错，不静默退回自动批准；不要通过删数据库绕过delivered保护。
审批子图没有工具节点，不启用工具自动重试。完整科学工作流仍从业务state/ledger回收对账后推进。

实现：orchestration/approval_graph.py；桥接：execution_layer/policy/native_approval.py。

## 状态契约与执行收据

生命周期等待门在回收和分析之后、派发新动作之前，读取execution_receipts并与业务invocations对账。收据返回状态必须与已保存业务执行状态一致才视为已记录；其他记录返回恢复清单并阻断新动作。只读检查不会创建数据库，也不自动删除收据或推断任务完成。

orchestration/contracts.py严格校验审批身份（动作ID、配置版本、方案hash、修订号）与执行身份（动作ID、配置版本、动作hash、工具）。拒绝空身份、多余字段及类型错误；审批修订号不接受字符串或布尔值。校验失败不会创建检查点。阶段标签有明确类型范围，但尚不等于完整业务状态已可持久化。

公共工具适配器execute_tool_action在有state_path时创建同级execution_receipts.sqlite。只读check_convergence与pause_search除外。该边界也覆盖模型刷新，不仅覆盖人工审批动作。

- started：调用工具前原子领取执行权；同一state路径及动作ID只能领取一次。
- returned：工具返回了成功或失败状态，不证明文件、任务与业务台账已完成入账。
- 相同动作ID改变参数或配置版本仍被拦截，不通过换hash绕过保护。

已有业务完成记录时走原幂等返回；没有完成记录却存在收据时，返回execution_reconciliation_required并停止动作循环。工具抛错、进程退出或返回后未入账均不自动重试；需核对已有文件、任务和state/ledger后再决定处理方式。SQLite串行领取不代表支持多个服务同时写业务状态。

两个数据库都属于流程状态：langgraph_approvals.sqlite保存审批交付，execution_receipts.sqlite保存执行边界。不要删除数据库来重试，也不要将收据视为科学结果。双写不是跨文件事务，也不承诺exactly-once。无state_path的低层调用没有持久恢复保护；完整科学图仍不支持任意节点自动恢复。

验证：py1隔离回归841项测试及5项子测试通过；包含并发领取、身份错误、异常/退出后的防重放以及返回后失账保护。未运行实际超算任务。
