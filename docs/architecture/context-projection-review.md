# 四 Agent 上下文管理审查与验证（2026-11-10）

## 核实结果与设计取舍

修改前 role_context 仅深复制同一快照并附权限；CodexTeam._view 在顶层及两份专家请求中重复完整 snapshot，且每次返回全部历史事件；普通 Simulation 收到身份、状态、预算、记忆与指导，原生 Simulation 只有 snapshot_id。decision_memory_context 已有共享审核知识与有界近期动作，2000 的检索 token_budget 不是整请求预算。额外负向测试复现：仅校验 action_hash 的 Simulation 旧计划可以跨修订登记到同一动作。

采用现有 evidence/agents/runtime 边界：投影前绑定快照；严格按 JSON 相等去重并记录字段去向；共享所有科学候选指标和跨阶段风险，角色仅略去其他角色的静态协议及其无权限的操作契约。DFT 数量上限等配置/资源字段仍共同保留，不能把整份 dft_selection_policy 当纯提示删掉。Strategic 咨询复用同一去重，最终普通提案仍保留原完整事实及渐进读取接口。

原生团队从当前权威状态复用 build_decision_context，补齐专家没有补读能力时所需的证据；强 state/control 来源身份加入原团队绑定，不新建事实库。当前台账解析的 branch_pool_facts 保持优先。新计划同时校验动作和上下文身份；已有冻结计划不追溯改变契约。所有科学执行、审批、回执、恢复和防重放仍属于原机制。本任务没有新增上下文 agent、四份记忆或调度链。

没有设置会截断科学证据的全局字数上限。知识检索预算仍只控制检索，不冒充整请求预算。用户约束、必要证据较大时请求可以变大；额外本地解析/证据派生/校验开销尚未做大项目延迟评估。

## 行为验证

py1：C:/ProgramData/anaconda3/envs/py1/python.exe。PYTHONPATH 前置 tests/fake_only_guard、源码仓库及 E:/Py-Code（Process_AL_MC 源码依赖）；启用 PYTHONDONTWRITEBYTECODE，pytest 禁用 cacheprovider，--basetemp E:/Py-Code/pdt；仓库 conftest 使用 TemporaryDirectory 隔离 TEMP/TMP 并清理。

新增 test_role_context_projection 覆盖：科学指标/约束/版本/未知风险不丢失、冲突值及 true/1/1.0 不合并、共享审核记忆别名不改历史、独立原生任务包、完整战略读取/单角色 MCP 读取、全量持久事件保留、空池不复活缓存、来源并发变化拒绝、普通/原生 Simulation 契约一致、任务就绪/输入校验/审批/资源/部分回收事实保留、无损表格精确还原、现有角色权限内的操作目录、旧 Simulation 意见跨轮及缺少上下文绑定拒绝。新增测试实际复现失败后完成修正。

既有测试继续覆盖：团队引用与失效、冻结动作、人工方向/执行审批分离、确认策略、旧计划恢复、失败诊断不自动重试、原生路径不调用项目模型、离线冷启动及回执/防重放。最终命令和结果在本文件末尾记录。

## 序列化大小

方法：使用任务开始前读取的 snapshot.py、codex_team.py、simulation/agent.py 和 simulation/contracts.py 源码在内存中创建基线模块；没有把 Git HEAD 当作基线（仓库已有大量未提交修改），也没有复制完整仓库。旧原生 Simulation 的动态 planning_request 导入绑定回旧实现和旧 schema，避免基线混入新逻辑。

统计 len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))。这是接口业务对象的紧凑 JSON UTF-8 字节，包含任务提示/输出 schema；不计 MCP 外层封装，不代表 provider 最终编码、token 或费用。未估算 token。

合成输入为 8 个候选、8 个 Branch、2 个任务，包含同帧指标、版本、限制、审核记忆和重复的 state.decision_context；静态协议取仓库现有生成/Branch 审阅/DFT 参考协议。Simulation 冻结动作是 prepare_local_batch_files(mode=relax_mc_results)。这不是所有操作或真实项目规模的性能评估。普通请求测量在既有 provider 传输压缩之前；原生请求包含本次复用的内联无损表格。

同一完整证据输入下：

| 对象 | 修改前 bytes | 修改后 bytes | 变化 |
| --- | ---: | ---: | ---: |
| 普通 Exploration 请求 | 45601 | 17726 | -61.13% |
| 普通 Reliability 请求 | 47468 | 23955 | -49.53% |
| 普通 Strategic 咨询请求 | 43104 | 28439 | -34.02% |
| 普通 Simulation 请求（补齐事实） | 11019 | 17628 | +59.98% |
| 原生 Exploration 请求 | 49129 | 16752 | -65.90% |
| 原生 Reliability 请求 | 51005 | 22955 | -54.99% |
| 原生 Simulation 请求 | 10547 | 5946 | -43.62% |
| 原生团队 consulting 响应 | 146030 | 40566 | -72.22% |

另通过真实 build_codex_context 和两版 CodexTeam.start/_view，读取同一临时合成 state，比较实际摘要入口与补齐证据后的入口（不是与上表相同的输入充分性）：

| 对象 | 修改前 bytes | 修改后 bytes | 变化 |
| --- | ---: | ---: | ---: |
| 原生 Exploration 请求 | 36430 | 39883 | +9.48% |
| 原生 Reliability 请求 | 38306 | 41822 | +9.18% |
| 原生 Simulation 请求 | 10547 | 29424 | +178.98% |
| 原生团队 consulting 响应 | 108013 | 82822 | -23.32% |

单个任务包增长来自补齐原摘要/仅 snapshot_id 中缺失的事实和上下文绑定，不能将这些缺口当成优化收益。两组测量不能混为一个普遍缩减比例。临时项目均在 TemporaryDirectory 内，退出后已确认删除；对同一合成 state 的原始字节核对未改变。

## 兼容与未验证范围

默认响应不再保证顶层 snapshot.facts 总是存在；原生消费者应直接传递自包含 requests，Strategic 需要完整事实时显式 request_role="strategic"。新生成 Simulation 计划要回显 context_binding；既有冻结方案的恢复仍兼容。历史讨论若显示 stale，应经原 revise 流程重新咨询，不删除持久讨论绕过。

没有运行真实项目、真实模型决策、HPC、真实服务或完整全仓测试；没有重启/部署。本次完成仅表示源码和合成针对性验证完成。强来源核验及完整咨询证据构建可能增加本地读取/CPU/内存开销，未测真实大项目端到端延迟或实际模型使用效果。


## 最终验证记录

最终源码 py1 回归：179 passed，14 warnings（第三方 Matplotlib/Pyparsing 弃用警告），40.53 秒。作者按任务开始时保存的范围内源码进行了差异自审；按用户要求没有派生审查/实施 Agent 或切换模型。

命令前置环境：PYTHONDONTWRITEBYTECODE=1，PYTHONPATH=tests/fake_only_guard;<repo>;E:/Py-Code。

```text
python -m pytest tests/test_role_context_projection.py tests/test_codex_native_team.py tests/test_decision_team.py tests/test_simulation_agent.py tests/test_four_agent_contracts.py tests/test_four_agent_closed_loop.py tests/test_codex_context_binding.py tests/test_codex_no_project_model.py tests/test_native_approval.py tests/test_single_action_dispatch.py tests/test_lifecycle_recovery_replay.py tests/test_memory_recovery_boundaries.py tests/test_review_memory_lifetime.py tests/test_decision_evidence_catalog.py tests/test_branch_evidence_transport.py tests/test_codex_branch_flow_fixes.py tests/test_role_permissions_and_research_plan.py tests/test_codex_offline_cold.py tests/test_codex_branch_evidence_binding.py tests/test_memory_integrity.py -q -p no:cacheprovider --basetemp E:/Py-Code/pdt --disable-warnings
```

python -m ruff check phase_agent --no-cache：通过；本次修改的 6 个生产文件 ruff format --check：通过；新增测试 ruff check：通过。已执行的全包 format --check 发现 137 个非本次改动文件已有格式差异，未批量格式化或改写这些文件。

最终源码再次测量得到本页相同字节结果。测量 TemporaryDirectory 退出后已删除；pytest TemporaryDirectory 正常清理，无清理警告。本任务开发验证脚本和内存基线的落地副本在交付前删除，不保留整仓副本、数据库、快照或失败现场。TEMP_PLAN.md 仅移除本任务标记段，保留原有其他任务计划。
