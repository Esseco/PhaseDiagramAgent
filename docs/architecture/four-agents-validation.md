# 四 Agent 验收记录

本页记录四 Agent 架构重构的验收；后续文件整理与当前目录见 [仓库架构](repository-layout.md)。原三角色文档及历史维护记录已在后续整理中外置备份并删除。

2026-11-01；默认环境：C:/ProgramData/anaconda3/envs/py1/python.exe。

## 最终架构

保留现有 PhaseDiagramAgent、LangGraph、Action Contract、科学工具和持久主图。Coordinator 源码职责迁入 Strategic，四个角色分别位于 agents/exploration、reliability、strategic、simulation。保存的旧模块描述符经集中映射读取，不保留旧包副本。不增加框架依赖。

两位专家独立读取同一事实快照，输出受角色权限与当前合法目录约束的 JSON 建议。Strategic 综合目标、资源、历史和人工意见，形成唯一 Action Proposal，可附最多 8 步研究路线；只有 active_step 对应当前动作，未来步骤没有执行权。Simulation 在审批前生成并冻结执行计划，调用一次现有注册工具，再基于实际返回检查和诊断。MC/Relax/DFT/MLIP/凸包、远端观察回收及资源账本均复用现有模块。

沿用现有 operation 名称；最终目录有 17 个 operation，其中补齐独立训练交接授权操作。Tool、Action 与 Execution 分开管理。普通已授权动作仅能由已确认 workflow.auto_approved_operations 允许；默认空列表。重大决策、严重冲突、配置的高成本阈值及已有敏感规则优先。人工方向接受、执行批准与长期记忆审核保持独立。超范围恢复和参数修改重新审核。

Agent 模式与 Codex 原生接管共享科学事实、权限、校验、预算、审批、执行回执和记忆。Codex 提供 phase_team_simulate 规划/诊断入口，不调用项目配置模型；只有精确绑定持久记录才能确认提交，不因返回状态重放动作。旧 callable、旧冻结审批和旧项目继续按兼容契约运行。

## 同轮修复

- 统一最终 Action 归一化、验证与实际派发，拒绝非法参数、非有限预算及冲突别名；非法修改不消耗持久审批。
- 记忆读改写共用项目生命周期锁，区分适用项目与来源，审核时复核证据、状态和指纹；跨项目替代被拒绝。
- 排序搜索复用最小传输包、清单与 SHA256，外部回收先验证绑定和真实消费者依赖，不整目录回传模型或历史。
- 新训练交接绑定原审批范围；范围扩大另行提案。旧任务不补写新授权。
- 明确专家操作权限、研究路线当前步骤绑定、自动授权来源、限次恢复及 Codex 提交防重放。

## 验证证据

冻结源码及测试后，全部默认测试按收集顺序的偶/奇索引分为两个互斥分片。核对两组收集的 4,143 个 nodeid 完全相同、选择集无交集、并集覆盖全部。两组退出码均为 0：

| 分片 | 通过 | 跳过 | 耗时 |
| --- | ---: | ---: | ---: |
| 0 | 2,054 | 18 | 438.87 秒 |
| 1 | 2,051 | 20 | 456.35 秒 |
| 合计 | 4,105 | 38 | 并行运行 |

另有 13 个 subtests 通过。冻结范围 phase_agent/tests/.codex/experiments 共 1,440 文件，运行前后内容和文件集合一致；映射摘要 SHA256：7b959910bedf30eabd02aa8644001b8c1ae241067f5e5a5c99ea97d55456d9f1。

分片仅是临时验证调度器，没有部署到应用。可用下列完整命令复现默认测试集：

~~~powershell
python -B -m pytest -q --basetemp E:/Py-Code/pdt --tb=short
python -m ruff check phase_agent
python -m ruff format --check phase_agent
~~~

ruff check 与 format 检查通过（807 个 Python 文件）。启动器及 Codex MCP 的 --help 入口检查通过。针对性闭环测试覆盖两专家→策略→方向修改/接受→执行批准→一次工具执行→更新事实→下一轮，以及冻结计划、授权重试和两种 Codex 新项目路径。

警告来自原 pymatgen 结构属性缺省和 Starlette/AnyIO 兼容性弃用提示；无测试失败。默认跳过的 38 项没有计为通过。最初普通沙箱的测试临时目录/本机套接字权限失败也未计为通过，最终测试使用允许短路径临时目录与本机假服务的环境重跑。

## 旧项目只读诊断

使用本次源码的 inspect_runtime 检查 E:/0-PhaseDiagram 下项目，不写项目、不启动服务：

| 项目 | 诊断 | issues |
| --- | --- | --- |
| O3-NiCo | ready | 空 |
| NaNiO2-phase | ready | 空 |
| O3-1ele-NaCrO2 | ready | 空 |
| O3-1ele-NaCoO2 | ready | 空 |
| O3-2ele-NaFeMn | ready | 空 |

## 发布与产物管理

原仓库位于 E:/jupyter notebook/1-AL_for_PhaseDiagram/Process_PhaseDiagram。发布仅涉及源码、测试、文档和原生角色定义，原 settings、科研项目及历史状态不迁移。部署前逐文件核对原始基线并在同级 _Process_PhaseDiagram_source_backups 保存源码 ZIP；删除仅限已知被迁走的旧源码。发布后核对目标哈希与入口。

本轮测试由 TemporaryDirectory 管理，结束后 E:/Py-Code/pdt 没有测试子目录；不保留临时项目、快照或失败现场。源码暂存副本在最终发布核验后删除，仅保留源码回退备份及本验收文档。

## 验证边界与使用

测试使用假模型、假超算和最小合成数据，未调用真实模型 API、未提交或重跑真实科研计算，未修改真实项目数据。ready 代表本地文件和配置检查通过，不等于真实 HPC 作业或科学结论已验证。未现场验证 Codex 客户端实际派生原生角色的能力。成功的工具返回不等于科学计算完成。

启动方式不变：py1 中运行 python -m phase_agent.runtime.local_project_launcher。Codex 按 docs/guides/codex-team.md 连接 codex-mcp.cmd。已有服务需用户手动重启，已有 MCP 客户端需重连加载代码；本轮没有自动重启服务。

## 原仓库发布复核

已在原仓库更新/新增 151 个源码、测试、文档及角色文件，移除 20 个被迁走的旧源码/旧架构文档文件。部署脚本先比对原始基线并校验源码 ZIP，再执行更新；目标文件 SHA256 与已验证副本一致，旧 coordinator 包不存在。保留的历史验收文档已标注为历史版本。

源码备份：E:/jupyter notebook/1-AL_for_PhaseDiagram/_Process_PhaseDiagram_source_backups/four-agent-20261101-211847.zip。

部署后在原仓库运行四角色权限、完整闭环、Codex 新项目/团队、Simulation、人工审阅、应用布局与实际编译图回归：66 passed（17.33 秒）。原仓库 ruff check --no-cache 和 format --no-cache --check 通过；启动器/MCP --help 通过。五个旧项目再次只读诊断，全部 ready、issues 为空。未启动真实服务或科学计算。

原仓库普通沙箱 ruff 首次因缓存目录权限失败，随后以 --no-cache 复核通过；缓存失败未作为通过证据。源码暂存副本、临时计划与测试产物在交付清理，不作为应用依赖。
