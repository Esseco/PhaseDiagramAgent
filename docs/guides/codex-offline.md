# Codex 进程内接管

`phase_agent.runtime.codex.offline.CodexOffline` 复用现有运行时和方案、审批接口，不启动 HTTP 服务，也不自动批准方案。创建实例时会按既有运行时逻辑初始化或读取台账；配置确认需显式 `confirm_config(explicit=True)`，方案使用 `propose(action)`，审批需提供原方案的 plan_id、state_version 和 proposal_hash。不得调用项目模型决定动作。

当前验证覆盖适配器传参、审批绑定，以及合成新项目从配置确认到首次 branch 提案、批准和结构登记；恢复、后续计算和完整科研周期尚未端到端验证。Codex 提案在审批和登记中以 codex 记录决策来源。提案保存后返回待审批列表；读取列表失败时保留原结果，不重提方案。当前 branch 方案的审批范围可能包括 Relax/MC 输入准备，应核对审批卡。P3 母结构的 Na 位点池已按候选位点处理：模板保留全部 Na 位点，目标 x 由输入结构决定；已通过合成 MC 占位交换测试。


## 首轮 branch 覆盖

Codex 接入时，读取 `phase_context` 后由当前 Codex 选择扩胞并提交 `phase_propose`，不通过 `chat` / `propose` / `continue` 请求项目配置的模型代替决策。使用原有 `phase_report` 生成 Markdown，用户批准该具体版本后才执行。

没有已登记 branch 的首轮只允许 `coverage`。选择范围由允许的扩胞矩阵及每区域的 `min_det_H/max_det_H` 决定；程序随后要求这些矩阵下所有合法相与整数 Na 组分均在方案中。Na 数量按 `x*N_O/2` 取整数，不能把候选 Na 位点数当作满钠化学计量；例如 P3 的候选位点多于 TM 时也保持此定义。端点与竞争相以项目 `boundary.P` 为准，不硬编码材料体系。若某相的母胞不同，只枚举该相在所选扩胞下可实现的整数组分，不伪造不合法的配对。

首轮方案不得把 O3、P3 分别裁成不同 Na 区间。资源不足应减少所选扩胞，而不是隐式丢掉 Na 组分或竞争相。覆盖完整性独立于初始 TM 近邻描述符；固定 TM ordering 不能成为过滤掉合法 Na 组分的理由。达到时间/次数保护阈值必须报告覆盖未完成并停止，不能自动准备 Relax–MC 输入；初始化失败也先复审。去重仍复用现有 branch 与结构两级规则，不按数量配额补齐。后续已有 branch 的轮次保留按证据选择局部区域的能力。

Codex 上下文和审核元数据中的 `initial_coverage` 给出首轮状态、允许策略与合法框架目录；区域预览冻结本次应覆盖的组合数。审计元数据存于 action 的 `codex_rule_audit`，不混入科学工具的 `parameters`。


## 每个项目的远端工作目录

初始化时询问并登记该项目的 Linux 绝对工作根目录，保存在 `agent_runtime.json` 的 `remote_workspace`。新项目该字段为空，不能从本地项目名猜测。Codex 使用 `phase_set_remote_workspace(remote_root=...)` 设置或更改，`phase_context` 返回当前值；也可手改运行配置后重新读取。这个路径是部署位置元数据，独立于科学参数快照；修改不会迁移已有文件、改写旧任务地址、验证服务器或授权上传/提交。实际交付仍须结合此目录核对目标路径清单并单独批准。

## 整理已登记分析产物

`phase_plot(refresh_analysis=true)` 通过现有项目控制入口，复用已保存、按模型绑定的相图和同帧 DFT 数据，补齐各 epoch 双图归档、累计 DFT 误差和输出索引。历史图从该 epoch 已保存凸包按当前绘图/审查规则重绘，manifest 明确记录此依据，不冒充当时原图。历史发布只更新本 epoch 的 latest，项目级 latest 始终属于当前激活模型。此入口只写派生分析文件，不保存改写 state、台账、配置、审批或任务，不触发模型推理、重算或 HPC；后续正常结果更新会登记相同版本。默认不传该选项时保留原绘图行为。

已有项目仅整理图表时使用 `python -m phase_agent.runtime.codex.mcp --runtime-config <agent_runtime.json> --analysis-only`，再调用 `phase_plot` 的显式刷新选项。此模式校验 state 与运行配置的输出目录绑定，只开放项目选择/查询和绘图，不创建完整科研运行时，因此不会执行旧路径迁移、状态修复或恢复训练；配置和工作流工具被拒绝。
