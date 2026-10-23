# 通用稳定性边界与恢复手册

目标：新组成/新结构沿相同生产流程运行；已知环境或数据故障由程序明确诊断、隔离和恢复，避免通过修改代码处理每次失败。故障不表示任务一定未执行。科学计算、质量判断、结果回收与页面展示分别报告。

## 入口

- 审批页 `/phase/approval`：分区独立加载，查看故障诊断、原方案审批、执行核对、事务续写和交互阻塞核对。单独图表故障不遮蔽方案；事务/当前状态不可核对时禁用科学审批。
- 认证 GET `/phase/diagnostics`：只读；即使 state.json 损坏，事务、操作历史、故障证据仍单独返回。各 section 的 available 表示能否读取，不表示科研任务成功。
- 离线诊断：`python -m phase_agent.runtime.project_diagnostics --state PROJECT/workflow_state/state.json`。无需服务、LLM、模型权重加载或超算。它不修复/重写状态、不扫描大型科学文件、不推断未知作业失败。
- `/phase/operations`：请求状态独立于科学任务状态。completed 表示交互返回；unavailable/interrupted/blocked 可能具有未知执行效果，不能改一个幂等键重放。

## 故障边界矩阵

| 边界 | 程序行为 | 处理方式 | 发布测试 |
|---|---|---|---|
| 非法相、超胞、比例、Na分数、缺母结构 | 生成/准备前拒绝；不修正已确认科学配置 | 修改配置草稿或恢复正确输入，再审阅 | generation_contracts / generation_runtime_guards / stability_new_structures |
| 缺模型、输入包损坏 | 子任务执行前预检失败，无子任务提交 | 核对原模型/包和路径，重新校验 | relax_mc_batch |
| Agent 返回无效协议/证据 | 有限校正后停留交互，不越过审批 | 修订方案；保留当前科研状态 | region_phase_contract / adaptive_branch_review / agent_retry_guard |
| 过期审批、重复请求、旧检查点 | 校验方案身份、状态版本与执行收据；拒绝重复科学效果 | 读取当前原方案，或先做执行核对 | status_approval_binding / lifecycle_checkpoint_resume / execution_recovery_scope |
| squeue缺失、sacct不可用 | 保持 scheduler_state_unknown；不当作完成/失败/重新提交依据 | 核对原job及已存在结果 | femn_scheduler_recovery / recovery_closure |
| 计算完成而控制器失败 | 科学结果校验与历史控制器诊断分离 | collect-only原结果；保留原job映射 | collect_only_recovery |
| 缺结果/坏JSON/凭据绑定或校验和不符 | 保留原文件，拒绝入账；JSON对象形状异常也返回验证失败 | 回传原结果和凭据；不要手改签名结果以绕过校验 | collect_only_recovery / stability_boundaries |
| 能量NaN/Inf/布尔值、错误单位/组成 | 账本修改前拒绝；坏项隔离，有效项继续 | 核对原计算输出/解析器；有效凭据重新导入 | stability_boundaries |
| 错模型/配置/结构/阶段绑定 | reconcile和feedback共用绑定校验，后续步骤不能重新引入拒绝结果 | 按原任务绑定回收；不能将历史结果改标成当前模型 | stability_boundaries |
| 非法MC步数/成本、非对象元数据 | 消费前校验，按项拒绝，不破坏后续项 | 核对原结果元数据，重新校验；未知成本保留未知 | stability_boundaries / unified_recovery_status |
| 未收敛Relax、相识别unknown/outside | 剔除或保留待识别，禁止冒充有效相图点；与计算状态分开 | 核对力/优化器/最终结构和搜索边界；重算须另行审批 | mlip_relaxation_convergence / phase_identification_cache / stability_new_structures |
| 打包中断、入账跨文件中断 | 保留原包、事务备份、CAS版本、旧写入屏障；不重放工具 | 重建同证据包，或续写原入账事务 | collect_only_recovery / result_recovery_commit |
| 权限/磁盘满/网络超时/锁占用 | 类型化故障码；执行效果未知时禁重放；诊断保存失败不替换原故障 | 处理环境问题后核对原事务和收据 | stability_diagnostics / stability_boundaries |
| 单个操作记录损坏、后台线程结束但完成未落盘 | 隔离单条；其他只读区域可见；未知效果阻塞新交互推进 | 原收据核对后解除交互阻塞，保留原记录 | stability_boundaries / operation_resolution |
| 图表/状态/方案某个接口失败 | HTTP结构化故障编号；页面并行加载、逐项到达即展示并逐区隔离；读取有有限超时，挂起图表不阻塞其他区域 | 单独刷新诊断；状态/事务不可核对时保持审批禁用 | stability_diagnostics（实际执行页面JS） |
| 内部未知异常 | 停在边界，返回incident_id、层、对象与调用位置；不猜测成功、不无限重试 | 先保留证据并排除数据/环境问题；真正程序缺陷才修代码 | stability_diagnostics |

## 核对后解除交互阻塞

页面操作卡填写实际核对依据，并选择外部作业与输出核对结论；点击“确认核对并解除交互阻塞（不重放）”。认证 POST `/phase/operations/resolve` 输入 operation_id、record_sha256、state_version、evidence_note、external_jobs、output_inventory。程序检查当前状态版本、原记录SHA、无未解决执行收据、原请求已停止，并在项目锁和请求锁内保存旁路核对记录。

此操作保留原操作文件、科学任务、审批、账本、费用和结果，不调用原callback。重复请求只返回原核对。新交互仍需新请求，科学动作仍需原审批。若存在未解决执行收据，先用已有“生成核对处理方案”与人工审阅流程核对，不能以解除UI阻塞代替执行核对。权限导致原记录不能读取时先修复权限；状态/文件版本变化后重新读取，不能使用旧哈希。

交互核对与原入账事务续写入口独立同步执行，避免被自身的未核对异步请求挡住。它只登记人工核对事实，不授予科研动作权限。

## 回收已有结果

使用现有工具，不调用模型生成恢复代码：

`python -m phase_agent.tools.relax_mc_batch.cli inspect --archive RESULTS.zip`

`python -m phase_agent.tools.relax_mc_batch.cli recover --original-archive OLD_RESULTS.zip --evidence-directory DOWNLOADED_WORK --destination RECOVERY_DIR --job-evidence JOB.json --remote-root ORIGINAL_REMOTE_WORK`

必须有原始任务、模型/配置、凭据与job-task映射。recover只构建验证过的回传包，不新提交。将有效结果放回已登记的回传位置，由现有回收生命周期导入；若路径/绑定变化，须先核对，不修改原凭据冒充有效。入账中断时页面的事务按钮 POST `/phase/recovery/transaction/resume` 只续写既有已批准事务，需精确revision。

## 发布门槛

py1运行：`python tests/recovery_release_gate.py --basetemp SHORT_WRITABLE_PATH --junitxml REPORT.xml`。

该命令覆盖原恢复套件、一般故障注入、实际页面JavaScript，以及Ni、Ti/V、Ni/Ti/V独立构建的新项目：真实结构生成、Na/空位静电初态、Na=0/0.5/1、生产输入准备、原生审批、fake计算/完成凭据、三轮MC、实际识别/Na层质量门、结果包校验导入、相图、重复防重、账本重载和只读诊断。fake仅替换计算能量和worker执行；不mock相识别或Na层质量检查。多相测试另外验证几何识别与Na0→O1规则；单相配置优先遵守明确边界。

未来发现新的确定缺陷，先把触发条件加入此公共门槛，再改公共边界；不增加项目名/元素名分支。代码部署须基线哈希一致并备份；现有真实服务由用户管理，部署不意味着运行中代码已刷新。

## 验证边界

不可能通过离线测试证明任意结构都物理收敛或外部系统永不失败。离线门槛验证流程和故障隔离，不验证新模型物理精度、真实GPU/DFT执行、超算账号/配额和硬件环境。未知内部缺陷仍可能需要维护，但应携带可定位的编号和证据；环境、数据、重复回传与已知中断不应要求用户反复找AI改代码。


## 回传故障证据与本地补提取

新Relax/MC回传包包含绑定bundle/plan/model/config的return_diagnostics.json。科学worker失败保留任务ID、作业号、原异常及有限堆栈；Slurm缺失凭据时补充状态、退出码和有限日志，但不伪造任务失败或科学完成。远端控制器、远端回收、调度作业与本地导入分别标记origin。控制器查询失败不等于GPU计算失败，也不授权重投。

Agent通过既有batch_facts读取remote_diagnostics、driver_diagnostic、return_diagnostic和local_recovery；状态页和离线诊断沿用同一来源。即使results.zip已导入，也读取本次diagnostics.zip中的控制器信息。旧包复用failure.json及有效worker凭据；缺少凭据的旧失败记录标为未核实，不计入已核验失败。

相同SHA256且已入账的回传包若在本地缺了解包文件，沿原清单补提取，不重放计算、入账或费用；不同内容的既有文件仍拒绝覆盖。原结果缺失或绑定不符不通过猜路径、改签名、篡改模型身份或整批重算来掩盖。

## 每轮交互Markdown

Studio与控制API聊天每轮在项目根目录“交互记录/”写 Markdown，按 `epoch_简短阶段_日期_序号.md` 命名，例如 `epoch1_DFT选点_20261020_01.md`。名称只用已有阶段事实；机器轮次 ID 保留在正文和过程 JSON。同日同阶段重名使用独占创建分配序号，不扫描历史目录；同一记录重试复用原地址。旧记录及其链接保留。正常答复和异常均留记录；发布失败清理本次空文件及临时文件。

短答复保留具体结果、阻碍原因及下一步执行者，不填充空泛的状态检查或无新预算套话；部分回收缺凭据显示“结果待核验”，不据此宣称计算失败。聊天仅保留结果、状态、下一步、必要预算和本轮文件的完整绝对地址（可直接复制打开）。Markdown 保存完整答复、本轮引用的 Branch/DFT 分布表、错误来源、原堆栈/日志证据及当前模型远程引用。当前待审批 DFT 即使没有文件链接，也直接投影保存方案的结构编号、相、Na/O₂、原子数、已有 H、预测 Ehull 及单位、理由、QBC 和原结构路径；缺失数据不补造，版本不符不混用当前候选。该投影不读大账本、不重新选点、不请求模型、不变更审批。可选展示 JSON 损坏时保留答复并标明文件，仍可展示已保存方案。凭据文本脱敏；文件保存失败不能重放已执行动作。

只读查询的状态仅在本轮传给展示层复用一次；复用前后核对 state 与行政控制文件的大小、修改时间和文件身份。文件发生变化时重新读取；普通业务读取仍各自独立，缓存不跨轮次、不写入检查点或过程记录。这样避免只为生成 Markdown 再解析一次大型状态。

工具返回的 canonical `state` 在合并后不再嵌入任务审计结果。既有生命周期载入和检查点恢复沿用历史精简入口，只剔除无科学 stage、ID 以 `:task` 结尾的工具审计行中重复的完整项目状态；保留任务身份、状态、费用、登记结构及科学输出。状态查看本身不落盘迁移；后续正常业务保存才持久化精简后的状态。

模型路径以训练回传的实际绝对位置和SHA256为依据，保留原有提交前存在性、大小与哈希预检。历史缺失错误必须与当前输入引用及后续成功记录分开解释；本地Path检查不能证明超算文件存在。旧冻结上传包不随本地源码升级而改变，已有作业只走原结果核验/回收入口，不为升级诊断而重投。
