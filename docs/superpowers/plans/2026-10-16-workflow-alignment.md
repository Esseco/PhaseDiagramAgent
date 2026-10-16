# MLIP 相图工作流对齐实施计划

> 执行方式：主 agent 使用 executing-plans 在本会话逐阶段实施，不自动换模型，不逐任务委派。用户已授权制定计划后继续修改和验证，不额外请求设计批准。

**Goal:** 将实际生产 `phase_agent` 与已确认的 MLIP 构图、阶段审批、100–1000 branch、DFT 检验/训练和可选数据共享规则对齐。

**Architecture:** 复用唯一项目图、单动作审批、Relax+最多三段 MC、现有训练/刷新、候选选择、短记忆和经验库。在当前工作区的隔离副本修改与测试，按基线哈希保护同步生产源码；没有服务操作和数据迁移。

**Tech Stack:** Python，py1，pytest，LangGraph，现有 MACE/DFT/Slurm 适配器（测试不执行真实后端）。

**Spec:** `docs/workflow-agent-rules-and-source-audit-2026-10-16.md`。最新政策覆盖早期综合相图方案。

## 全局约束

- 不启动、停止、重启真实服务；不调用真实 LLM/MLIP/DFT/HPC。
- 不修改真实项目的数据、配置快照、任务、审批、模型、输入包和历史结果。测试只用隔离临时目录和假后端。
- 不重建 scheduler、不改封装内部 MC 科学分配算法、不改变候选为全 pool、不取消审批或幂等。
- 新政策仅影响后续运行时行为/新方案；已有批准动作按原冻结身份复用。原记录不迁移、不重编号。
- 科学成本仅参考，物理上限、文件完整性和批准范围仍是硬约束。不得静默更改用户确认的科学容差。
- 普通管理和测试用 `C:/ProgramData/anaconda3/envs/py1/python.exe`；测试临时目录和日志全部在工作区；封装测试使用 Windows 扩展短路径，SQLite 测试使用普通短路径。
- 生产根已有其他任务的 TEMP_PLAN.md：保留原内容，不覆盖；本计划及隔离副本 TEMP_PLAN 是本次唯一进度依据。若需要生产根登记，追加具名引用且部署时守护原哈希。
- 先写有意义的行为回归、确认旧行为失败，再改实现；不强制逐任务 commit 或 worktree，不删除有效实现。

## 审查重点

1. 历史同模型数据/训练几何/共享标签不能伪装成新独立验证；覆盖验证率与误差样本是不同资格。
2. 重复回收、部分返回、等待豁免不能计完整反馈循环或绕过切换屏障。
3. 新旧方案兼容：不能用新政策修改已批准输入与审批，缓存不能沿用已变化事实。
4. 共享未启用时不得创建数据库或导入；启用后只从来源名单及批准快照读取，不篡改旧方案。
5. 发布前检查生产源码未并发变化；仅同步白名单源码/测试/文档，先备份，不触碰 E:/0-PhaseDiagram 或服务进程。

## 阶段 0 基线和隔离

- [x] 比对文档的 656 文件快照和当前源码，记录差异。
- [x] 将 phase_agent、tests、必要静态配置复制到 `build/workflow_alignment_20261016/repo`；不复制真实项目、数据库、模型、输入或状态。
- [x] 记录生产源码/静态文档基线及真实项目关键记录哈希；只读记录服务相关文件，不管理服务。
- [x] 副本写入 TEMP_PLAN.md、progress.md；按文件实际内容选择相关基线测试，日志保存工作区。

## 阶段 1 MLIP 相图与独立收敛证据

Files: `analysis/convergence/check_global_convergence.py`, `single_point_validation.py`, `tools/workflows/apply_scientific_feedback.py`, `lifecycle_recovery.py`, `configuration/defaults/default_layered_search_config.py`, `runtime/post_dft_presentation.py`；需要时新增窄职责证据函数，不新建决策器。

Interfaces: 复用 phase_diagrams.mlip、DFT 配对表、训练记录和 model_registry；完整循环证据按模型和 DFT 批次身份去重，旧 model_update_epochs 保留作历史，不冒充新反馈循环。

- [x] 测试 DFT 能量不改变正式 hull、同模型完整反馈才计数、重复/部分回收不计数。
- [x] 实施 MLIP 唯一正式图、反馈循环计数、新模型基线；不删除旧 combined 产物。
- [x] 测试/实施基态匹配 DFT >60%、新独立样本能量 MAE <0.005 eV/atom、力分量 RMSE <0.05 eV/Å；训练数据排除、排序冲突阻止交付、未知资格不得放行。
- [x] 沿用已确认 hull 变化容差、用户最终验收与关键竞争相规则；旧状态无新证据则明确不足。
- [x] 运行新回归、single_point_convergence、版本化 hull、反馈和汇报相关现有测试。

## 阶段 2 branch 与正式路线

Files: `tools/policy/region_sampling.py`, `adaptive_branch_count.py`, `tools/workflows/region_controller.py`, `decisions/agent/propose_tool_action.py`, `scientific_decisions.py`, `post_dft_review.py`, `tools/workflows/scientific_module_scope.py`, 相关展示/校验器。

Interfaces: 保留区域表达与合法性，新增方案显式规划 total_quota（100–1000）及区域分配；旧 region-v2 冻结方案保持原规则。生成后先汇报、封装输入另审批。小超胞优先是策略与解释，不是强制穷举。

- [x] 新回归覆盖 99/100/1000/1001、实际不足、POSCAR 数不等于 branch 数、资源保护停止、旧审批兼容。
- [x] 修正新提案默认契约和实际生成数量控制，保留去重及结构约束。
- [x] 明确首轮 DFT 后首次微调前置；后续 MC 后 branch/DFT、DFT 后探索/微调/交付；沿用现有收益比较契约。
- [x] 检查生成后的单独审批与包内无额外审批，运行 branch/region/科学决策/审批回归。

## 阶段 3 DFT 选样及异常

Files: `decisions/qbc_selection/select_dft_candidates.py`, `recommend_post_mc_dft.py`, `tools/workflows/validate_dft_agent_decisions.py`, `analysis/state/task_failure_facts.py`, 候选指标构建和汇报。

- [x] 新回归：建议预算不裁剪候选；小批独立抽查至少一个或明确跨批证据；合法性与物理范围仍校验。
- [x] Relax 必须有关键相图影响及未优化结构证据，复用残余力/相竞争事实和 agent理由；普通 static 不增加前置 Relax。
- [x] DFT >50% 失败仅提供警报事实；个别失败不自动重试，未知回传不算失败；每批汇报等待审批。
- [x] 运行 DFT 选样、候选/成本/去重/回收相关测试。

## 阶段 4 训练 切换与配置排空

Files: `tools/local/training_cv_review.py`, `training_agent_review.py`, `tools/workflows/create_model_update_handler.py`, `persistence/models/activate_validated_model.py`, 训练图及配置修订既有入口。

- [x] 新回归：训练有效完成即提出切换供批准，不因 MAE未改善而再选补DFT；执行失败仍保留旧模型。
- [x] 所有训练/切换入口等待真实未结束任务，等待豁免不等于任务实际终结。
- [x] 复用切换后旧记录失效、刷新、首次新branch屏障、五折和累计数据，不以五折充当独立交付。
- [x] 超胞上限人工修订待当前任务结束生效，不写已有任务/批准包；在现有配置版本机制接入。
- [x] 运行训练handoff、CV、单审批、刷新与配置相关回归。

## 阶段 5 可选共享与交互

Files: `persistence/shared_dft.py`, `runtime/diagram_projects.py`, `tools/workflows/lifecycle_recovery.py`, `reference_dft_startup.py`, `decisions/agent/create_deepseek_client.py`, `post_dft_review.py`, 原有配置/审批入口。

- [x] 新回归：共享默认关闭且不创建库；只选明确来源；审批快照绑定数据ID；新增标签仅下一次审批纳入。
- [x] 既有协议/TM/边界筛选和数据来源沿用，不共享模型，不迁移已导入历史数据。
- [x] 统一新调用零自动重试/修复重发；一次失败直接报告并记录用量，批准不再推理；保留按需证据读取、超时、短记忆和缓存。
- [x] 经验引用/草稿/发布沿用，补充必要的契约提示；不新建经验系统。
- [x] 运行共享、客户端、审批、缓存、短记忆和经验回归。

## 阶段 6 集成验证与同步

- [x] 新回归与相关现有套件绿色；变更导致的旧政策测试只按新规格更新，不能删去合法性/审批断言。
- [x] `python -m ruff check phase_agent`；变更文件 format check；更大范围格式若有基线问题不得无关重写。
- [x] 用假模型/假后端跑首轮和后续完整路线、失败、恢复、批准复用及旧冻结方案场景。
- [x] 逐项回填 R/M/G 对齐清单；记录仍未验证的真实API/科学计算性能边界。
- [ ] 对待同步生产文件核对基线哈希，备份到工作区 `originals`，只同步白名单文件；生产根旧TEMP_PLAN不覆盖。
- [ ] 核对同步后哈希、项目数据/任务/审批记录未变，无服务生命周期调用；正式计划记录完成，删除本次副本临时计划，保留日志和进度。

## 验证命令与日志

在隔离副本执行：`C:/ProgramData/anaconda3/envs/py1/python.exe -m pytest tests/test_workflow_alignment.py -q -p no:cacheprovider --basetemp <workspace-short-temp>`，预期目标行为测试全部通过。

各阶段追加对应现有 tests/test_*.py，完整输出保存 `build/workflow_alignment_20261016/logs`。禁止将真实项目路径交给假测试。

## 进度

阶段 0–5 已实施，阶段 6 验证/同步凭据见实施记录；逐阶段证据见 `build/workflow_alignment_20261016/progress.md`。本计划不授权真实计算、服务操作、数据迁移或重写已有审批。
