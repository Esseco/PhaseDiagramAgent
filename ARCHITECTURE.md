# 项目架构与功能入口

## 当前边界

目录已经按职责分层，不宜为统一名称迁移整套目录。优化重点是缩小入口职责、统一事实来源与复用功能接口。本文描述实际实现，后续目标不等于已完成。

| 层 | 职责 | 不应承担 |
| --- | --- | --- |
| run | 项目启动、服务、对话、运行依赖装配 | 科学计算、直接跳过审批改任务 |
| config_layer | 可编辑配置、确认快照、版本与迁移校验 | 覆盖已有任务的科学参数 |
| data_layer | 结构台账、阶段历史、长期记忆及证据 | 替 Agent 做下一步决策 |
| analysis_layer | 相识别、凸包、覆盖与收敛、成本分析、上下文证据 | 提交任务或批准方案 |
| decision_layer | LLM 结合记忆和现状选择方案、修订建议 | 修改台账、运行计算 |
| execution_layer | 审批、校验、预算预留、输入准备、回收、状态协调 | 伪造缺失科学证据或静默替换 LLM 决策 |
| scientific_layer | 结构与 MC、DFT、模型计算的科学接口 | 对话和服务生命周期 |

## 功能地图

| 功能 | 主要入口/模块 | 归属与约束 |
| --- | --- | --- |
| 项目启动 | run/local_project_launcher.py | 项目独立；不接回占用端口的后台进程 |
| HTTP 服务适配 | run/local_http_server.py | 路由、认证、响应编码；显式注入聊天与页面依赖，不反向导入运行入口 |
| 聊天与兼容入口 | run/open_webui_api.py | 对话协调及旧服务/装配接口薄包装 |
| 运行依赖装配 | run/runtime_composition.py | 配置会话恢复、台账和后端创建；显式注入聊天与重新装配工厂 |
| 计算后端接线 | run/runtime_backends.py | dispatcher、runner、manual upload 与只读 collector 装配；不准备、提交或回收任务 |
| 运行配置工具 | run/runtime_config_io.py | JSON 对象读取、路径与工厂引用、密钥字段检查、历史配置恢复 |
| 配置会话初始化 | run/runtime_session.py | 加载已有会话、从可信状态恢复确认配置、初始化和升级草稿；不自动确认配置 |
| 客户端角色参数 | run/runtime_client_settings.py | 配置对话、搜索与意图识别分别展开参数；不读密钥、不调用模型；数值展开在装配捕获前执行 |
| 新运行隔离 | run/isolated_run_paths.py | 标准输出、独立描述文件与重启路径；内置handler完整接管回调，自定义后端无隔离协议时拒绝新建 |
| 草稿工作区路径 | run/draft_workspace_paths.py | 编辑文件定位、异常路径恢复、一次性模板创建；具名返回路径并明确保留 session 恢复写入 |
| 配置对话 | run/configuration_chat.py | 源配置与确认快照分开；旧任务保留版本 |
| 状态与回复展示 | run/chat_state_presentation.py、run/workflow_reply_presentation.py | 只读展示；旧导入兼容 |
| 敏感建议分类 | run/chat_approval_rules.py | 展示和聊天共用；最终授权仍由 execution policy 决定 |
| 工作流推进 | execution_layer/workflows/run_event_loop.py、run_tool_step.py | 回收→形成证据→提案→审批→校验→执行；不能绕过门槛 |
| DFT 预览与修订 | execution_layer/workflows/prepare_dft_proposal.py、attach_dft_preview.py、dft_template_review.py | 同源预览；模板确认与科学选择各自保留 |
| 重新生成规划 | execution_layer/local/identify_rerun_plan.py | 先定位轮次/action并说明影响；不能把规划当删除授权 |
| 远端结果发布 | execution_layer/remote/export_batch_result.py | 结构、结果先发布，完成标记最后发布 |
| 结果接收与分析协调 | execution_layer/workflows/apply_scientific_feedback.py | 验收、入账、识别与分析协调；算法留在分析层 |
| 相识别缓存 | analysis_layer/phase/ensure_phase_identification.py、identify_result_phase.py | 文件哈希及分类范围缓存；未知结果不得当已识别 |
| 相图与 CSV | analysis_layer/phase/update_phase_diagram.py、export_current_phase_diagram.py | TM/O₂ 一致、Na 端点形成能；新数据才更新；MLIP 与 DFT 不混能量 |
| 真实运行统计与估算 | execution_layer/cost/runtime_observation.py、analysis_layer/cost/predict_runtime.py、calibrate_relative_cost.py | 开始/结束轻量记录；实际样本优先，初始值仅参考；秒数不直接冒充相对预算 |
| 台账与记忆 | data_layer/ledger/phase_data_manager.py、data_layer/memory/decision_memory.py | 原始事实与决策建议分开，记忆更新保留审核 |

## 状态与数据的单一来源

- 配置：可编辑 project JSON → 用户确认快照 → 运行绑定版本。迁移必须单独校验，不能重启时隐式改版本。
- 执行状态：持久化 state 中的任务、预留、审批与处理编号；阶段台账记录结构和结果历史，不另建平行任务系统。
- 科学事实：校验通过的结果及结构文件 → 相识别证据 → phase_records → 版本化相图与 CSV。未知结构/相不能靠 branch 标签补造。
- 决策：当前状态证据、长期记忆、用户要求 → LLM 提案；校验只负责权限、版本、科学前提与硬预算。
- results 是计算交付物，不是另一套任务状态库；processed_task_ids 等现有机制负责回收幂等。

## 下一步拆分顺序（尚未实施）

1. HTTP、运行装配、计算后端接线及草稿路径恢复已独立。后续细分配置会话初始化与客户端创建；不要仅将整段巨型函数搬家当作功能模块化完成。
2. 从 run_tool_step 拆出阶段提案上下文与审批协调；复用现有 DFT 模块，不另造一个策略引擎。
3. 在既有 action/result/preview 字典接口边界添加契约校验；旧 state 通过已有迁移入口读取，不一次性重写历史。
4. 将回归记录整理成最新摘要加历史附录；各层 README 与本图核对，过期说明逐条改，不把目标写成当前行为。

BOHB 暂缓；旧备份文件和生产目录不在本次清理范围。不得仅为了文件整齐移动用户结果或删除备份。全量测试与剩余问题见 ARCHITECTURE_REVIEW.md。
