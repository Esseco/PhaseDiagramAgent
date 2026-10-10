# 项目开发原则

## 默认环境

- 未特别说明时，普通项目管理、分析和测试代码使用 `py1` 环境。
- 科学计算后端继续使用项目已配置的原有环境。

## Codex 可读性

- 代码应让 Codex 和开发者都能快速理解：使用清楚的命名、明确的数据流和简短的函数；避免隐式副作用、过度抽象和重复实现。
- 修改应贴合现有架构与接口；不确定的数据显式报告，不猜测，不用静默回退掩盖错误。
- 不留下明知会误导后续开发的占位实现、死代码或过期说明。

## 功能模块化

- 每个模块承担清晰、单一的职责；优先复用已存在的实现和公共接口，避免循环依赖、巨型文件及平行重复流程。
- 所有功能优先采用满足需求的最小实现；先检查已有代码和调用方，只改必要范围，并用针对性验证覆盖行为。
- 不引入可避免的技术债。若确实存在暂时无法解决的权衡，必须说明原因、影响和后续处理条件，不得把临时绕行伪装成完整实现。
- 跨多个模块、改变接口/数据格式/运行流程或存在显著设计取舍的复杂功能，开始编码前必须在仓库根目录创建 `TEMP_PLAN.md`，写明目标、现有实现复用点、拟改文件、接口/状态影响、风险和验证方式。完成后删除临时规划书；若计划仍有未完成事项，应先转成正式文档并在交付说明中指出。


# Working on this project

- 默认使用 py1：`C:/ProgramData/anaconda3/envs/py1/python.exe`。
- 先读 `docs/architecture/graph.md` 和对应 `graph.py`，再修改节点或业务适配器。
- 普通用户语义由 LLM 理解；不要向在线路径增加关键词意图分类器或隐藏动作选择规则。
- 配置可直接编辑草稿；计算、输入准备、提交、模型激活等受现有审批策略约束。批准复用原方案，不再次请求模型。
- 每轮最多一个科学动作。等待、答疑和拒绝不得准备任务或提交作业。
- 图 State 只保存可序列化事实；模型、manager、锁和回调放在 Runtime context。
- 用假模型/假超算验证；没有明确授权不得运行真实计算或提交 HPC 作业。
- 实际程序源可能在 E:/jupyter notebook/1-AL_for_PhaseDiagram/Process_PhaseDiagram；修改暂存副本不等于部署。部署前比对基线哈希并备份原文件。
- 不删除项目 structures、parameters、workflow_state、结果或模型。源码入口删除前检查所有引用并跑相关回归。
- Windows 测试使用短 basetemp，TEMP/TMP 指向工作区可写目录。

## 源码布局约束

- 唯一生产应用包是 phase_agent；不要恢复顶层 run、orchestration 或 *_layer 包。
- graphs 中的 graph.py 组织连线，nodes.py 适配业务，state.py 定义状态；配置、分析与科学算法放在各自包。
- 启动命令：python -m phase_agent.runtime.local_project_launcher；Studio 配置注册 phase_agent 包内入口。
- settings 保存运行时示例；真实科学项目的数据与源码独立，不加入应用包。
- 保存的旧模块描述符只由 module_references.py 转换，不增加旧模块转发文件。
- 历史源码和测试产物在同级 _Process_PhaseDiagram_source_backups，不能作为应用依赖。

## 可读性与检查

- 源码阅读入口见 docs/architecture/development.md，各包 README 指向实际模块。
- 使用 ruff.toml 的统一格式；提交前运行 python -m ruff check phase_agent 和 python -m ruff format --check phase_agent。
- 框架升级不保留平行 scheduler 或自动科学提交链；有效科学断言迁到当前生产接口测试。
- 数据迁移兼容与旧执行框架分别判断，不能因名字带 legacy 删除历史结果读取能力。
