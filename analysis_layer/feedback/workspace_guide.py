"""Publish navigation only, not computed scientific results."""
from pathlib import Path


def publish_workspace_guide(root):
    from analysis_layer.feedback.export_dft_products import _publish_bytes
    root = Path(root)
    text = '# 相图 Agent 工作区\n\n本目录保存一个独立项目的配置、结构、提交文件、结果和记忆。程序源码另存于代码仓库；启动服务后，在 Studio 中与本项目 Agent 对话。\n\n## 启动与日常操作\n\n在 **源码目录** 激活 `py1` 后运行 `python -m run.local_project_launcher`，选择本项目的 [agent_runtime.json](agent_runtime.json)。不要在只有结构和数据的项目目录中运行这个模块。\n\n| 你要做什么 | 对 Agent 说什么 |\n| --- | --- |\n| 查看进度，不推进任务 | 查看状态 |\n| 回收新结果并推进到下一处等待 | 继续 |\n| 手动修改文件后先检查 | 读取配置 JSON |\n| 检查配置，通过后继续 | 读取配置 JSON 并继续 |\n| 修改长期使用的参数 | 修改运行配置：把……改为…… |\n| 批准独立验证通过的模型 | 激活候选 `<版本>` 原因：`<审阅理由>` |\n\n“继续”用于回收与推进，不等于批准模型激活。具体科学计算由你在超算提交；Agent 准备输入并说明提交位置和回传文件。\n\n## 两份可编辑配置\n\n- [初始配置](parameters/search_config.project.json)：本地与超算环境、搜索边界、体系组成、母结构、模型路径和必要计算设置。\n- [运行配置](parameters/run_config.project.json)：预算、采样、MC、微调、验证和收敛参数。旧项目未拆分时仍使用原科学配置文件。\n- 两份均可手动编辑，也可告诉 Agent 要修改的值。保存后重新读取检查；通过后生成新版本快照，历史计算仍归属原配置版本。\n\n初始必要信息齐全即可进入搜索建议；缺项或冲突会再次询问。后续阶段缺少的验证标准等信息，到使用该阶段时再补齐。\n\n## 超算提交与回传\n\n上传完整轮次目录，在对应 `inputs` 中提交作业。计算、模型读取和 SHA256 计算均通过计算节点作业执行；登录节点只负责提交与查询。\n\n| 阶段 | 提交位置与文件 | 回传内容 |\n| --- | --- | --- |\n| Relax / MC | `inputs` 内对应批次的 `GPU.sh` | 对应 `results` |\n| DFT | 对应单任务目录中的提交脚本 | 对应 `results` |\n| 微调训练 | 训练轮 `inputs/GPU.sh`，提交一次 | K折及逐点CSV、`models.json`、完成标记 |\n| 补模型清单 | 原训练轮 `inputs/GPU_manifest.sh` | `results/models.json` |\n| 独立模型验证 | 原训练轮 `inputs/GPU_validation.sh` | 指定的 `results/validation-<request_id>.json` |\n\n回传后说“继续”。模型和大日志保留在超算，不必下载；独立验证数据必须与训练数据区分，K折结果不能替代最终主模型独立测试。\n\n## 等待与恢复\n\nStudio 主图可展开 `training_lifecycle`，查看回收、清单检查、作业准备、验证和等待节点。训练交接通过 LangGraph 原生中断暂停，检查点保存在 `workflow_state/langgraph_training.sqlite`。重启后“继续”会带入最新业务状态并重新核对文件。\n\n[运行状态](workflow_state/state.json) 是业务状态事实源，建议通过 Agent 修改。检查点记录恢复位置，不替代业务状态或激活审批。其他搜索阶段沿用各自现有恢复机制。\n\n## 文件导航\n\n- [结果与分析索引](analysis_outputs/output_index.md)\n- [跨epoch流程总览](analysis_outputs/round_summary.csv)\n- [提交与结果布局说明](documentation/FILE_LAYOUT.md)\n- [项目记忆](agent_memory/)：事实与建议经审阅管理，文件视图不直接反向导入状态。\n\n| 目录 | 内容 |\n| --- | --- |\n| parameters | 初始/运行配置、配置会话和版本快照 |\n| structures/reference_structures | 初始母结构 |\n| structures/candidate_structures | 候选结构池 |\n| submissions/epochN_model | 提交输入与原始回传结果 |\n| analysis_outputs/epochN_model | 相图、误差、训练数据和诊断 |\n| workflow_state | 状态、台账、缓存、审批和检查点 |\n| agent_memory | 项目记忆视图 |\n| logs | 服务日志 |\n| history_backups | 历史备份 |\n| documentation | 项目使用说明 |\n\n`epoch` 表示已激活模型的代际；同一模型下的多个搜索轮次不增加 epoch。每个 Search-group 对应一批 branch，Relax、MC 和 DFT 按该组关联；微调轮属于原模型 epoch。每轮 `inputs` 放提交文件，`results` 放原始回传，`analysis_outputs` 放本地处理产品。\n'
    if (root / "runtime/state.json").is_file() and not (root / "workflow_state/state.json").is_file():
        # Existing projects keep their explicit layout until authorized migration.
        from config_layer.session.workspace_directory_names import DIRECTORY_RENAMES
        for old, new in sorted(DIRECTORY_RENAMES.items(), key=lambda item: -len(item[1])):
            text = text.replace(new, old)
    _publish_bytes(root / "README.md", text.encode("utf-8"))
    if not (root / "runtime/state.json").is_file():
        details = """# 提交、结果与分析文件

## 对应关系

epoch对应实际激活的模型版本；Search-group对应一次branch生成与搜索链。
Relax-0001与Relax-0001_MC-round-0001、0002为同一组的浅/深MC。
DFT-round归入该Search-group；MLIP-finetune-round位于原模型epoch下。
未执行且已明确废弃的微调草稿不占编号，不重编号真实运行历史。

## 回传到哪里

submissions保存提交输入和各阶段原始results。Relax/MC提交批次GPU.sh，
DFT提交单任务脚本，微调在训练轮inputs提交GPU.sh一次。回传该轮必要results后由Agent读取。
不把原始results放到analysis_outputs，不手动修改task_id或完成标记。

## 分析文件在哪里

analysis_outputs/epoch…/phase_diagrams：MLIP、DFT、combined三种相图及history。
analysis_outputs/epoch…/Search-group…/DFT-round…：

- comparisons：能量、受力逐点CSV，误差指标；plots是对角线图。
- training：可供MLIP训练的结构和标签。
- diagnostics：逐原子磁矩与检查信息。

当前版本不因重复请求重新分析；有新有效数据时由Agent更新。
母结构在structures/reference_structures，候选池在structures/candidate_structures。
参数经Agent复核确认后生效，记忆视图不直接反向导入state。
完整文件位置见[文件索引](../analysis_outputs/output_index.md)。
"""
        _publish_bytes(root / "documentation/FILE_LAYOUT.md", details.encode("utf-8"))
