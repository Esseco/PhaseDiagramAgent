"""Publish navigation only, not computed scientific results."""
from pathlib import Path


def publish_workspace_guide(root):
    from analysis_layer.feedback.export_dft_products import _publish_bytes
    root = Path(root)
    text = """# 相图 Agent 工作区

- [修改科学参数](parameters/search_config.project.json)：修改后让Agent读取、审核与确认。
- [查看提交、回传与分析文件索引](analysis_outputs/output_index.md)。
- [跨epoch流程总览](analysis_outputs/round_summary.csv)。
- [运行状态](workflow_state/state.json)：状态唯一事实源，建议通过Agent修改。
- [记忆视图](agent_memory/)：经Agent审阅批准后修改，不反向导入视图。
- [启动绑定](agent_runtime.json)：启动器继续选择此文件，无需另建项目。

| 目录 | 内容 |
| --- | --- |
| parameters | 可编辑参数、配置会话、冻结快照 |
| workflow_state | 活状态、台账、缓存、审批、提交清单 |
| agent_memory | 可读记忆视图 |
| structures/reference_structures | 原始母结构与相参考 |
| structures/candidate_structures | 生成的候选结构池 |
| submissions/epochN_model | 本epoch提交输入与原始results |
| analysis_outputs/epochN_model | 本epoch相图、误差、图、训练数据和诊断 |
| logs | 服务日志 |
| history_backups | 有独立内容的历史备份；已迁移重复副本删除 |
| documentation | 说明文档 |

epoch表示模型代际，模型名使用实际版本。微调round按有效轮次编号，未执行且已废弃的旧草稿不占编号。

每个Search-group对应一批branch：Relax-0001与Relax-0001_MC-round-0001/0002配套；DFT-round按实际轮次归属；MLIP-finetune-round放在其原模型epoch下。

每轮inputs放提交文件，results放本地必需回传文件，analysis_outputs放本地处理产品。上传完整轮目录；Relax/MC在inputs中的批次目录提交GPU.sh，DFT在单任务目录提交，微调在训练轮inputs提交GPU.sh一次。回传同级results；模型和大日志留在超算。历史快照与当前相图各有用途，不按内容相同删除。
"""
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
DFT提交单任务GPU.sh，微调提交训练轮根GPU.sh一次。回传完整results后由Agent读取。
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
