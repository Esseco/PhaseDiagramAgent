# 项目文件管理

源码只在 Process_PhaseDiagram/phase_agent；五个科学项目保留在 E:/0-PhaseDiagram。新项目使用同一模板，不复制应用代码。现有入口 `python run_project.py` 不变。

```text
项目/
├── run_project.py               打开此项目
├── agent_runtime.json           启动绑定
├── parameters/                  初始配置、运行配置和确认版本
├── structures/                  母结构、候选结构
├── workflow_state/              科学状态、账本、审批和执行凭据
├── interaction_records/
│   ├── epoch0/                  本轮交互；阶段_YYYYMMDD_序号.md
│   ├── epoch1/
│   ├── project/                 项目级维护说明
│   └── unassigned/              无法可靠确定轮次的旧记录
├── submissions/
│   ├── README.md                提交与回传索引
│   ├── epoch0/
│   │   ├── README.md            完整模型版本与各目录用途
│   │   ├── transfers/           普通计算、训练的配对上传和回传
│   │   ├── relax_mc/            弛豫与蒙特卡洛配对批次
│   │   ├── verified_results/    已校验读取的原始结果
│   │   ├── recovery/            仅回收资料和故障凭据
│   │   ├── storage_archives/    完整归档包
│   │   └── G0001_*/             登记的计算工作目录；inputs 与 results
│   └── epoch1/
├── analysis_outputs/
│   ├── output_index.md          已有产物索引
│   ├── round_summary.csv        跨轮汇总
│   ├── gpu_usage.csv            跨轮 GPU 用量
│   ├── epoch0/
│   │   ├── phase_diagrams/      mlip、dft、combined 及 history
│   │   ├── Search-group-*/DFT-round-*/
│   │   │   ├── comparisons/    比较表、指标和 plots
│   │   │   ├── training/       训练结构与标签
│   │   │   └── diagnostics/    磁矩和检查
│   │   └── recovery/           本轮历史恢复分析（若存在）
│   └── epoch1/
└── history_backups/             人工维护快照
```

只创建实际需要的目录；尚未产生科学结果的 Mn/Ni 项目有导航文件，没有伪造的结果或任务。

## 日常使用

1. 阅读本轮答复中的实际路径。上传完整 `_input` 文件夹，在超算执行其中的 `sbatch submit_all.sh`。
2. 将远端对应 `_output` 文件夹整体放回原登记目录，再告诉 Agent“已回传”。程序沿登记的回传路径校验和读取，不扫描其他项目寻找结果。
3. `verified_results` 是已校验的科学结果与凭据；分析图表看 `analysis_outputs/epochN/`。历史已提交任务不重投。
4. 交互详细记录在 `interaction_records/epochN/`；答复保留本轮文件的完整地址。项目级维护文档放 `project`，无法可靠确定轮次的旧记录放 `unassigned`。

## 维护实现

模型与 epoch 的对应仍由 `upload_layout.model_rounds` 登记。目录采用简短 epoch 名称，完整模型版本与任务、批次身份不变。重复编号对应不同模型会显式报错，防止目录合并。

- `tools/remote/model_upload_directory.py`：轮次目录的共享命名。
- `runtime/interaction_report_paths.py`：交互目录和独占序号；未知模型不会猜测 epoch。
- `tools/remote/submission_handoff.py`：普通计算和训练的 transfers 配对包。
- `tools/relax_mc_batch/prepare.py`、`results.py`：Relax/MC 配对批次和 verified_results。
- `analysis/feedback/workspace_guide.py` 与 templates：新项目和现有项目共用说明。
- `tools/local/organize_epoch_folders.py`：显式迁移、冲突预检、字节核验与归档代码副本清理。

迁移直接使用真实 epoch 目录。运行记录与索引更新为新路径；历史 manifest 与冻结配置保持原始内容，读取时使用显式路径迁移记录。旧目录链接与重复文件名核对后移除。

清理仅删除 SHA256 与保留 inputs.zip 对应成员一致的解压运行代码；删除清单记录归档路径、成员名和哈希。内容不同的文件、原始结果和恢复资料保留。修改源码后由用户手动重新启动服务加载，维护工具不控制服务或提交计算。

目录被占用而无法移动时，迁移明确报错并保留原数据；不再创建目录链接。

旧路径的读取转换复用 `configuration/session/relocate_workspace_paths.py`，支持多次搬迁与循环检测；冻结配置保持原样。当前状态持久化新路径，工作流控制记录保留迁移版本，旧检查点在读取时转换，过期状态写入会被拒绝。迁移不依赖目录 junction 或文件硬链接。
