# 默认交互 Relax + MLIP-MC 上传封装

统一流程的新 Branch 交互默认推荐此封装，通过现有审批工具准备。
明确说“每一步我自己调用，分别分配 Relax 和 MLIP-MC”时使用原逐步流程；Agent 沿用该明确偏好，直到你要求改回封装。
已有在途任务和已批准方案保持原回收路径，不自动转换或重复生成。默认封装不免除审批，也不自动上传或提交。
默认由 Agent 提出封装方案，审批后准备输入；旧任务保留原回收路径。
先生成并登记 Branch，再让 Agent 推荐具体 Branch，提出 `prepare_relax_mc_batch`。
操作绑定 `prepare_local_batch_files`，参数 `mode=relax_mc_inputs`，`target_ids` 为 Branch ID。
审批前自动诊断去重状态、结构/满 Na 模板、相参考、当前模型、历史相图和任务重复情况，展示工作量上限。
模型和配置版本被冻结；需要改变模型、规则或范围时生成新提案重新审批。

## 计算规则

已有当前模型相图时，携带其能量/组分证据和可用结构；新增 Relax 结果合并后按现有初始 MC 分档。
没有相图时，所有成功 Relax 且可做 Na/V MC 的 Branch 首轮统一 `100/10`，不虚构 Ehull。
每段后复用相识别、结果回收和相图模块。后两段使用共享 Ehull＋改善幅度表：
高档 `200/20`、中档 `150/15`、低档 `100/10`、探索档 `50/5`。
高档全保留、基态/近简并保护、约 10% 低档探索和至少 10 个有效追加任务规则与原流程一致。
无有效相图、追加池不足或无追加价值时可提前结束/暂停并返回已有结果；不保证强制执行三段。
停止改善阈值不覆盖计算库默认值。Na0 Branch 可参与 Relax/相图，但不执行 Na/V MC。
MC 失败暂停，不自动重投；提交回执不明确时禁止自动重复提交。

## 上传与回传

批准后本地生成独立目录，含 `inputs.zip`、`submit_all.sh`、`workload.json`、`batch.json` 和说明。
上传输入包，解压到独立、较短的超算目录，进入该目录执行一次：

```bash
sbatch submit_all.sh
```

控制作业自动派发现有分组 GPU 作业，等待 Relax 与最多三段 MC 并保存检查点。
估算区分结构计算任务与 Slurm 作业，另外计 1 个控制作业；默认计算并发为 1。
沿用当前 GPU 队列、资源、环境初始化及 Relax/MC 分组配置。
可在项目配置 `supercomputer.relax_mc_batch.max_concurrent_jobs` 或提案同名参数指定并发。
预算默认建议值；显式硬预算仍须遵守现有校验。

输入包含结构、满 Na 模板、相参考、可用历史相图能量、模型路径/SHA/大小证据、应用代码和现有 MC 后端。
模型权重不复制，必须已在超算配置位置。超算原科学环境须有 MACE、torch、ase、pymatgen、numpy、scipy 等现有依赖。
模型和输入校验失败不会派发子作业。使用同一目录恢复，不删除检查点，不重投仍在运行的控制作业。

下载远端根目录 `results.zip` 到 Agent 显示的本地 `expected_results_archive` 位置，再说“继续”。
回收核验包与任务校验和、模型/配置/计划绑定，合并原台账，不覆盖其他历史。
返回结构、MC 占据/检查点、任务结果、相图 JSON/CSV、候选池/QBC及实际作业数，供 Agent 选择继续 Branch 或 DFT。

## 离线检查

```bash
export PYTHONPATH="$PWD/runtime/mc_backend:$PWD/runtime:${PYTHONPATH:-}"
python3 run_batch.py run --root "$PWD" --dry-run
python3 run_batch.py inspect --archive results.zip
```

Dry-run 不提交、不计算、不写执行检查点。开发测试使用假科学执行器和假超算；不加载真实模型或调用 sbatch。
远端实机环境、排队和物理计算仍需首次实际上传后验证。
