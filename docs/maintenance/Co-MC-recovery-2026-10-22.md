# Co MC 回收修复与操作说明（2026-10-22）

## 已确认的原因

Co epoch1 RelaxMC_Batch01 的两个 MC 作业 3752097、3752098 已正常结束。远端 27 个唯一 MC 任务都有 completed 结果、完成标记及校验和一致的最终结构。旧上传包将 outputs.pool_summary 错误校验为 dict，而 worker 实际输出 list，因此 27 项全部被 result_output_object_required:pool_summary 拒收。控制器仍推进 phase，并以 completed / tasks_in_progress 提前结束；压缩回传包只包含已接纳任务的结果，因而漏掉 MC。

本地已修正字段类型；旧上传包使用冻结源码，不会自动获得本地更新。

## 本次项目修复

- 依赖式控制器在反馈校验入账后检查任务状态和拒收记录，再决定是否完成一轮。空或缺失作业范围不能算成功。
- 同步运行入口使用相同完成条件；拒收后停止推进，不登记后续轮次。
- 最终 completed 受到未完成任务及校验错误约束；tasks_in_progress 不再落到 completed。
- 回传诊断保留 remote_validation 来源、原拒收代码、任务及作业映射。
- 已导入但仍有待核验任务的批次始终进入补收提示，包括旧 completed 回传。
- 仅补收工具生成补收包前校验科学字段，并根据已有任务的 segment_index 核对完成轮次。旧包缓存与新校验规则通过构建版本区分。

## 现在如何处理 Co

1. 方便时由用户手动重启 Co 项目的本地服务，加载新源码。本次修复没有重启任何服务；不必重启超算任务。
2. 对现有远端结果执行“仅补收 Co 这 27 项 MC，不重算、不提交后续轮次”。下一次操作应使用当前源码中的 collect-only 工具，保留旧 results.zip 和远端原工作目录。
3. 原工作目录为 `/data/home/lichaoyue/26-10-PhaseDiagramAgent/O3-1elm-NaCoO2/epoch1_RelaxMC_Batch01_input/work`。需要读取 workflow_state.json、两个 MC 批次的 task.json/result.json/task.finished.json 及其引用结构、候选池和检查点。对应操作目录是 `jobs/epoch0_remote-94dca56f3378809f-8000f88ca5fd/G0001_Relax-0001_MC-round-0001`。按需要收集这些证据，无需复制整个项目、runtime 或模型权重。
4. 使用 sacct 对 3752097、3752098 分别获取终态记录，并从原 workflow_state.json 提取各自 task_ids（20 项和 7 项），不能凭作业正常退出直接把任务状态改成成功。
5. 复用 `python -m phase_agent.tools.relax_mc_batch.cli recover`：两个作业分别提供 job-evidence，第二次以第一次生成的 recovered.results.zip 作为 original-archive。evidence-directory 保持原相对目录结构；destination 与证据目录必须分开。使用当前 py1 环境和生产源码，不运行旧上传包中的 run_batch.py。
6. 生成新补收包后，先核对 27 项结果、绑定、结构和签名，再导入。本批原 config_version 为 config-0013-b4201d1daf1b9deb，可能与当前配置不同；必须按原登记批次的冻结绑定处理，不能为了导入修改当前配置或绕过模型/配置校验。
7. 通过标准：27 项已有 MC 全部登记，重复导入不重复结算；这只证明已提交的第一轮 MC 补收成功。后续轮次未提交时应显示 partial/待后续处理，而不是整个多轮流程完成。后续计算需另行确认原方案与当前状态。

普通“已回传”只能重读已有压缩包；旧 results.zip 本身没有这 27 项文件，重复该操作不能补出结果。也不要重新执行原输入目录的 sbatch submit_all.sh。

## 操作边界

本次只修复应用源码、合成回归测试及文档，没有改 Co 科学任务状态、预算、结果、配置，没有覆盖旧上传/回传包，也没有提交或取消超算作业。源码修复不等于历史结果已经补收。

## 交付验证

140 related regression tests passed; 39 targeted tests on the final code passed; ruff check passed; ruff format check passed (739 files).

只读真实回归：远端 27/27 项 MC 通过当前科学字段校验、原任务绑定、结果签名及最终结构校验和。未执行历史补收、未改项目 state.json。

源码备份：`E:\jupyter notebook\1-AL_for_PhaseDiagram\_Process_PhaseDiagram_source_backups\co_recovery_completion_20261022`。


## 历史配置补收入口（追加修复）

新增 `recovery_receipt.py` 和 CLI `register-recovery`。登记文件 `RelaxMC_Batch01/recovery_return.json` 选择独立恢复包并固定原包与恢复包 SHA256，保留原始 `_output/results.zip`。原有 `recover_available -> import_results` 自动使用已登记恢复包。

配置修订不再阻断此类受限补收：必须同一活动模型、同一原登记批次、方案校验和一致、原包所有任务范围一致，且每个任务已存在于当前台账，任务键、结构、分支、阶段、作业批次、任务校验和、模型和原配置全部一致。取消或被替代批次、已取消任务、跨模型、范围新增及损坏文件仍拒绝。普通历史包保留原配置隔离。导入继续执行原有签名、结构、科学字段、台账与费用防重流程，不改当前配置，不新增计算。

Co 恢复包已登记：`submissions/epoch1/relax_mc/RelaxMC_Batch01_recovery/a5b191c954138c8f/recovered.results.zip`。当前真实任务登记预检通过，27 项 MC 和 287 项 Relax 共 314 项保持原绑定；摘要为 partial，后续轮次未提交。

**尚未执行正式台账导入，也未启动/停止服务。** 用户手动重启 Co 本地服务加载新代码后，在原项目发送“仅回收已登记的恢复包，不重算、不提交后续轮次”。回收后应核对本批 27 项 MC 已登记、待核验数为 0、重复回收不重复结算；多轮整体仍不能标记完成。

验证：最终新增专项 21 passed（含自动回收调用链与重复导入，科学反馈使用合成替身）；相关恢复回归 68 passed；原批处理与恢复组合 79 passed；ruff check 全包通过，format check 739 文件通过。真实 Co 只读校验及恢复入口解析通过，正式 UI 与入库效果待用户重启后验证。

备份：`_Process_PhaseDiagram_source_backups/historical_mc_recovery_20261022`。本次只新增恢复包登记文件，未改科学状态、配置、账本或原结果包。
