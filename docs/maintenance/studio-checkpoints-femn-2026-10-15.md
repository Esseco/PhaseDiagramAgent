# FeMn Studio 检查点与卡顿修复（2026-10-15）

范围：仅 runtime/persistence 与离线检查点维护；没有生成结构、准备计算输入、执行计算、提交作业、改变模型，未启动/停止/重启真实服务。

## 修复

- Studio 历史读取的 unpack-hook saver 共享已有存储，避免先重复加载 GB 级文件再丢弃；保留原解包钩子。
- 类型序列化压缩兼容旧格式；检查点与待完成写入共享锁。无变更时跳过周期写盘；API 直接复制/删除对话也会更新持久化版本，保留 PersistentDict 身份。
- 每个命名空间保留最近 8 个检查点、每线程最近 32 个已结束子命名空间；保留当前中断/待执行命名空间，以及现有父检查点的明确引用。未持久化父图的追踪引用保持原样，不伪造检查点。活跃历史读取期间推迟清理。
- 业务 SQLite 沿用已有字段级压缩去重，新增无引用载荷回收；请求路径不执行 VACUUM。
- 流程页读取最近原子保存的状态，不等待业务锁，IO/渲染放入后台线程；一轮回复的两处展示复用一次最终状态读取。

## 离线维护证据

完整原件备份目录：`E:\0-PhaseDiagram\O3-2ele-NaFeMn\workflow_state\maintenance_backups\studio_storage_20261015_143012`。
机器记录：该目录的 `manifest.json`（原件校验）、`maintenance_report.json`（替换与保护项校验）。
源码备份：`E:\jupyter notebook\1-AL_for_PhaseDiagram\_Process_PhaseDiagram_source_backups\studio_storage_20261015_143613`。

| 文件 | 整理前字节 | 整理后字节 |
| --- | ---: | ---: |
| .langgraph_checkpoint.1.pckl | 377787 | 158454 |
| .langgraph_checkpoint.2.pckl | 4656458254 | 419734763 |
| .langgraph_checkpoint.3.pckl | 5135713198 | 499215036 |
| langgraph_lifecycle.sqlite | 7177322496 | 153853952 |
| langgraph_training.sqlite | 348839936 | 291680256 |
| langgraph_batches.sqlite | 50315264 | 659456 |

六个运行文件合计 17369026935 → 1365301917 字节。
另将遗留写盘临时文件原样移入 `orphaned_temporary_files`，不删除。
所有原件和校验值保留在 `original`；`prepared` 是验证中间产物，`maintenance_report.json` 中的 prepared 路径记录当时来源，已替换的来源文件不再位于该路径。

Studio 所有 27 个线程/命名空间最新恢复点：检查点、元数据、所引用通道字节、最新待完成写入的类型与内容校验一致；原始全部历史仍在备份。
各 SQLite 的所有最新恢复点，迁移前后完整状态、元数据和 pending writes 指纹一致；整库 integrity_check 与替换后 quick_check 通过。
当前 state.json、控制修订、科学账本、所有 branch 方案、原生审批库、主执行收据及 Studio 网关收据的 SHA256 均未改变。

## 验证与边界

最终生产文件定向回归 54 passed，4 deselected。四项旧失败已在未修改基线复现，涉及展示文案/流事件/子图身份断言；没有冒充通过。新增测试覆盖压缩、去重载荷回收、旧格式读取、恢复保留、无变更不落盘、共享视图、复制删除、父追踪引用和繁忙业务锁下的流程页。
只对已改文件执行 Ruff 检查与格式验证。没有启动真实项目服务，所以真实启动耗时、交互速度和 Windows 连接重置是否复发尚未实测。
该适配器针对当前安装的 LangGraph dev runtime 内部接口；升级依赖时需重跑针对性测试。会话根历史、有效审批及真实增长的科学数据不以大小为由自动删除；备份不会自动清理。

## 恢复与启动

由用户手动启动 FeMn 服务加载新代码。若需回滚，先由用户停止服务，在项目锁保护下从 original 恢复对应文件；SQLite 若存在 WAL，需按备份清单成套恢复，不混用新旧 WAL。源码回滚使用上述源码备份；不要删除科学账本、结构、模型、审批或收据。

仓库 TEMP_PLAN.md 在维护期间被另一项 Interaction Efficiency 任务替换；已确认不是本任务计划，因此保留，未删除或改写。以上源码哈希与验证结果对应本次交付时点。
