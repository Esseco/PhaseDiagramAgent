# 冻结审计、检查点、报告和模型修复热点审查（2026-11-10）

## 范围与已确认事实

本轮先读取 AGENTS.md、decision-flow-review.md、native-team-performance-review.md、graph.md 及对应实现。已有角色任务、指导替代、草案修改、单次澄清、就绪信息、索引、单次字段复用和团队响应改进继续使用，不重复实现。基线来自修改前磁盘源码及先行基准，不使用 Git HEAD。仅开发源码、测试与本文变化；没有真实科研状态读取/写入、真实模型、HPC、部署或服务重启。临时本机测试服务器仅用于一个被替换回调的HTTP测试，随机端口，结束关闭。

本轮没有核实真实服务加载模块的身份。所有性能结果来自 py1 新验证进程及磁盘源码，不能视为已运行服务生效。

### 冻结动作与审计

_action 把当前 advice 放入 _decision_team，同时把全部 session.events 放进 human_review，另保留原 guidance 与 effective_guidance。原生动作随后进入 proposal.raw_action、策略 original_action/final_action、pending 和 action_records；包含这些状态的业务帧也会进入检查点。按不同字段持有和序列化会重复审计正文；不能据此推出指数增长。

实际通过团队入口进行每修订一次澄清、两次草案/Simulation计划登记、再修订，最后仅接受方向，未 propose 或执行：

|修订数|事件数|持久讨论文件 bytes|冻结 action bytes|当前意见数|历史 Simulation 计划数|
|---|---:|---:|---:|---:|---:|
|1|8|7154|4316|1|2|
|5|44|25772|16450|1|10|
|10|89|49021|31633|1|20|

另用固定4条事件/修订、固定约4 KiB正文的压力场景，1/10/100修订 action 为17,332/170,496/1,702,568 bytes，近似线性。本轮没有改变这些审计字段，也未把它们换成引用。不可变审计对象、自包含导出、损坏/缺失恢复与旧记录契约尚需共同设计，不能只留下易丢失文件路径。

记忆候选并非复制整段审计：collect_memory_candidates 从已明确关联任务的结果选择 decision 字段、参数名和稳定 action_parameters_ref；decision_memory_context 的近期动作仅取5项摘要和team_id。没有证据表明当前记忆候选自动把全讨论递归回灌成指数增长。

### 检查点

project.nodes 在持久节点调用 checkpoint_frame 并比较业务帧；_snapshot 会先比较内容、复用已验证相等的旧字段，变化列表仍需分离。CheckpointBlobSerde 已有同批对象memo、内容hash、数据库去重、流式列表、压缩与历史解码。未新建持久层。

确认热点：流式大列表在数据库命中前已完成编码、hash、临时写入和压缩。16384行压力样本每次65次 JsonPlus 编码（分块+外层），1次流式hash、1个流式压缩器；命中并不免除这些前置成本。本轮只把压缩移到“完整编码/hash后确认miss”再做，事务写入仍重新检查，避免用可变对象身份认定内容不变。没有跨请求完整对象缓存。

真实 lifecycle.snapshot 来自已确认配置，通常不是完整科研state。基础压力帧特意加入 snapshot.state 验证可序列化嵌套状态开销；它不是生产帧常态，不能把此压力形状当成已证实的生产重复。_frame 对 loaded_state/result/feedback/pre_reconciled 的既有去重路径保留，未删除恢复字段。

### 报告

_propose_codex 的 awaiting_approval 路径已经自动返回 report_path，并捕获生成失败，不把展示失败改成业务重试；codex_report 对现有pending只读读取状态并渲染，不重放提案。同一方案三次 save_codex_report 确实三次生成交互报告，目录文件数为4（含源state）。简单pause报告首次约0.86秒，预热后约2.6/2.2毫秒；首轮主要有库导入高水位。

本轮未给通用报告加缓存：format_workflow_reply/save_interaction 会查看待审状态、引用产物、DFT选点展示及恢复相关信息，外部文件依赖没有完整manifest。仅以proposal hash缓存可能展示过期审批信息。已有自动响应路径可直接使用；缺失/失败时仍可 phase_report。上述简单样本不能说明 DFT 报告便宜，也未验证所有昂贵分析/图表分支。

### 普通模型路径

request_validated_action 原来依次可能调用 schema、branch、只读对话、post-DFT及scientific绑定修复；各自“一次”不能保证合计一次。DeepSeek 每个独立请求原来重新建立 total_timeout，内部推理截断还允许一次恢复。普通图包含咨询和最终提案，Simulation在后续规划阶段；原生Codex和冻结适配器不属于项目模型咨询。

先行假模型复现：initial → proposal_contract_repair → branch_review_repair，共3次，既有usage能合计但没有共享修复额度。已存在逐provider model_requests、usage_summary、同轮精确请求缓存和API硬等待期限，本轮沿用这些设施。

## 实际改动与契约

1. with_advice 在复制前排除将被替换的 decision_context，保留其他字段深复制并独立复制一次context。原来先深复制整个payload又复制context，第一份被丢弃。仍隔离调用方、两轮意见与其它别名；没有一律改浅复制。
2. 流式CAS命中跳过压缩；编码/hash仍每次执行。miss会从有界临时raw文件压缩，第一次写入增加一次raw文件读取的取舍；旧schema、内容ID、原子写入、回滚重查和解码保持。外层 CompressedSerde 的压缩仍存在，不把“流式压缩器0次”说成全部压缩0次。
3. api_deadline 增加只作用于模型的共享scope：普通咨询图、最终提案与修复加入同一scope；有record_turn时同轮多个请求链及后续DeepSeek规划也共享。无turn记录时只保证单个请求链，不跨相互独立的调用保存额度。
4. 默认共享2次修复。schema/branch/dialogue/post-DFT/scientific修复及DeepSeek输出截断恢复共用额度；独立的初次专家分析不是修复。新增客户端限额元数据由原 copy_model_capabilities 传播。后续scope不能重置已有deadline/额度。
5. DeepSeek每次HTTPdeadline=min(原单请求deadline,共享deadline)，保留原传输slot、取消、超时逻辑。存在turn时预算在首个模型阶段开始；不覆盖此前读盘和后续报告的总耗时。普通callable采用调用前后合作式检查，不能中断任意正在阻塞的回调；DeepAgents内部provider迭代仍受其已有机制，本轮未统一其全部网络超时/修复计数。
6. model_call_chain 仅保存call_id、parent_call_id、mode原因、repair、invoked、耗时、已知usage；被额度拒绝也可区分未调用。DeepSeek物理model_requests记录其logical_call_id、已编码HTTP正文 input_bytes、网络耗时、原有attempt/finish_reason/已知tokens。逻辑请求未编码时 input_bytes=null，不伪造0或额外完整编码科学状态。费用未知保持null；预算耗尽后保留已付出调用的usage。
7. schema已经失败时汇总同时可判定的branch/region错误，交给同一次模型修复；原分支与科学校验仍在后续执行。程序不选择候选或修改参数，不截断证据。在同一假模型响应规则下，初始同时有负预算和错误证据引用，调用3→2，修复2→1，最终仍为相同generate_branches及合法审阅。模型在后一次输出新错误时仍可能用到第二份额度，超额即失败。
8. 原生codex/_codex_structured适配器不进入这些项目模型scope；冻结方案复用不会因此调用模型。没有科学动作超时重放、自动retry或新调度器。运行时budget在记录结束时释放，只持久保存元数据链；现有精确请求缓存仍按原生命周期释放。

## 对照测量

tests/benchmark_remaining_hotspots.py 使用受控TemporaryDirectory，固定独立维度：任务512/4096/16384；压力讨论修订1/10/100；实际讨论修订1/5/10；动作历史0/10/100；检查点1/5/20。最大基础请求约5.53 MB，没有复制真实项目或无界profiling。

墙钟为perf_counter。每次操作前GC取RSS基线，独立线程5ms采样进程RSS/private，释放结果并GC后取结束RSS。短操作可能漏采尖峰，采样窗口还包含输出大小统计；秒数只涵盖callback本身。不是tracemalloc。固定BLAS/OMP/MKL为1；未控制整机负载或给出统计置信区间。前后为相同合成内容、同样的计数方法，各新进程执行一次配对系列；不能外推真实项目。

|操作|前→后秒|RSS 基线/峰值/结束 MiB（前）|RSS 基线/峰值/结束 MiB（后）|
|---|---:|---:|---:|
|with-advice-512|0.006278 → 0.003677|50.91/51.31/51.31 |50.98/51.17/51.17|
|with-advice-4096|0.040845 → 0.019616|58.27/60.74/59.14 |58.42/58.49/58.47|
|with-advice-16384|0.167385 → 0.081779|76.92/91.10/80.80 |76.45/88.62/76.91|
|checkpoint-16384-0|0.030664 → 0.025284|77.80/82.70/77.88 |76.91/82.13/77.30|
|checkpoint-16384-1|0.022677 → 0.017302|77.88/77.90/77.88 |77.30/82.13/77.30|
|checkpoint-16384-2|0.023022 → 0.016086|77.88/82.71/77.88 |77.30/82.13/77.30|
|checkpoint-16384-3|0.022784 → 0.017364|77.88/82.71/77.88 |77.30/82.13/77.30|


with_advice最大样本输出均为5,530,113 bytes，科学内容未裁剪。checkpoint-*数字后缀0为第一次写入，其后为相同内容命中：encode=65、hash=1保持，流式compressor由1→0。此处基础探针output_bytes=72是“统计摘要对象”的JSON大小，不是checkpoint wire大小；不能混淆。

实际配置快照形状的补充探针（无前后修复比较）：
- 动作历史0/10/100时，原始帧JSON为73,398/162,056/960,176 bytes；wire为1,203/2,179/1,315 bytes。100项时达到blob阈值，额外blob为2,823 bytes，wire下降并非历史丢失。
- 4096任务、相同内容1/5/20个检查点，单wire均308 bytes，内容blob保持1个/17,472 bytes；数据库32,768/32,768/40,960 bytes。没有声称检查点表或数据库不增长，也没有删除科学历史。

完整数字见 remaining-hotspots-metrics.json；其中private、RSS、耗时和计数按case保存。大小使用 len(json.dumps(..., ensure_ascii=False, separators=(",", ":"), default=str).encode())，不是token、provider最终费用或MCP外层字节。物理HTTP input_bytes则直接取已经生成的Request.data长度。

## 时间线与连续内存

改后通过真实run_tool_step、假四角色客户端、临时state/报告跑到awaiting_approval，无工具执行；不是完整真实project初始化或服务端端到端对照：
- read: 0.004336秒，RSS 193.35/193.35/193.35 MiB。
- consult-strategy-simulation-validation: 0.588931秒，RSS 193.35/194.29/194.27 MiB。
- persist: 0.091493秒，RSS 194.27/194.33/194.31 MiB。
- report: 0.077529秒，RSS 194.31/194.45/194.45 MiB。
- return: 0.000017秒，RSS 194.45/194.45/194.45 MiB。

其中模型阶段依次为咨询范围、Exploration、Reliability、Strategic、Simulation；对应请求业务JSON为62523/56825/66343/123830/50507 bytes。假客户端只用于流程观测，耗时不代表真实API等待。校验与冻结包含在run_tool_step阶段，检查点另由专门探针测量；没有把聚合时间拆成虚构的独立科学耗时。

连续20次4096行请求构建，结束RSS从191.93 MiB，经前几次预热到193.29 MiB，后段进入平台。报告首次库导入增加内存亦单独显示。此短测未见持续线性增长，不能证明或排除真实长期泄漏、进程间/客户端保留或其他项目的占用。

## 验证、兼容与保留方向

综合回归：352 passed、1 deselected、2 subtests passed，151.26秒；14条第三方Matplotlib/Pyparsing弃用警告。被排除的一项临时本机HTTP测试在受限环境因WinError 10013失败，单独允许本机测试权限后1 passed（0.65秒），不接触真实服务。最后高分辨率计时调整后，新热点和API期限19项再次通过（4.83秒）。新增test_remaining_hotspots共15项。没有运行全仓或真实项目回归。

Ruff check phase_agent及新增测试/基准通过；本次13个代码文件的格式检查通过。全包format检查仍报告135文件需格式化、770文件已符合，未扩散改写其他任务文件。按用户要求由当前主agent自审，没有派生审查agent或更换模型。

测试命令统一前置PYTHONDONTWRITEBYTECODE=1，BLAS/OMP/MKL线程1，PYTHONPATH=tests/fake_only_guard;<repo>;E:/Py-Code；pytest -q -p no:cacheprovider --basetemp E:/Py-Code/pdt --disable-warnings。基准额外设置TEMP/TMP=E:/Py-Code/pdt，运行tests/benchmark_remaining_hotspots.py及--timeline/--history/--audit。conftest按TemporaryDirectory隔离；基准SQLite连接显式closing。开发中一次SQLite句柄未关闭导致临时目录删除失败，修复并定点清理确切自有目录后重测成功，交付检查无phase-hotspots-*残留。没有清理其他任务目录或数据。


保留旧冻结动作、讨论和checkpoint；没有存量状态迁移。测试覆盖无变化、局部变化、同对象内容变化、rollback后重建、缺失/损坏、旧解码、暂停与跨进程恢复、输入/别名隔离、不同轮意见、额度/期限/截断/用量、客户端适配、Codex不调用项目模型、报告失败不重提案、审批/回执/防重放。

尚未实施：冻结审计引用；通用报告缓存；根据identity跳过checkpoint遍历；局部列表结构共享diff；对科学binding错误一概合并；DeepAgents全部内层网络链的统一取消。原因分别为恢复/导出契约不足、依赖manifest不足、无法证明可变对象不变、额外算法缺少显著收益证明，以及接口与科学边界不宜混改。科学证据、全部审计与公共入口继续可取得。

未验证真实模型质量、真实API延迟/费用、DFT复杂报告、大项目端到端性能、真实服务加载代码、长期泄漏或部署效果。源码完成不表示运行中已生效。

## 补充原语测量与源身份

同一16384行列表独立运行5次的中位数如下。编码按原流式256行块构造相同msgpack字节，校验与完整编码逐字节相等；hash与压缩按相同块顺序。它们是独立原语计时，不是实际节点互斥分解，不能直接相加，也没有把它们伪造为修改前各阶段耗时。raw=5,035,165 bytes，压缩=73,963 bytes。

- equality: 0.000547 s
- detach: 0.037887 s
- encode_64_chunks: 0.003449 s
- hash_stream: 0.002098 s
- compress_stream: 0.005066 s

- strategic/context.py before: bc0b859f788ad5152c93d4fadcb6aed57937d78b1ea7ab1862eeca674b5770b8
- checkpoint_blobs.py before: 79e1768bc956ed37d1ff7c714472ef2dbd6b908d3cb0c5632981e2ae81f97128
- api_deadline.py before: 0f8a98b4658f710da06dacffb005beb4585c8863fea25e9df4ef9c953a9f7ff5
- proposal_schema_repair.py before: 150538698eaab95f1bce10c16f491633b233dbd3fe6dc2f207088554fc62d449
