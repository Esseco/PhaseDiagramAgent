# FeMn DFT 重新选点取证修复（2026-10-22）

## 根因与修复

FeMn 最近一次“重新生成方案”停在只读取证阶段，未生成新 DFT 方案。原待审 16 点方案绑定 config-0017；当前生效 config-0018 更新了 DFT 每轮数量限制，旧方案不能沿用旧批准依据。这不是本次取证报错的根因。

本次定位到一个可复现的客户端组合缺陷：FeMn 的 agent_runtime.json 配置 proposal_harness=hybrid。hybrid_proposal.create_hybrid_client 会给返回值附加 _proposal_harness 和 _analysis_route，而旧 summary_request 将除 _llm_usage 外的全部字段交给 extra=forbid 的 ScientificEvidenceRequest 校验，导致合法取证回复也因内部元数据被拒绝。旧日志又把详细原因包装成“取证请求不符合只读契约”，因此无法从那次已保存日志直接读出原始字段错误。已用实际 hybrid 包装器和假模型复现、修复。

修复只在取证 schema 边界排除这两个已知运行元数据字段（及原有 _llm_usage）；最终方案的运行来源记录保留。其他未知字段、越界工具、错误主题和字段仍严格校验。

真正的模型取证格式错误允许一次模型自行纠正：返回具体字段错误、现有索引和契约，校验通过前不展开新证据。整轮最多一次纠正，不重置或扩大原有取证读取上限；纠正回复必须仍为 inspect_evidence，不能直接返回科学动作。若仍失败，保存具体原因并停止。用量累计包含纠正，诊断不保存原始回复或字段输入值。

## 环境配置核对

只读合并当前 FeMn config_session 与 agent_runtime 后确认 config-0018-ff374eacab09b088、python_environments.remote_mlip=mace，DFT 输入准备环境准入检查返回无错误。旧快照里未配置该字段并不等于当前运行配置仍缺失；无需修改快照或重新猜测超算环境。这仅验证配置能被正确读取，未启动远端环境或计算。

## 用户接下来的操作

1. 用户手动重启 FeMn 本地服务，加载本次源码修复。程序不会自动重启服务。
2. 在 FeMn 对话中发送：“基于当前生效配置重新生成 DFT 选点方案，展示数量、结构清单和逐点理由；先不准备输入、不提交计算。”
3. 查看新方案是否绑定当前配置与模型、候选来源是否有效，再确认输入准备。100 是每轮数量上限，不要求选满 100 点。
4. 仍报错时，查看本轮过程中的“取证请求校验未通过”：应显示具体字段、错误类型和纠正机会，而不是只有通用错误。本次修复没有追认旧方案、替换候选、准备输入或提交超算。

## 验证边界

验证使用假模型及合成数据，包含实际 hybrid 包装器、完整证据加载、越权拒绝、纠正次数、用量与诊断隐私、DFT 环境准入和配置修订。没有调用真实模型重新选点，因此不将源码修复宣称为新选点已成功。

验证结果：190 related regression tests passed; ruff check passed; ruff format check passed (738 files).

源码备份：`E:\jupyter notebook\1-AL_for_PhaseDiagram\_Process_PhaseDiagram_source_backups\femn_evidence_contract_20261022`。


## 追加：DFT 取证目录修复

再次检查时，最新保存的重新生成失败记录仍为 2026-10-22 11:11，早于此前 14:15 的取证修复；没有保存的新一轮失败记录，不能把旧报错视为新代码失败。

在当前重新选点代码中另外复现：`initial_tool_proposal` 生成 `dft_candidate_costs`、`dft_stage_limits`、`independent_audit_candidate_ids`，但 `summary_request.TOPICS` 未登记这些字段，导致它们默认属于 history。模型以 dft 主题请求这些科学决策依据会被拒绝为 topic mismatch，消耗纠正机会并可能中断方案准备。新增 4 项回归在修改前全部失败，修改后三字段按 dft 主题可索引并完整读取；实际 hybrid 包装器下无需纠正或切换模型，保留原始成本精度和当前配置，旧方案及原 payload 不变。

改动仅为三项明确字段的主题登记；没有放宽未知字段/主题校验，没有替用户选点、修改配置、追认旧审批、生成输入或提交作业。74 项选点/取证/配置回归通过，ruff check 和 format check（739 文件）通过。真实模型的重新选点结果尚未执行验证。

实际源码已更新。由用户手动重启 FeMn 本地服务后，发送：“按当前生效配置重新生成 DFT 选点方案，展示数量、结构清单与逐点理由；先不准备输入、不提交计算。”旧配置的待审方案不能直接批准。每轮 100 是数量上限，不要求选满。

追加备份：`_Process_PhaseDiagram_source_backups/femn_dft_topic_20261022`。

补充验证：摘要分层取证、hybrid、调用次数、方案契约和 DFT 环境准入组合 105 passed（19.11s）。
