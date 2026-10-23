# Co DFT 当前候选池与抽查依据修复

最新记录 `无运行中任务_20261022_01.md`（15:45）证明 27 项 MC 已回收，待核验 0 项；新错误在后续 DFT 方案预览：unknown_candidate:S-dba16f90ccf7 与 independent_dft_audit_required。

只读真实核对：当前 qbc_candidates 178 项，原 dft_audit_candidate_ids 10 项全部不在池内。旧代码仅在完整 MC 后或 post-DFT assessment 分支更新抽查和推荐；部分回收后沿用当前候选池的首次提案及待审方案修订可携带旧抽查名单。模型据此引用了不存在于当前池的结构，预览正确拒绝。另有 dft_selection_policy、dft_validation_feedback 未列入 DFT 取证主题，修订时不能保证按 DFT 主题读全。

修复复用 recommend_post_mc_dft 和原成本估计器，将候选、成本、推荐、抽查集中到 dft_proposal_evidence.refresh_dft_evidence；首次提案和旧 DFT 方案修订均先同步当前池。仍有效的原抽查名单保留约束；空名单或越界名单按现有抽查算法重新生成，空池清理旧引用。同步顶层和 decision_context，避免一轮模型读到两套依据。取证目录登记抽查别名、DFT 策略和核验反馈。

不改变 LLM 最终选点权，不移除未知候选或独立抽查硬校验，不自动修正科学选择，不准备/提交计算，不修订配置或正式台账。真实只读预检得到 5 个当前池内抽查参考、178 行成本，state.json 哈希不变；这不是已生成新 DFT 方案。

验证：118 项合成回归通过，覆盖部分 MC 首次提案、当前/空池、保留有效抽查、修订一次后失败保留旧方案、抽查硬限制、配置修订及混合客户端取证。ruff check 全包通过，format check 742 文件通过。

用户手动重启 Co 本地服务后发送：按当前已回收结果重新评估下一步；若提出 DFT，使用当前候选池和有效独立抽查依据，展示新方案，先不准备输入、不提交计算。后续 MC 阶段尚未提交，仍由模型根据当前事实与审批规则比较路线，不将本次部分回收标记为整个搜索完成。

备份：_Process_PhaseDiagram_source_backups/co_dft_pool_binding_20261022。未控制服务。日志中的 WinError10054 与健康检查并存，并非这次 DFT 候选校验失败的直接证据。
