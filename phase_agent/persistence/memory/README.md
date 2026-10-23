# 记忆模块

所有记忆存取、检索与审核实现集中在此目录。对话图负责组织本轮上下文，运行时负责调用；它们不另存一套记忆。科学状态、账本和审批回执仍是事实与执行权限的来源。

| 记忆类型 | 代码入口 | 保存和使用边界 |
| --- | --- | --- |
| 近期对话 | `conversation.py` 的 `remember_turn` / `recent_turns` | 项目状态目录下 `dialogue_memory/<线程哈希>.json`；最多读取 6 轮、10,000 字符；坏行隔离、原子写入 |
| 用户长期要求 | `decision_memory.py` | 明确人工填写后进入项目 state，保留版本；普通评论不自动固化 |
| 知识候选与审批 | `collect_memory_candidates.py`、`review_queue.py`、`knowledge_record.py` | 候选与已批准知识分开，证据与用户批准不可省略 |
| 相关知识检索 | `retrieve_relevant_knowledge.py` | 只读；按项目/体系/模型/适用条件过滤，以 ID 索引处理替代关系，限制数量及字符预算 |
| 可复用领域经验 | `build_domain_skill_draft.py`、`propose_domain_skill_import.py`、`publish_domain_skill.py` | 草稿、导入、发布分别受原审核流程约束 |
| 人可读视图 | `publish_memory_views.py`、`publish_user_preferences.py` | 展示已有记录，不成为新的审批来源 |

`phase_agent/text_safety.py` 提供共享文本脱敏；不依赖运行时、图或记忆状态。会话记录沿用原路径与格式，移动源码无需迁移已有记忆文件。近期对话可解释“刚才”“继续”等引用，不能证明任务已执行，也不能替代方案审批。

使用说明见 [上下文与记忆](../../../docs/guides/memory.md)；流程扩展见 [源码指南](../../../docs/architecture/development.md)。
