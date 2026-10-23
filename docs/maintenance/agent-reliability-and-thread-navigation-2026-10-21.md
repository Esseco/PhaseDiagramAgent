# Agent可靠性与thread会话连续性修复

日期：2026-10-21（Asia/Shanghai）。本次只修程序与离线验证，未调用真实模型、浏览器或项目服务，未准备/提交计算。

## 已确认问题与处理

1. **模型修订上下文丢失**：DeepAgents材料投影遗漏human_comment/current_proposal，导致修订时看不到具体人工意见和原动作。现在保留二者完整语义，并用既有transport剥离已过期preview、用量和重复上下文；原动作参数不经过会丢字段的科学指标白名单。
2. **常见输出错误直接退出**：错误枚举、参数类型、缺失/空分析字段、显式旧generation_plan冗余计数冲突等会在返回前退出。现在只在新提案的公共契约边界把精确错误和原动作交模型修正一次，再过同样严格校验；程序不猜候选、不填科学分析、不替模型改配额。region_adaptive保持原审查。已保存的review仍只做校验，不因本次改动重新调用模型；网络错误也不经此逻辑重试。累计API用量保留。
3. **错误记录不够具体**：failure_summary原只有类型和阶段，难定位代码；loaded-code guard又在记录作用域外直接使Studio run失败。现在故障MD包含脱敏原因、失败阶段、异常类型和代码位置（文件/函数/行号，不取frame局部变量或模型隐藏思维）；已知源码不一致返回“本轮未执行”和实际报告地址，仍阻止任何handler/模型/科学任务调用。监控摘要保持小字段契约；未知异常继续抛出并留档，不伪装成功。
4. **会话选择入口缺失**：本地会话页原来只有暂停/删除，启动器链接只到Studio根页，未保存所选thread_id。现在项目面板有“选择／继续对话”，启动输出也指向本机/phase/conversations。页面可打开旧thread、继续上次、分页查更早记录；打开前核验ID仍在，过滤状态页与内部子图。按当前项目的baseUrl+phase_chat隔离浏览器选择，不取聊天正文，也不自动新建thread。显式点击“新对话”才进入Studio根页。

## 日志证据与边界

各项目最近6份Studio日志共18份，按run_id去重为5个失败run：3个是运行中源码发生变化的保护拦截，2个是上次已修复的旧恢复路径范围校验；最新均止于2026-10-21 19:16，没有更新后的复发日志可供判断。78条Windows连接关闭回调异常记录不能当作78次业务失败，本次不靠屏蔽日志假装修复。新增模型契约缺口由最小假client重现确认。

三个项目旧会话文件仍在workflow_state/studio/.langgraph_api；没有修改cwd、存储身份或移动/删除会话。只核对文件存在与大小，未反序列化pickle或宣称已验证每条历史完整性。官方Studio thread深链依据：[LangChain官方说明](https://www.langchain.com/blog/how-we-deployed-our-multi-agent-flow-to-langgraph-cloud-2)。

## 验证结果

- 30模块联合测试：**301 passed**；另region_phase契约：**11 passed**，合计312项针对性回归，无本轮排除项。
- thread页面含10个Node行为场景：上次会话、删除后ID、内部namespace、多标签、分页、打开时已删除、跨baseUrl、storage受限等；另用假浏览器验证实际面板打开入口。
- 全部phase_agent：Ruff check通过，737个源码文件format --check通过。
- Co/Cr/FeMn真实数据仅只读检查，状态回复与runtime doctor均ready；Co/Cr分别发现35/15条旧复核结果并验证7/6个返回归档绑定。
- 三项目正式state/config/ledger及审批引用等本次检查涉及的文件哈希未变；模型、相图、任务与已保存待审方案均未调整。测试TemporaryDirectory均已清理。

## 后续使用

下次手动启动项目后，从面板“选择／继续对话”进入，选旧thread继续；不要从无thread的旧Studio根书签判断旧历史消失。实际在线UI和新版真实模型响应本次未启动验证。

常见格式问题由项目模型在上述边界内修正一次；仍失败时查看回复中的“交互记录地址”及/phase/process。报告保存于项目interaction_records目录，异常MD包含本轮具体原因和代码位置。需要人工重启或核查执行回执时会明确说明，不由程序重启或重放科研任务。

## 备份与可追溯收据

- `E:\jupyter notebook\1-AL_for_PhaseDiagram\_Process_PhaseDiagram_source_backups\agent_runtime_failures_20261021_204006`。
- `E:\jupyter notebook\1-AL_for_PhaseDiagram\_Process_PhaseDiagram_source_backups\studio_thread_navigation_20261021`。
- `E:\jupyter notebook\1-AL_for_PhaseDiagram\_Process_PhaseDiagram_source_backups\agent_contract_reliability_20261021_203700`；契约修复哈希见contracts/source_before.json及source_after.json。

详见combined.json/.log、region_phase.json/.log、final_real_projects.json、formal_files_after_final_check.json、final_source_manifest.json及threads/verification.json。
