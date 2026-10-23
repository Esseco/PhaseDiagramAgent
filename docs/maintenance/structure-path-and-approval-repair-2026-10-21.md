# 结构路径与审批展示修复

Co / Cr / FeMn 路径与审批展示修复核验（2026-10-21）

结果：已部署生产源码并修正三个项目当前可验证的迁移路径。没有操作真实服务，没有调用真实模型、准备或提交科学计算，也没有删除结构。

路径根因：已有明确 verified → epochN/verified_results 迁移映射，但旧路径含 Windows 扩展前缀 \\?\，原映射函数未识别。此次补齐扩展盘符路径识别。
写入范围：每个项目的 state.json、phase_data.json、branch_energy_pools.json 和当前 epoch1 phase_diagram.csv。仅更改已存在且通过 SHA256 校验的精确目标引用。
科学数值、模型/结构标识、任务状态、签名方案与历史审计未变。程序维护修订号正常推进，已有路径迁移控制元数据保留。

真实项目复核：
- O3-1ele-NaCoO2：修正当前相图引用 214 条；当前 538 条结构引用全部存在并匹配 SHA256；运行配置诊断 ready。
  选点证据：范围内 84 条 D/H 绑定可用；范围外 442 条未强行赋值；此前排除的 12 条仍被过滤。
  新诊断缓存：E:\0-PhaseDiagram\O3-1ele-NaCoO2\analysis_outputs\epoch1\phase_diagrams\mlip\quality\phase_quality_02fb992c8573fb114558.json
  备份与审计入口：E:\0-PhaseDiagram\O3-1ele-NaCoO2\history_backups\structure_path_repair_20261021_185647_477462
  原始 JSON 事务备份：E:\0-PhaseDiagram\O3-1ele-NaCoO2\workflow_state\backups\result-recovery-49cf67a03c2041099df1cf06b52a8b68
- O3-1ele-NaCrO2：修正当前相图引用 16 条；当前 770 条结构引用全部存在并匹配 SHA256；运行配置诊断 ready。
  选点证据：范围内 96 条 D/H 绑定可用；范围外 674 条未强行赋值；此前排除的 0 条仍被过滤。
  新诊断缓存：E:\0-PhaseDiagram\O3-1ele-NaCrO2\analysis_outputs\epoch1\phase_diagrams\mlip\quality\phase_quality_dd29a53f71c1e67afc8c.json
  备份与审计入口：E:\0-PhaseDiagram\O3-1ele-NaCrO2\history_backups\structure_path_repair_20261021_185659_642610
  原始 JSON 事务备份：E:\0-PhaseDiagram\O3-1ele-NaCrO2\workflow_state\backups\result-recovery-a843c6e3fa064057ae034e378a0014db
- O3-2ele-NaFeMn：修正当前相图引用 244 条；当前 526 条结构引用全部存在并匹配 SHA256；运行配置诊断 ready。
  选点证据：范围内 72 条 D/H 绑定可用；范围外 454 条未强行赋值；此前排除的 0 条仍被过滤。
  新诊断缓存：E:\0-PhaseDiagram\O3-2ele-NaFeMn\analysis_outputs\epoch1\phase_diagrams\mlip\quality\phase_quality_ca88abb5f0fc67368499.json
  备份与审计入口：E:\0-PhaseDiagram\O3-2ele-NaFeMn\history_backups\structure_path_repair_20261021_185712_058541
  原始 JSON 事务备份：E:\0-PhaseDiagram\O3-2ele-NaFeMn\workflow_state\backups\result-recovery-60f4c42057eb4469932961d3c2494dde
合计修正当前相图结构引用 474 条，验证 1834 条真实结构文件；各备份哈希和原 CSV 压缩包均核验通过。

范围说明：诊断沿用当前近凸包及递归暴露范围。分组内未单独触发骨架检查的成员保持 not_triggered；已触发检查的最终文件均可读且哈希匹配。刷新仅生成派生诊断 JSON，未执行诊断建议中的移除。
历史审批、已确认配置、历史清单和已归档报告保持原样。原任务输入、checkpoint 等缺少实际文件的旧来源引用保留原始证据，不猜测目标；它们不属于本次当前相图结构路径修复。

审批展示：已保存模型激活方案同时展示同意/拒绝提示及本机审批页；尊重 PHASE_CONTROL_PORT（默认8765）。其他敏感审查仍走审批页，普通审查仍按原方式；查看不批准、不执行。另修正已保存微调方案的过期展示文字为生成超算微调训练输入。
审批身份、方案哈希和执行权限逻辑保持原样。没有新增关键词意图规则。

验证：py1 下 27 个相关测试模块，268 passed；ruff check phase_agent 通过，ruff format --check phase_agent 733 files already formatted。覆盖路径迁移、恢复事务、审批边界、相图/D/H、选点、传输和假进程启动流程。
验证使用假模型/假进程；测试临时目录均清理。本次仅保留用户要求的正式维护备份及小型脚本、日志、核验回执。
实际服务未重启，因此当前进程是否加载新源码未经在线验证；新代码由用户下次手动启动加载。三个项目的数据记录已落盘。

源码备份：
E:\jupyter notebook\1-AL_for_PhaseDiagram\_Process_PhaseDiagram_source_backups\extended_path_repair_20261021_184806_549592
E:\jupyter notebook\1-AL_for_PhaseDiagram\_Process_PhaseDiagram_source_backups\approval_display_repair_20261021
E:\jupyter notebook\1-AL_for_PhaseDiagram\_Process_PhaseDiagram_source_backups\approval_display_wording_repair_20261021

机器核验与源码差异：C:\Users\Esseco\Documents\ChatGPT\AL_for_FM\outputs\path_and_approval_repair_20261021
