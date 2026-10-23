# Co / Cr恢复修复与DFT选点证据模块验证

生产源码：`E:/jupyter notebook/1-AL_for_PhaseDiagram/Process_PhaseDiagram`。

- Co/Cr报错根因：目录迁移改变本地名称，而恢复归档保留原名称。现在仅从已验证的精确迁移映射恢复原标签，同时保留原归档校验和、完整任务范围、结果缓存键和返回归档绑定；状态诊断包含实际状态渲染。
- DFT候选补充当前模型/凸包绑定的审查结构和必要的D上方参考；精确几何哈希绑定QBC，缺失不填零，不挪用旧模型。FeMn有效池从51补到105，此数字是候选池大小，未选为正式DFT任务。
- agent同时取得固定D、带前提的递归H、正式Ehull、各自单位的QBC和覆盖证据。独立且相关的高D、高H疑点多时，应按验证需要扩大数量及相/Na/结构覆盖；不按100凑数，也不按重复结构数同比放大。最终数量与结构选择由项目agent决定，无新自动打分排名/删除规则。
- 每轮单点与DFT优化合计最多100，包含已登记本轮任务。预览与执行前均校验。取消旧本轮DFT费用硬上限，估价和显式通用hard预算机制保留；这三个项目沿用advisory预算。
- 正常提案与人工修订取得相同参考，修订也获得已登记独立抽检候选。校验失败最多再交模型修订一次，再失败保留旧pending，不由代码补选，不执行handler。
- 配置编辑器明确支持缺失字段写入None，三个项目以新配置版本确认本次策略，未追认此前editable配置中的其他变化。

## 实际验证

最新组合回归：**172 passed**，Ruff检查通过，**736**个源码文件格式检查通过。使用最小合成数据/假agent及TemporaryDirectory；详见final_checks.json。另有数量上限专项35 passed（1条旧展示fixture因已有简短回复契约不一致排除，源断言原样保留，不作为通过项）。未声称全仓库全测试通过。

三个真实项目仅只读核验，状态回复与runtime doctor均ready；Co发现35条旧复核结果、验证7个返回归档绑定；Cr分别15与6；FeMn无此类旧恢复目录。归档哈希不变，详见final_real_projects.json。

- O3-1ele-NaCoO2: `config-0016-0f6c6825bf9983cc`；候选池109；正式任务、pending方案、模型和凸包均与本次变更前一致。
- O3-1ele-NaCrO2: `config-0017-aeec85e4ecefad16`；候选池122；正式任务、pending方案、模型和凸包均与本次变更前一致。
- O3-2ele-NaFeMn: `config-0018-ff374eacab09b088`；候选池105；正式任务、pending方案、模型和凸包均与本次变更前一致。

## 真实模型边界与后续操作

用户明确授权后曾调用FeMn已配置的deepseek-flash（https://api.deepseek.com，hybrid），原生校验以independent_dft_audit_required拒绝该次方案。**没有发布新方案或替换旧方案，没有创建科学任务。**据此修复了修订上下文中抽检约束不可见及一次模型自修的流程，并通过假agent验证。按用户随后要求，未再次调用真实模型；最终新版提示下的实际模型选点质量尚未重新实测。

后续通过项目原入口让agent依据当前D/H/Ehull/QBC重新规划或修改DFT方案，再由human审批。旧proposal的配置绑定已过期，不能直接把原方案当作新策略批准；原生修改路径允许重拟并生成新hash。无需运行本次维护脚本。

全程未启动/停止/重启服务，未准备或提交DFT/MLIP/MC任务，未删除结构或修改科学数据。当前新加载源码的本地入口已验证；既有长驻进程是否已热加载未通过重启验证。

## 备份与收据

- O3-1ele-NaCoO2: `E:\0-PhaseDiagram\O3-1ele-NaCoO2\history_backups\dft_selection_policy_20261021_201114`。
- O3-1ele-NaCrO2: `E:\0-PhaseDiagram\O3-1ele-NaCrO2\history_backups\dft_selection_policy_20261021_201154`。
- O3-2ele-NaFeMn: `E:\0-PhaseDiagram\O3-2ele-NaFeMn\history_backups\dft_selection_policy_20261021_201224`。

源文件备份路径与前后哈希见source_deployment.json、dft_evidence_source_deployment.json、agent_policy_source_deployment.json、dft_guidance_deployment.json以及dft_task_cap、dft_config_policy、dft_revision_repair子目录。final_source_manifest.json记录最终源码哈希。
