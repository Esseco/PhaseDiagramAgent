# 2026-10-21 公共结构检查验收

实现：`E:/Py-Code/Process_AL_MC/structure_check.py`；接口和数据分析详见同目录 `STRUCTURE_CHECK.md`。

## 实施结果

- 公共 `check_structure` / `LayeredFrameworkReference`，passed/invalid/uncertain 三态；只依赖已有 numpy/pymatgen，不触发模型或写文件。
- 原相图骨架判据迁入公共模块；`phase_framework` 保留文件校验和、组成及序号绑定，不重复实现算法。`run_structure_check` 复用基础检查并修复完全重合原子漏检，避免完整距离矩阵。
- 原 MC 和独立 TM/Na+TM 排序搜索都在链/最优值/候选池/排名更新之前检查。明确无效候选排除后继续；已耗步骤、访问记录和预算保留，patience 和 checkpoint 正常工作。初始结构明确无效则结束单任务，不造一个可继续的状态。
- 骨架未知显式 `retain_with_evidence`，不伪装成通过。原始科学数据和文件不删除。
- 参考拓扑每次运行/每条独立链计算一次；短程邻居列表替代全距离矩阵；能量突变或可能入池才做骨架检查；每个候选三个截断共用一次邻居表，trace/pool/CSV 只保存摘要。
- 新准备远端包携带共享模块；worker 返回检查计数。旧包和历史结果不重写，也不声称已检查。

## 阈值与成本证据

只读扫描 E:/0-PhaseDiagram：1291 份 trace 文件、约 1.09 MB（压缩/原始文件总大小）；按解压内容 SHA256 去重。Co/Cr/FeMn 共 12572 个有限能量差步骤，绝对差 p99 为 0.038856/0.021297/0.040547 eV/atom。暂定 0.05 eV/atom，只用于触发核查，共超过 28 步（约0.223%）。没有坏结构人工标注和 Mn/Ni 轨迹，不能估计召回率或声称材料通用阈值。初始/入池检查不依赖能量触发，可覆盖缓慢变化。

py1，本机合成周期层状结构，15 次中位数：96 原子基础 0.56 ms / 含骨架 9.35 ms / 初始参考一次13.0 ms；216 原子分别1.19/20.0/24.6 ms。不是超算或真实 MLIP 性能测量。

## 验证

合并运行 `test_shared_structure_check.py`（13项）、Py-Code `test_ordering_search.py` 和应用现有 `test_phase_framework_review.py`、`test_phase_quality.py`、`test_phase_quality_review.py`、`test_exact_mc_occupations.py`、`test_mc_hb_policy.py`、`test_relax_mc_batch.py`、`test_mlip_relaxation_convergence.py`：**164 passed, 1 deselected**。仅使用合成结构、假模型、假超算。测试目录由 TemporaryDirectory 和项目 conftest 清理。

单独运行被排除的既有 `test_prepared_mc_payload_omits_inherited_threshold_and_is_not_submitted` 仍失败；在未修改的生产源码上也复现。打印原始原因是 `input_admission_failed:structure_format:.../full_na_structure.vasp`，属于既有测试输入格式验收，本轮不放宽科学检查或修改该断言。最初暂存打包测试缺少相邻模板的问题已补齐，并用短临时路径完成回归。

`ruff check` 公共库及新增测试通过；应用全包 `ruff check --config ruff.toml phase_agent` 通过；改动的应用文件、公共模块和新测试 `ruff format --check` 通过。API 索引使用实际 `tools/project_catalog.py` 在暂存目录静态生成，仅更新 Process_AL_MC 的 API 与总目录中该项目条目，未重写其他项目文档。

未运行真实 MLIP、DFT、HPC 或服务；未改项目配置、模型、台账、数据库、既有上传包或科学结果。部署前逐文件校验基线并备份，部署后核对字节/导入并运行已安装模块的针对性测试。备份与部署清单路径附于后续记录。
