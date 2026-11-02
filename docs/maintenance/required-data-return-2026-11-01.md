# 必需数据回传：修复全部MC pool被扩展为依赖

问题发生在Relax+MC聚合封装，不是普通export_batch_result。analysis.candidate_pool把全部历史pool_summary复制为mc_pool；return_payload.result_files又逐项扩展所有pool几何文件。逻辑虽然名为依赖筛选，但把统计列表误当成本地消费者的必需输入。collect-only恢复导出另有整个jobs扫描，会绕开这一限制。

修复：

- candidate_pool只导出实际branch_candidates，兼容mc_pool为空列表。pool_summary统计仍保留在原签名result.json和任务元数据中，不改科学能量/原始字节。
- result_files只跟随最终结构、branch_candidates明确选择的结构，以及既有身份校验、DFT配对/训练、checkpoint、occupation和资源审计依赖；pool_summary及历史mc_pool不再隐式触发结构回传。
- 控制器失败、结果尚未反馈时仍从raw结果保留最终结构，避免只裁剪成功路径；被明确选中的pool结构缺失仍拒绝打包。
- collect-only复用同一result_files，不再整目录扫描导出；支持签名远端路径映射到恢复目录，保留原始字节。占据增量放到现有history_seed读取的result.json同级目录，冲突拒绝。恢复缓存版本升为collect-only-v3。
- prepare的冻结runtime来源指纹加入analysis.py，旧冻结封装不会被当成修复后的代码复用。继续使用原create_bundle/完整性校验/正式回收，不新增平行打包器。

保留的关键文件：result.json、task.finished.json、任务最终结构、明确选中的后续候选、原续跑checkpoint、mc_occupations.bin、训练/配对预测依赖、资源收据及包级状态/校验清单。未选pool、轨迹、日志、代码与模型留在服务器；不能仅因元数据提到一个文件就把它提升为本地依赖。

验证：修改前新增测试复现5个失败；修改后最终8个相关测试模块共89项通过（153.61秒）。覆盖未选pool大小独立、显式候选缺失拒绝、签名JSON不改、未收集最终结构、collect-only必需文件边界、占据历史、冻结代码身份、正常自动打包到正式回收与双图归档、依赖链、完成门槛及兼容交接。全应用ruff check通过；四个修改源码模块format --check通过。

只读NiCo包盘点：18222文件变为7595必需文件，剔除10627未选pool结构；保留2007项最终结构和403个已选候选所需依赖。原12634结构内容各异，不属于相同内容复制。此数值针对该历史包，是只读筛选结果，没有重新打包、删除或修改正式数据。压缩JSON中原mc_pool统计冗余也不再生成。

源码基线备份：同级_Process_PhaseDiagram_source_backups/required_return_20261101_224344。测试临时数据由项目conftest的TemporaryDirectory清理。修复未部署到历史远端冻结runtime，未启动或重启真实服务；后续新封装会携带该源码，旧封装重新准备时会被来源校验拦住。

NiCo本批已通过公开入口正式回收2007项，旧1893项能量未变；MLIP归档版本e70f7626747479998c1de35e、epoch0/latest与项目latest均校验通过。当前无同模型同帧DFT配对，DFT误差latest明确not_ready，无伪造误差图。remote已通过公开入口登记为/data/home/lichaoyue/26-10-PhaseDiagramAgent/O3-NiCo-1TMOrder并只读确认目录存在，旧NiMn地址不存在。
