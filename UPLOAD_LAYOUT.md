# 上传和收回目录

新生成的文件按 MLIP 版本、branch 生成来源（Search-group）、组内计算轮次组织。例如：

```text
upload_batches/
  MLIP-round-0001_mace-mh-1/
    Search-group-0001/
      Relax-0001/
        Relax-submission-0001_remote-000001/
        results/
      Relax-0001_MC-round-0001/
        MC-sampling-0001_remote-000006/
        MC-sampling-0002_remote-000007/
        results/
      Relax-0001_MC-round-0002/
        MC-sampling-0013_remote-000018/
        results/
      DFT-round-0001_<标识>/
        DFT-single-point/
          DFT-single-point-submission-.../
          results/
        DFT-relax/
          DFT-relax-submission-.../
          results/
    Search-group-0002/
      Relax-0001/
      Relax-0001_MC-round-0001/
```

Search-group 根据保存的 branch 生成记录确定。第一组 Relax 与其两轮 MC
放在同一 Search-group，下一组 branch 使用新组编号。
MC 文件夹前缀 Relax-0001 明确表示父 Relax；MC-round 表示该父 Relax 的 MC 段序号。
批次还保存 parent_relax_round 和 parent_relax_directory；不同父 Relax 不能混批。
标识来自批准方案，重启和继续准备时复用。
MC 每个 GPU 批次最多 10 个算例；同一次分配的全部 GPU 批次共享一个 results。
不同分配不混批，也不共享 results。上传时保留整个 allocation 文件夹，
每个提交批次根目录的 GPU.sh 提交一次；结束后下载该 allocation 的 results。
组内使用 DFT-round-0001_<标识> 等明确轮次目录，下设 DFT-single-point 和 DFT-relax。
同一次 DFT 选择包含单点和弛豫时，两种阶段使用相同轮次编号；下一次选择递增。
每个阶段的该轮有独立 results，但仍在单任务目录提交 GPU.sh。

第二段 MC 使用上一段回传的最终结构文件。账本 structure_id 用于定位 branch
和满 Na 模板，不能替代实际的 MC 输入文件。现有已分配任务缺编号时，准备器
从所属 branch 补全账本引用并验证结构文件和 branch 一致性。

重生成只允许定位最新已批准分配的目录，保留早期段的任务与结果。
旧式共享目录含有其他分配时，需要先整理目录，自动删除会被阻止。

迁移已有第一轮可使用 execution_layer.local.migrate_allocation_layout。
默认只预检；--apply 才移动目录并修正状态、能量池、相识别缓存和相图文件的路径。
--search-groups 将已归档的单组旧任务合并进同一个 Search-group-0001。
输入 task.json、manifest、result.json 和完成标记保持原始字节，科学数值及版本不变。
备份保存到项目 layout_backups/<时间>/，migration.json 记录全部移动和文档更新。
