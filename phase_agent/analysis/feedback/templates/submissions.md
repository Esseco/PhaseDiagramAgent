# 提交与回传

## 先读这里

1. 先看本页 `README.md`：只决定当前要交付/回传的目录。
2. 再读 `submission_index.json`：读取精确路径、命令、状态和存在性；使用前重新核对文件。
3. 最后看 `epochN/README.md` 与 `layout_manifest.json`：确认模型版本和该轮每个直接子目录的职责。

| 层 | 入口 | 用途 |
| --- | --- | --- |
| 交付层 | `epochN/delivery/transfers/`、`epochN/delivery/relax_mc/` | 只处理登记的 `_input` 上传和 `_output` 回传；无后缀旧包由 manifest 标记为隔离历史 |
| 计算层 | `epochN/calculations/<operation>/inputs/`、`epochN/calculations/<operation>/results/` | 保留本地计算工作目录，不作为上传入口 |
| 证据层 | `epochN/evidence/verified_results/`、`epochN/evidence/recovery/`、`epochN/evidence/storage_archives/` | 保存已校验结果、回收凭据和完整归档 |
| 隔离层 | `previously_used/` | 历史/未登记材料，只读追溯，禁止提交或回收 |

本轮只在 `epochN/` 下保留 `calculations/`、`delivery/`、`evidence/` 三个物理层；Agent 先读 `layout_manifest.json` 后才进入登记计算目录。无后缀旧交付包即使因历史登记暂留在 `delivery/`，也只能按隔离历史读取，不能作为当前提交入口；其他旧目录和未登记材料放到 `../previously_used/<epoch>/`。

本地未回收不代表尚未提交或远端未运行；先核对原提交记录，勿重复提交。以下路径与命令仅供核对，不构成提交授权。回传放入对应目录后说“已回传”，无需手动解压。

## 当前待回传/核对

当前没有登记的本模型未完交付。
