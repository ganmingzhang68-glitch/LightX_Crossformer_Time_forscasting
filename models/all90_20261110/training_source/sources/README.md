# 字段来源的可移植副本

`added_fields.json` 从原工作区 `docs/PLC_NEW90_TO_128_EXPERIMENT_PLAN.html`
的 `queries.fields.rows` 提取全部 90 条记录，仅保留 `field`、`tier`、`caution`。
它不包含原始观测值或逐设备数据画像。原 HTML 和旧 128 字段 manifest 的原始
SHA-256 保存在 `provenance` 中，路径是原工作区的历史来源标识。

`feature_manifest_content_sha256` 是对完整旧 manifest 解析后，使用
`common.digest` 计算的内容摘要，避免 Git 在 Windows/Linux 间转换换行符造成误判。

独立克隆后，`build_contract.py` 使用本清单和仓库的
`artifacts/feature_manifest.json` 重建完全相同的契约。若 `--root` 指向原工作区且
原 HTML 存在，还会核验 HTML 文件哈希和全部提取记录的一致性。

原始 Parquet 数据需另外放入授权的数据目录。`vendor/SOURCE.json` 中的路径同样
是历史来源记录；对应实现已经随 `vendor/` 和 `amp_step.py` 提交，无需原工作区代码。
