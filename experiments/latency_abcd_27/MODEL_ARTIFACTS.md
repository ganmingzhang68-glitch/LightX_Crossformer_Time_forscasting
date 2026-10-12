# 模型产物

C/D 的探索性模型权重不提交到普通 Git 历史。代码、scaler、字段清单、调参结果、确认结果、训练历史和设备 04 指标已随本分支保存。

| 实验 | 文件 | 字节 | SHA-256 |
| --- | --- | ---: | --- |
| C | `model.pth` | 11,873,863 | `9e008d634f7c77ece1be2873160bdd2fe535edaf5ca0af031deed27f3679df24` |
| D | `model.pth` | 11,874,247 | `11d89030b726b241f1d74e0e9166378230ee773d93713f37172ef9b52dad0366` |

对应 scaler 已提交：

- C：`65ccca3f7d2b7f7582adde11cf948b25881cf5f3fbfeee2ea568838f54dd5244`
- D：`429472abda9454cc4697b5d0973b5524d0375af453fa2d11c74ac81a1fa593d2`

如果后续决定发布模型，应使用 Git LFS 或 GitHub Release，并把权重与本文件中的哈希、对应 `config.json`、`scaler.npz` 和 `data_feature_manifest.json` 作为一个不可拆分的版本化模型包。
