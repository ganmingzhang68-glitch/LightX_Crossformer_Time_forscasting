# ALL90正式模型：218输入 → 原128输出

本目录发布原ALL90实验已完成的12套最终模型，G1–G4各三个预注册种子。不是随机初始化权重，也不是旧128输入Epoch45模型。

|组别|历史 / 空档 / 输出|固定重训轮数|种子|
|---|---|---:|---|
|G1_300_10_120|300s / 10s / 120s|22|20260928、20260929、20260930|
|G2_180_10_60|180s / 10s / 60s|44|20260928、20260929、20260930|
|G3_180_10_120|180s / 10s / 120s|40|20260928、20260929、20260930|
|G4_300_10_60|300s / 10s / 60s|40|20260928、20260929、20260930|

输入是原128加全部新增90，共218个唯一业务字段；主干读取130动态历史字段，条件分支读取全部90字段，两分支有2字段重叠。输出固定原128（56连续、72布尔），顺序见每个模型的 `feature_manifest.json`。

10秒空档表示跳过5个2秒采样点。首个标签为最后输入后+12秒；G1/G3最后标签为+130秒，G2/G4为+70秒。

## 已发布文件

每个 `G*/seed_*/` 都包含：

- `best.pt`：原训练最终checkpoint，保留原始字节和SHA-256。
- `preprocessor.json`：最终重训scaler、训练类别词表、常量/活跃字段与拟合范围。
- `feature_manifest.json`：源字段、输出字段、动态/条件分支和类型的有序清单。
- `model_config.json`：窗口、种子、固定epoch和模型结构参数。
- `spec.json`、`result.json`、`history.json`：原训练签名、结果和逐轮记录。
- `bundle.json`：完整模型契约、预处理参数及逐文件SHA-256。
- `runtime/`、`predict.py`、`forecast_windows.py`：与原权重逐值验证一致的冻结推理实现。

顶层 `MODEL_INDEX.json` 索引12套模型的权重校验值和验证结果；`contract.json` 是完整218字段冻结契约；`final.json`、`frozen.json` 保留原实验设计与开发设备选轮次记录；`training_source/` 是经核验与checkpoint匹配的原训练源码。

权重来源代码提交为 `662dc1b854ffe3911d41e7d4a5d043260e996261`，原代码身份为 `cdc770138f6d5ee32989c80257dc238d52ebc812cfdfff9838efa58dd94de540`。代码、checkpoint、预处理器和字段契约已逐一核对；12套模型均通过严格加载和CPU前向，冻结推理实现与原模型输出最大绝对差为0。原始权重总大小245,402,652字节，约234MiB。

## 使用一个模型

完整克隆该分支后，在仓库根目录执行：

```bash
python -m pip install -r models/all90_20261110/requirements.txt
python models/all90_20261110/G4_300_10_60/seed_20260928/predict.py --input /data/aligned_numeric.parquet --output /results/predictions.parquet --mode rolling --include-target --device cpu
```

也可只下载一个模型（建议输出到仓库以外的新目录）：

```bash
python models/all90_20261110/download_model.py --group G4_300_10_60 --seed 20260928 --output /exports/all90_g4_seed20260928
python /exports/all90_g4_seed20260928/predict.py --input /data/aligned_numeric.parquet --output /results/latest.parquet --mode latest --device cpu
```

下载器会验证全部文件。私有仓库可通过环境变量 `GITHUB_TOKEN` 提供已有读取权限，程序不记录token。复制整个模型子目录后可离开训练仓库独立推理；依赖只有NumPy、Pandas、PyArrow和PyTorch。

## 输入和使用限制

- 输入为经过2秒对齐及历史时点类型解码的宽表CSV/Parquet，不是原始Influx长表。
- 必须有UTC时间和218源字段；原128必须有限，新增字段允许缺失，由90个掩码表示。
- 必须有 `observed_mask`、`new_observed`、`new_age_seconds`，质量通道不能伪造。建议保留 `device_id`、`job_id`、`segment_id`，窗口按身份与时间连续性隔离。
- `设铺粉模式`、`负压测试结果`、`正压测试结果`使用已冻结训练词表；未知类别有明确编码。日期字段需已按源快照时刻转换为 `log1p(距检测天数)`，未来/无效检测日期及对应结果置空。
- 不得在04或其他外部设备重新拟合scaler、词表或选择最佳种子。12套全部发布，不根据04只挑某一个权重。
- `best.pt` 是开发设备固定轮数重训后保存的最终权重，不代表使用设备04进行早停选择。
- 本次发布未重新训练模型。设备04历史评估是探索性业务验证；CPU打包验证不等于新的训练成绩或目标硬件延迟保证。
- 原分支根目录 `predict.py` 和 `artifacts/checkpoint.pth` 属于旧Epoch45发布包；ALL90请使用本目录的模型入口。

模型目录的 `.gitattributes` 禁止Git自动转换换行，以保持逐文件校验值。请不要用编辑器重新格式化已发布的模型包JSON或Python文件。
