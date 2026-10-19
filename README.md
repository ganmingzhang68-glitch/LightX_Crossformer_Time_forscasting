# LightX Crossformer 时间序列预测

这是用于 LightX SLM 3D 打印机 PLC 传感器的 Crossformer 部署项目。仓库包含最佳模型权重、固定特征契约、训练集归一化参数、原始 InfluxDB Parquet 预处理脚本和推理脚本，可以在其他同构设备上执行与训练阶段一致的预测任务。

仓库现在有两套用途：

- 根目录的 `artifacts/`、`prepare_influx.py` 和 `predict.py` 用于已经发布的固定128输入、128输出模型推理。
- [`experiments/input_window_matrix_v1/`](experiments/input_window_matrix_v1/) 用于重新训练不同输入特征组合，并在G1–G4四种时间窗口下统一预测原128特征。自定义特征清单功能位于 `feature/custom-feature-manifest` 分支。

两套流程的模型、特征契约和预处理器不能混用。实验训练产生的新checkpoint不会覆盖根目录发布模型。

## 预测任务

- 输入：过去 300 秒数据，即 150 个点 × 128 个特征。
- 采样间隔：2 秒。
- 输出：未来 10 秒，即 5 个点 × 128 个特征。
- 滚动预测默认步长：5 个点，即每 10 秒移动一次窗口。
- 连续特征输出为原始物理量。
- 布尔特征在反归一化后使用 `0.5` 阈值，最终输出严格为 `0/1`。

时间关系如下：

```text
过去 300 秒真实观测                     未来 10 秒模型输出
[t-298s, ..., t-2s, t]  ────────────>  [t+2s, t+4s, t+6s, t+8s, t+10s]
```

模型不会用前 300 秒“预测自身”。前 300 秒只是模型输入，预测结果从输入窗口结束后的第 2 秒开始。

## 模型与训练信息

- 模型：Crossformer + RevIN + 最后值残差预测。
- 最佳 Epoch：45。
- 训练设备：`01-X450-1`、`02-X450-2`、`03-X450-3`。
- `04-X450-4` 未参与梯度更新或最佳模型选择。
- 最佳分组验证损失：`0.0120321285`。
- 连续特征相对最后值基线的 MSE skill：`66.73%`。
- 布尔特征宏平均 F1：`0.80065`，最后值基线为 `0.72977`。

完整指标见 [`artifacts/final_metrics.json`](artifacts/final_metrics.json)，训练历史见 [`artifacts/training_history.json`](artifacts/training_history.json)。

全部训练特征、固定顺序、类型划分和预处理规则见 [`FEATURES.md`](FEATURES.md)。

## 外部设备适用条件

可以部署到其他设备，但需要满足以下条件：

1. 数据属于同一种 `PLC-Sensors` 测量，传感器字段语义和单位一致。
2. 必须包含 [`artifacts/feature_manifest.json`](artifacts/feature_manifest.json) 中按固定顺序定义的全部 128 个特征。
3. 原始数据应以 2 秒为主要采样周期。脚本只在线性/前向方式补一个缺失点，长缺口会切成新的独立片段。
4. 新设备不得重新拟合归一化器。推理始终使用训练设备 01/02/03 的 [`artifacts/scaler.json`](artifacts/scaler.json)。
5. 额外字段会被忽略；缺失训练字段会直接报错，不会静默填零。

模型在同构设备上可以直接运行，但新设备的传感器校准、控制逻辑或工作区间如果明显不同，精度可能下降。正式上线前建议保留一段带真实未来值的数据，使用 `rolling` 模式先评估模型相对最后值基线的提升。

## 仓库结构

```text
.
├── artifacts/
│   ├── checkpoint.pth          # Epoch 45 最佳权重
│   ├── feature_manifest.json   # 128 特征顺序及布尔/连续分组
│   ├── scaler.json             # 仅在训练拆分上拟合的 StandardScaler
│   ├── model_config.json       # 固定模型结构和预测配置
│   ├── final_metrics.json
│   └── training_history.json
├── config/
│   └── plc_feature_quality_rules.json
├── FEATURES.md                    # 128个训练特征的完整说明
├── crossformer/
│   └── model.py                # 精简、可独立加载权重的 Crossformer
├── experiments/
│   └── input_window_matrix_v1/ # 多输入方案、自定义特征、G1–G4训练及04评估
├── prepare_influx.py           # Influx 长表 Parquet → 128 特征宽表
├── predict.py                  # 最新窗口或滚动预测
└── validate_bundle.py          # checkpoint 严格加载及前向冒烟测试
```

原始训练数据、InfluxDB token 和设备访问凭据不包含在仓库中。

## 安装

建议 Python `3.10` 或 `3.11`。

```bash
git clone https://github.com/ganmingzhang68-glitch/LightX_Crossformer_Time_forscasting.git
cd LightX_Crossformer_Time_forscasting

python -m venv .venv
source .venv/bin/activate                 # Linux
# .venv\Scripts\Activate.ps1             # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python validate_bundle.py
```

`validate_bundle.py` 正常输出应包含：

```json
{
  "strict_checkpoint_load": true,
  "input_shape": [2, 150, 128],
  "output_shape": [2, 5, 128]
}
```

CPU 可以推理；如果安装的 PyTorch 支持 CUDA，默认会自动使用 GPU。

## 训练新的输入特征模型

训练代码不在默认 `main` 分支。使用自定义特征清单时应明确切换分支：

```bash
git clone https://github.com/ganmingzhang68-glitch/LightX_Crossformer_Time_forscasting.git
cd LightX_Crossformer_Time_forscasting
git switch feature/custom-feature-manifest

cd experiments/input_window_matrix_v1
python -m pip install -r requirements.txt
python -m pytest tests -q
```

训练任务固定遵循以下约束：

- 输入存储契约为原128字段加新增90字段，共218个有序源字段；每个模型只选择其中一部分。
- 自定义模型至少包含原128字段，因为模型使用这128个字段的历史值完成末值残差预测。
- 输出始终为原128字段，其中56个连续量、72个布尔量。
- 每个输入方案都覆盖G1–G4四种历史长度和预测长度。
- 设备01/02/03用于选轮次和最终训练；设备04只用于全部模型完成后的外部评估。
- `plan` 和 `init` 不训练；只有显式执行 `train_queue.py run` 才会更新模型权重，并且需要CUDA。

训练数据不会随Git仓库发布。训练前必须准备旧128数据集和新增90字段宽表，再由实验目录的 `prepare.py` 融合成218字段、按打印任务分片的数据集。完整的数据格式、特征类型处理、自定义清单格式、训练与恢复命令见：

[`experiments/input_window_matrix_v1/README.md`](experiments/input_window_matrix_v1/README.md)

## 输入数据

### 方式一：InfluxDB 下载的长表 Parquet

输入文件需要包含：

```text
time, measurement, field, value_type, value_bool, value_float
```

每个文件应只包含待预测的一台设备。执行：

```bash
python prepare_influx.py \
  --input data/external_device.parquet \
  --output outputs/external_device_prepared.parquet
```

预处理会：

- 只选择 `measurement = PLC-Sensors`；
- 按固定的 128 特征契约转为宽表；
- 检查必需字段是否完整；
- 按 2 秒时间网格整理数据；
- 连续特征最多线性插值一个缺失点；
- 布尔特征最多前向填充一个缺失点；
- 对无法安全填补的缺口切分片段；
- 不使用训练百分位裁剪外部设备的真实物理量。

### 方式二：已经整理好的宽表

可以直接使用 CSV 或 Parquet，但必须包含：

- `time_beijing`、`time_utc` 或 `time` 中的一个时间列；
- 特征清单中的全部 128 列；
- 每个连续片段内严格 2 秒一个点；
- 不含 NaN 或无穷值。

## 预测最新 10 秒

使用最后 150 个连续点预测其后的 5 个点：

```bash
python predict.py \
  --input outputs/external_device_prepared.parquet \
  --output outputs/latest_prediction.csv \
  --mode latest
```

输出包含：

- `forecast_origin_time`：输入窗口的最后时刻；
- `prediction_time`：预测点对应时刻；
- `horizon_step`：1 至 5；
- 128 个反归一化后的物理量预测值。

## 从头到尾滚动预测

每 10 秒移动一次 300 秒窗口，并预测下一个 10 秒：

```bash
python predict.py \
  --input outputs/external_device_prepared.parquet \
  --output outputs/rolling_predictions.parquet \
  --mode rolling \
  --stride 5 \
  --batch-size 64 \
  --include-target
```

`--include-target` 会为已存在于输入文件中的未来点增加 `actual__特征名` 列，便于计算外部设备指标。部署到实时流时使用 `latest` 模式，不需要真实未来值。

当预处理结果含有 `observed_mask` 时，`rolling` 模式会排除未来5个目标点中包含插值行的窗口，与训练阶段的验证口径保持一致；输入窗口本身仍允许包含预处理规则许可的单点插值。

## 固定模型参数

| 参数 | 数值 | 含义 |
|---|---:|---|
| `seq_len` | 150 | 过去 300 秒 |
| `pred_len` | 5 | 未来 10 秒 |
| `enc_in` | 128 | PLC 特征数 |
| `d_model` | 128 | 隐状态维度 |
| `n_heads` | 2 | 注意力头数 |
| `e_layers` | 3 | Crossformer 编码尺度数 |
| `d_ff` | 512 | 前馈网络维度 |
| `seg_len` | 10 | 每段 10 个采样点 |
| `win_size` | 2 | 跨尺度合并窗口 |
| `dropout` | 0.0 | 推理时关闭随机丢弃 |

完整配置见 [`artifacts/model_config.json`](artifacts/model_config.json)。

## 模型文件校验

发布的最佳 checkpoint SHA-256：

```text
8365b802d2fef8bc8d5c54f7398ff74459cc1b987664ddf4db0ef0e8351aee11
```

模型基于 THUML Time-Series-Library 中的 Crossformer 实现修改，沿用其 MIT License。
