# 历史输入方案 × G1–G4：统一原128输出

2026-10-13 建立实验代码，2026-10-19 增加自定义特征清单。用户已明确要求：**所有新模型统一输出原128字段（56连续、72布尔），四个时间窗口均要训练。**

本目录提交的是训练、恢复和评估代码以及CPU/替身检查结果，不包含由该矩阵新训练出的checkpoint。旧 Epoch45、A/B/C/D、new90 和已完成 all90 的代码、权重、数据均保留；服务器上是否存在历史队列应以服务器实时状态为准。

## 已核实的历史输入

详见 [HISTORICAL_INPUTS.md](HISTORICAL_INPUTS.md)。`catalog.json` 保留历史输出数量用于追溯；**训练计划会将其全部转换为原128输出**。例如 `B133_27` 中的 `_27` 是历史名称，新模型实际输出128，以 `plan.json` 和 checkpoint 契约为准。

默认包含已核实有训练产物的23个历史输入/编码角色。统一输出后，A与BASE128完全相同，合并为 **22个方案**，而不是重复训练两份。N90_R0与BASE128虽然均有128个业务输入，但前者还看3个源质量通道，保留为独立编码对照。

F133_133只有历史代码、本次未定位到完成产物；其输入与B相同，统一128输出后也会合并。三个事件编码方案有历史代码但未找到已完成训练结果，默认不加入；可以用 `--profiles` 显式选用，代码已覆盖其四种窗口。

## 时间与规模

|时间组|历史/空档/输出（秒）|点数|标签相对最后输入时刻|
|---|---|---|---|
|G1_300_10_120|300 / 10 / 120|150 / 5 / 60|+12…+130s|
|G2_180_10_60|180 / 10 / 60|90 / 5 / 30|+12…+70s|
|G3_180_10_120|180 / 10 / 120|90 / 5 / 60|+12…+130s|
|G4_300_10_60|300 / 10 / 60|150 / 5 / 30|+12…+70s|

默认22方案 × 4窗口 × 3配对种子 = **264个最终模型**。另有264次内部选轮次训练，共 **528次训练任务**。默认种子20260928/20260929/20260930；可以在初始化前显式修改，初始化后不可改变预算。

名义300秒窗口包含150个2秒采样点，首末时间差298秒；180秒同理为178秒。10秒空档表示跳过5点，首个标签位于+12秒，不是+10秒。

## 从克隆仓库到开始训练

自定义清单代码在 `feature/custom-feature-manifest` 分支。建议Python 3.10或3.11，正式训练需要可用的CUDA PyTorch环境。

```bash
git clone https://github.com/ganmingzhang68-glitch/LightX_Crossformer_Time_forscasting.git
cd LightX_Crossformer_Time_forscasting
git switch feature/custom-feature-manifest
cd experiments/input_window_matrix_v1

python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests -q
```

一次完整训练分为五步：

1. 准备旧128字段数据和新增90字段数据。
2. 用 `prepare.py` 生成冻结的218字段训练数据集。
3. 用 `train_queue.py plan` 检查模型数量、输入数量和G1–G4配置。
4. 用 `train_queue.py init` 冻结字段、任务、训练/验证拆分和scaler；此步骤不训练。
5. 用 `train_queue.py run` 在CUDA上逐个训练，完成后再单独评估设备04。

Git仓库不包含原始训练数据、准备后的训练数据、InfluxDB凭据或新模型权重。只有代码和字段契约并不足以直接开始训练。

## 训练数据要求

### 数据源一：旧128字段数据集

`--old-root` 必须指向既有的128字段预处理目录，至少包含：

```text
old_root/
├── feature_manifest.json
├── train.parquet
├── validation.parquet
└── test.parquet
```

三个Parquet需要包含 `time_utc`、`device_id`、`segment_id`、`observed_mask` 和完整原128字段。`feature_manifest.json` 的 `feature_order` 必须与本实验 [`contract.json`](contract.json) 的 `targets` 完全同名、同顺序，且不能声明连续值已被裁剪。这里的 `train/validation/test` 是旧数据存储分片；新实验会重新按照完整打印任务及时间顺序构造自己的训练、内部验证和最终重训集合。

根目录的 `prepare_influx.py` 面向发布模型的单设备推理输入，不会生成上述多设备训练目录，不能替代这个训练数据源。

### 数据源二：新增90字段宽表

`--added-root` 使用以下文件名。开发集准备需要设备01/02/03，外部评估准备需要设备04：

```text
added_root/
├── influx_sensors_added_01-X450-1_2026-07.parquet
├── influx_jobs_added_01-X450-1_2026-07.parquet
├── influx_sensors_added_02-X450-2_2026-07.parquet
├── influx_jobs_added_02-X450-2_2026-07.parquet
├── influx_sensors_added_03-X450-3_2026-07.parquet
├── influx_jobs_added_03-X450-3_2026-07.parquet
├── influx_sensors_added_04-X450-4_2026-07.parquet
└── influx_jobs_added_04-X450-4_2026-07.parquet
```

传感器宽表必须包含：

- `time_utc`：可解析且带UTC语义的采样时间。
- `device_id`、`job_id`：设备和打印任务身份。
- `source_row`：同一时间戳去重时使用的稳定原始行序号。
- [`contract.json`](contract.json) 中 `added` 定义的全部90个字段。允许同时保留与原128同名的源列，但不允许混入无法解释的额外业务列。

任务表至少需要 `device_id`、`job_id`、`job_start_utc` 和 `job_end_utc`。每个设备内 `job_id` 必须唯一；同一时间范围不能同时匹配多个任务。字段名称、文件名和当前硬编码的 `2026-07` 月份必须一致。如需处理其他月份或不同命名方式，应另建数据准备版本，不要直接把文件改名后假定语义相同。

本仓库当前提供的是“旧128数据 + 新90宽表”的融合程序，不包含从任意原始CSV或数据库自动生成上述新增宽表的通用采集程序。数据提供者需要先按这个契约导出宽表，并核实单位、类型、时区和任务对应关系。

### 融合和规则采样

生成开发数据集：

```bash
python prepare.py \
  --old-root /data/processed_crossformer_multidevice_july_5m_10s_128features \
  --added-root /data \
  --output /runs/input_matrix/data_dev
```

生成设备04外部评估数据集：

```bash
python prepare.py --external \
  --old-root /data/processed_crossformer_multidevice_july_5m_10s_128features \
  --added-root /data \
  --output /runs/input_matrix/data_04
```

`prepare.py`执行以下处理：

- 按打印任务读取数据，绝不跨任务或跨连续段生成窗口。
- 将新增源按2秒时间网格对齐；相同源时间保留 `source_row` 最后一行，对齐后相同网格时间也保留最后一行。
- 原始间隔或对齐后间隔超过4秒时切成新连续段。
- 只对插入的一行网格缺口整体前向保持一次；真实源快照里的显式空值不会被填补。
- 旧128字段只在原连续段内进行一次因果前向保持，随后与新增数据按时间内连接。
- 检查 `available_time_utc <= time_utc`，防止未来才可获得的数据进入历史窗口。
- 生成完整218字段的 `values.npy`、时间和质量信息 `rows.parquet`、共同预测起点 `origins.npy`，并为每个文件保存SHA-256。
- 默认 `--stride 5`，即相邻预测起点间隔10秒。`--added-delay-seconds` 可模拟新增源的固定到达延迟，默认0；实际到达时间未知时不要把该模拟值描述成真实延迟。

输出目录必须不存在，并且不能放在代码目录、旧数据目录或新增数据目录内部。生成后以 `manifest.json` 的 `status=complete` 为成功标志；每台设备的 `alignment.json` 记录任务行数、窗口数、缺失数和排除原因。

## 不同类型的特征如何进入模型

模型有三个输入通道：

|清单配置|实际模型路径|适用特征|处理方式|
|---|---|---|---|
|`dynamic`|Crossformer主干|原128历史、连续传感器、需要主干直接建模的时变量|使用训练任务拟合的均值和尺度标准化，保持清单顺序|
|`conditions`|条件GRU修正分支|设定值、工艺条件、类别、日期年龄等辅助信息|连续量标准化；已声明类别使用训练词表和4维embedding|
|`quality` / `missing_masks` / `events`|条件GRU辅助通道|采集质量、缺失状态和历史变化事件|生成辅助编码，不计入原始业务字段数量|

原128字段必须全部位于 `dynamic`，因为Crossformer主干使用它们的最后历史值进行残差预测，同时评估使用同一最后值作为持久性基线。自定义方案因此最少是128个原始字段，不能只选其中一部分却仍声称完成原128输出任务。

### 连续数值特征

- 原128中的56个连续输出和新增连续输入都使用训练任务拟合的标准化参数。
- 连续新增字段放入 `dynamic` 时参与Crossformer多变量历史建模；放入 `conditions` 时通过GRU产生对128输出的修正。
- 常量字段不会自动删除。新增常量字段使用稳定尺度处理，但通常不会提供时间变化信息，应通过对照实验确认价值。

### 布尔状态特征

- 原128中的72个布尔字段仍作为数值历史进入Crossformer，并作为128输出的一部分训练。
- 损失在标准化空间计算；设备04评估时反标准化后以 `0.5` 为阈值计算F1、准确率和状态变化指标。
- 对开关边沿敏感时可以启用 `events=true`。程序为原72个布尔字段及全部条件字段生成上升、下降、距最近变化时间、变化年龄是否已知四类通道。该选项会明显增加辅助通道和推理开销，应单独做延迟对照。

### 类别特征

当前契约声明的类别字段是 `设铺粉模式`、`负压测试结果`、`正压测试结果`。

- 推荐把类别字段放在 `conditions`；程序只用训练任务建立词表，缺失值编码为0、训练中未见类别编码为1、已知类别从2开始，并使用4维embedding。
- 如果使用简易 `features` 清单，所有字段都会进入 `dynamic`，类别编号将被当作普通数值标准化，不会使用embedding。包含类别字段时通常应使用高级 `dynamic`/`conditions` 格式。
- 类别编码必须在设备之间保持同一业务含义，不能让同一数字在不同设备代表不同模式。

### 日期与检测结果

`负压测试日期`、`正压测试日期`不会把时间字符串直接送入模型。预处理会按新增快照时刻计算“距检测日期天数”，转换为 `log1p(days)`；未来日期、无效日期及其对应检测结果会被置为缺失，避免使用未来信息。此类字段推荐放在 `conditions` 并启用缺失掩码。

### 缺失、质量和事件辅助通道

- `missing_masks=true`：为所选新增字段生成逐字段已知掩码；数值缺失位置在标准化后填0，由掩码告诉模型该0不是实测值。
- `missing_masks=false`：所有未带掩码的新增输入都必须在最长300秒历史内完整，否则相应预测起点会从所有选定方案的共同样本集合中删除。
- `quality=true`：增加3个历史质量量，分别是旧128快照是否实测、新90快照是否实测、新90快照年龄除以300秒。
- `events=true`：只使用历史相邻快照计算变化，不跨未知快照推断边沿，也不读取未来标签。

所有scaler、类别词表和活跃字段统计只从设备01/02/03的训练任务拟合。不得用设备04、验证未来值或完整数据重新拟合。

## 新实验的共同规则

- 统一自包含 Crossformer + RevIN + 末值残差；条件历史/质量历史用GRU修正。各方案保持其业务字段和分支划分，并采用同一套固定超参数重新训练，不直接加载历史权重，也不声称复现历史HPO的最佳结果。
- 存储仍是完整218源字段，模型实际只取各方案选择的字段。ALL90保留全部90新增字段、90个缺失掩码和3个源质量通道；常量不筛掉，日期/检测结果按历史时点屏蔽。
- A/B及纯旧128基线不增加质量修正分支；C/D使用其条件分支；N90系列保留3个历史源质量通道。事件方案仅产生历史变化，不使用未来事件或任务最终信息。
- **跨全部选定输入和四个时间组共享同一批预测起点**。不带缺失掩码的方案要求其新增历史字段完整；ALL90允许带掩码的缺失。因此共同集合可能比上次单独all90评估更严格，不能直接复用旧04分数。只选择部分输入方案，也会形成不同的队列/共同集合。
- 01/02/03按完整任务的起始时间划分前80%训练、后20%内部验证；所有模型使用同一划分，训练窗口按设备平衡。scaler/词表仅拟合相应训练任务。
- 每个输入×时间组的3种子在同一50轮上限、patience8预算下选轮次；冻结其最佳epoch中位数。随后使用01/02/03全部开发任务重新拟合预处理器，三种子固定epoch重训。没有04早停或选种子。
- 损失固定为0.6连续Huber+0.4布尔Huber。J与上一轮定义相同：0.7连续相对末值MSE+0.3(1−训练活跃布尔宏F1)。完整输出及共同前60秒分开报告。逐字段指标保留全部128。
- 04仅在整个矩阵最终训练全部完成后评估；历史已被查看，仍明确标为探索性业务评估。

## 一个一个训练：支持

`train_queue.py run` 同步调用一个模型的训练函数，返回并验证模型产物后才进入下一个，**没有并行训练**。每次清理模型引用与CUDA空闲缓存。

顺序为：所有264次选轮次训练 → 按输入/时间冻结epoch → 所有264次最终训练 → `final.json`及总完成状态。

- `status.json`：总数、完成数、当前模型、失败信息；每个模型独立日志。
- `last.pt`：模型、optimizer、scheduler、随机状态和当前epoch。恢复时用同一运行加`--resume`。
- 完成模型按spec、签名和文件SHA256复核后跳过；修改数据、字段、代码、预算或种子会拒绝续跑。
- 遇到OOM、非有限值等错误，队列停止，后面的模型不会被误标为完成。修复环境后从同一任务/最近完整epoch恢复；需要改变预算或实现时另建运行目录。
- 同一队列有操作系统独占锁，重复启动会失败。不同输出目录的队列不会共享此锁，部署时只启动一条GPU训练队列。
- 总完成要求全部输入×4时间组×全部种子齐全；评估入口同时核查status、final、checkpoint身份，不能用部分完成模型冒充完整比较。

## 不训练的命令

本目录在独立克隆后可直接使用；不依赖工作区其他实验模块。依赖见 `requirements.txt`；不要升级正在训练的环境。

```bash
python train_queue.py plan
python -m pytest tests -q
```

`plan`只打印完整矩阵，不导入CUDA训练模块、不写checkpoint。可先选少量输入查看计划，例如：

```bash
python train_queue.py plan --profiles BASE128 B133_27 C129_4_27 D129_5_27 ALL90
```

此示例为5输入 × 4时间 × 3种子 = 60最终模型；默认全量仍为264。减少输入是显式用户选择，不会被程序悄悄改变。

## 自定义特征清单

`--feature-manifest` 可以按有序字段清单定义新的输入方案。它可以单独使用，也可以和 `--profiles` 同时使用；参数可以重复，以便在一个队列中比较多个自定义方案。只提供自定义清单而不提供 `--profiles` 时，不会隐式加入默认22方案。

简易格式把清单里的全部字段送入Crossformer动态分支。必须包含完整原128字段，可以从冻结的 `contract.json` 生成清单，随后按需要追加0至90个新增字段：

```json
{
  "version": 1,
  "name": "CUSTOM_135",
  "features": ["这里填写有序字段名"],
  "quality": true,
  "missing_masks": true,
  "events": false,
  "note": "原128加自选7个新增字段"
}
```

高级格式可以明确分配动态分支和条件GRU分支。`dynamic` 必须包含完整原128字段；同一字段允许同时出现在两个分支，原始输入数量按并集计算：

```json
{
  "version": 1,
  "name": "CUSTOM_CONDITION_133",
  "dynamic": ["这里填写完整原128有序字段"],
  "conditions": ["这里填写5个新增字段"],
  "quality": true,
  "missing_masks": false,
  "events": false
}
```

不要手工复制128个字段名。可以从冻结契约生成清单，并只维护希望新增的字段：

```bash
python - <<'PY'
import json
from pathlib import Path

contract = json.loads(Path("contract.json").read_text(encoding="utf-8"))
selected_added = ["打印层数", "设松粉S", "设铺粉模式", "设前螺旋时长", "设后螺旋时长"]
manifest = {
    "version": 1,
    "name": "CUSTOM_CONDITION_133",
    "dynamic": contract["targets"],
    "conditions": selected_added,
    "quality": True,
    "missing_masks": False,
    "events": False,
    "note": "原128动态历史加5个工艺条件",
}
Path("custom_condition_133.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
PY
```

如果新增字段需要直接参与Crossformer主干，可改用 `"features": contract["targets"] + selected_added`，或者把对应字段加入高级格式的 `dynamic`。同一个方案不要同时提供 `features` 和 `dynamic`/`conditions`。

查看计划不会开始训练：

```bash
python train_queue.py plan --feature-manifest /path/to/custom_135.json
python train_queue.py plan --profiles BASE128 ALL90 \
  --feature-manifest /path/to/custom_135.json \
  --feature-manifest /path/to/custom_condition_133.json
```

初始化时使用同一个参数：

```bash
python train_queue.py init --data-root /path/to/prepared --output /path/to/run \
  --feature-manifest /path/to/custom_135.json
```

程序会拒绝未知字段、字段重复、缺少原128动态历史、内置方案重名和不安全的方案名。原始字段数量自动计算，输出固定为原128。`quality` 会额外启用3个质量通道；`missing_masks` 和 `events` 产生的编码通道不计入原始字段数量。解析后的完整方案、来源路径和文件SHA-256会冻结进 `plan.json`，训练恢复不依赖原清单文件继续存在。

## 后续初始化（不更新模型权重）

复用已有all90完整218字段、四组共同起点的prepared数据即可，`contract.json`相同且入口会核验文件哈希。也可以用本目录 `prepare.py` 另建数据：

```bash
python prepare.py --old-root /path/to/data/processed_crossformer_multidevice_july_5m_10s_128features \
  --added-root /path/to/data --output /path/to/new_matrix/data_dev

python train_queue.py init --data-root /path/to/new_matrix/data_dev --output /path/to/new_matrix/run
```

使用自定义方案时，`plan` 和 `init` 应传入同一组内置方案与清单。例如：

```bash
python train_queue.py plan \
  --profiles BASE128 ALL90 \
  --feature-manifest ./custom_condition_133.json

python train_queue.py init \
  --data-root /path/to/new_matrix/data_dev \
  --output /path/to/new_matrix/run_custom133 \
  --profiles BASE128 ALL90 \
  --feature-manifest ./custom_condition_133.json
```

`init`做完整开发数据核验、计算矩阵共同集合、冻结计划/任务划分/两套scaler，并写`preflight.json`，不训练、不读04观测。输出必须新建，不能放入代码、源数据或旧运行目录。查看`plan.json`确认预期模型数量、原128输出和正确预算后再启动；默认22方案是264个最终模型，自定义方案按实际方案数计算。

初始化后至少检查：

- `preflight.json` 中 `status` 为 `passed`、`training_started` 为 `false`。
- `plan.json` 中每个方案的 `raw_input_count`、`dynamic`、`conditions`、`targets` 和四个时间组正确。
- `inputs.json` 中设备和完整任务拆分合理，共同窗口数量不是0。
- 任务总数符合“方案数 × 4时间组 × 种子数 × 2阶段”。默认22方案和3种子是528次任务；单个自定义方案和3种子是24次任务。

## 后续启动/恢复（会实际训练；本次未运行）

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_queue.py run --output /path/to/new_matrix/run
python train_queue.py status --output /path/to/new_matrix/run

# 中断后
CUDA_VISIBLE_DEVICES=0 python -u train_queue.py run --output /path/to/new_matrix/run --resume
```

服务器可用nohup/tmux保留进程，避免SSH断开终止队列：

```bash
nohup /path/to/verified/venv/bin/python -u train_queue.py run \
  --output /path/to/new_matrix/run > /path/to/new_matrix/run/queue.log 2>&1 < /dev/null &
```

`run`需要已冻结的初始化目录和CUDA。训练开始后只读代码、字段契约和数据；不修改plan来追加/删模型。

训练输出主要包括：

```text
run/
├── plan.json                 # 冻结的输入、窗口、预算、种子和代码身份
├── inputs.json               # 冻结的任务拆分、scaler和共同样本集合
├── preflight.json
├── status.json               # 当前任务和完成数量
├── logs/                     # 每个任务的独立日志
├── selection/                # 内部验证选轮次阶段
├── models/                   # 固定轮次最终重训模型
└── final.json                # 只有全部最终模型完成后才生成
```

如果训练异常停止，先运行 `status` 和查看对应日志。只有确认代码、字段、数据和预算未改变时才使用 `--resume`；需要修改任何实验条件时应建立新的输出目录。

## 后续04评估和计时

先等所有最终模型完成，再准备外部数据或复用相同contract的既有04 prepared数据：

```bash
python prepare.py --external --old-root /path/to/data/processed_crossformer_multidevice_july_5m_10s_128features \
  --added-root /path/to/data --output /path/to/new_matrix/data_04
python evaluate04.py quality --model-root /path/to/new_matrix/run --data-root /path/to/new_matrix/data_04 \
  --output /path/to/new_matrix/quality04 --device cuda
python evaluate04.py benchmark --model-root /path/to/new_matrix/run --data-root /path/to/new_matrix/data_04 \
  --output /path/to/new_matrix/timing04 --device cuda --samples 256 --rounds 3 --warmup 50
```

计时FP32、batch1、单线程、CUDA同步，记录逐请求前向和服务管线耗时。为避免数百模型同时驻留，按轮次随机排列模型块，一次只加载一个；每次加载后预热，加载不计入请求耗时。这种块计时仍可能有温度/时间漂移，不能直接和上次12模型驻留交错的计时结果作精细差异结论。

管线从已对齐、已解码内存输入开始，不包含磁盘、网络、采集融合、日期解析、模型加载或名义10秒空档。服务器04数据计时不等于设备04部署硬件计时。GPU有其他计算进程时拒绝计时。

质量及计时是独立显式命令，训练完成后不会自动读取04或额外启动评估。当前评估输出目录必须新建，中断评估需另建目录重跑；训练队列本身支持断点恢复。

## 训练前注意事项

1. **保持设备04隔离。** 设备04不能进入梯度更新、早停、epoch选择、特征选择或scaler拟合。只有 `final.json` 完成后才执行04评估。
2. **字段名称相同不代表数据相同。** 加入新字段前核实单位、采集位置、缩放、枚举含义、固件版本和设备间一致性。
3. **字段数量不能代表方案。** 两个133字段模型可能使用不同字段、不同顺序或不同动态/条件分支，必须保存完整清单和SHA-256。
4. **特征必须在预测时可获得。** 不要加入任务结束后才生成的检测结论、人工补录信息或未来设定值。日期和检测结果应经过当前快照有效性屏蔽。
5. **缺失策略会改变样本集合。** 不启用掩码会删除包含缺失历史的预测起点；启用掩码会保留样本并增加模型输入。比较方案时应使用同一个队列生成的共同起点。
6. **选择更多方案会增加训练量。** 每个方案默认产生4窗口 × 3种子 × 选轮次/最终重训两阶段，共24次训练。先用 `plan` 核对总数和GPU预算。
7. **不要修改已初始化队列。** 代码、字段契约、数据、预算、种子和模型产物都有签名或哈希校验。需要改变实验时创建新目录，不要手改 `plan.json`。
8. **输出目录必须全新。** `prepare`、`init`、质量评估和计时都拒绝覆盖已有目录，以避免混入旧结果。
9. **训练结果需与末值基线比较。** 连续量报告相对末值MSE，布尔量报告宏F1和状态变化指标；只看总体MSE或大量常量布尔准确率会产生误导。
10. **推理延迟需在目标环境实测。** 特征数、条件GRU、缺失掩码和事件通道都会改变参数量与延迟。训练服务器GPU计时不等于设备04实际部署硬件计时。
11. **不要提交数据和凭据。** 原始Parquet、prepared目录、checkpoint、SSH密码和数据库token应保存在Git仓库之外。
12. **新checkpoint不是根目录发布包。** 实验模型需要使用本目录的运行契约加载和评估；在完成独立导出、验证和版本化前，不要覆盖根目录 `artifacts/`。

## 本次验证边界

CPU合成数据检查覆盖全部默认输入/四组的128输出形状、可选事件编码、相同起点、精确gap标签、缺失/类别编码、训练统计隔离；替身队列覆盖串行执行、失败停止、断点恢复、完成屏障、文件篡改拒绝和互斥锁；合成Parquet检查真实初始化入口。

没有执行CUDA反向传播、真实全量初始化、正式训练或04质量/计时。测试通过不代表新模型已有训练成绩；最终CUDA环境与真实数据预算需在部署前核验。
