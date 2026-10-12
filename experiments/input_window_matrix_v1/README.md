# 历史输入方案 × G1–G4：统一原128输出

2026-10-13 新实验代码。用户已明确要求：**所有新模型统一输出原128字段（56连续、72布尔），四个时间窗口均要训练。**

本次只编写代码及执行CPU/替身检查，没有启动CUDA训练、构造真实全量数据或写入服务器。旧 Epoch45、A/B/C/D、new90 和已完成 all90 的代码、权重、数据均保留。

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

## 后续初始化（不更新模型权重）

复用已有all90完整218字段、四组共同起点的prepared数据即可，`contract.json`相同且入口会核验文件哈希。也可以用本目录 `prepare.py` 另建数据：

```bash
python prepare.py --old-root /path/to/data/processed_crossformer_multidevice_july_5m_10s_128features \
  --added-root /path/to/data --output /path/to/new_matrix/data_dev

python train_queue.py init --data-root /path/to/new_matrix/data_dev --output /path/to/new_matrix/run
```

`init`做完整开发数据核验、计算矩阵共同集合、冻结计划/任务划分/两套scaler，并写`preflight.json`，不训练、不读04观测。输出必须新建，不能放入代码、源数据或旧运行目录。查看`plan.json`确认264个最终模型、原128输出和正确预算后再启动。

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

## 本次验证边界

CPU合成数据检查覆盖全部默认输入/四组的128输出形状、可选事件编码、相同起点、精确gap标签、缺失/类别编码、训练统计隔离；替身队列覆盖串行执行、失败停止、断点恢复、完成屏障、文件篡改拒绝和互斥锁；合成Parquet检查真实初始化入口。

没有执行CUDA反向传播、真实全量初始化、正式训练或04质量/计时。测试通过不代表新模型已有训练成绩；最终CUDA环境与真实数据预算需在部署前核验。
