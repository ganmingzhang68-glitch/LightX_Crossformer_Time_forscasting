# 全部 90 新字段：218 输入 → 旧 128 输出

独立实验版本；方案见 `../../docs/ALL90_LATENCY_EXPERIMENT_PLAN_20260928.md`。本次交付不启动训练。旧 Epoch45、A/B/C/D、新90筛组选组和其运行目录均不作为本实验的写入目标。

|组|历史/空档/输出（秒）|点数|
|---|---|---|
|G1|300 / 10 / 120|150 / 5 / 60|
|G2|180 / 10 / 60|90 / 5 / 30|
|G3|180 / 10 / 120|90 / 5 / 60|
|G4|300 / 10 / 60|150 / 5 / 30|

首个标签是最后输入点之后 +12 秒。四组使用相同预测起点；完整输出质量和共同前 60 秒质量分别报告。

G4 于 2026-09-29 加入。四组构成历史长度 180/300 秒与输出长度 60/120 秒的 2×2 比较，仍使用三个配对种子；计划为 12 次选择训练和 12 次最终训练，共 24 次。新增组已接入配置驱动的训练、04评估和计时入口，本次不启动训练。

## 文件

- `contract.json` / `build_contract.py` / `sources/`：从原 docs HTML 提取的完整 90 字段，以及旧 128 输出顺序、编码与预算；包含可独立克隆使用的精简字段来源清单。
- `prepare.py`：独立任务分片、2 秒因果融合、全 218 源矩阵和共同起点。
- `data.py`：训练侧统计、日期/类别/常量/缺失的编码，90 个字段均有值位置和掩码。
- `model.py` / `vendor/`：130 动态通道的 Crossformer，以及包含全部90绝对历史的条件分支，输出旧128。
- `training.py` / `amp_step.py`：CUDA 训练、严格完整状态续跑、可选 AMP 保护。
- `run.py`：显式的 plan、preflight、select、refit、evaluate04 命令。
- `benchmark.py`：04 数据上的配对 batch1 前向与服务流程耗时；不训练。
- `metrics.py` / `compare.py`：原单位质量、持久性、共同预测区间、任务级 bootstrap、质量速度并列表。
- `check_real_sources.py` / `tests/`：不训练的验证入口。

## 只读检查，可现在执行

在本目录运行。将 Python 解释器换成实际环境；本地已有 CPU PyTorch 可做前向检查，正式训练需要服务器 CUDA。独立代码不依赖 einops；vendor 的张量布局操作已改为 PyTorch 原生等价操作。

Git 分支中的代码可在独立克隆后执行 `plan` 和合成测试，不需要原工作区 HTML。
`preflight` 与真实片段检查的 `--root` 则必须指向含授权 `data/` 数据的工作区。
字段来源及哈希校验方式见 `sources/README.md`。

```powershell
python run.py plan
python run.py preflight --root D:\PATCHTST
python -m pytest tests -q
python check_real_sources.py --root D:\PATCHTST --output D:\PATCHTST\output\all90_check\real_sources.json
```

真实短片段检查只读 01/02/03，不构造全量数据、不读 04 观测、不训练。已经存在的检查输出不会被覆盖。

## 后续数据构造（本次不执行）

以下以服务器目录举例，路径需替换。依赖见 requirements.txt；使用独立环境或已验证的 CUDA 环境，不在正在训练的环境里升级依赖。

```bash
ROOT=/path/to/PATCHTST
WORK=/path/to/all90_latency_v1_run
python prepare.py --old-root "$ROOT/data/processed_crossformer_multidevice_july_5m_10s_128features" --added-root "$ROOT/data" --output "$WORK/data_dev"
```

默认只构造 01/02/03。输出必须是新的目录，且不能嵌入任何源数据或代码目录。源表、旧权重不会被覆盖。构造完成后所有训练/评估入口都会核验文件哈希、时间网格、窗口边界和观测标签。

## 后续训练（这些命令会训练，本次未运行）

```bash
python run.py select --data-root "$WORK/data_dev" --output "$WORK/selection" --precision fp32
python run.py refit --data-root "$WORK/data_dev" --frozen "$WORK/selection/frozen.json" --output "$WORK/final"
```

select 为 4 组 × 3 种子，内部验证选择轮次；refit 为同样 12 个最终模型，固定轮数、全部开发任务拟合预处理器、按设备平衡窗口。最终目录的 `final.json` 仅在全部 12 个模型完成后生成。恢复同一运行需使用原命令加 `--resume`，修改代码/数据/预算后应另建版本与目录。

## 后续 04 质量与计时（要求全部最终模型已完成）

```bash
python prepare.py --external --old-root "$ROOT/data/processed_crossformer_multidevice_july_5m_10s_128features" --added-root "$ROOT/data" --output "$WORK/data_04"
python run.py evaluate04 --data-root "$WORK/data_04" --model-root "$WORK/final" --output "$WORK/quality_04_fp32" --device cuda --precision fp32
python benchmark.py --data-root "$WORK/data_04" --model-root "$WORK/final" --output "$WORK/timing_04_cuda_fp32" --device cuda --precision fp32 --warmup 50 --samples 256 --rounds 3
python compare.py --quality-dir "$WORK/quality_04_fp32" --benchmark-dir "$WORK/timing_04_cuda_fp32" --output "$WORK/comparison_cuda_fp32.json"
```

CPU 计时使用单独目录，并指定 `--device cpu --threads 1 --precision fp32`。每组报告全部种子，不用04挑最优种子。计时不包含磁盘、网络、采集/对齐、原始日期字符串解析、模型加载和10秒空档；服务流程从已对齐且类型解码完成的内存窗口开始。所有逐次耗时保存在 `latency_samples.csv`。

04 只用于冻结后的业务比较；历史曾被查看，因此结果明确标为探索性评估。没有真实训练结果前，不推测新增90字段带来的增益或四组具体耗时。

## 本次验证结果

2026-09-29：加入 G4 后，16 项本地测试通过。测试覆盖全90来源契约、日期/结果的未来屏蔽、缺失掩码、训练侧统计隔离、四组相同起点与标签边界、四组 CPU 前向、分片构造与哈希校验、8 项配对质量比较及12次替身训练编排。

原工作区版本对 01/02/03 各抽查一个 12 分钟真实片段，每设备得到 360 个融合点和 30 个共同窗口，全部218字段及四组编码均通过。未执行优化器更新、CUDA训练、全量构造、04评估或正式计时。`VALIDATION.json` 区分本次 Git 打包复测与原工作区真实片段检查，后者的完整本地证据不随分支上传。
