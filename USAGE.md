# 独立克隆后的安装、数据准备、训练与推理

本说明使用仓库内代码，不需要作者电脑的 `D:/PATCHTST`、服务器路径或外部 Time-Series-Library。命令从仓库根目录执行，示例 `/data`、`/runs`、`/results` 均需替换为自己的绝对路径。输出目录使用新目录。

## 选择分支

|分支|训练用途|推理用途|
|---|---|---|
|`main`|ALL90 的218输入→128输出；历史A/B/C/D的27输出|本分支有12套ALL90正式模型；也支持自己新训练的模型|
|`experiment/all90-latency-4groups`|ALL90，全部90新增字段入模，G1–G4|本分支有12套ALL90正式模型|
|`experiment/input-window-matrix-g1-g4`|内置历史输入方案，统一原128输出，G1–G4，串行队列|自己训练后导出；现成ALL90模型可按下文单独下载|
|`feature/custom-feature-manifest`|上述矩阵及自定义有序特征清单|本分支有12套ALL90模型；自己训练后也可导出|
|`experiment/latency-abcd-27`|A/B/C/D历史角色，新训练统一10秒空档，输出27字段|新训练模型使用 `predict_legacy.py`；现成ALL90模型可单独下载|

每个分支只包含其目录里实际存在的训练入口。矩阵默认22个去重输入方案；自定义最少保留原128输入，新增字段只能从冻结90字段中选。它不是任意新字段、任意输出维度的通用训练框架。矩阵和新10秒A/B/C/D没有随仓库发布已完成的新训练权重，需自行训练；不要把历史30秒模型的元数据改成10秒。

## 环境与检查

建议Python3.10–3.12。CPU可做检查与推理；正式训练需适合自己显卡驱动的CUDA PyTorch。先按PyTorch官方安装器选择对应CUDA版本，再安装本仓库依赖；不要在正在训练的环境中升级。

```bash
git clone https://github.com/ganmingzhang68-glitch/LightX_Crossformer_Time_forscasting.git
cd LightX_Crossformer_Time_forscasting
git switch feature/custom-feature-manifest  # 按上表选择
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python tools/check_install.py --training --models
python -m pytest -q
```

`check_install.py`不训练：检查依赖、当前分支计划入口和历史权重严格加载；`--models`额外校验已发布模型文件。`cuda_available=false`表示这个环境不能正式训练，但仍可CPU推理。测试使用合成数据和前向运算，不代表完成CUDA训练或测得新的模型质量。分支的验证记录见 `BRANCH_VALIDATION.json`。

## 从原始数据生成训练来源

仓库不包含原始设备数据、数据库凭据和用户自己训练的checkpoint，使用者需提供合法取得的数据。不要把密钥、密码或原始设备数据提交到Git。

旧128字段必须与 `artifacts/feature_manifest.json` 的名称、单位、类型和顺序一致。可直接提供符合以下格式的已有旧128数据目录：`feature_manifest.json`及 `train.parquet`、`validation.parquet`、`test.parquet`；行包含 `time_utc`、`time_beijing`、`device_id`、`segment_id`、`observed_mask`和完整128字段，连续值不能被百分位裁剪。

没有这个中间目录时，现在可从每设备一个Influx长表Parquet生成：

```bash
python tools/prepare_old128.py --source 01-X450-1=/data/influx_sensors_01-X450-1_2026-07.parquet --source 02-X450-2=/data/influx_sensors_02-X450-2_2026-07.parquet --source 03-X450-3=/data/influx_sensors_03-X450-3_2026-07.parquet --source 04-X450-4=/data/influx_sensors_04-X450-4_2026-07.parquet --output /runs/old128
```

长表要求 `time`、`host`、`measurement`、`field`、`value_type`、`value_bool`、`value_float`，且每个文件只含所声明设备。也支持已按原始完整快照展开的宽表：`time_utc`或`time`加全部128字段，可保留任务和分段身份。时间必须已落在UTC的2秒网格上；重复时间戳、混合设备、类型错误会拒绝，需先核查来源。工具不猜测偏移，不标准化、不裁剪、不插值，只能在缺失的一整行网格上因果保持一次，并将其标为非实测。显式空值、长缺口及任务/段边界不会被跨越。它是新的可复现来源构造方式，不保证得到与作者历史预处理完全相同的样本数。

输出的 `train/validation/test`仅为存储分片（01/02/03、04、空表）；ALL90/矩阵训练会重新按完整打印任务及时间拆分，不能把这些文件名直接当成模型选择口径。

新增源可以直接提供每设备一份 `influx_sensors_added_<设备>_2026-07.parquet`和 `influx_jobs_added_<设备>_2026-07.parquet`。传感器需要 `time_utc`、`device_id`、`job_id`、`source_row`、`source_file`及冻结90字段；任务表需要设备/任务键、`job_start_utc`、`job_end_utc`。ALL90允许新增字段值缺失，但不能漏列；旧128不能以新增CSV中的同名字段替代。

若提供的是作者使用的按任务命名的CSV，可使用补齐的转换器：

```bash
python tools/convert_added_csv.py --input-dir /data/ValuePLCLog --data-dir /data
```

要求CSV为UTF-8，文件名形如 `ValuePLCLog_2026-07-01_08-00-00_Machine01.csv`，首列为北京时间 `记录时间`；`/data`中还需对应的原始 `influx_jobs_<设备>_2026-07.parquet` Jobs长表用于匹配。转换器保留原始行和缺口，不重采样；已有输出拒绝覆盖。它面向这个PLC导出格式和四台设备，不是任意CSV转换器。其他月份、设备编号和字段语义需要先修改契约及导出规则，新建实验，不能只改文件名。

## ALL90：G1–G4训练

存在 `experiments/all90_latency_v1/` 的分支支持：

```bash
python experiments/all90_latency_v1/run.py plan
python experiments/all90_latency_v1/prepare.py --old-root /runs/old128 --added-root /data --output /runs/all90_data
python experiments/all90_latency_v1/run.py select --data-root /runs/all90_data --output /runs/all90_selection --precision fp32
python experiments/all90_latency_v1/run.py refit --data-root /runs/all90_data --frozen /runs/all90_selection/frozen.json --output /runs/all90_final
```

只有`select/refit`会训练。先按开发设备01/02/03的完整任务选轮次，再对全部开发任务重训；设备04不参与拟合、早停和选种子。恢复同一运行需同一代码、数据和预算，并在原命令上加`--resume`。本次升级修正包导入及跨平台代码身份，旧签名队列需在原代码版本继续；新版本应新建队列，禁止改签名绕过检查。已发布模型保留自己的冻结源码和推理实现。

四组窗口分别为G1=300+10+120、G2=180+10+60、G3=180+10+120、G4=300+10+60秒。采样2秒，空档跳过5点，首个标签在最后输入后+12秒；60秒输出结束于+70秒，120秒输出结束于+130秒。各组输出原128字段（56连续、72布尔）。

## 矩阵及自定义输入训练

存在 `experiments/input_window_matrix_v1/` 的分支支持以下内置方案队列，默认执行全部方案；用`--profiles`可选子集：

```bash
python experiments/input_window_matrix_v1/prepare.py --old-root /runs/old128 --added-root /data --output /runs/matrix_data
python experiments/input_window_matrix_v1/train_queue.py plan --profiles BASE128 ALL90
python experiments/input_window_matrix_v1/train_queue.py init --data-root /runs/matrix_data --output /runs/matrix_queue --profiles BASE128 ALL90
python experiments/input_window_matrix_v1/train_queue.py run --output /runs/matrix_queue
python experiments/input_window_matrix_v1/train_queue.py status --output /runs/matrix_queue
# 断点恢复：上一条run命令加 --resume
```

队列会逐个执行所有方案×G1–G4×种子的选择训练，再逐个执行最终重训。出错即停，完成且通过校验的模型不会重复训练。计划和初始化不训练；正式`run`需要CUDA。共同起点会按所选方案取交集，所以改变方案集合也会改变可比较的数据口径。

仅自定义分支支持额外参数 `--feature-manifest /path/to/my_features.json`，在`plan`和`init`时传入，`run`读取冻结计划。可重复传多份清单。完整例子见该分支的 `experiments/input_window_matrix_v1/README.md`；简易`features`会全部送入动态分支，高级`dynamic/conditions`可以分别指定角色。清单必须保留原128动态字段，不能含重复或未知名称，顺序会写入训练契约和模型包。

连续量仅用训练任务拟合尺度；布尔量以数值进入主干，反归一化后按0.5阈值输出。类别字段建议放条件分支，用训练词表和embedding；检测日期转换为该快照时已知的`log1p(检测年龄天数)`，未来或无效日期及对应结果置缺失。新增缺失用掩码表达，质量/事件通道只用历史，不算额外业务字段。单位或物理含义不明时不能合并同名/近义字段。

## A/B/C/D：27输出模型

这些是单独的历史角色对照，不能当成统一128输出矩阵。现在可直接构造所需数据：

```bash
python tools/prepare_abcd.py --profile ABC --old-root /runs/old128 --new-data-dir /data --output /runs/abc133
python tools/prepare_abcd.py --profile D --old-root /runs/old128 --new-data-dir /data --output /runs/d134
python experiments/latency_abcd_27/C_129_history_plus_4_conditions_to_27/validate_contract.py --data-root /runs/abc133
python experiments/latency_abcd_27/C_129_history_plus_4_conditions_to_27/tune.py --data-root /runs/abc133 --output /runs/c_hpo
python experiments/latency_abcd_27/C_129_history_plus_4_conditions_to_27/confirm.py --data-root /runs/abc133 --best-trial /runs/c_hpo/best_trial.json --output /runs/c_confirm
python experiments/latency_abcd_27/C_129_history_plus_4_conditions_to_27/train_final.py --data-root /runs/abc133 --best-trial /runs/c_hpo/best_trial.json --confirmation-result /runs/c_confirm/confirmation_result.json --output /runs/c_model
python predict_legacy.py --group C --model-dir /runs/c_model --input /runs/abc133/devices/04-X450-4/rows.parquet --output /results/c.parquet --mode rolling --device cpu
```

A/C用133来源，D用134来源。B同样用133来源，其脚本名为 `tune_latency_133_to_27.py`、`confirm_latency_133_to_27.py`、`train_final_latency_133_to_27.py`，还需要 `--targets experiments/latency_abcd_27/B_133_history_to_27/selected_targets_27.json`。具体参数可先用`--help`检查。新最终模型会保存 `model.pth`、`scaler.npz`、`feature_manifest.json`、`config.json`；转移模型时需要完整复制。旧C/D记录没有新10秒权重，不能补改元数据后推理。

## ALL90和矩阵模型的推理

已经发布的ALL90模型可直接按 `models/all90_20261110/README.md` 使用。所有分支都附带根目录 `download_model.py`。没有模型目录时也可直接下载一套，不需要手动复制权重或切换分支：

```bash
python download_model.py --group G4_300_10_60 --seed 20260928 --output /exports/all90_g4
python /exports/all90_g4/predict.py --input /data/aligned_numeric.parquet --output /results/g4.parquet --mode latest --device cpu
```

自己训练的模型先导出。矩阵在`final/<方案>/<组>/seed_<种子>`中选择一个已完成目录，具体路径以队列`plan.json`为准；ALL90在`final/<组>/seed_<种子>`中选择：

```bash
python export_model.py --experiment all90 --model-dir /runs/all90_final/G4_300_10_60/seed_20260928 --output /exports/my_all90
# 矩阵使用 --experiment matrix，--model-dir换成对应已完成目录
python predict.py --bundle /exports/my_all90 --input /data/aligned_numeric.parquet --output /results/my_all90.parquet --mode rolling --include-target --device cpu
```

推理输入必须是2秒对齐且类型解码后的宽表，不能直接传原始Influx长表或日期字符串。可从`prepare.py`的任务分片直接导出：

```bash
python tools/dataset_to_inference.py --data-root /runs/all90_data --job-id <manifest.json中的真实任务ID> --output /data/aligned_numeric.parquet
```

这个工具按冻结字段顺序读取`values.npy`和`rows.parquet`、验证哈希，保留UTC时间、设备、任务、分段、质量掩码和新增源年龄，不拟合scaler。外部实时数据需按同样的因果规则准备，不能伪造质量信息。推理复用模型包的scaler、词表和顺序；不跨任务或长缺口。`latest`只需历史，`rolling`还需足够的未来实测标签范围，输出文件默认拒绝覆盖。

根目录`artifacts/`为历史无空档Epoch45档案，`validate_bundle.py`可检查原权重和契约；当前`predict.py`要求训练时就有10秒空档的模型，不能把旧权重直接用于新任务。历史复现应检出对应历史提交，在独立目录运行，不修改档案权重。
