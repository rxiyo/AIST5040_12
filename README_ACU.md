# 从 XRD 反演金红石 (a, c, u)

`acu_benchmark.py` 从一维 XRD 谱估计金红石 TiO₂ 的晶胞参数 a、c（Å）和氧原子分数坐标 u（无量纲）。CVAE 学习给定观测谱的参数分布，再用 XRD 正演给候选排序。对照方法包括 MDN、等候选预算网格和密集网格。训练、推断和物理评分都在 CPU 上运行。

## 课程使用限制

**此原型和对原 MDN 的扩展由 AI 生成。课程讲义禁止将 AI 生成代码作为学生自己的代码提交；请先与教师确认允许的使用方式。** 不应将此代码或其结果作为学生独立完成的成果提交。其他 AI 用途及受影响的部分也应按课程要求如实声明。

## 工作流程与代码

| 阶段 | 输入、处理和输出 | 代码 |
|---|---|---|
| 1. 生成模拟谱 | 对 a ∈ [4.50, 4.69]、c ∈ [2.90, 3.02]、u ∈ [0.2953, 0.3153] 独立均匀采样。构建金红石晶胞，用 pymatgen 计算衍射峰，再以固定 σ = 0.12° 的高斯峰形展宽到 20–70°、步长 0.1° 的 2θ 网格。 | [`ACUSimulator`](acu_benchmark.py#L31)、[`ANGLES` 与 `MASKS`](physics.py#L9) |
| 2. 划分与构造观测 | 分别生成训练、验证、测试晶胞。每个晶胞构造完整谱、24–33°、33–42° 三个观测，留在同一划分。窗口内按最大强度归一化后加入相对强度高斯噪声，窗口外填零。 | [`split`](acu_benchmark.py#L47)、[`run` 的划分](acu_benchmark.py#L181) |
| 3. 输入特征与目标 | 谱每隔两个网格点取样，追加观测掩码特征，得到 272 维输入 x。只用训练谱估计标准化参数。以各参数的中心和半范围将实值坐标转换为 [−1, 1] 中的 y。 | [`SpectrumFeatures`](models.py#L8)、[特征与目标归一化](acu_benchmark.py#L197) |
| 4. 训练 CVAE | 条件先验 p(z ∣ x) 只看谱，编码器 q(z ∣ x, y) 同时看谱和真值。两者输出对角高斯分布。训练时由编码器重参数化采样 z，解码器根据 x、z 输出三个参数在 logit 空间的高斯均值和标准差。用 Adam 更新三个网络，以验证损失最低的轮次保存检查点。 | [`CVAE`](acu_benchmark.py#L61)、[`CVAE.loss`](acu_benchmark.py#L81)、[`train_cvae`](acu_benchmark.py#L105)、[保存检查点](acu_benchmark.py#L212) |
| 5. 生成参数候选 | 给定测试谱，只用条件先验采样 z，再从解码器采样 t。将 `2 * sigmoid(t) - 1` 反归一化成 (a, c, u)。每条谱固定生成 216 组候选，与 6×6×6 网格的点数一致。 | [`CVAE.sample`](acu_benchmark.py#L96)、[反归一化](acu_benchmark.py#L244) |
| 6. 物理重排 | 对每个候选重新计算 XRD，按对应窗口归一化，只比较可见角度的强度 MSE。按 MSE 从低到高输出前五组。 | [物理评分与重排](acu_benchmark.py#L249) |
| 7. 评估与保存 | 计算参数 MAE、联合 Top-1/Top-5、完整候选池命中率、归一化能量得分和边界样本比例。原始 Top-5 使用前五个随机样本，物理 Top-5 使用同一候选池排序。真值用于计算指标。 | [`summarize`](acu_benchmark.py#L156)、[`summarize_pool`](acu_benchmark.py#L169)、[写入结果](acu_benchmark.py#L274) |

CVAE 的隐藏层宽度为 64，潜变量为 4 维。训练损失是参数的负对数似然，加上编码器后验相对于条件先验的 KL 散度，正式实验中 KL 权重为 1。负对数似然包含 sigmoid 变换的 Jacobian，使训练的概率密度与推断的有界输出分布一致。

## 安装和运行（macOS / Linux）

在项目目录运行，Python 3.12 是本次验证使用的版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt torch
python acu_benchmark.py
```

Windows 可先执行 `.venv\Scripts\activate`，然后使用同样的 pip 和 Python 命令。训练和推断都在 CPU 上运行。小规模流程检查可以用以下命令。

```bash
python acu_benchmark.py --train 30 --validation 10 --test 5 --epochs 3 --mdn-iterations 5 --output outputs/acu_smoke
```

默认实验用 600/80/40 个晶胞，训练 80 轮。下方正式实验每个种子用 1800/240/160 个训练、验证、测试晶胞，训练 150 轮，批量大小 64，学习率 0.001，噪声标准差 0.01。等预算网格为 6×6×6，三个轴包含端点，步长分别为 0.038 Å、0.024 Å、0.004。CVAE 和 MDN 各生成 216 个候选。密集网格为 41³ = 68,921 点。网格使用预计算模板，生成候选的物理重排在线模拟候选谱。

等预算网格和 CVAE、MDN 的候选数由 [统一的预算常量](acu_benchmark.py#L27) 固定。独立的密集网格基线使用 `--dense-side` 设置每轴点数。

```bash
for seed in 42 43 44; do
  python acu_benchmark.py --seed "$seed" --train 1800 --validation 240 --test 160 \
    --epochs 150 --mdn-iterations 260 \
    --dense-side 41 --output "outputs/acu_matched216/seed-$seed"
done
```

## 216 候选等预算结果

用种子 42、43、44 各测试 160 个晶胞。三种等预算方法每条谱都对 216 个参数组合评分。下表列出三个种子的平均联合 Top-5 命中率，命中要求同一个候选的 a、c 误差均不超过 0.02 Å，u 误差不超过 0.002。

| 测试窗口 | 216 点网格 | MDN + 物理重排 | CVAE + 物理重排 | 密集网格 |
|---|---:|---:|---:|---:|
| 完整谱 | 89.8% | 91.7% | 92.1% | 97.7% |
| 24–33° | 18.1% | 27.7% | 23.1% | 16.7% |
| 33–42° | 56.9% | 92.7% | 92.7% | 97.5% |

24–33° 的 CVAE 全池命中率为 100.0%，重排后 Top-5 为 23.1%，表明该窗口的谱残差难以区分部分参数组合。完整谱的批量在线耗时约为 CVAE 124 ms/谱、MDN 120 ms/谱、预建 216 点网格 0.05 ms/谱。测试数据来自同一套理想化模拟器，候选分布尚未做概率校准。 [完整指标与跨种子标准差](results/local/matched216/summary.csv)

## u 多样性重排与 KL 预热

Benchmark 为 CVAE 和 MDN 各记录两种重排。 `*_physics` 按谱 MSE 取前五。 `*_diverse_u` 保留首位，在谱误差最低的 20 个候选中贪心扩大前五个候选的 u 间距。两种重排使用同一候选池，候选池生成和物理评分只算一次。 [重排实现](acu_benchmark.py#L141)

训练选项 `--kl-warmup-epochs 30` 将训练里的 KL 权重从目标值的 1/30 逐轮升至目标值，在第 30 轮达到 1。所有轮次的验证损失都用相同的目标权重 1 计算。对照命令使用原来的三个种子和固定的 216 候选配置。 [训练实现](acu_benchmark.py#L105)

```bash
for seed in 42 43 44; do
  python acu_benchmark.py --seed "$seed" --train 1800 --validation 240 --test 160 \
    --epochs 150 --mdn-iterations 260 --dense-side 41 --kl-warmup-epochs 30 \
    --output "outputs/acu_warmup_diversity/seed-$seed"
done
```

三个种子的平均联合 Top-5 命中率如下。原训练的多样性结果直接复用原有候选池和谱误差，KL 预热则重新训练 CVAE 后评分。两种排序交出的首位候选相同。

| 模型与训练 | 重排 | 完整谱 | 24–33° | 33–42° |
|---|---|---:|---:|---:|
| MDN | 谱误差 | 91.7% | 27.7% | 92.7% |
| MDN | u 多样性 | 96.3% | 31.9% | 97.5% |
| CVAE 固定 KL 权重 | 谱误差 | 92.1% | 23.1% | 92.7% |
| CVAE 固定 KL 权重 | u 多样性 | 96.9% | 26.0% | 98.8% |
| CVAE KL 预热 | 谱误差 | 93.1% | 26.0% | 95.6% |
| CVAE KL 预热 | u 多样性 | 96.3% | 27.5% | 98.5% |

固定训练时，完整谱的 CVAE 前五个候选的平均 u 跨度由 0.0052 增至 0.0100，三个窗口 Top-5 均提高。固定物理排序时，KL 预热使 33–42° 的 CVAE Top-1 从 66.0% 升至 71.9%。已经使用 u 多样性排序时，KL 预热仅在 24–33° 窗口提高了平均 Top-5。 [逐种子结果](results/local/warmup_diversity/ablation_by_seed.csv) · [均值与样本标准差](results/local/warmup_diversity/summary.csv)

## 对一条曲线生成候选

加上 `--ranking diverse-u` 可在前五个候选中增加 u 的覆盖范围。首位仍是谱误差最小的候选，剩余四席从谱误差最低的 20 个候选里依次选取，与已选 u 值的最小间距越大越优先。CSV 的行顺序表示输出顺序，`physics_rank` 列记录每个候选按谱误差的原始名次。默认 `--ranking physics` 按谱误差排序。

训练后，输入 CSV 必须有 `angle_deg,intensity` 两列表头。角度须与项目的 20–70°、步长 0.1° 网格严格对应，并包含选定窗口内所有角度；可只提供窗口里的点，或提供完整曲线。脚本在窗口内按最大强度归一化，生成候选并按物理谱残差排序：

```bash
python predict_acu.py your_spectrum.csv --window full --method cvae \
  --model-dir outputs/acu_matched216/seed-42 --output candidates.csv
```

`--window` 可选 `full`、`low_information`、`contrast`，`--method` 可选 `cvae` 或 `mdn`。[`predict_acu.py`](predict_acu.py#L12) 读取 `feature_stats.npz` 和所选权重文件，生成 216 个候选并按窗口内谱 MSE 对全部候选排序。模型加载使用 Python 序列化格式，请使用可信的本地模型文件。应用到实验谱还需要处理背景、峰形与标定，并验证候选在实验数据上的可信度。

## 输出

- `dataset.npz`：分组参数和带噪谱；`predictions.npz`：候选池和每个候选的可见谱误差。
- `mdn_weights.npz`、`cvae.pt` 和 `feature_stats.npz`：模型权重与输入标准化信息。`cvae_history.csv` 记录训练 KL 权重、验证损失、重建项和 KL。
- `metrics.csv`：各窗口和方法的 a、c、u 的 Top-1 误差，a/c、u 和联合 Top-5 命中率，谱误差及在线时间。
- `proposal_metrics.csv`：重排之前的完整候选池联合命中率、归一化能量得分（越低越好）和各参数的边界样本比例。能量得分用参数半范围归一化距离，结合候选到真值的距离和候选之间的距离。
- `comparison.png`：u 误差与三参数联合 Top-5；`run.json`：范围、随机种子、训练配置、离线建库与训练时间。

参数范围定义了合成实验的采样空间。真实样品的结构稳定性、相组成、峰形和仪器偏差需要另外验证。

## 接入组内 xrdbench

相邻的 [`xrd_inverse`](../xrd_inverse/README.md) 统一模拟数据、训练与测试划分、提交格式和评分。其金红石任务估计 a、c，固定氧坐标 u = 0.305。盲测提供计数谱、观测掩码和曝光，隐藏仪器参数。后验提交要求每条谱提供 256 个未重排的样本，由 evaluator 用样本边缘中位数评估点误差，并评估 Energy Score、区间覆盖率等分布指标。该赛道的共享评分器目前不支持物理重排。 [联合评测协议](../xrd_inverse/docs/v0.2实验协议.md)

[`xrdbench_acu.py`](xrdbench_acu.py) 提供两个接入口。

- `FrozenAdapter` 将观测谱对齐到原项目的 20–70° 谱轴，用旧权重生成 (a, c, u) 后只输出 (a, c)，保留旧模型的参数范围。该对照用于检查直接迁移。
- `Adapter` 读取在 xrdbench public/train 和 public/val 上重新训练的二维模型。CVAE、MDN 使用相同的谱分箱、观测掩码、曝光特征和数据划分。用验证集 Energy Score 选检查点，每条观测使用 32 个固定随机样本。提交和打分分别调用 xrdbench 的 `run` 与 `score`。

本地联合 pilot 使用训练种子 0、5,000 个训练结构和 512 个保留测试结构。目录里提供了发布索引，本次数据由组件本地生成，软件依赖和 dataset_id 与正式发布不同。正式组内比较应使用协调者发布的同一份 public 数据，并由协调者读取 private/test 评分。

```bash
export PYTHONPATH="../xrd_inverse:$PYTHONPATH"
XRD_PUBLIC=/path/to/rutile/blind/v0.1+v0.2/public
python -m xrdbench_acu --public "$XRD_PUBLIC" --method cvae \
  --kl-warmup-epochs 30 --output outputs/shared_cvae
python -m xrdbench.cli run --public "$XRD_PUBLIC" --adapter xrdbench_acu:Adapter \
  --checkpoint outputs/shared_cvae/model.pt --metadata outputs/shared_cvae/metadata.json \
  --kind posterior --output outputs/shared_cvae_prediction
```

| 本地联合 pilot，盲测 | 平均相对误差 ↓ | 0.5% 联合成功率 ↑ | Energy Score ↓ |
|---|---:|---:|---:|
| 旧 MDN 权重直接迁移 | 2.076% | 9.16% | 0.30009 |
| 旧 CVAE 权重直接迁移 | 1.976% | 9.55% | 0.27990 |
| 旧 CVAE 预热权重直接迁移 | 1.960% | 9.38% | 0.27933 |
| MDN 重新训练 | 0.697% | 41.05% | 0.07862 |
| CVAE 重新训练 | 0.616% | 51.53% | 0.06991 |
| CVAE 重新训练并预热 30 轮 | 0.547% | 58.68% | 0.06158 |

后验样本数诊断使用相同样本流的前缀。预热 CVAE 从 64 增至 216 个原始样本时，平均相对误差从 0.553% 变为 0.548%，Energy Score 接近 0.0616。原三参数实验中的预算诊断同样使用既有 216 个候选的前缀，并在每个预算内重新排序。它衡量的是物理评分预算。

[组件完整评分报告](results/xrdbench/report/report.md) · [候选数诊断（组内后验）](results/xrdbench/posterior_sample_budget.csv) · [候选数诊断（原三参数物理重排）](results/xrdbench/acu_rerank_budget.csv)

### CVAE 与 MDN 的三种提交格式

保持同一盲测 pilot 和重训后的模型权重，分别用 `xrdbench_acu:PointAdapter --kind point`、`xrdbench_acu:Adapter --kind posterior`、`xrdbench_acu:CandidateAdapter --kind candidates` 接入 `xrdbench run`。单点取 256 次抽样的边缘中位数，与组件给后验计算点误差时使用的是同一个值。候选集合抽 64 组，按各参数以先验半宽归一化后到样本中位数的绝对距离之和排序，并提交首位作为点预测。候选排序只使用模型的预测样本。

| 格式 | MDN 联合平均相对误差 | CVAE 预热联合平均相对误差 | MDN 首选命中率 | CVAE 预热首选命中率 |
|---|---:|---:|---:|---:|
| 单点 | 0.697% | 0.547% | 41.05% | 58.68% |
| 256 后验样本 | 0.697% | 0.547% | 41.05% | 58.68% |
| 64 候选及首选 | 0.705% | 0.556% | 40.20% | 57.86% |

命中要求 a、c 的相对误差都不超过 0.5%。64 个候选中至少有一组命中的比例是 MDN 91.89%、预热 CVAE 98.19%，该指标借助测试真值检查候选池。256 样本的 Energy Score 分别为 0.07862 和 0.06158。 [三种格式的分域指标](results/xrdbench/report_three_formats/formats.csv)

### 同一盲测集的网格对照

组件内置的 `reference-grid` 需要已公开的仪器参数。在同一盲测 pilot 上，[`xrdbench_blind_grid.py`](xrdbench_blind_grid.py) 在 21×21 个 (a, c) 格点上对计数谱评分。每个格点用 32 组从公开仪器参数先验抽出的设置平均似然，再从加权网格抽取协议所需的 256 个预测样本。已知的观测掩码决定使用哪个公开先验。此对照给出了近似网格结果，尚未做网格和仪器参数积分的收敛验证。

| 方法 | 联合平均相对误差 ↓ | 0.5% 联合成功率 ↑ | Energy Score ↓ | 90% 区间实际覆盖率 |
|---|---:|---:|---:|---:|
| 盲测近似网格 | 0.266% | 86.52% | 0.03423 | 16.4% |
| CVAE 预热 | 0.547% | 58.68% | 0.06158 | 88.7% |
| MDN | 0.697% | 41.05% | 0.07862 | 84.3% |

区间覆盖率为 a、c 两个边缘区间的平均值，目标接近 90%。粗网格的权重集中在少数离散格点上，区间覆盖偏低。此次本地运行中，网格预计算约 9.7 秒，在线约 0.99 ms/谱，预热 CVAE 约 0.15 ms/谱。 [含网格的组件评分报告](results/xrdbench/report_with_grid/report.md)
