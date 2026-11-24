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

[`xrd_inverse`](../xrd_inverse/README.md) 使用固定氧坐标的金红石模拟谱，评估晶格参数 a、c。盲测提供 XRD 计数、掩码与曝光，仪器参数保持未知。数据按晶胞划分为训练、验证和测试。

当前接入使用卷积 MDN 和卷积 CVAE。两者使用相同结构的谱编码器，并分别训练。组件支持单点、256 个原始后验样本、64 个候选及其首选三种提交方式。详细参数、结果、组员基线对照和复跑命令见 [主 README](README.md#当前版本结果)。

正式 rutile 盲测联合 pilot 的数据指纹为 `73a358b50688e514a8d7f95d45069edc70a5c23bcbd2ab5b9127cef3b62111c3`。预测使用训练集训练、验证集选检查点，由组件对测试集评分。 [完整结果](results/xrdbench_official/shared_conv/comparison.csv) · [数据版本](results/xrdbench_official/dataset.json)
