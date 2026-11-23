# 实验原型：XRD → (a, c, u) 与 cVAE

此目录新增 `acu_benchmark.py`，独立于原来的二维实验。模型输入一维模拟 XRD 和观测窗口掩码，输出金红石 TiO₂ 的晶胞参数 a、c（Å）和氧分数坐标 u（无量纲）。MDN 和小型 cVAE 从条件分布采样，按模拟谱误差可选物理重排；对照包含等 27 候选的 3×3×3 网格与默认 9×9×9 密集网格。cVAE 使用 CPU，可避免 AMD GPU 或 CUDA 配置；不是三维卷积模型。

## 课程使用限制

**此原型和对原 MDN 的扩展由 AI 生成。课程讲义禁止将 AI 生成代码作为学生自己的代码提交；请先与教师确认允许的使用方式。** 不应将此代码或其结果作为学生独立完成的成果提交。其他 AI 用途及受影响的部分也应按课程要求如实声明。

## 安装和运行（macOS / Linux）

在项目目录运行，Python 3.12 是本次验证使用的版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt torch
python acu_benchmark.py
```

Windows 可先执行 `.venv\Scripts\activate`，然后使用同样的 pip 和 Python 命令。PyTorch 默认 CPU；这里并不使用 MPS 或 CUDA。小规模流程检查：

```bash
python acu_benchmark.py --train 30 --validation 10 --test 5 --epochs 3 --mdn-iterations 5 --proposals 8 --grid-side 2 --dense-side 3 --output outputs/acu_smoke
```

默认实验为 600/80/40 晶胞、80 个 cVAE 训练轮次、27 个生成候选；三个窗口属于同一晶胞的不同观测，同一晶胞只出现在同一数据划分中。示例研究规模：

```bash
python acu_benchmark.py --train 1800 --validation 240 --test 160 --epochs 150 --mdn-iterations 260 --output outputs/acu_seed42
```

改变 `--seed` 可做重复实验。默认预算网格每个方向使用 `--grid-side` 指定的点数；也可用 `--grid-shape NA NC NU` 单独指定每个方向的点数。`--proposals` 必须等于预算网格的点数乘积，以保持候选数量相等。密集网格可由 `--dense-side` 指定。相同候选数不意味着运行时间相同：网格使用预计算模板，生成候选重排要在线模拟。

对照原项目的数据划分、36 个生成候选与每个方向 41 个密集网格点：

```bash
python acu_benchmark.py --train 1800 --validation 240 --test 160 --epochs 150 --mdn-iterations 260 --proposals 36 --grid-shape 3 3 4 --dense-side 41 --output outputs/acu_original_scale_seed42
```

二维原项目的 6×6 预算网格改为三维的 3×3×4；密集网格从 41²=1,681 点变为 41³=68,921 点，因此无法同时保持每轴分辨率和总网格数不变。网格覆盖和候选去重规则也与原二维脚本不同，不应直接比较两次 Top-5。150 轮 cVAE 是新设定，原项目无 cVAE 对应轮数。

## 对一条曲线生成候选

训练后，输入 CSV 必须有 `angle_deg,intensity` 两列表头。角度须与项目的 20–70°、步长 0.1° 网格严格对应，并包含选定窗口内所有角度；可只提供窗口里的点，或提供完整曲线。脚本在窗口内按最大强度归一化，生成候选并按物理谱残差排序：

```bash
python predict_acu.py your_spectrum.csv --window full --method cvae --output candidates.csv
```

`--window` 可选 `full`、`low_information`、`contrast`，`--method` 可选 `cvae` 或 `mdn`。输出每个候选的 a、c、u 和谱误差。只能加载可信的本地模型文件；模型加载使用 Python 序列化格式。推断假设样品符合训练中的金红石原型及固定参数范围，输出的多个候选不代表真实数据上的置信度或已验证结构。实验谱还需要专门的背景、峰形与标定处理。

## 输出

- `dataset.npz`：分组参数和带噪谱；`predictions.npz`：候选池和每个候选的可见谱误差。
- `mdn_weights.npz`、`cvae.pt` 和 `feature_stats.npz`：模型权重与输入标准化信息。`cvae_history.csv`：验证目标和组成项。
- `metrics.csv`：各窗口和方法的 a、c、u 的 Top-1 误差，a/c、u 和联合 Top-5 命中率，谱误差及在线时间。
- `comparison.png`：u 误差与三参数联合 Top-5；`run.json`：范围、随机种子、训练配置、离线建库与训练时间。

Top-5 同时命中阈值为 a、c 各 0.02 Å，u 为 0.002（无量纲）；是试验性评价条件，不是实验精度或公认标准。候选按各方法原顺序或谱 MSE 排序，不做候选去重。单参数和联合指标不能混为一谈。两个模型生成 27 组候选；物理重排使用同一个候选池，raw 的谱误差仅用于事后诊断，不参与原始排序。cVAE 解码均值加固定尺度的高斯噪声后裁剪到参数范围，边界可能出现堆积。cVAE 使用验证集选择检查点，尚未校准分布。

u 在 [0.2953, 0.3153] 只是围绕参考 CIF 的示例合成范围，训练中的 a、c、u 独立均匀采样；不代表真实稳定结构的分布。当前是单相、固定原型、理想化 Gaussian 峰及合成高斯噪声；u 对当前低信息单峰窗口经归一化后几乎不可识别。不可将这组结果推广到实验数据。使用强网格或优化基线及多种种子、验证数据、参数阈值与真实结构合理性后，才可讨论模型性能。
