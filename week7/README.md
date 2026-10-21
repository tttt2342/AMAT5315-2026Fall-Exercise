# Week 7：Diffusion and flow matching

讲义：[week7-learning-sheet.pdf](week7-learning-sheet.pdf)，下载自
https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/week7-learning-sheet.pdf 。
原文提取保存在 `week7-learning-sheet.txt`，未改动题目规格。

全部命令从 `week7/` 执行。环境：Python 3.12、uv、CPU、float32。

```sh
../week5/.bootstrap/bin/uv sync --locked
export JAX_PLATFORMS=cpu
export MPLCONFIGDIR=artifacts/mpl
```

系统已安装 uv 时可直接使用 `uv`。`inputs/` 是下载，`artifacts/` 是原始数组、模型检查点、训练曲线和日志；两者及 `.venv/` 均不进 Git。

## Part 1：DDPM

```sh
uv run --locked python scripts/mixture.py train-noise
uv run --locked python scripts/mixture.py part1
uv run --locked python -m unittest discover -s scripts -p test_transport.py -k test_part1 -v
```

前两条命令生成本节全部证据。

![Forward diffusion](evidence/forward.gif)
![Training stages](evidence/training.gif)
![Reverse diffusion](evidence/reverse.gif)
![Generated distribution](evidence/ddpm_distribution.png)
![Training history](evidence/training.png)

四个中心是 (3,0)、(0,3)、(-3,0)、(0,-3)，等权且协方差 0.09I。
使用 1000 步线性 β∈[0.0001,0.02]，DDPM 方差 β，最终步不加噪，全程不裁剪。
前向动画跟踪同一批 2000 个点逐步加噪，并保持坐标尺度；t=1000 的均值为 (0.0186,0.0107)，方差为 (1.0119,0.9828)。

网络为 34→128→128→128→2，三个 SiLU 隐藏层；输入拼接两个坐标与16对正弦/余弦时间特征。
每次用512个新混合样本，t在1至1000均匀抽样，最小化每坐标噪声 MSE。
Adam 初始学习率0.001，25000次更新，最后一次学习率为0。初始化种子7101；训练随机流7201。
保存0、100、250、500、1000、2500、5000、10000、17500、25000更新的10个阶段。
详见 [noise_training.json](evidence/noise_training.json)。

训练动画中每阶段固定权重，重复同一10个高斯起点和全部反向噪声；阶段之间只改变权重。
初始轨迹最大坐标绝对值约3351，超出固定显示范围；最终端点围绕目标簇。
最后1000次平均损失0.2463。最优回归给出 E[ε|x_t,t]，无法知道一次加噪的真实ε，因此条件方差使损失不必为0。

生成2000个点，簇比例为0.2685、0.2530、0.2330、0.2455；最大中心欧氏偏差0.02215，最大宽度误差0.01207。
比例、中心、宽度三个验收均通过。全部中心与逐坐标标准差见 [part1.json](evidence/part1.json)。
只检查散点图或比例并不足够：正确比例也可能掩盖中心偏移或簇宽度错误。
