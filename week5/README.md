# Week 5：自动微分与检查点

按 [课程讲义](week5-learning-sheet.pdf) 顺序完成四个 Part 的必做 DO / VERIFY。
这是补做实操的真实运行记录，不是到课或教师验收记录；没有执行 bonus。

## 验收结果

| 项目 | 实测 | 讲义要求 |
|---|---:|---:|
| 手写 forward 最大误差 | 2.13e-14 | < 1e-12 |
| 手写 reverse 最大误差 | 1.42e-14 | < 1e-12 |
| 中心有限差分最大误差 | 4.95e-9 | 大于两种 AD |
| 3072 输入时 forward / reverse 耗时 | 1094 倍 | > 100 倍 |
| 所有 cluster 梯度最大相对误差 | 1.95e-15 | < 1e-12 |
| 正向 traces L2 | 11.5747695036 | 11.574770，相对误差 < 1e-4 |
| 外侧 / 中央炮点峰值 | 0.6080951434 / 0.5927139735 | 匹配参考，均在 trace index 83 |
| Born / adjoint 内积两边 | 0.03484789021516374 | 相对差 < 1e-9；实测 0 |
| 反射层图像峰值 | 2.1 km，深度误差 0 | 距 2.1 km 不超过一格 |
| 四种 checkpoint 图像相对 L2 误差 | 全部 0 | < 1e-9 |
| 所有调度文件顺序 / 恢复 / 超额计数 | 全部 0 | 全部 0 |
| Marmousi 图像 L2 | 6.703774060378255e-4 | 6.7037741e-4，相对误差 < 1e-4 |
| Marmousi 保存状态 | 6 个，20,788,320 字节 | 完全匹配 |

| 额外保存槽预算 | 每炮调度前向步数 | 峰值状态数 | 保存字节 |
|---:|---:|---:|---:|
| 1 | 28,680 | 2 | 53,792 |
| 3 | 1,695 | 4 | 107,584 |
| 5 | 990 | 6 | 161,376 |
| 10 | 642 | 11 | 295,856 |
| full | 240 | 241 | 6,481,936 |

保存字节只计检查点内的两帧压力，不是进程总内存；工作区、导数临时量、接收数据和图像另外占内存。
Marmousi 从未运行 full history（每炮本来需要 4,161,128,720 字节）；程序会拒绝超过 1 GB 的 full-history 请求。

## 从头复现

所有命令从 `week5/` 执行。需要 Rust/rustup、Python 3.12、uv、curl 和 unzip。
依赖由 `uv.lock` 和 `seismic/Cargo.lock` 锁定。当前验证平台是 Apple Silicon macOS。

```bash
rustup toolchain install nightly-2026-09-05 --profile minimal --component enzyme
uv sync --locked --python 3.12
bash scripts/reproduce.sh
```

本次安装的 uv 位于 `.bootstrap/bin/uv`；若当前 PATH 没有 uv，可先执行
`export PATH="$PWD/.bootstrap/bin:$PATH"`。从干净克隆复现则先安装官方 uv。
`rust-toolchain.toml` 会选择指定 nightly。`cargo install --path seismic --locked`
安装命令 `seismic`。自动脚本复现数值结果和 matplotlib 图；三张 viewer PNG 按下面操作导出。

输入原始下载命令（数据和 `.npy` 均在 `.gitignore` 中）：

```bash
curl -fLO https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week5-inputs.zip
unzip -o week5-inputs.zip
rm week5-inputs.zip
```

许可证保留为 `MARMOUSI-LICENSE`。讲义来源：
https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/week5-learning-sheet.pdf 。
Enzyme 技能已安装到 `../.codex/skills/enzyme-setup/SKILL.md`，并在本会话直接读取执行。
原 Week 2 的规划技能包在本机未找到，因此明确保存 `PLAN.md`、先运行失败测试、再实现并验收；不声称调用了不存在的技能。

## 按讲义解释证据

**Part 1。** `src/ad.py` 按 r → a → b → c → U 逐节点调用 JAX 的 JVP/VJP，
不是用一次 `jax.grad` 代替手写链式法则。反向经过共享节点 a 时，把 c=b-a 的 -4
和 b=a² 的 8a 相加；`grad-graph.png` 的 `add_any` 就是这一步。
在 r=1.3，U=-0.6570169144600471，dU/dr=2.239979929791143，
a 的 adjoint=-2.3425903117359734。力是 -dU/dr。
`modes.png` 在 r=2^(1/6) 附近过零；误差图中的零误差仅为显示截到 1e-16。
cluster 用固定随机种子，forward 通过 `lax.map` 顺序计算每个坐标的 JVP，
reverse 一次得到所有坐标梯度。计时先编译、等待设备完成，再取多次运行中位数。
本机 reverse/energy 比约 2.0–4.3，是几个 energy 的开销；forward 随输入数大幅上升。

**Part 2。** `seismic.design.toml` 保留讲义三个绿色面板的接口约定。
`seismic/src/kernel.rs` 使用讲义式 (9) 的中心差分阻尼，而不是额外乘衰减系数。
q 在 n*dt 计算，receiver 在更新后采样；traces[83] 对应第 84 次更新。
录制则标记完整状态 u^n 的 n*dt，并含正向初始零帧。
`inputs.png` 显示三个源、十四个接收器、2.1 km 反射层和 1.5 s 脉冲峰值。
`gathers.png` 的峰值与参考一致；150 步 echo 是两次正向运行之差，
最大值 0.0062247，相对直接波 0.2552434 约 2.44%。

**Part 3。** 完整重启状态是一个长度 2*nx*nz 的向量，前半为 u^(n-1)，后半为 u^n。
这两帧和步号 n 足够重新计算固定的源项；只有一帧无法恢复二阶时间更新。
一个真实的局部 Enzyme 调用是 `Kernel::vjp` → `wave_vjp` → 编译器生成的 `step_vjp`。
`step` 的 state 和 velocity 是 active，网格、dt、源和阻尼都是 Const。
安全封装在调用 C ABI 前检查每个缓冲区的长度。
`build.rs` 仅对 no_std 内核开启 `-Zautodiff=Enable`。
冒烟测试实测 cube(2)=(8,12,12)；去掉该编译开关时编译失败，见 negative log。

Born 循环把扰动作为 velocity tangent，每步一次 JVP；逆循环先向新压力的 adjoint
注入 receiver weights（重复 receiver 会累加），再一次 VJP，同时累加对 c 的灵敏度。
这给出 d=Jm 和 image=Jᵀd。不能用单纯的正反压力乘积替代局部导数：
它会漏掉 2c、Laplacian、dt² 和阻尼分母等链式法则因子。
`adjoint/checks.json` 核验内积，`image.png` 的指定窗口 row L2 峰值为 2.1 km。
另外测试了随机 weights 的转置关系及整条轨迹有限差分，避免只靠 w=Jm 的单例自检。

**Part 4。** `seismic/src/reverse.rs` 直接移植
[TreeverseAlgorithm.jl 的 treeverse! 和 mid](https://github.com/GiggleLiu/TreeverseAlgorithm.jl/blob/master/src/treeverse.jl)，
包括端点修正、每次递归最后 reverse 首步及保留初始状态的规则。
预算 5 的动作图显示锯齿状 replay；每个 `grad` 的输入都实际保存在槽中。
`restore` 复制已有完整状态，`call` 只调用 primal `Kernel::step`，不访问 `image`；
只有 `grad` 中的 VJP 累加 image。因此一次 replay 不改变已累加的图像。
四个预算最终逐元素相同的结果也验证了这一点。小测试同时验证讲义 N=6、预算=2
的例子确实使用 8 次前向推进和 3 个状态，以及 N=1 的端点。
调度器统计不计 Enzyme 在 VJP 内部重算的单步成本。

`marmousi.png` 用深度无关的统一幅度标尺，显示 8–14 km、上部 3 km 窗口内的倾斜条带。
浅层受到源和接收器照明，深层信号弱。精确求导保证计算的是离散模型的 JᵀJm，
并不保证 JᵀJ 是恒等矩阵；有限带宽、有限接收覆盖和照明不足使 image 与原扰动 m
在宽度、旁瓣和幅度上不同。这是成像分辨率限制，不是自动微分误差。

## 课程 viewer 操作

打开 https://giggleliu.github.io/AMAT5315-2026Fall/week5-viewer.html ，用文件选择器
同时选数组和同目录 `run.json`，点击 Play 查看传播，再选指定帧，点击 **Save PNG**：

| 输入数组 | slider frame 值（从 0 起） | 步号 / 时间 | 保存为 |
|---|---:|---|---|
| `artifacts/forward/wavefield.npy` | 50 | 150 / 3.00 s | `artifacts/forward/wavefield.png` |
| `artifacts/forward/echo.npy` | 50 | 150 / 3.00 s | `artifacts/forward/echo.png` |
| `artifacts/adjoint/wavefield.npy` | 35 | 132 / 2.64 s | `artifacts/adjoint/wavefield.png` |

本次三张图均实际从课程 viewer 的 Save PNG 导出，包含文件名、步号、时间和数值读数；
不是 matplotlib 仿制。紧凑 run.json 不含速度数组，因此 viewer 显示 velocity hidden，
几何信息另见 `inputs.png`。伴随录制按时间递减顺序播放。

## 检查命令

```bash
cargo test --release --manifest-path seismic/Cargo.toml --locked
.venv/bin/pytest -q tests
.venv/bin/python scripts/forward_evidence.py
.venv/bin/python scripts/adjoint_evidence.py
.venv/bin/python scripts/checkpoint_evidence.py
.venv/bin/python scripts/verify_inventory.py
```

最后一个命令打开全部 JSON/PNG、核对所有本地数组的 shape/dtype/有限值、元数据和文件大小。
`artifacts/tests/red.log` 是实现前的预期失败，`green.log` 和 `trajectory.log` 是通过记录。
PNG 已逐图检查。下一次上课可用本目录展示数值检查、存储–重算曲线和地下成像；
讲义明确没有线上提交，教师现场查看需由学生下次上课完成。

## 完整证据清单与生成命令

下表路径相对于 `week5/`。`.npy` 行对应本地保留、没有提交的数组。
命令缩写：

- A：`.venv/bin/python scripts/ad_evidence.py`
- F：`seismic --experiment inputs/reflector.json --mode forward --every 3 --out artifacts/forward`
- FP：`.venv/bin/python scripts/forward_evidence.py`
- B：`seismic --experiment inputs/reflector.json --mode born --out artifacts/born`
- R：`seismic --experiment inputs/reflector.json --mode adjoint --data artifacts/born/born_data.npy --every 3 --out artifacts/adjoint`
- RP：`.venv/bin/python scripts/adjoint_evidence.py`
- C(b)：`seismic --experiment inputs/reflector.json --mode adjoint --data artifacts/born/born_data.npy --storage treeverse --checkpoints b --out artifacts/checkpoint-b`
- MB：`seismic --experiment inputs/marmousi.json --mode born --out artifacts/marmousi-born`
- MR：`seismic --experiment inputs/marmousi.json --mode adjoint --data artifacts/marmousi-born/born_data.npy --storage treeverse --checkpoints 5 --out artifacts/marmousi-image`
- CP：`.venv/bin/python scripts/checkpoint_evidence.py`
- V：上面的课程 viewer 导出步骤。

| 文件 | 生成命令 | 保存方式 |
|---|---|---|
| `artifacts/ad/checks.json` | A | 提交 |
| `artifacts/ad/derivatives.json` | A | 提交 |
| `artifacts/ad/grad-graph.png` | A | 提交 |
| `artifacts/ad/graph.png` | A | 提交 |
| `artifacts/ad/modes.png` | A | 提交 |
| `artifacts/ad/scaling.png` | A | 提交 |
| `artifacts/ad/verification.log` | A（stdout 重定向） | 提交 |
| `artifacts/adjoint/checks.json` | RP | 提交 |
| `artifacts/adjoint/image.npy` | R | 本地数组 |
| `artifacts/adjoint/image.png` | RP | 提交 |
| `artifacts/adjoint/result.json` | R | 提交 |
| `artifacts/adjoint/run.json` | R | 提交 |
| `artifacts/adjoint/wavefield.npy` | R | 本地数组 |
| `artifacts/adjoint/wavefield.png` | V | 提交 |
| `artifacts/born/born_data.npy` | B | 本地数组 |
| `artifacts/born/result.json` | B | 提交 |
| `artifacts/born/run.json` | B | 提交 |
| `artifacts/checkpoint-1/actions-0.json` | C(1) | 提交 |
| `artifacts/checkpoint-1/actions-1.json` | C(1) | 提交 |
| `artifacts/checkpoint-1/actions-2.json` | C(1) | 提交 |
| `artifacts/checkpoint-1/image.npy` | C(1) | 本地数组 |
| `artifacts/checkpoint-1/result.json` | C(1) | 提交 |
| `artifacts/checkpoint-1/run.json` | C(1) | 提交 |
| `artifacts/checkpoint-10/actions-0.json` | C(10) | 提交 |
| `artifacts/checkpoint-10/actions-1.json` | C(10) | 提交 |
| `artifacts/checkpoint-10/actions-2.json` | C(10) | 提交 |
| `artifacts/checkpoint-10/image.npy` | C(10) | 本地数组 |
| `artifacts/checkpoint-10/result.json` | C(10) | 提交 |
| `artifacts/checkpoint-10/run.json` | C(10) | 提交 |
| `artifacts/checkpoint-3/actions-0.json` | C(3) | 提交 |
| `artifacts/checkpoint-3/actions-1.json` | C(3) | 提交 |
| `artifacts/checkpoint-3/actions-2.json` | C(3) | 提交 |
| `artifacts/checkpoint-3/image.npy` | C(3) | 本地数组 |
| `artifacts/checkpoint-3/result.json` | C(3) | 提交 |
| `artifacts/checkpoint-3/run.json` | C(3) | 提交 |
| `artifacts/checkpoint-5/actions-0.json` | C(5) | 提交 |
| `artifacts/checkpoint-5/actions-1.json` | C(5) | 提交 |
| `artifacts/checkpoint-5/actions-2.json` | C(5) | 提交 |
| `artifacts/checkpoint-5/image.npy` | C(5) | 本地数组 |
| `artifacts/checkpoint-5/result.json` | C(5) | 提交 |
| `artifacts/checkpoint-5/run.json` | C(5) | 提交 |
| `artifacts/checkpoint-actions.png` | CP | 提交 |
| `artifacts/checkpoint-checks.json` | CP | 提交 |
| `artifacts/checkpoint-verification.log` | CP（stdout 重定向） | 提交 |
| `artifacts/checkpoint-work.png` | CP | 提交 |
| `artifacts/forward/checks.json` | FP | 提交 |
| `artifacts/forward/echo.npy` | F | 本地数组 |
| `artifacts/forward/echo.png` | V | 提交 |
| `artifacts/forward/gathers.png` | FP | 提交 |
| `artifacts/forward/result.json` | F | 提交 |
| `artifacts/forward/run.json` | F | 提交 |
| `artifacts/forward/traces.npy` | F | 本地数组 |
| `artifacts/forward/wavefield.npy` | F | 本地数组 |
| `artifacts/forward/wavefield.png` | V | 提交 |
| `artifacts/inputs.png` | FP | 提交 |
| `artifacts/inventory-check.log` | `.venv/bin/python scripts/verify_inventory.py`（stdout 重定向） | 提交 |
| `artifacts/marmousi-born/born_data.npy` | MB | 本地数组 |
| `artifacts/marmousi-born/result.json` | MB | 提交 |
| `artifacts/marmousi-born/run.json` | MB | 提交 |
| `artifacts/marmousi-checks.json` | CP | 提交 |
| `artifacts/marmousi-image/actions-0.json` | MR | 提交 |
| `artifacts/marmousi-image/actions-1.json` | MR | 提交 |
| `artifacts/marmousi-image/actions-2.json` | MR | 提交 |
| `artifacts/marmousi-image/actions-3.json` | MR | 提交 |
| `artifacts/marmousi-image/actions-4.json` | MR | 提交 |
| `artifacts/marmousi-image/actions-5.json` | MR | 提交 |
| `artifacts/marmousi-image/actions-6.json` | MR | 提交 |
| `artifacts/marmousi-image/actions-7.json` | MR | 提交 |
| `artifacts/marmousi-image/actions-8.json` | MR | 提交 |
| `artifacts/marmousi-image/image.npy` | MR | 本地数组 |
| `artifacts/marmousi-image/result.json` | MR | 提交 |
| `artifacts/marmousi-image/run.json` | MR | 提交 |
| `artifacts/marmousi.png` | CP | 提交 |
| `artifacts/tests/enzyme-negative.log` | 按 Enzyme 技能编译 kernel，但不传 `-Zautodiff=Enable`（预期失败） | 提交 |
| `artifacts/tests/green.log` | `cargo test --release --manifest-path seismic/Cargo.toml --locked` | 提交 |
| `artifacts/tests/red.log` | 实现前 `cargo test --manifest-path seismic/Cargo.toml`（预期失败） | 提交 |
| `artifacts/tests/trajectory.log` | `.venv/bin/pytest -q tests` | 提交 |
