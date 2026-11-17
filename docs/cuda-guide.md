# C 同学交付：MPI + CUDA

这份代码在最新 A/B 源码上添加 C 的实现。A 的 `serial.cpp`、`heat_common.h`、`validation.h`，B 的 MPI 实现和通信接口均保持原样。独立的 `Makefile.cuda` 供 A 后续合并到总构建文件。

## 当前验证状态

- 已完成 CUDA 源码、MPI 集成、串行 CSV 对照、实验采集和绘图工具。
- 2026-11-16 已建立 Windows + Docker/WSL2 本地环境，在 RTX 3070 Laptop GPU 上成功编译，并通过 32×8 block 的单 GPU pilot：10 个逐点/解析验证用例通过，6 个负面用例按预期失败。
- 本地 pilot 标识为 `local_pilot_20261116_091046_5fc69fe6`；结果在 `D:\DSA5208-CUDA-results`，最大绝对误差约 5.55e-17，不能标为 Vanda 数据。后续性能结果以各次完整运行日志为准。
- Vanda 实测和多 GPU 路径仍待验证；CPU 辅助检查与单 GPU 验证都不能证明多 GPU halo 通信正确。
- 原来空的 `results/mpi_cuda.csv` 和 `cuda_scaling.png` 不是运行产物。运行入口与数据来源说明见 [本地指南](cuda-local-guide.md)。

## 文件与接口

| 文件 | 用途 |
|---|---|
| `include/cuda_solver.h`, `src/cuda_kernel.cu` | GPU 双缓冲、二维五点差分、固定页主机 halo 缓冲、可选 event 计时 |
| `src/mpi_cuda.cu` | 参数广播、rank/GPU 映射、B 的 `exchange_rows`、收集、验证、计时 |
| `include/cuda_host_utils.h` | CPU 独立参考、公共验证器适配、CSV 输入输出；不依赖 CUDA/MPI |
| `Makefile.cuda` | 构建 CUDA 程序、A 的串行程序及 CPU 辅助测试 |
| `scripts/test_cuda.py` | 在真实 GPU 上逐点比对 A 的串行输出，含失败用例 |
| `scripts/benchmark_cuda.py` | 预热、重复计时、原始记录和运行元数据 |
| `analysis/cuda_analysis.py` | 实测汇总、强弱扩展、时间/效率/诊断图 |

统一使用 `double`、`r=0.1`、0/1 归一化温度、固定零物理边界、恰好 K 步。C 在 host 上用 `heat::initial_value(global_row,j,N)` 初始化拥有行。A 的 `initialize_grid` 遍历完整 N×N，不能直接传入局部网格。

每个 rank 使用一张 GPU。节点内共享 GPU 可见列表时按节点内 rank 选卡；每个 rank 只看得到自己分配的卡时选择可见编号 0。程序交换 GPU UUID，拒绝同节点两个 rank 误用同一张卡。不要设置超额分配或让一个 GPU 同时对应多个 rank。

多 rank 每步执行：设备边界行复制到 host -> B 的 `exchange_rows` 完成收发 -> 真实邻居 halo 复制回设备 -> kernel 更新 -> 交换两块设备缓冲。只传每侧一行，整个场常驻设备。单 rank 没有邻居，跳过这些 MPI/halo 复制。

基线使用 host-staged MPI，无需 CUDA-aware MPI。B 的接口只接受主机指针，不能传 `cudaMalloc` 地址。共享内存 tiling、通信计算重叠和直接设备 MPI 均不是此版本已实现的优化。

## 在 Vanda 第一次运行

1. 使用已开通的登录账号登录 Vanda。在登录节点查看 `hpc project`、`hpc gpu` 和 `hpc gstat`。SSH 成功不代表个人项目映射已验证；遇到 project not found 应跟进 IT 工单。
2. 按账号显示的课程项目和 GPU 示例申请小规模计算资源，建议先一张 GPU。队列、project、CPU/内存配比以实际输出为准。本仓库没有替你提交作业，也没有猜测队列名。
3. 在获分配的 GPU 计算节点检查 `hostname`、`nvidia-smi -L`、`module list`，加载匹配的 CUDA Toolkit、C++ 编译器与 MPI 模块。课件列出的模块要以现场可用版本为准。
4. 进入包含 `SPEC.md` 的项目目录，执行：

```bash
command -v nvcc mpicxx mpirun python3
nvcc --version
mpicxx --version
mpirun --version
make -f Makefile.cuda show-config
make -f Makefile.cuda -j4 all serial
```

`CUDA_HOME` 默认从 `nvcc` 的实际路径查找，`CUDA_LIBDIR` 查找 cudart；Open MPI 头文件选项来自 `mpicxx --showme:compile`。若环境不同，可显式设置 `CUDA_HOME`、`CUDA_LIBDIR`、`MPICXX`、`NVCC`、`MPI_CPPFLAGS`、`CUDA_HOST_CXX`。host 编译器必须受所选 CUDA 版本支持，并和 MPI wrapper ABI 兼容。需要指定架构时用 `CUDA_ARCH=sm_XX`，XX 必须和实际 GPU/编译器支持情况匹配。

首次编译后，先跑一个很小的测试。下面命令在**已分配的单 GPU 作业内**执行：

```bash
mpirun -np 1 ./heat_mpi_cuda --n 8 --steps 1 --validate
./heat_serial --n 64 --steps 100 --validate --output serial_64_100.csv
mpirun -np 1 ./heat_mpi_cuda --n 64 --steps 100 \
  --reference serial_64_100.csv --output cuda_64_100.csv
```

`--validate` 使用 C 的独立 CPU stencil，并调用 A 的 `heat::compare_results`。`--reference` 接受 A 输出的 N×N、无表头 CSV，并自动启用验证。两条路径都另查 finite、严格零边界、[0,1] 范围、K=0 初始化和 N=8/K=1 解析值。参考计算、收集和文件输出均在迭代计时外。

## GPU 正确性验证

```bash
python3 scripts/test_cuda.py --allocated-gpus 1 --ranks 1 \
  --output results/runs/cuda_validation_single
```

默认测试零步、单步解析值、N=17、奇偶步数、较长迭代，并自动生成 A 的串行 CSV。测试脚本还导出 GPU 完整场，由 Python 再比较一次，并验证错误参数/错误参考被拒绝。目录必须尚不存在，以防覆盖旧证据。

获得足够的 GPU 后，按实际资源增加 `--ranks 1 2 3` 等配置，并设置真实的 `--allocated-gpus`。一个进程需要一张独占 GPU；每节点 rank 数也必须匹配资源分布。3 ranks 检查不均匀分域；4/8 ranks 可覆盖小网格两行/一行子域。没有资源的配置应在报告里标为未测试，不能以单 GPU 结果代替跨节点验证。调度器需要特殊启动/绑定参数时传 `--launcher-args='现场确认的参数'`。

Windows 本地只预览计划，不启动 GPU 或申请资源：

```text
python scripts/test_cuda.py --allocated-gpus 1 --output preview --dry-run
```

## 性能实验

首次可只用 1 GPU 做规模实验。N 和 K 是起始方案，要按试跑时间、显存和作业时限调整：

```bash
python3 scripts/benchmark_cuda.py --mode size --sizes 256 512 1024 2048 4096 \
  --steps 1000 --ranks 1 --allocated-gpus 1 \
  --output results/runs/cuda_size_b32x8
```

每个配置先做独立小网格正确性 gate，然后一次预热、五次新进程测量；所有测量均从原始初值开始。不剔除慢值。`raw.csv` 保留预热和全部重复，`summary.csv` 给中位数和 min/max。默认大网格测量的 `validation=not_run` 会如实保留；小网格 gate 通过不代表每个大网格已逐点验证。如需每次额外执行串行验证，可用 `--validate-measurements`，其总作业时间会明显增加。

支持以下实验，分别保存到新目录：

- `--mode strong --n 4096 --steps 1000 --ranks 1 2 ...`：固定全局网格和步数。
- `--mode weak --n 1024 --steps 1000 --ranks 1 4 ...`：N≈1024√P，保持每 GPU 工作量近似固定。
- 同一规模实验分别用 `--block-x 32 --block-y 8` 与 `--block-x 16 --block-y 16`，比较 kernel block 选择。

省略号是说明，不能原样复制。`--allocated-gpus` 必须填实际获得的总 GPU 数，不是希望申请的数量。启动时会拒绝 rank/GPU 数超出声明值，设备 UUID 检查还会检测实际节点内重用。

整体耗时为各 rank 完成 K 次更新的最大墙钟时间，包括每步 host-device halo 复制、MPI 通信、kernel、必要等待及交换；排除分配、初始化、首次上传、最终下载、收集、验证和输出。结束计时前明确等待 GPU 完成。该时间是迭代时间，不是含 MPI 启动的程序端到端时间。

## 诊断 kernel / 传输 / MPI 开销

在独立目录运行相同实验并添加 `--profile`。或直接执行：

```bash
mpirun -np 1 ./heat_mpi_cuda --n 1024 --steps 1000 \
  --profile-output cuda_profile_single.csv
```

诊断 CSV 每个 rank 一行，记录 kernel、D2H、H2D、MPI 及整体迭代时间。CUDA 阶段使用 event；MPI 阶段用墙钟。profile 增加阶段同步，必须和正常 benchmark 分开分析，不能拿它替换正常 scaling 时间。

这里只计**迭代内 halo 传输**，不计初始/最终整场传输。单 GPU 没有 halo 通信，D2H/H2D/MPI 为零是正常的。各阶段有未计入的 host 调用/同步开销，不能强求阶段之和等于总时间，也不能把不同 rank 的各阶段最大值相加冒充同一个 rank 的分解。

## 汇总和报告

绘图需要 matplotlib；汇总 CSV 可以只使用 Python 标准库：

```bash
python3 analysis/cuda_analysis.py --input results/runs/cuda_size_b32x8 \
  --output results/cuda_analysis_b32x8 --summary-only
# 有 matplotlib 时移除 --summary-only，选择另一个尚不存在的输出目录。
```

实际 GPU scaling 的基准是 1 rank/1 GPU。默认 `cuda_job.py` 的本地和 PBS 汇总都只分析本次作业新产生的 GPU 测量，不再自动读取旧 CPU 性能 CSV；源码快照只保留 host-test 所需的两个数值参考场，不依赖历史性能文件。旧 `serial_scaling_clean.csv` 缺少完整平台记录，已从干净交付包排除。

需要额外 CPU/GPU 对比时，可显式传 `--serial-csv PATH`，但输入须是已经确认来源、计时口径、N/K/精度相符的独立串行测量。该分析器只检查 CSV 配置，不会自动证明两份数据来自同一机器。当前已完成的本机配对实验位于 `results/local_20261116/`，用 `analysis/local_comparison_analysis.py` 重画其三张图；`comparison_repetitions.csv` 同时包含串行和 GPU 行，不能直接当成只接受纯串行行的 `--serial-csv` 输入。

如确实需要跨资源对比，可另外显式传 `--mpi-csv results/mpi_strong.csv --mpi-ranks 1`（或明确选择 8、256 等 CPU rank 数）。B 的此文件来自 Vanda；和本机 GPU 相除只能得到跨系统时间比，不能用来隔离 GPU 加速效果。分析器只匹配 N 和 K，所选 CPU 资源预算及平台差异必须另外解释。B 的 256 核 MPI 数据及四进程异常继续保留。

脚本保留二进制哈希、源码文件哈希、实际启动命令、PBS 作业信息、模块、编译器、启动节点硬件和每个 rank 的设备映射。ZIP 快照若位于其他 Git 仓库内，不会将父仓库 SHA 冒充源码提交号。多节点硬件信息仍需按现场补充保存。

报告对应 5.1 设计、5.2 验证、5.3 GPU 性能、5.4 同机 CPU/GPU 比较、6.2 GPU 加速与局限。当前已有本地误差表、规模/block 性能图及五个尺寸的配对串行/GPU重复测量；N=4096、K=1000 的求解循环中位时间比为约41.56。这个结果不含计时区间外的初始化及整场传输，不是端到端程序加速比。Vanda GPU 和多 GPU 数据仍未实测。个人 AI 声明应反映实际辅助编码、执行、分析和人工审阅情况。

## 本地可执行检查

```bash
make -f Makefile.cuda host-test
python3 -m unittest discover -s tests -p test_cuda_tools.py -v
```

CPU 测试包括独立解析值、公共容差与严格物理边界的区别、CSV 形状/非法值、A 提供的两个真实串行参考场。Python 测试中的合成计时只用于统计/解析器单元测试，不是 GPU 实测。

## 技术依据

- NVIDIA CUDA Runtime API: https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
- NVIDIA nvcc compilation: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
- Open MPI communicator split: https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Comm_split_type.3.html
- Open MPI variable-size gather: https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Gatherv.3.html
