# GPU 排队期间的准备及后续自动运行

本流程基于用户实际提供的 Vanda `hpc gpu`、`module avail` 和 `qstat` 输出。2026-11-16 已完成本地 RTX 3070 Laptop GPU 的 nvcc 编译、单 GPU 验证和规模/block 实验，见 [本地指南](cuda-local-guide.md)。Vanda 环境尚未实测，当前已有批处理作业 1555991.stdct-mgmt-02 在排队；不要因阅读本指南再次提交同一 pilot。以下交互操作仅适用于另行获批的交互会话。

## 排队期间可以做什么

当前保留的是批处理作业，退出 SSH 不影响其排队和自动启动。登录节点可以上传/解压源码、读取文档、查询模块和队列；学校欢迎信息明确要求在计算节点编译。

```bash
cd ~/project3_cuda/DSA5208-Project3-main
ls Makefile.cuda src/mpi_cuda.cu src/cuda_kernel.cu
module avail CUDA
module avail OpenMPI
```

补充包上传并解压到同一根目录后，可以在登录节点预览自动流程：

```bash
python3 scripts/cuda_job.py --phase pilot --dry-run
python3 scripts/cuda_job.py --phase benchmark --dry-run
```

预览只打印命令，不编译、不运行 GPU、不生成结果、不申请资源。Python 需要 3.8 或更新版本。首次 GPU 验证前可以学习 `docs/cuda-guide.md` 的分域、halo 传输、双缓冲和计时定义，并先写报告设计部分；不能先填写 GPU 误差和速度结论。

## 已观察到的配置与候选模块

- `auto_free` 是启用的路由队列，目标包括 `gpu`；学校示例允许使用它而省略 `-P`。是否能提交/运行仍以实际调度结果为准。
- 当前交互作业实际记录为 1 GPU、1 CPU 核心，因此脚本默认 `make -j1`。不能把旧帮助文字中的 36 核当成这次实际 CPU 配额。
- `scripts/cuda_env_vanda.sh` 加载 `CUDA/12.1.1` 和 `OpenMPI/4.1.5-GCC-12.2.0`。这两项出现在用户提供的模块列表中；[NVIDIA CUDA 12.1.1 指南](https://docs.nvidia.com/cuda/archive/12.1.1/cuda-installation-guide-linux/index.html#system-requirements) 支持 GCC 12.2。
- 模块存在不等于计算节点已经验证成功。仍需确认实际 GPU 驱动、模块路径、`mpicxx --showme:command` 和编译/运行兼容性。脚本使用普通主机 MPI，不要求 CUDA-aware MPI。
- 新 PBS 模板申请 1 GPU、1 CPU、1 小时，没有因为批处理就假设可以使用 24 小时。改时限前应查看目标队列规则。

## 已有交互作业启动后

仅在原排队窗口已进入计算节点后执行：

```bash
hostname
echo "$PBS_JOBID"
nvidia-smi -L
cd ~/project3_cuda/DSA5208-Project3-main
source scripts/cuda_env_vanda.sh
module list
mpicxx --showme:command
g++ --version
nvcc --version
```

确认环境后执行自动试运行：

```bash
python3 scripts/cuda_job.py --phase pilot
```

流程为：GPU/编译器预检 → 在独立目录编译 → 调用 A 的串行程序生成参考场 → 完整单 GPU 正确性测试。任何一步失败都会停止，保留相应 `.stdout`、`.stderr`、`.command.json` 和 `job_status.json`。

一次 pilot 通过只表示此环境下单 GPU 测试通过，不代表多 GPU/MPI 通信正确，也不代表已有正式性能结果。不要同时在一个 GPU 分配里启动两份 pilot/benchmark。

## 后续批处理

建议先在交互作业中解决环境和编译问题，再提交这些脚本。一次成功的 `qsub` 会新建一个独立作业，不是给已有交互作业附加步骤。

```bash
cd ~/project3_cuda/DSA5208-Project3-main
# 后续独立的编译与验证作业：
qsub scripts/vanda_cuda.pbs
# 或后续正式单 GPU 规模/block 实验，按所需范围二选一：
qsub -v CUDA_PHASE=benchmark scripts/vanda_cuda.pbs
```

`pilot` 是默认模式，只编译和验证。`benchmark` 模式另外验证 16×16 block，并依次测量 32×8、16×16 两种 block；N=256、512、1024、2048、4096，K=1000，每配置一次预热、五次测量。它不是多 GPU scaling，也不包含 profile。该完整计划不保证在一小时内完成，应依据 pilot/小规模试跑的实际耗时调整计划或合法 walltime。

普通性能测量的 `validation=not_run` 会被保留；单独验证通过不等于每次大规模计时都做过逐点验证。汇总使用 Python 标准库，GPU 节点不需要 matplotlib。汇总同时匹配 A 的 serial 和 B 的 256 CPU ranks 基准；这是不同资源预算之间的时间对比，硬件/编译器差异需在报告解释。

## 结果和完整性

每次运行打印全新的 `results/runs/cuda_<job-id>_<随机后缀>` 路径。目录包含所用源码和环境脚本快照、SHA-256、独立构建目录与两个程序、每阶段日志，以及 validation/size/analysis 子目录。并发的不同作业不会覆盖同一个二进制。

`job_status.json` 只有所有阶段返回成功时才是 `complete`。超时、手动结束或节点故障可能留下 `running`、`preparing` 或 `failed`；这些都不能当作完整实验。PBS 会执行整项作业的 walltime 限制，各串行/MPI 子进程另有 120 秒超时。编译阶段的独立超时为 900 秒。外层实验脚本不再额外套一层可能遗留 MPI 子进程的超时。

批处理不需要保持提交终端连接；当前的交互作业则需要保持连接。结果及时下载备份，`/scratch` 不作为唯一保存位置。

## 本地验证范围

新增 Python 流程的测试验证：预览不执行、登录节点拒绝实际运行、失败后停止并保留日志、源码快照包含参考场、独立二进制路径不冲突。这些是 CPU 工具测试，不能代替 GPU 编译/运行。新增 shell/PBS 文件还需要计算节点上的现场环境验证。
