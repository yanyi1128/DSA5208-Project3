# B 阶段性交付说明：供 A、C 开发与集成使用

更新日期：2026-11-11。

## 1. 交付结论与边界

B 已完成可运行的 CPU MPI 求解器，以及可复用的按行分块、邻居定位和 Halo 交换模块。A 可以独立继续串行实现、公共工具和项目整合；C 可以开始 CUDA 与 MPI 通信集成，无需等待 B 的正式性能实验结束。

这不是全部项目完成声明，也不是所有公共功能均已封装的声明：初始化、全局收集、串行参考和误差验证目前仍在 `src/mpi.cpp` 内部，不能直接作为外部库函数调用。C 可参考其实现，配合 A 的公共工具接入自己的程序；如果需要直接复用这些函数，应先与 B 协调提取，避免各自维护不一致的副本。

算法与行为约定以 `SPEC.md` 为准，完整人员分工以团队已共享的分工 PDF 为准。

## 2. 文件清单

| 文件 | 用途 |
|---|---|
| `SPEC.md` | 数学问题、数据布局、通信顺序、输入输出与计时规则 |
| `src/mpi.cpp` | CPU MPI 可执行程序，包含 main、初始化、迭代、计时及独立参考验证 |
| `include/mpi_interface.h` | C 可以调用的公共 MPI 接口及缓冲区约定 |
| `src/mpi_interface.cpp` | 公共 MPI 接口实现，不包含 main 或 CUDA 依赖 |
| `tests/mpi_interface_test.cpp` | 分离主机缓冲区、单行子域、重复交换和重排通信器测试 |
| `scripts/test_mpi.ps1` | Windows 上的完整 CPU MPI 正确性用例 |
| `scripts/run_mpi.sh`、`scripts/benchmark_mpi.py` | 性能数据采集，预热一次、测量至少五次 |
| `scripts/vanda_mpi.pbs`、`scripts/submit_vanda_256.sh` | Vanda 正式实验作业主体与提交入口 |
| `docs/mpi-handoff.md` | 英文接口补充说明 |

不要把 Windows 的 exe 上传到集群运行；在目标平台重新编译。结果目录与二进制被 Git 忽略，拉取源码不会自动获得 B 的集群实验数据。

## 3. 给 A：已经可以独立开展的工作

- 按 SPEC 实现正式串行程序、初始化、验证工具及串行实验。
- 使用本说明中的命令编译、运行 B 的 MPI 版本，纳入项目构建和 README。
- 后续用相同 N、K、初始化及边界条件交叉核对串行、MPI、MPI+CUDA 的完整结果。

MPI 当前自带的 `serial_reference()` 是 B 独立验证使用的参考实现，不代表 A 的正式串行实现已经完成，也不要求 A 将其作为自己的性能基准。A 的串行实验应独立计时，不能把 MPI 的参考验证耗时当作串行性能数据。

当前统一构建的 Makefile 和总 README 尚未完成；本文提供的是 MPI 部分可直接使用的命令。SPEC 中的 `--output PATH` 尚未实现，所以目前不能通过该参数导出 MPI 温度网格，也不能仅比较程序打印的校验指标来替代三版本逐点验证。

## 4. A、C 都必须保持一致的数值规则

- 网格为 N×N，包含物理边界，N≥4；四周始终为 0。
- 使用 double；温度为归一化数值，1 对应原设定的 100°C，0 对应 0°C。
- 中心热区：行、列均在 `[N/4, (3*N)/4)`，端点使用整数除法；计算 3*N 时避免整数溢出。
- 扩散系数 r=0.1，执行恰好 K 次五点差分更新。
- 双缓冲：只读 current，写入 next，再交换；交换 K 次后的 current 是最终结果。
- 连续一维行优先存储，索引乘法使用足够宽的类型。
- 初始化时两数组全部清零，只向 current 的实际拥有行写初始热区。

## 5. 给 C：可复用接口

在源文件中包含 `mpi_interface.h`，编译时添加 `-Iinclude`。公共接口位于 `heat_mpi` 命名空间。

```cpp
MPI_Comm comm = MPI_COMM_WORLD;
// 调用前必须已执行 MPI_Init，所有进程使用相同的 N。
auto domain = heat_mpi::make_domain(N, comm);
```

| 字段/函数 | 含义 |
|---|---|
| `domain.rows` | 本进程实际拥有的行数，不含 Halo |
| `domain.start` | 第一行拥有数据的全局行号，从 0 开始 |
| `domain.upper`、`domain.lower` | 同一通信器中的上下邻居，无邻居时为 MPI_PROC_NULL |
| `decompose(N, rank, size)` | 纯分块计算，可用于构造各进程的收集计数和位移 |
| `make_domain(N, comm)` | 查询通信器中的 rank/size 并计算本进程分块 |
| `exchange_halos(current, N, domain, comm)` | 连续 CPU 网格的 Halo 交换适配器 |
| `exchange_rows(top, bottom, recv_top, recv_bottom, N, domain, comm)` | 四个独立 CPU 行缓冲区的交换接口，供 CUDA 主机暂存路径使用 |

要求 `1 <= P <= N`，包括对物理上下边界行的分块。N=10、P=3 时，拥有行数为 4、3、3，起始全局行为 0、4、7。

每个本地网格分配 `(domain.rows+2)*N` 个 double：第 0 行是上 Halo，第 1..rows 行是真实拥有行，第 rows+1 行是下 Halo。局部行 ell 对应全局行 `domain.start+ell-1`；全局行 0 和 N-1 不更新。

### CUDA 每轮调用顺序

1. 等待 GPU 对旧 current 的写入完成。
2. 将 current 的第一行和最后一行拥有数据复制到两个主机发送缓冲区，各 N 个 double；复制完成后才调用 MPI。
3. 调用：

```cpp
heat_mpi::exchange_rows(host_top, host_bottom,
                       received_top, received_bottom,
                       N, domain, comm);
```

4. 对真实邻居，把接收行复制到 GPU 的相应 Halo；无邻居时保持外侧 Halo 为零。
5. 确保复制完成或通过同一 CUDA stream 保证先后顺序，再启动更新 next 的 Kernel。
6. 交换设备缓冲区，下一轮继续使用最新 current。

接口接收的是 CPU 可访问指针，不能直接传 cudaMalloc 得到的设备指针。主机接收缓冲区初始清零；两个接收缓冲区互不重叠，也不能与发送区重叠。单行子域允许两个发送指针相同。

接口内部先发起接收，再发起发送，并等待四个请求全部完成后返回。因此返回时发送缓冲区可重用，接收数据可读取。接口不负责分配/释放内存、CUDA 同步、MPI_Init/MPI_Finalize，也不执行全局 barrier。

消息标签 100、101 为此通信预留。所有参与进程必须使用一致通信器和迭代顺序，不能在同一通信器中并发启动使用相同标签的其他交换。MPI 调用失败时模块会 MPI_Abort；分块参数非法会抛异常，调用方需统一处理，避免其他进程阻塞。

### 编译链接边界

C 应链接 `src/mpi_interface.cpp` 编译出的对象文件。不要把整个 `src/mpi.cpp` 链入 CUDA 可执行程序：它有自己的 main，并且内部函数不是公共接口。

```bash
mpic++ -std=c++17 -O2 -Iinclude -c src/mpi_interface.cpp -o mpi_interface.o
```

CUDA 对象与该对象文件的最终链接需加入匹配的 MPI 和 CUDA 库，具体参数由 C 根据其构建方式确定；以上命令只编译 CPU MPI 模块，不是完整 CUDA 构建命令。

## 6. 编译与复现

以下命令在仓库根目录执行。集群上的 mpirun 测试必须在已分配的计算资源内运行。

### Vanda/Linux

```bash
module load ucx/1.10.1
module load openmpi/gcc/64/4.1.8
export LD_LIBRARY_PATH="/opt/pbs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mpic++ -std=c++17 -O2 -Wall -Wextra -Wpedantic src/mpi.cpp src/mpi_interface.cpp -o heat_mpi
mpirun -np 3 ./heat_mpi --n 17 --steps 7 --validate
mpirun -np 2 ./heat_mpi --n 64 --steps 100 --validate
```

库路径用于解决此前观察到的 PBS 动态库加载问题，UCX 模块用于补齐相应通信插件依赖；其他机器请使用当地安装环境，不要直接照搬 Vanda 路径。

### Windows PowerShell（已有 MinGW 和 Microsoft MPI SDK）

```powershell
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -I "$env:MSMPI_INC" src/mpi.cpp src/mpi_interface.cpp -L "$env:MSMPI_LIB64" -lmsmpi -o heat_mpi.exe
if ($LASTEXITCODE -ne 0) { throw "编译失败" }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\test_mpi.ps1
```

### 验证和输出含义

`--validate` 在计时外收集完整网格，与独立串行参考逐点比较。通过条件含有限值、物理边界精确为零、温度范围检查，以及 `abs(error) <= 1e-12 + 1e-10*abs(reference)`。N=8、K=1 还检查四个解析单步值。

stdout 为一条带表头的 CSV，包含耗时、validation、最大绝对误差和相对 L2 误差；诊断文字写 stderr。未启用验证时 validation=not_run，不代表验证失败，也不能称为已验证。并行耗时取各 rank 的最大迭代时间，排除初始化、收集与验证。

## 7. 已有验证证据

以下来自 B 在本人终端及 Vanda 执行后反馈的结果，而非本文新增的运行声明：

- Windows CPU 正确性脚本的 15 组数值用例及非法输入检查通过。
- 重构后的求解器测试通过；独立暂存缓冲区接口测试在 1、3、4 个进程下通过。
- Vanda 作业 `1536975`：单节点 1、2、4 进程试跑，18 条含预热记录均验证通过。
- Vanda 作业 `1537163`：免费资源池双节点八进程试跑完成。
- Vanda 作业 `1541976`：项目队列双节点八进程试跑完成，Exit_status=0；所展示的六组跨节点正确性检查均通过，最大绝对误差和相对 L2 误差为 0。

这些证据验证 CPU MPI 和通信接口；不代表 C 的 CUDA 程序已经通过验证。两组较大网格计时没有逐次运行串行参考，不应混同为大网格逐点验证证据。

## 8. 仍待完成

- B：正式 Strong/Weak Scaling 至 256 核的数据采集与检查、分析绘图脚本和性能解释。正式作业已给出提交方式，本交付时尚未收到其完成结果。
- 大网格四进程偏慢在两次试跑中重复出现，原因尚未确认；不得删除该点或提前宣称环境修复已解决此现象。
- CPU MPI 的可选温度网格文件输出 `--output` 尚未实现。
- A、C：正式串行/公共工具、CUDA 实现与集成、三版本交叉验证及各自实验，按原分工继续开展。
- 整合阶段统一 Makefile、README 和最终报告；B 提供 MPI 部分内容。

B 的性能实验未结束，不阻碍 A 的开发或 C 开始集成。C 若发现 MPI 模块本身的缺陷，由 B 修复；设备内存、Kernel、主机暂存复制和 MPI+CUDA 同步由 C 负责。

## 9. 发给组员时的版本信息

推送本次交付文档与源代码后，使用 `git rev-parse HEAD` 获取对应提交号，发给 A、C。上述集群作业并未自动证明未来提交的所有改动也经过测试；交付应以明确提交号关联测试记录。不要在程序行为发生变化后继续沿用旧测试结论。
